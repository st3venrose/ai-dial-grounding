import asyncio
import json
from typing import Any, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from pydantic import BaseModel, Field, SecretStr

from task._constants import API_KEY, DIAL_URL
from task.user_client import UserClient

# Info about app:
# HOBBIES SEARCHING WIZARD
# Searches users by hobbies and provides their full info in JSON format:
#   Input: `I need people who love to go to mountains`
#   Output:
#     ```json
#       "rock climbing": [{full user info JSON},...],
#       "hiking": [{full user info JSON},...],
#       "camping": [{full user info JSON},...]
#     ```
# ---
# 1. Since we are searching hobbies that persist in `about_me` section - we need to embed only user `id` and `about_me`!
#    It will allow us to reduce context window significantly.
# 2. Pay attention that every 5 minutes in User Service will be added new users and some will be deleted. We will at the
#    'cold start' add all users for current moment to vectorstor and with each user request we will update vectorstor on
#    the retrieval step, we will remove deleted users and add new - it will also resolve the issue with consistency
#    within this 2 services and will reduce costs (we don't need on each user request load vectorstor from scratch and pay for it).
# 3. We ask LLM make NEE (Named Entity Extraction) https://cloud.google.com/discover/what-is-entity-extraction?hl=en
#    and provide response in format:
#    {
#       "{hobby}": [{user_id}, 2, 4, 100...]
#    }
#    It allows us to save significant money on generation, reduce time on generation and eliminate possible
#    hallucinations (corrupted personal info or removed some parts of PII (Personal Identifiable Information)). After
#    generation we also need to make output grounding (fetch full info about user and in the same time check that all
#    presented IDs are correct).
# 4. In response we expect JSON with grouped users by their hobbies.
# ---
# This sample is based on the real solution where one Service provides our Wizard with user request, we fetch all
# required data and then returned back to 1st Service response in JSON format.
# ---
# Useful links:
# Chroma DB: https://docs.langchain.com/oss/python/integrations/vectorstores/index#chroma
# Document#id: https://docs.langchain.com/oss/python/langchain/knowledge-base#1-documents-and-document-loaders
# Chroma DB, async add documents: https://api.python.langchain.com/en/latest/vectorstores/langchain_chroma.vectorstores.Chroma.html#langchain_chroma.vectorstores.Chroma.aadd_documents
# Chroma DB, get all records: https://api.python.langchain.com/en/latest/vectorstores/langchain_chroma.vectorstores.Chroma.html#langchain_chroma.vectorstores.Chroma.get
# Chroma DB, delete records: https://api.python.langchain.com/en/latest/vectorstores/langchain_chroma.vectorstores.Chroma.html#langchain_chroma.vectorstores.Chroma.delete
# ---
# TASK:
# Implement such application as described on the `flow.png` with adaptive vector based grounding and 'lite' version of
# output grounding (verification that such user exist and fetch full user info)

SYSTEM_PROMPT = """
You are a RAG-powered assistant that assists users with their questions about user information.

## Structure of User message:
`RAG CONTEXT` - Retrieved documents relevant to the query.
`USER QUESTION` - The user's actual question.

## Instructions:
- Use information from `RAG CONTEXT` as context when answering the `USER QUESTION`.
- Cite specific sources when using information from the context.
- Answer ONLY based on conversation history and RAG context.
- If no relevant information exists in `RAG CONTEXT` or conversation history, state that you cannot answer the question.
- Be conversational and helpful in your responses.
- When presenting user information, format it clearly and include relevant details.
## Response Format:
{format_instructions}
"""
USER_PROMPT = """## RAG CONTEXT:
{context}

## USER QUESTION:
{query}"""

class HobbiesExtraction(BaseModel):
    """Hobbies extraction model"""
    hobbies: dict[str, list[int]] = Field(
        description="Mapping of hobby name to list of user IDs who match this hobby.",
        default_factory=dict,
    )

class UserHobbiesRAG:
    """User hobbies RAG"""
    def __init__(self, embeddings: AzureOpenAIEmbeddings, llm_client: AzureChatOpenAI, user_client: UserClient):
        self.embeddings = embeddings
        self.llm_client = llm_client
        self.user_client = user_client
        self.vector_store: Optional[Chroma] = None

    async def __aenter__(self):
        await self._init_vector_store()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    async def _init_vector_store(self) -> None:
        """Initialize vector store by fetching all users and creating batches to save in vector store"""
        print('Fetch all users')
        users = self.user_client.get_all_users()
        user_documents = [Document(id=user.get('id'), page_content=self._format_user_content(user)) for user in users]
        user_document_batches = self._create_batches(user_documents, 50)

        self.vector_store = Chroma(collection_name="user_collection", embedding_function=self.embeddings)
        tasks = [
            self.vector_store.aadd_documents(user_document_batch) for user_document_batch in user_document_batches
        ]

        await asyncio.gather(*tasks)
        print('Vector store is READY')

    def _format_user_content(self, user: dict[str, Any]) -> str:
        return f"User:\n id: {user.get('id')},\nAbout user: {user.get('about_me')}\n"

    def _create_batches(self, document_list: list[Any], batch_size: int = 50) -> list[list[Any]]:
        """Create batches from list"""
        return [document_list[i:i + batch_size] for i in range(0, len(document_list), batch_size)]

    async def _sync_vector_store(self) -> None:
        """Sync vector store with current users"""
        if self.vector_store is None:
            raise RuntimeError("Vectorstore is not initialized")

        db_users = self.user_client.get_all_users()
        vector_store_users = self.vector_store.get()
        db_user_by_ids = {int(user['id']): user for user in db_users}
        current_db_user_ids = set(db_user_by_ids.keys())
        vector_store_user_ids = set(int(id) for id in vector_store_users.get('ids', []))

        new_ids = current_db_user_ids - vector_store_user_ids
        deleted_ids = vector_store_user_ids - current_db_user_ids

        if new_ids:
            new_documents = [
                Document(
                    id=user_id,
                    page_content=self._format_user_content(db_user_by_ids[user_id]),
                ) for user_id in new_ids]
            new_document_batches = self._create_batches(new_documents)
            tasks = [self.vector_store.aadd_documents(user_document_batch) for user_document_batch in new_document_batches]
            await asyncio.gather(*tasks)

        if deleted_ids:
            self.vector_store.delete(ids=[str(id) for id in deleted_ids])

    def augment_user_prompt(self, relevant_context: str, query: str) -> str:
        """Augment user prompt with relevant context"""
        return USER_PROMPT.format(context=relevant_context, query=query)

    async def get_context_from_vector_store(self, user_prompt: str, k: int = 20, score: float = 0.2) -> str:
        """Get context from vector store"""
        if self.vector_store is None:
            raise RuntimeError("Vectorstore is not initialized")

        await self._sync_vector_store()

        relevant_docs: list[tuple[Document, float]] = self.vector_store.similarity_search_with_relevance_scores(user_prompt, k=k, score_threshold=score)
        context: list[str] = []

        for doc, score in relevant_docs:
            context.append(doc.page_content)

        text_context = "\n\n".join(context)

        return text_context

    def gather_users_based_on_hobbies(self, augmented_prompt: str) -> HobbiesExtraction:
        """Gather users based on hobbies"""
        parser = PydanticOutputParser(pydantic_object=HobbiesExtraction)
        messages = [
            SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT).format(
                format_instructions=parser.get_format_instructions()
            ),
            HumanMessage(content=augmented_prompt),
        ]
        prompt = ChatPromptTemplate.from_messages(messages=messages)

        return (prompt | self.llm_client | parser).invoke({})

class OutputGrounder:
    """Checks if users exist and fetches full user info"""
    def __init__(self, user_client: UserClient):
        self.user_client = user_client

    async def ground_llm_answer(self, llm_answer: HobbiesExtraction) -> None:
        """Ground LLM answer by checking if users exist and fetching full user info"""
        for hobby in llm_answer.hobbies:
            print(f'Hobby: {hobby}')
            users = await self._get_users(llm_answer.hobbies[hobby])
            print(json.dumps(users, indent=2))
            print('-' * 100)

    async def _get_users(self, user_ids: list[int]) -> list[dict[str, Any]]:
        tasks = [self._get_user(id) for id in user_ids]
        results = await asyncio.gather(*tasks)

        return [user for user in results if user is not None]

    async def _get_user(self, user_id: int) -> Optional[dict[str, Any]]:
        """Get user by ID, returns None if user not found (halucinated id or record was deleted)"""
        try:
            return await self.user_client.get_user(user_id)
        except Exception as e:
            if '404' in str(e):
                return None
            raise e

async def main():
    embeddings = AzureOpenAIEmbeddings(
        model="text-embedding-3-small-1",
        azure_endpoint=DIAL_URL,
        api_key=SecretStr(API_KEY),
        dimensions=384,
    )
    llm_client = AzureChatOpenAI(
        temperature=0,
        azure_deployment="gpt-4o",
        azure_endpoint=DIAL_URL,
        api_key=SecretStr(API_KEY),
        api_version="",
    )
    user_client = UserClient()
    output_grounder = OutputGrounder(user_client)

    async with UserHobbiesRAG(embeddings, llm_client, user_client) as rag:
        while True:
            print("Query samples:")
            print(" - I need people who love to go to mountains")
            print(" - I need people who like hiking and camping")
            print(" - I need fans of painting and music")
            user_prompt = input("> ").strip()
            if user_prompt.lower() in ['quit', 'exit']:
                break

            context = await rag.get_context_from_vector_store(user_prompt)
            augmented_prompt = rag.augment_user_prompt(context, user_prompt)
            llm_answer = rag.gather_users_based_on_hobbies(augmented_prompt)
            await output_grounder.ground_llm_answer(llm_answer)

asyncio.run(main())