from enum import StrEnum
from typing import Any
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import SystemMessagePromptTemplate, ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
from openai import BaseModel
from pydantic import SecretStr, Field
from task._constants import DIAL_URL, API_KEY
from task.token_tracker import TokenTracker
from task.user_client import UserClient

# Before implementation open the `api_based_grounding.png` to see the flow of app

QUERY_ANALYSIS_PROMPT = """You are a query analysis system that extracts search parameters from user questions about users.

## Available Search Fields:
- **name**: User's first name (e.g., "John", "Mary")
- **surname**: User's last name (e.g., "Smith", "Johnson")
- **email**: User's email address (e.g., "john@example.com")

## Instructions:
1. Analyze the user's question and identify what they're looking for
2. Extract specific search values mentioned in the query
3. Map them to the appropriate search fields
4. If multiple search criteria are mentioned, include all of them
5. Only extract explicit values - don't infer or assume values not mentioned

## Examples:
- "Who is John?" → name: "John"
- "Find users with surname Smith" → surname: "Smith"
- "Look for john@example.com" → email: "john@example.com"
- "Find John Smith" → name: "John", surname: "Smith"
- "I need user emails that filled with hiking" → No clear search parameters (return empty list)

## Response Format:
{format_instructions}
"""

SYSTEM_PROMPT = """You are a RAG-powered assistant that assists users with their questions about user information.

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
"""

USER_PROMPT = """## RAG CONTEXT:
{context}

## USER QUESTION:
{query}"""


# 1. Create AzureChatOpenAI client
# 2. Create UserClient
llm = AzureChatOpenAI(
    temperature=0,
    azure_deployment="gpt-4o",
    azure_endpoint=DIAL_URL,
    api_key=SecretStr(API_KEY),
    api_version="",
)
user_client = UserClient()
token_tracker = TokenTracker()

# Now we need to create pydentic models that will be user for search and their JSON schema will be passed to LLM by
# langchain. In response from LLM we expect to get response in such format (JSON by JSON Schema)
# 1. SearchField class, extend StrEnum and has constants: name, surname, email
# 2. Create SearchRequest, extends pydentic BaseModel and has such fields:
#       - search_field (enum from above), also you can provide its `description` that will be provided with JSON Schema
#         to LLM that model will be better understand what you expect there
#       - search_value, its string, sample what we expect here is some name, surname or email to make search
# 3. Create SearchRequests, extends pydentic BaseModel and has such fields:
#       - search_request_parameters, list of SearchRequest, by default empty list
class SearchField(StrEnum):
    name = "name"
    surname = "surname"
    email = "email"
class SearchRequest(BaseModel):
    search_field: SearchField
    search_value: str
class SearchRequests(BaseModel):
    search_request_parameters: list[SearchRequest] = Field(default_factory=list)

def retrieve_context(user_question: str) -> list[dict[str, Any]]:
    """Extract search parameters from user query and retrieve matching users."""
    # 1. Create PydanticOutputParser with `pydantic_object=SearchRequests` as `parser`
    # 2. Create messages array with:
    #       - use SystemMessagePromptTemplate and from template generate system message from QUERY_ANALYSIS_PROMPT
    #       - user message
    # 3. Generate `prompt`: `ChatPromptTemplate.from_messages(messages=messages).partial(format_instructions=parser.get_format_instructions())`
    # 4. Invoke it: `(prompt | llm_client | parser).invoke({})` as `search_requests: SearchRequests` (you are using LCEL)
    # 5. If `search_requests` has `search_request_parameters`:
    #       - create `requests_dict`
    #       - iterate through searched parameters and:
    #           - add to `requests_dict` the `search_request.search_field.value` as key and `search_request.search_value` as value
    #       - print `requests_dict`
    #       - search users (**requests_dict) with `user_client`
    #       - return users that you found
    # 6. Otherwise print 'No specific search parameters found!' and return empty array
    search_field_parser = PydanticOutputParser(pydantic_object=SearchRequests)
    messages = [
        SystemMessagePromptTemplate.from_template(QUERY_ANALYSIS_PROMPT).format(format_instructions=search_field_parser.get_format_instructions()),
        HumanMessage(content=user_question),
    ]
    prompt = ChatPromptTemplate.from_messages(messages=messages)

    # Refactored to not use chain function call.
    intermediate_prompt = prompt.invoke({})
    intermediate_llm = llm.invoke(intermediate_prompt)
    search_requests = search_field_parser.invoke(intermediate_llm)
    total_tokens = intermediate_llm.usage_metadata['total_tokens'] if intermediate_llm.usage_metadata else 0
    token_tracker.add_tokens(total_tokens)

    # search_requests = (prompt | llm | search_field_parser).invoke({})

    if search_requests.search_request_parameters:
        requests_dict = {request.search_field.value: request.search_value for request in search_requests.search_request_parameters}
        print(f"Requests dict: {requests_dict}")
        users = user_client.search_users(**requests_dict)
        return users
    print("No specific search parameters found!")
    return []


def augment_prompt(user_question: str, context: list[dict[str, Any]]) -> str:
    """Combine user query with retrieved context into a formatted prompt."""
    # 1. Prepare context from users JSONs in the same way as in `no_grounding.py` `join_context` method (collect as one string)
    # 2. Make augmentation for USER_PROMPT
    # 3. print augmented prompt
    # 3. return augmented prompt
    context_string = "\n\n".join([f"User:\n  {user['name']}\n  {user['surname']}\n  {user['email']}  {user['about_me']}" for user in context])
    augmented_prompt = USER_PROMPT.format(context=context_string, query=user_question)
    print(f"Augmented prompt: {augmented_prompt}")
    return augmented_prompt


def generate_answer(augmented_prompt: str) -> str:
    """Generate final answer using the augmented prompt."""
    # 1. Create messages array with:
    #       - SYSTEM_PROMPT
    #       - augmented_prompt
    # 2. Generate response, use invoke method with llm_client
    # 3. Return response content
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=augmented_prompt),
    ]
    prompt = ChatPromptTemplate.from_messages(messages=messages)
    response = (prompt | llm).invoke({})
    total_tokens = response.usage_metadata['total_tokens'] if response.usage_metadata else 0
    token_tracker.add_tokens(total_tokens)
    return str(response.content[0]) if isinstance(response.content, list) else str(response.content) if response.content else "No response content"


def main():
    print("Query samples:")
    print(" - I need user emails that filled with hiking and psychology")
    print(" - Who is John?")
    print(" - Find users with surname Adams")
    print(" - Do we have smbd with name John that love painting?")

    # 1. Create infinite loop
    # 2. Get input from console as `user_question`
    # 3. retrieve context
    # 4. if context is present:
    #       - make augmentation
    #       - generate answer with augmented prompt
    # 5. Otherwise print `No relevant information found`
    while True:
        user_question = input("> ").strip()
        if user_question.lower() in ['quit', 'exit']:
            break
        context = retrieve_context(user_question)
        if context:
            augmented_prompt = augment_prompt(user_question, context)
            answer = generate_answer(augmented_prompt)
            print(answer)
        else:
            print("No relevant information found")
        print(f"Token tracker: {token_tracker.get_summary()}")
        # Used tokens for "Who is John?" query:
            # {'total_tokens': 2322, 'batch_count': 2, 'batch_tokens': [546, 1776]}



if __name__ == "__main__":
    main()


# The problems with API based Grounding approach are:
#   - We need a Pre-Step to figure out what field should be used for search (Takes time)
#   - Values for search should be correct (✅ John -> ❌ Jonh)
#   - Is not so flexible
# Benefits are:
#   - We fetch actual data (new users added and deleted every 5 minutes)
#   - Costs reduce