import asyncio
from typing import Any
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import AzureChatOpenAI
from pydantic import SecretStr
from task._constants import DIAL_URL, API_KEY
from task.token_tracker import TokenTracker
from task.user_client import UserClient

#TODO:
# Before implementation open the `flow_diagram.png` to see the flow of app

BATCH_SYSTEM_PROMPT = """You are a user search assistant. Your task is to find users from the provided list that match the search criteria.

INSTRUCTIONS:
1. Analyze the user question to understand what attributes/characteristics are being searched for
2. Examine each user in the context and determine if they match the search criteria
3. For matching users, extract and return their complete information
4. Be inclusive - if a user partially matches or could potentially match, include them

OUTPUT FORMAT:
- If you find matching users: Return their full details exactly as provided, maintaining the original format
- If no users match: Respond with exactly "NO_MATCHES_FOUND"
- If uncertain about a match: Include the user with a note about why they might match"""

FINAL_SYSTEM_PROMPT = """You are a helpful assistant that provides comprehensive answers based on user search results.

INSTRUCTIONS:
1. Review all the search results from different user batches
2. Combine and deduplicate any matching users found across batches
3. Present the information in a clear, organized manner
4. If multiple users match, group them logically
5. If no users match, explain what was searched for and suggest alternatives"""

USER_PROMPT = """## USER DATA:
{context}

## SEARCH QUERY:
{query}"""


#TODO:
# 1. Create AzureChatOpenAI client
#    hint: api_version set as empty string if you gen an error that indicated that api_version cannot be None
# 2. Create TokenTracker
llm = AzureChatOpenAI(
    temperature=0,
    azure_deployment="gpt-4o",
    azure_endpoint=DIAL_URL,
    api_key=SecretStr(API_KEY),
    api_version="",
)

# Create TokenTracker
token_tracker = TokenTracker()


def join_context(context: list[dict[str, Any]]) -> str:
    # You cannot pass raw JSON with user data to LLM (" sign), collect it in just simple string or markdown.
    # You need to collect it in such way:
    # User:
    #   name: John
    #   surname: Doe
    #   ...
    return "\n".join([f"User:\n  {user['name']}\n  {user['surname']}\n  {user['about_me']}" for user in context])


async def generate_response(system_prompt: str, user_message: str) -> str:
    print("Processing...")
    # 1. Create messages array with system prompt and user message
    # 2. Generate response (use `ainvoke`, don't forget to `await` the response)
    # 3. Get usage (hint, usage can be found in response metadata (its dict) and has name 'token_usage', that is also
    #    dict and there you need to get 'total_tokens')
    # 4. Add tokens to `token_tracker`
    # 5. Print response content and `total_tokens`
    # 5. return response content
    # 1. Create messages array with system prompt and user message
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message),
    ]

    response = await llm.ainvoke(input=messages)

    total_tokens = response.usage_metadata['total_tokens'] if response.usage_metadata else 0
    token_tracker.add_tokens(total_tokens)
    content = str(response.content[0]) if isinstance(response.content, list) else str(response.content) if response.content else "No response content"

    print(f"Response content: {content}")

    return content

async def main():
    print("Query samples:")
    print(" - Do we have someone with name John that loves traveling?")

    user_question = input("> ").strip()
    if user_question:
        print("\n--- Searching user database ---")

        # 1. Get all users (use UserClient)
        # 2. Split all users on batches (100 users in 1 batch). We need it since LLMs have its limited context window
        # 3. Prepare tasks for async run of response generation for users batches:
        #       - create array tasks
        #       - iterate through `user_batches` and call `generate_response` with these params:
        #           - BATCH_SYSTEM_PROMPT (system prompt)
        #           - User prompt, you need to format USER_PROMPT with context from user batch and user question
        # 4. Run task asynchronously, use method `gather` form `asyncio`
        # 5. Filter results on 'NO_MATCHES_FOUND' (see instructions for BATCH_SYSTEM_PROMPT)
        # 5. If results after filtration are present:
        #       - combine filtered results with "\n\n" spliterator
        #       - generate response with such params:
        #           - FINAL_SYSTEM_PROMPT (system prompt)
        #           - User prompt: you need to make augmentation of retrieved result and user question
        # 6. Otherwise prin the info that `No users found matching`
        # 7. In the end print info about usage, you will be impressed of how many tokens you have used. (imagine if we have 10k or 100k users 😅)
    users = UserClient().get_all_users()
    user_batches = [users[i:i+100] for i in range(0, len(users), 100)]
    tasks = [generate_response(BATCH_SYSTEM_PROMPT, USER_PROMPT.format(context=join_context(batch), query=user_question)) for batch in user_batches]
    results = await asyncio.gather(*tasks)
    filtered_results = [result for result in results if "NO_MATCHES_FOUND" not in result]

    print(f"Filtered results: {filtered_results}")

    if filtered_results:
        combined_results = "\n\n".join(filtered_results)
        final_response = await generate_response(FINAL_SYSTEM_PROMPT, combined_results)
        print(f"Final response: {final_response}")
    print(f"Token tracker: {token_tracker.get_summary()}")
    # Used tokens for "Do we have someone with name John that loves traveling?" query:
        # {'total_tokens': 42002, 'batch_count': 12, 'batch_tokens': [4045, 4027, 4002, 4056, 4078, 4060, 4018, 4018, 4041, 4100, 885, 672]}


if __name__ == "__main__":
    asyncio.run(main())


# The problems with No Grounding approach are:
#   - If we load whole users as context in one request to LLM we will hit context window
#   - Huge token usage == Higher price per request
#   - Added + one chain in flow where original user data can be changed by LLM (before final generation)
# User Question -> Get all users -> ‼️parallel search of possible candidates‼️ -> probably changed original context -> final generation