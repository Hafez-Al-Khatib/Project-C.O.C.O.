import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

from agent.react_agent import stream_llm_react

async def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    messages = [
        {"role": "user", "content": "How many employees do I need in Jnah for today?"}
    ]
    
    try:
        async for event in stream_llm_react(messages, api_key):
            print(event)
    except Exception as e:
        import traceback
        traceback.print_exc()

asyncio.run(main())
