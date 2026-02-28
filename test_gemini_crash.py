import asyncio
import os
from dotenv import load_dotenv
from agent.react_agent import stream_llm_react

load_dotenv()

async def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    messages = [
        {"role": "user", "content": "Graph the lift for the Chimney & Latte combo"},
        {"role": "user", "content": "Graph the lift for the Chimney & Latte combo"},
        {"role": "assistant", "content": ""}
    ]
    
    try:
        async for event in stream_llm_react(messages, api_key):
            print(event)
    except Exception as e:
        print("EXCEPTION CAUGHT:", e)

asyncio.run(main())
