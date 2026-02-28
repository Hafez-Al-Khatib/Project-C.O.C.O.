import asyncio
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")

from agent.react_agent import build_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage

async def main():
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, google_api_key=api_key)
    app = build_react_agent(llm)
    
    # SvelteKit sends this array: [Human, Assistant="", Human]
    # In my patch, I collapse this to a single HumanMessage saying PREV CONTEXT + QUERY
    messages = [
        HumanMessage(content="PREVIOUS CONVERSATION CONTEXT:\nUser: Graph the lift for the Chimney & Latte combo\n\nACTUAL QUERY:\nGraph the lift for the Chimney & Latte combo")
    ]
    
    try:
        async for event in app.astream_events({"messages": messages}, version="v2"):
            print(event["event"], event.get("name"))
    except Exception as e:
        import traceback
        traceback.print_exc()

asyncio.run(main())
