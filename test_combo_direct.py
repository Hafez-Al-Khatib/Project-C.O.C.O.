import asyncio
import os
import sys
from dotenv import load_dotenv

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent.react_agent import build_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

async def main():
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, google_api_key=api_key)
    app = build_react_agent(llm)
    
    msgs = [HumanMessage(content="What combo goes best with CHIMNEY CAKE?")]
    try:
        async for ev in app.astream_events({"messages": msgs}, version="v2"):
            if ev["event"] == "on_tool_start":
                print(f"TOOL START: {ev['name']}", flush=True)
            elif ev["event"] == "on_tool_end":
                print(f"TOOL END: {ev['name']}", flush=True)
            elif ev["event"] == "on_chat_model_stream":
                chunk = ev["data"]["chunk"]
                if hasattr(chunk, "content") and chunk.content and not chunk.tool_calls:
                    c = chunk.content
                    if isinstance(c, str):
                        print(c, end="", flush=True)
    except Exception as e:
        import traceback
        print(f"\nCRASH: {e}", flush=True)
        traceback.print_exc()

asyncio.run(main())
