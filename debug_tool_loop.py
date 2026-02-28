import asyncio
import os
import sys
from dotenv import load_dotenv

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent.react_agent import build_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage

async def main():
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, google_api_key=api_key)
    app = build_react_agent(llm)
    
    messages = [HumanMessage(content="Graph the lift for the Chimney & Latte combo")]
    
    try:
        async for event in app.astream_events({"messages": messages}, version="v2"):
            kind = event["event"]
            if kind == "on_chat_model_start":
                input_data = event["data"].get("input")
                if isinstance(input_data, list):
                    msgs = input_data
                elif isinstance(input_data, dict):
                    msgs = input_data.get("messages", [])
                else:
                    msgs = []
                for i, m in enumerate(msgs):
                    ct = getattr(m, 'content', str(m))
                    tc = getattr(m, 'tool_calls', [])
                    print(f"  MSG[{i}] type={type(m).__name__} content={repr(ct)[:80]} tool_calls={bool(tc)}")
                print("---")
            elif kind == "on_tool_start":
                print(f"  TOOL START: {event['name']}")
            elif kind == "on_tool_end":
                print(f"  TOOL END: {event['name']}")
    except Exception as e:
        import traceback
        print(f"\nCRASH: {e}")
        traceback.print_exc()

asyncio.run(main())
