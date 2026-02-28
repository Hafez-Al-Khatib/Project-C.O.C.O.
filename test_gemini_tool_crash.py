import asyncio
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool

load_dotenv()

@tool
def sql_engine(query: str) -> str:
    """Mock SQL tool"""
    return "Mock Result"

async def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, google_api_key=api_key)
    llm_with_tools = llm.bind_tools([sql_engine])
    
    sys_msg = SystemMessage(content="You are a helpful AI.")
    
    # Simulate Gemini answering with MULTIPLE tool calls in one AIMessage
    messages = [
        sys_msg,
        HumanMessage(content="Graph the lift for the Chimney & Latte combo"),
        AIMessage(
            content="",
            tool_calls=[
                {'name': 'sql_engine', 'args': {'query': "SELECT 1"}, 'id': 'call_1'},
                {'name': 'sql_engine', 'args': {'query': "SELECT 2"}, 'id': 'call_2'}
            ]
        ),
        ToolMessage(content="SQL Error 1", tool_call_id="call_1"),
        ToolMessage(content="SQL Error 2", tool_call_id="call_2")
    ]
    
    try:
        response = llm_with_tools.invoke(messages)
        print("SUCCESS MULTI-TOOL:", response)
    except Exception as e:
        print("CRASH MULTI-TOOL:", e)

asyncio.run(main())
