import asyncio
import sys
from typing import TypedDict, Optional, Annotated
from dotenv import load_dotenv, find_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langgraph.graph import StateGraph, END
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from rag import rag_answer, _is_rate_limited, RATE_LIMIT_MESSAGE
from langchain_core.tools import tool
from datetime import date
import os
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

load_dotenv(find_dotenv())

# Must match the embedding model that built lc_chroma_db (see rag.py)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
chroma_collection = Chroma(persist_directory=os.getenv("PERSIST_DIR", "./lc_chroma_db"), embedding_function=embeddings)

#LLM SetUp
llm=ChatOpenAI(model="gpt-4o-mini")
# More results = more chances of a snippet with concrete figures, not just generic teaser text
web_search_tool=DuckDuckGoSearchRun(api_wrapper=DuckDuckGoSearchAPIWrapper(max_results=8))


# STATE DEFINITION (one combined class)
class AgentState(TypedDict):
    query:str
    retrieved_chunks:Optional[list]
    top_score:Optional[float]
    answer:Optional[str]
    route: Optional[str]           # This will hold "rag_answer"/"web_search"/"clarify"
    messages:Annotated[list,add_messages]    # New field for tool-calling loop


# TOOL DEFINITION
@tool
def get_todays_date()->str:
    """Use this to get today's current date. Call this whenever the user asks what
    today's date is or asking something depends on knowing the current date
    (e.g., 'how many days until X'). Takes no input.
    Return the date as a string in YYYY-MM-DD format."""
    return str(date.today())

@tool
def retrieve_docs(query:str)->str:
    """Use this to answer questions about the internal knowledge base (e.g., email
    marketing, segmentation strategies, Marketo/Eloqua topics). Retrieves the most
    relevant chunks from the vector store so you can ground your answer in them.
    Takes the user's question as input. Returns the raw retrieved text."""
    results=chroma_collection.similarity_search_with_relevance_scores(query,k=3)
    return "\n\n".join(doc.page_content for doc,_ in results)


# BIND TOOLS TO LLM
llm_with_tools=llm.bind_tools([get_todays_date,retrieve_docs])

# NODE FUNCTIONS
def call_model_node(state:AgentState) -> AgentState:
    try:
        response=llm_with_tools.invoke(state["messages"])
    except Exception as exc:
        if _is_rate_limited(exc):
            response=AIMessage(content=RATE_LIMIT_MESSAGE)
        else:
            raise
    return {**state,"messages":[response]}

def has_tool_call(state:AgentState)->str:
    last_message=state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return "end"



def retrieve_node(state:AgentState)->AgentState:
    results=chroma_collection.similarity_search_with_relevance_scores(state["query"],k=3)
    top_score=results[0][1] # relevance score - higher means more similar
    retrieved_chunks=[doc.page_content for doc,_ in results]
    return {**state,"retrieved_chunks":retrieved_chunks,"top_score":top_score}

CONFINDENCE_THRESHOLD=0.65

def route_decision(state:AgentState)->str:
    if len(state["query"].split())<4:
        return "clarify"
    if state["top_score"]>=CONFINDENCE_THRESHOLD:
        return "rag_answer"
    else:
        return "web_search"

def rag_answer_node(state:AgentState)->AgentState:
    answer,_=asyncio.run(rag_answer(state["query"]))
    return {**state,"answer":answer,"route":"rag_answer"}

def web_search_node(state:AgentState)->AgentState:
    result=web_search_tool.invoke(state["query"])
    return {**state,"answer":result,"route":"web_search"}

def clarify_node(state:AgentState)->AgentState:
    return {**state,"answer":"Could you clarify what you'd like to know more about?","route":"clarify"}

# GRAPH WIRING : Day10 stand alone tool-calling graph (test this in isolation first)
tool_graph=StateGraph(AgentState)
tool_node=ToolNode([get_todays_date,retrieve_docs])

tool_graph.add_node("call_model",call_model_node)
tool_graph.add_node("tools",tool_node)
tool_graph.set_entry_point("call_model")
tool_graph.add_conditional_edges(
    "call_model",
    has_tool_call,
    {"tools":"tools","end":END}
    )
tool_graph.add_edge("tools","call_model")

checkpointer=MemorySaver()
app_tools=tool_graph.compile(checkpointer=checkpointer)    # <-- separate compiled app, just for today's test


# GRAPH WIRING
# Existing router graph
graph=StateGraph(AgentState)

graph.add_node("retrieve",retrieve_node)
graph.add_node("rag_answer",rag_answer_node)
graph.add_node("web_search",web_search_node)
graph.add_node("clarify",clarify_node)

graph.set_entry_point("retrieve")

graph.add_conditional_edges(
    "retrieve",  #afterr this node runs...
    route_decision, # ... calls this function...
    {
        "rag_answer":"rag_answer",
        "web_search":"web_search",
        "clarify":"clarify",
    }
)

graph.add_edge("rag_answer",END)
graph.add_edge("web_search",END)
graph.add_edge("clarify",END)

app=graph.compile(checkpointer=checkpointer)


def message_text(message)->str:
    # .content is a plain string for simple replies (and our rate-limit fallback),
    # but some providers/response types return a list of {"type":"text","text":...} parts.
    if isinstance(message.content,list):
        return "".join(part.get("text","") if isinstance(part,dict) else str(part) for part in message.content)
    return message.content

if __name__=="__main__":
    config = {"configurable": {"thread_id": "conv-1"}}
    # Turn 1 — RAG question
    r1 = app_tools.invoke(
        {"messages": [HumanMessage(content="What email platforms are mentioned in the docs for segmentation?")]},
        config
    )
    print(message_text(r1["messages"][-1]))

    # Turn 2 — follow-up that only makes sense with memory of turn 1
    r2 = app_tools.invoke(
        {"messages": [HumanMessage(content="Which of those did you mention first?")]},
        config
    )
    print(message_text(r2["messages"][-1]))

    # Turn 3 — should trigger the tool
    r3 = app_tools.invoke(
        {"messages": [HumanMessage(content="What's today's date?")]},
        config
    )
    print(message_text(r3["messages"][-1]))

    # Turn 4 — same question as turn 2, but switch thread_id mid-conversation
    config_broken = {"configurable": {"thread_id": "conv-2"}}
    r4 = app_tools.invoke(
        {"messages": [HumanMessage(content="Which of those did you mention first?")]},
        config_broken
    )
    print(message_text(r4["messages"][-1])) 