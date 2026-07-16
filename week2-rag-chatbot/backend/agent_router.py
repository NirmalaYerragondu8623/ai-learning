import asyncio
from typing import TypedDict, Optional
from dotenv import load_dotenv, find_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from rag import rag_answer

load_dotenv(find_dotenv())

# Must match the embedding model that built lc_chroma_db (see rag.py)
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
chroma_collection = Chroma(persist_directory="./lc_chroma_db", embedding_function=embeddings)

llm=ChatGoogleGenerativeAI(model="gemini-2.5-flash")
# More results = more chances of a snippet with concrete figures, not just generic teaser text
web_search_tool=DuckDuckGoSearchRun(api_wrapper=DuckDuckGoSearchAPIWrapper(max_results=8))

class AgentState(TypedDict):
    query:str
    retrieved_chunks:Optional[list]
    top_score:Optional[float]
    answer:Optional[str]
    route: Optional[str]           # This will hold "rag_answer"/"web_search"/"clarify"

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
    search_results=web_search_tool.invoke(state["query"])
    response=llm.invoke(
        "Answer the question using the web search context below. "
        "If any specific figures (numbers, dates, prices, etc.) are present anywhere in the "
        "context, state them directly instead of describing that the information exists.\n\n"
        f"Context:\n{search_results}\n\nQuestion:{state['query']}"
    )
    return {**state,"answer":response.content,"route":"web_search"}

def clarify_node(state:AgentState)->AgentState:
    return {**state,"answer":"Could you clarify what you'd like to know more about?","route":"clarify"}

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

app=graph.compile()

result=app.invoke({"query":"Explain in few more lines"})
print({"query":result["query"],"route":result["route"],"answer":result["answer"]})