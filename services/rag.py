from langchain_mistralai import ChatMistralAI
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, MessagesState
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.runnables import RunnableConfig
from database.vector_store import get_retriever
from config import MISTRAL_CHAT_MODEL, OLLAMA_TOOL_MODEL
import os
from dotenv import load_dotenv

load_dotenv()



def chat_model():
    model = ChatMistralAI(model=MISTRAL_CHAT_MODEL)
    # model = ChatOllama(model=OLLAMA_TOOL_MODEL, model_kwargs={"enable_thinking":False})
    return model 



@tool
def rag_tool(query1: str, query2: str = None, query3: str = None,
             retrieved_docs=2, page_content=True, metadata=False, config: RunnableConfig = None) -> list:
    """
    Retrieve relevant information from uploaded PDFs.
    - Always return factual excerpts (with humour).
    - Always include page number citation.
    - If nothing found, say explicitly no answer in documents.
    """

    user_id = config['configurable']['user_id']
    thread_id = config['configurable']['thread_id']
    print(f"User id: {user_id} - Thread_id: {thread_id}")
    
    document = []

    queries = [q for q in [query1, query2, query3] if q]
    for query in queries:
        docs = get_retriever().similarity_search( query=query, k=retrieved_docs, 
            filter={'user_id': user_id, 'thread_id': thread_id}
        )
        for doc in docs:
            extracted_content = doc.page_content
            doc_metadata = doc.metadata
            page_number = doc.metadata['page_label']

            if page_content and ((extracted_content + f" (Page:{page_number})") not in document):
                document.append(extracted_content + f" (Page:{page_number})")

            if metadata and (doc_metadata not in document):
                document.append(doc_metadata)
    
    if len(document) == 0:
        document.append('Sorry! No document found - First upload the document')
        return document

    return document


tools = [rag_tool]
model_with_tool = chat_model().bind_tools(tools)


def chat_node(state: MessagesState):
    message = state['messages']
    response = model_with_tool.invoke(message)
    return {'messages': [response]}


tool_node = ToolNode(tools)

checkpointer = InMemorySaver()

def rag_graph():
    graph = StateGraph(MessagesState)

    graph.add_node('chat_node', chat_node)
    graph.add_node('tools', tool_node)

    graph.add_edge(START, 'chat_node')
    graph.add_conditional_edges('chat_node', tools_condition)
    graph.add_edge('tools', 'chat_node')

    rag_workflow = graph.compile(checkpointer=checkpointer)
    return rag_workflow