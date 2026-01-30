# checking Hallucination of the LLM in answer
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from pydantic import BaseModel, Field
import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langsmith import traceable
from langchain_ollama import ChatOllama
from config import MISTRAL_CHAT_MODEL, GROQ_CHAT_MODEL, OLLAMA_CHAT_MODEL

load_dotenv()

 
# API models
# verification_model = ChatMistralAI(model=MISTRAL_CHAT_MODEL)
verification_model = ChatGroq(model=GROQ_CHAT_MODEL)

# local model 
# verification_model = ChatOllama(model=OLLAMA_CHAT_MODEL)


class Verification(BaseModel):
    hallucination: bool = Field(..., description='False if the answer fully supported by the context document, if did not then True')
    confidence: float = Field(..., ge=0.0, le=1.0, description='Confidence score (0-1) in the hallucination assessment')
    description: str = Field(description='Short (Max 40 words) explanation of why the response is or is not hallucinated')


structured_llm = verification_model.with_structured_output(Verification)


# fetching only required data from all the list containing data
@traceable
def get_verification_data(messages: list) -> dict:
    ai_response = None
    user_query = None
    context_docs = set()

    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            ai_response = msg.content

            for i in reversed(messages):
                if isinstance(i, HumanMessage):
                    user_query = i.content
                    break

            for j in reversed(messages):
                if isinstance(j, ToolMessage) and j.content:
                    docs_str = j.content  # in string format

                    try:
                        docs_list = json.loads(docs_str)  # converting into list
                        for doc in docs_list:
                            if isinstance(doc, dict):
                                context_docs.add(json.dumps(doc))
                            else:
                                context_docs.add(doc)
                    except Exception as e:
                        print(f"Unexpected content in hallucination's tool message (in fetching section): {e}")
                        context_docs.add(str(docs_str))
            break

    return {
        "user_query": user_query,
        "ai_response": ai_response,
        "context_docs": context_docs
    }


VERIFICATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a fact-checking assistant. Your task is to verify if the AI's response is 
    fully supported by the provided context documents.
    Instructions:
    - Compare the AI's response against the context documents
    - Be strict - if it's not in the documents, it's hallucination
    Remember: The AI might rephrase or add humour, which is acceptable as long as the meaning matches the context.
    """),
    ("human", 
     """USER QUERY: {user_query}
        
        AI RESPONSE: {ai_response}
        
        CONTEXT DOCUMENTS (for verification):
        {context_docs}
        
        Output your assessment in the specified format.""")
])


# generating final report of hallucinaton using LLM
@traceable
def verification_report(data: list) -> dict:
    messages = get_verification_data(data)

    query = messages['user_query']
    ai_answer = messages["ai_response"]
    context_docs = messages['context_docs']

    if not (query and ai_answer and context_docs):

        return {
            'hallucination': None,
            "confidence": 0.8,
            "description": "Insufficient data for verification (probably no document fetched)",
            "user_query": query,
            "ai_response": ai_answer,
            "context_used": list(context_docs)
        }

    prompt = VERIFICATION_PROMPT.format_messages(
        user_query=query,
        ai_response=ai_answer,
        context_docs=context_docs
    )

    response = structured_llm.invoke(prompt)

    return {
        "hallucination": response.hallucination,
        "confidence": response.confidence,
        "description": response.description,
        "user_query": query,
        "ai_response": ai_answer,
        "context_used": list(context_docs)
    }


# giving format to final output 
def formatted_hallucination_report(report):
    formating = """
    1. Hallucinating(T/F): {hallucination}
    2. Confidence: {confidence} %
    3. Description: {description}\n
    Document:{context_document}   
    """

    formatted_report = formating.format(
        hallucination=report['hallucination'],
        confidence=float(report['confidence']) * 100,
        description=report['description'],
        context_document=report['context_used']
    )

    return formatted_report
