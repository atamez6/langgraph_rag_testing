from typing import List
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama

class Routerquery(BaseModel):
    """Route a user query to the most relevant datasource"""
    datasource : Literal["vectorstore","websearch"] = Field(
        description="Given a user question choose to route it to web search or a vectorstore"
    )

llm = ChatOllama(temperature=0)
structured_llm_output = llm.with_structured_output(routerquery=Routerquery)

system = """You are an expert at routing a user question to a vectorstore or web search.
The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.
Use the vectorstore for questions on these topics. For all else, use web-search."""

route_prompt = ChatPromptTemplate.from_messages(
    [("system", system),
     ("human", "{question}")]
)

question_router = route_prompt | structured_llm_output