from typing import List, TypedDict, Any


class GraphState(TypedDict):
    """
    Represents the state of a graph,
    Attributes:
        question:question
        generation: LLM generation
        web_search: wether to use web search or not
        documents: list of documents
    """
    question: str
    generation: str
    web_search: bool
    documents: List[str]