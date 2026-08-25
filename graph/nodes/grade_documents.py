from typing import Any, Dict

from graph.chains.retrieval_grader import GradeDocuments, retrieval_grader
from graph.state import GraphState

def grade_documents(state: GraphState) -> Dict[str, Any]:
    """determines wether the retrieved documents are relevant to the question if any document is nto relevant, we will set a flag to run web search
    Args:
        state (dict): The current graph state
    Returns:
        state  (dict): Filtered out irrelevant documents and updated web_search state 
    """

    print("grading documents")
    question = state["question"]
    documents = state["documents"]

    filtered_documents = []
    web_search = False

    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
            )
        grade = score.binary_score
        if grade == "yes": 
            filtered_documents.append(d)
            print(f"Document is relevant to the question: {d.page_content}")
        else:
            print(f"Document is NOT relevant to the question: {d.page_content}")
            web_search = True
            continue
    return {"documents": filtered_documents, "question": question, "web_search": web_search}