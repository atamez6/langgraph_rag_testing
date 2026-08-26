from dotenv import load_dotenv
load_dotenv()
from ingestion import retriever
from graph.chains.retrieval_grader import GradeDocuments, retrieval_grader
from graph.chains.generation import generation_chain
from ingestion import retriever
from pprint import pprint


def test_retrieval_grader_as_yes() -> None:
    question = "system memory"
    docs = retriever.invoke(question)
    doc_txt = docs[0].page_content
    res: GradeDocuments = retrieval_grader.invoke(
        {"question": "What is the main topic of the article?", "document": doc_txt}
    )
    assert res.binary_score == "yes"

def test_retrieval_grader_as_no() -> None:
    question = "system memory"
    docs = retriever.invoke(question)
    doc_txt = docs[0].page_content
    res: GradeDocuments = retrieval_grader.invoke(
        {"question": "What pizza?", "document": doc_txt}
    )
    assert res.binary_score == "no"


def test_generation_chain() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    generation = generation_chain.invoke(
        {"question": question, "context": docs}
    )
    pprint(generation)