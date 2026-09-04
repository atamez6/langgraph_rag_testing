from dotenv import load_dotenv
load_dotenv()
from graph.chains.hallucination_grader import hallucination_grader, GradeHallucinations
from ingestion import retriever
from graph.chains.retrieval_grader import GradeDocuments, retrieval_grader
from graph.chains.generation import generation_chain
from ingestion import retriever
from pprint import pprint
from graph.chains.router import question_router

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



def test_hallucination_grader_yes() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    generation = generation_chain.invoke(
        {"question": question, "context": docs}
    )
    res : GradeHallucinations = hallucination_grader.invoke({"documents": docs, "generation": "agent memory a new technique"})
    assert res.binary_score
    pprint(res)

def test_hallucination_grader_no() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    generation = generation_chain.invoke(
        {"question": question, "context": docs}
    )
    res : GradeHallucinations = hallucination_grader.invoke({"documents": docs, "generation": "pizza is good"})
    assert not res.binary_score
    pprint(res)



def test_question_router() -> None:
    question = "agent memory"
    route = question_router.invoke({"question": question})
    pprint(route)