from dotenv import load_dotenv
load_dotenv()
from ingestion import retriever
from graph.chains.retrieval_grader import GradeDocuments, retrieval_grader


def test_retrieval_grader_as_yes() -> None:
    question = "What is the main topic of the article?"
    docs = retriever.invoke(question)
    doc_txt = docs[0].page_content
    res: GradeDocuments = retrieval_grader.invoke(
        {"question": question, "document": doc_txt}
    )
    assert res.binary_score == "yes"