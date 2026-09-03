from dotenv import load_dotenv
load_dotenv()
from langgraph.graph import END,StateGraph
from graph.nodes import generate, retrieve, web_search, grade_documents
from graph.consts import RETRIEVE, GRADE_DOCUMENTS, GENERATE, WEBSEARCH
from graph.state import GraphState
from graph.chains.answer_grader import answer_grader
from graph.chains.hallucination_grader import hallucination_grader


def decide_generate(state):
    print("asses graded documents")
    if state["web_search"]:
        print("web search invoken, not all documents are graded, so we will not generate")

        return WEBSEARCH

    else:
        print("all documents are graded, we can generate")
        return GENERATE


def grade_generation_grounded_in_documents_and_question(state: GraphState) -> str:
    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )

    if hallucination_grade := score.binary_score:
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        print("---GRADE GENERATION vs QUESTION---")
        score = answer_grader.invoke({"question": question, "generation": generation})
        if answer_grade := score.binary_score:
            print("---DECISION: GENERATION ADDRESSES THE QUESTION---")
            return "useful"
        else:
            print("Decision: GENERATION DOES NOT ADDRESS THE QUESTION")
            return "not_useful"
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS---")
        return "not_supported"
    


workflow = StateGraph(GraphState)

workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(GENERATE, generate)
workflow.add_node(WEBSEARCH, web_search)

workflow.set_entry_point(RETRIEVE)
workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)
workflow.add_conditional_edges(GRADE_DOCUMENTS, decide_generate, {WEBSEARCH: WEBSEARCH, GENERATE: GENERATE})
workflow.add_conditional_edges(GENERATE, grade_generation_grounded_in_documents_and_question, {"useful": END, "not_useful": WEBSEARCH, "not_supported": GENERATE})
workflow.add_edge(WEBSEARCH,GENERATE)
workflow.add_edge(GENERATE, END)

app= workflow.compile()


app.get_graph().draw_mermaid_png(output_file_path="graph.png")