from typing import Any, Dict, List, Optional
from graph.state import GraphState
from graph.chains.generation import generation_chain


def generate(state: GraphState) -> Dict[str, Any]:
    print("generate")
    question = state["question"]
    documents = state["documents"]
    generation = generation_chain.invoke(
        {"question": question, "context": documents}
    )

    return {
        "question": question,
        "documents": documents,
        "generation": generation,
    }