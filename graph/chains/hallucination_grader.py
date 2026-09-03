from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableSequence
from langchain_ollama import ChatOllama

llm = ChatOllama(model= "qwen3",temperature=0)

class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer"""

    binary_score : bool = Field(..., description="Answer is grounded in the facts, 'yes' or 'no'")


structured_llm_grader = llm.with_structured_output(GradeHallucinations)


system = """you are a grader assesing whether an llm generation is grounded in / supported by the facts provided. and you should provide a binary score indicating if the answer is hallucinated or not, supported by the facts and documents."""


hallucination_prompt = ChatPromptTemplate.from_messages([
    ("system", system),
    ("user", "Given the facts and {{documents}}, determine if the following {{generation}} is hallucinated: {{answer}}")
])



hallucination_grader : RunnableSequence = hallucination_prompt | structured_llm_grader