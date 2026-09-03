from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableSequence
from langchain_openai import ChatOpenAI


class GradeAnswer(BaseModel):
    binary_score: bool = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )


llm = ChatOpenAI(temperature=0)
structured_llm_grader = llm.with_structured_output(GradeAnswer)

system = """You are a grader assessing whether an answer addresses / resolves a question.
    Give a binary score 'yes' or 'no'. 'Yes' means that the answer resolves the question."""

answer_prompt = ChatPromptTemplate.from_messages([
    ("system", system),
    ("user", "Given the question {{question}} and the answer {{answer}}, determine if the answer addresses the question.")
])

answer_grader : RunnableSequence = answer_prompt | structured_llm_grader