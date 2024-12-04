from typing import List
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from core.prompts import QA_PROMPT
from langchain.docstore.document import Document
from pydantic import BaseModel
from core.faiss import FAISSStore


class AnswerWithSources(BaseModel):
    answer: str
    sources: List[Document]


def query_answer(query,vectorstore,llm,embeddings):

    chain = load_qa_with_sources_chain(
        llm=llm,
        chain_type="stuff",
        prompt=QA_PROMPT,
    )

    relevant_docs = FAISSStore(embeddings=embeddings).query_faiss(vectorstore,query)
    result = chain(
        {"input_documents": relevant_docs, "question": query}, return_only_outputs=True
    )

    answer = result["output_text"].split("SOURCES: ")[0]
    return AnswerWithSources(answer=answer, sources=relevant_docs)