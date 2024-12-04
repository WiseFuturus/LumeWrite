from tenacity import retry, stop_after_attempt, wait_fixed
from typing import List
from core.prompts import ARTICLE_PROMPT,POLISH_PROMPT
from langchain.docstore.document import Document
from pydantic import BaseModel
from core.faiss import FAISSStore
from langchain.chains import LLMChain

class AnswerWithSources(BaseModel):
    answer: str
    sources: List[Document]


@retry(stop=stop_after_attempt(5), wait=wait_fixed(1))  # 尝试3次，每次等待2秒
def generate_section(query,vectorstore,llm,embeddings):

    chain = LLMChain(llm=llm, prompt=ARTICLE_PROMPT)
    relevant_docs = FAISSStore(embeddings=embeddings).query_faiss(vectorstore,query)

    formatted_docs = [
        {
            "metadata": doc.metadata,
            "page_content": doc.page_content
        }
        for doc in relevant_docs
    ]

    formatted_docs_str = "\n".join([
        f"Source: {doc['metadata']['name']}\nContent: {doc['page_content']}"
        for doc in formatted_docs
    ])

    input_data = {
        "outline": query,
        "summaries": formatted_docs_str,
    }

    content = chain.run(input_data)

    return {
        "content": content,
        "sources": formatted_docs
    }


@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
def polish_content(article,llm):

    chain = LLMChain(llm=llm, prompt=POLISH_PROMPT)

    input_data = {
        "article": article,
    }

    article = chain.run(input_data)

    return article