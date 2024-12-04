import os
from dotenv import load_dotenv
import hashlib
from tenacity import retry, stop_after_attempt, wait_fixed
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

load_dotenv()

class FAISSStore:
    def __init__(self,embeddings):
        self.vectorstore = None
        self.embeddings = embeddings

    def generate_chunk_id(self,chunk_content: str) -> str:
        sha256_hash = hashlib.sha256(chunk_content.encode('utf-8'))
        return sha256_hash.hexdigest()   

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
    def initialize_vectorstore(self,file_list):

        chunked_docs = []
        for file in file_list:

            for doc in file.docs:
                text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                    model_name="gpt-3.5-turbo",
                    chunk_size=300,
                    chunk_overlap=0,
                )

                chunks = text_splitter.split_text(doc.page_content)
                for i, chunk in enumerate(chunks):
                    unique_id = self.generate_chunk_id(chunk)
                    doc = Document(
                        page_content=chunk,
                        metadata={
                            "name":file.name,
                            "page": doc.metadata.get("page", 1),
                            "chunk": i + 1,
                            "source": f"{file.name}-{doc.metadata.get('page', 1)}-{i + 1}",
                            "unique_id": unique_id,
                        },
                    )
                    chunked_docs.append(doc)

        chunked_file = file.copy()
        chunked_file.docs = chunked_docs

        self.vectorstore = FAISS.from_documents(
            documents=chunked_file.docs,
            embedding=self.embeddings,
        )

        return self.vectorstore,chunked_docs

        
    def query_faiss(self,vectorstore,query):

        relevant_docs = vectorstore.similarity_search(query, k=5)

        return relevant_docs