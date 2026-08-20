from dotenv import load_dotenv
load_dotenv()
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings

embedding_ollama = OllamaEmbeddings(model="nomic-embed-text")

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

#docs = [UnstructuredLoader(web_url=url,chunking_strategy="basic",max_characters=1000000).load() for url in urls]
docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]


text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=250, chunk_overlap=0)

doc_splits = text_splitter.split_documents(docs_list)
'''
vectorstore= Chroma.from_documents(
    documents=doc_splits,
    embedding=embedding_ollama,
    persist_directory="./.chroma",
    collection_name="rag-chroma"
)'''
retriever = Chroma(
    persist_directory="./.chroma",
    embedding_function=embedding_ollama,
    collection_name="rag-chroma"
).as_retriever()

