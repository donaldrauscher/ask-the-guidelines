import chromadb
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader


if __name__ == "__main__":

    # load pdf
    loader = PyPDFLoader("guidelines.pdf")
    pages = loader.load_and_split()

    # embed and load into chroma
    # each page = one document
    # by setting ids for each document, will prevent loading more than once
    vs = Chroma.from_documents(
        collection_name="guidelines",
        documents=pages,
        ids=[str(i) for i in range(len(pages))],
        embedding=OpenAIEmbeddings(), 
        persist_directory="./chroma_db"
    )

    # confirm that we have a document for each page
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    assert chroma_client.get_collection(name="guidelines").count() == len(pages)
