import re, os
from typing import List

import chromadb
from langchain_openai import OpenAIEmbeddings

from langchain_core.documents.base import Document

from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders.llmsherpa import LLMSherpaFileLoader


NLM_INGESTOR_URL = os.environ.get('NLM_INGESTOR_URL', "http://localhost:5010/api/parseDocument?renderFormat=all")


# add chapter information to metadata
def add_chapter_metadata(docs: List[Document]):

    last_chapter = None
    for doc in docs:
        chapter_match = re.compile(r'^([0-9]+(?:\.[0-9]+)*) -').search(doc.metadata['section_title'])
        if chapter_match:
            chapter = chapter_match.group(1)
            doc.metadata['chapter'] = chapter
            last_chapter = chapter
        else:
            if last_chapter is not None:
                doc.metadata['chapter'] = last_chapter


# add chapter information to metadata
def add_hcpcs_metadata(docs: List[Document]):
    for doc in docs:
        hcpcs = re.compile(r'[0-9A-Z][0-9]{4}').findall(doc.page_content)
        doc.metadata['hcpcs_codes'] = ' '.join(list(set(hcpcs)))


if __name__ == "__main__":
    # load document
    loader = LLMSherpaFileLoader(
        file_path="docs/medicare_claims_processing_manual_ch5.pdf",
        new_indent_parser=True,
        apply_ocr=True,
        strategy="sections",
        llmsherpa_api_url=NLM_INGESTOR_URL
    )
    docs = loader.load()

    # add metadata
    add_chapter_metadata(docs)
    add_hcpcs_metadata(docs)

    # embed and load into chroma
    # each page = one document
    # by setting ids for each document, will prevent loading more than once
    vs = Chroma.from_documents(
        collection_name="guidelines",
        documents=docs,
        ids=[str(i) for i in range(len(docs))],
        embedding=OpenAIEmbeddings(), 
        persist_directory="./chroma_db"
    )

    # confirm that we have a document for each page
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    assert chroma_client.get_collection(name="guidelines").count() == len(docs)
