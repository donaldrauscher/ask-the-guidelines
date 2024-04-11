from typing import List

import chromadb
import streamlit as st

# from langchain import hub
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from retriever import DocumentationRetriever, make_hcpcs_codes_chain


# from langchain.globals import set_debug
# set_debug(True)


def sort_docs_by_section_number(docs: List[Document]) -> List[Document]:
    return sorted(docs, key=lambda x: x.metadata['section_number'])  


@st.cache_resource(show_spinner=False)
def get_vector_store():
    persistent_client = chromadb.PersistentClient(path="./chroma_db")

    return Chroma(
        client=persistent_client,
        collection_name="guidelines",
        embedding_function=OpenAIEmbeddings(),
    )


def get_rag_chain(prompt_str: str, score_threshold: float = 0.5):
    llm = ChatOpenAI(model_name="gpt-4-0125-preview", temperature=0)

    vector_store = get_vector_store()

    retriever = DocumentationRetriever(
        vectorstore=vector_store, 
        search_type="similarity_score_threshold", 
        search_kwargs={"score_threshold": score_threshold}, 
        hcpcs_codes_chain=make_hcpcs_codes_chain(llm)
    )

    # prompt = hub.pull("rlm/rag-prompt")
    prompt = PromptTemplate(
        input_variables=['context', 'question'],
        template=prompt_str
    )

    def format_docs(docs):
        def format_doc(doc):
            return "Chapter {} Section {}:\n{}".format(
                doc.metadata.get('chapter', 'None'),
                doc.metadata['section_title'],
                doc.page_content
            )
        return "\n\n".join(format_doc(doc) for doc in docs)

    rag_chain = (
        RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
        | prompt
        | llm
        | StrOutputParser()
    )

    rag_chain_with_source = RunnableParallel(
        {"context": retriever | sort_docs_by_section_number, "question": RunnablePassthrough()}
    ).assign(answer=rag_chain)

    return rag_chain_with_source
  

@st.cache_data(show_spinner=False)
def ask_question(question: str, prompt_str: str) -> dict:
    rag_chain = get_rag_chain(prompt_str)
    return rag_chain.invoke(question)


# page title
st.set_page_config(page_title='Ask the Guidelines')
st.title('🦜🔗 Ask the Guidelines')

st.markdown("""    
            **What does this app do?** You can use this app to answer questions about how CMS processes outpatient rehab claims.\n
            Source: [Medicare Claims Processing Manual Chapter 5 - Part B Outpatient Rehabiliation 
                and CORF/OPT Services](https://www.cms.gov/regulations-and-guidance/guidance/manuals/downloads/clm104c05.pdf)
            """)

# form
prompt = st.text_area(
    label='RAG Prompt:', 
    value=(
        "You are a medical billing specialist who is helping healthcare providers answer questions related "
        "to claims billing. You will be provided excerpts from CMS' Medicare claims processing manual, which "
        "you will use to answer the question. Only use information from the context provided below.  If you cannot "
        "determine the answer based on the context provided, just say that you don't know. "
        "Answers should be detailed and thorough.\n\n"
        "Question: {question} \n\n"
        "Context: {context} \n\n"
        "Answer:"
    ),
    height=300
) 

question = st.text_area(
    label='Question:',
    value=(
        "What guidance does Medicare Claims Processing Manual provide on when and how a healthcare provider "
        "can bill for HCPCS 97116? Please format response as a list of guidelines for a healthcare provider "
        "to follow."
    ),
    height=150
)

if st.button("Ask", type="primary", disabled=not question):
    with st.spinner("Calculating..."):
        result = ask_question(question, prompt)

        # show answer
        st.info(result["answer"])

        # show supporting documents
        for doc in sort_docs_by_section_number(result["context"]):
            expander_title = "Chapter={}; Section={}".format(
                str(doc.metadata.get('chapter', 'None')), 
                str(doc.metadata['section_title'])
            )
            with st.expander(expander_title):
                st.markdown(doc.page_content, unsafe_allow_html=True)
