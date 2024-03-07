from typing import List

import chromadb
import streamlit as st

# from langchain import hub
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


def sort_docs_by_page_number(docs: List[Document]) -> List[Document]:
    return sorted(docs, key=lambda x: x.metadata['page'])  


@st.cache_resource(show_spinner=False)
def get_vector_store():
    persistent_client = chromadb.PersistentClient(path="./chroma_db")

    return Chroma(
        client=persistent_client,
        collection_name="guidelines",
        embedding_function=OpenAIEmbeddings(),
    )


def get_rag_chain(prompt_str: str, k: int = 5):
    vector_store = get_vector_store()

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    # prompt = hub.pull("rlm/rag-prompt")
    prompt = ChatPromptTemplate.from_messages([("human", prompt_str)])

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
        | prompt
        | llm
        | StrOutputParser()
    )

    rag_chain_with_source = RunnableParallel(
        {"context": retriever | sort_docs_by_page_number, "question": RunnablePassthrough()}
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
        "you will use to answer the question. If you don't know the answer, just say that you don't know. "
        "Answers should be detailed with a maximum length of 8 sentences. \n\n"
        "Question: {question} \n\n"
        "Context: {context} \n\n"
        "Answer:"
    ),
    height=300
) 

question = st.text_input(
    label='Question:',
    value="When can the KX modifier be used?"
)

if st.button("Ask", type="primary", disabled=not question):
    with st.spinner("Calculating..."):
        result = ask_question(question, prompt)

        # show answer
        st.info(result["answer"])

        # show supporting documents
        for doc in sort_docs_by_page_number(result["context"]):
            with st.expander(f"Page {str(doc.metadata['page'])}"):
                st.write(doc.page_content)
