from typing import List

from langchain_community.vectorstores import Chroma

from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

from langchain_core.runnables.base import RunnableSequence
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel


class HCPCSCodes(BaseModel):
    codes: List[str] = Field(description="A list of 5-character HCPCS codes")


class DocumentationRetriever(BaseRetriever):
    vectorstore: Chroma
    search_type: str
    search_kwargs: dict
    hcpcs_codes_chain: RunnableSequence

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:

        # determine if any HCPCS codes in the query
        hcpcs_codes = self.hcpcs_codes_chain.invoke({"query": query})

        # if HCPCS codes present, add to search_kwargs
        search_kwargs = {**self.search_kwargs}
        if len(hcpcs_codes.codes) > 1:
            search_kwargs["where_document"] = {
                "$or": [{"$contains": code} for code in hcpcs_codes.codes]
            }
        elif len(hcpcs_codes.codes) == 1:
            search_kwargs["where_document"] = {"$contains": hcpcs_codes.codes[0]}

        # get docs
        if self.search_type == "similarity":
            docs = self.vectorstore.similarity_search(query, **search_kwargs)
        elif self.search_type == "similarity_score_threshold":
            docs_and_similarities = (
                self.vectorstore.similarity_search_with_relevance_scores(
                    query, **search_kwargs
                )
            )
            docs = [doc for doc, _ in docs_and_similarities]
        else:
            raise NotImplementedError("`search_type` should be 'similarity' or 'similarity_score_threshold'")

        return docs


# create chain to get the HCPCS codes mentioned in a question
def make_hcpcs_codes_chain(model: BaseChatModel):
    parser = PydanticOutputParser(pydantic_object=HCPCSCodes)
    prompt = PromptTemplate(
        template=(
            "Identify any HCPCS codes contained in the following question.\n\n"
            "{format_instructions}\n\n"
            "Question: {query}"
        ),
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    return (
        prompt
        | model
        | parser
    )
