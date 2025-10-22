"""
RAG Tool module for LangChain + Qdrant

Usage:
- Place this file in your project and ensure .env contains OPENAI_API_KEY, QDRANT_API_KEY, QDRANT_URL
- Ensure qdrant-client compatible version (1.8.2) is installed if using langchain-qdrant that expects it.

Features:
- Initializes QdrantVectorStore from existing collection
- Provides `RAGTool` class with .retrieve(query, k) and .ask(query, k)
- Provides example how to wrap .ask into a LangChain Tool for use by an Agent

"""

import os
from typing import List, Optional
from dotenv import load_dotenv

# LangChain / Qdrant imports
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain.chains import RetrievalQA
from langchain.schema import BaseRetriever

# Optional: for Tool wrapper depending on LangChain version
try:
    # newer unified tools API
    from langchain.tools import Tool
except Exception:
    try:
        from langchain.agents import Tool
    except Exception:
        Tool = None


class RAGTool:
    """RAG helper that loads an existing Qdrant collection and exposes retrieval + QA.

    Example:
        rag = RAGTool(collection_name="resume_documents")
        answer, sources = rag.ask("Apa pengalaman kandidat dengan skill data science?", k=3)

    Notes:
    - Requires .env with OPENAI_API_KEY, QDRANT_API_KEY, QDRANT_URL (or env vars set)
    - Make sure the collection already exists in Qdrant and contains vectors (from earlier pipeline)
    """

    def __init__(
        self,
        collection_name: str,
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
        embedding_model: str = "text-embedding-3-small",
        llm_model: str = "gpt-4o-mini",
        embedding_dim: int = 1536,
        check_compatibility: bool = True,
    ) -> None:
        # load .env if present
        load_dotenv()

        self.collection_name = collection_name
        self.qdrant_url = qdrant_url or os.getenv("QDRANT_URL")
        self.qdrant_api_key = qdrant_api_key or os.getenv("QDRANT_API_KEY")

        if not self.qdrant_url:
            raise ValueError("QDRANT_URL must be set via parameter or .env")
        if not self.qdrant_api_key:
            raise ValueError("QDRANT_API_KEY must be set via parameter or .env")

        # Initialize embeddings and LLM
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        self.llm = ChatOpenAI(model=llm_model, temperature=0)

        # Create Qdrant client
        # If your qdrant-client <-> langchain mismatch causes compatibility check warnings,
        # you can set check_compatibility=False when constructing QdrantClient (see qdrant-client docs)
        client_kwargs = {"url": self.qdrant_url, "api_key": self.qdrant_api_key}
        if not check_compatibility:
            client_kwargs["check_compatibility"] = False

        self.qdrant_client = QdrantClient(**client_kwargs)

        # Load vectorstore that uses the collection (does not recreate)
        # Use from_documents with empty docs is OK for creating instance from existing collection
        self.vectorstore = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=self.collection_name,
            embedding=self.embeddings,
        )

        # Prepare retriever and QA chain lazily
        self._retriever: Optional[BaseRetriever] = None
        self._qa_chain: Optional[RetrievalQA] = None

    def _ensure_retriever(self, k: int = 3):
        if self._retriever is None:
            self._retriever = self.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})
        return self._retriever

    def _ensure_qa_chain(self, k: int = 3):
        if self._qa_chain is None:
            retriever = self._ensure_retriever(k=k)
            self._qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                retriever=retriever,
                return_source_documents=True,
            )
        return self._qa_chain

    def retrieve(self, query: str, k: int = 3) -> List[Document]:
        """Return top-k documents (langchain Document objects) from the vectorstore."""
        retriever = self.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})
        results = retriever.get_relevant_documents(query)
        return results

    def ask(self, query: str, k: int = 3) -> dict:
        """Run a RetrievalQA query and return dict with 'answer' and 'source_documents'."""
        qa = self._ensure_qa_chain(k=k)
        output = qa({"query": query})
        # output typically includes keys: 'result' and 'source_documents'
        return output


# ------------------------- Helper: create Tool for Agent -------------------------

def make_langchain_tool(rag_tool: RAGTool, name: str = "RAGTool", description: str = None):
    """Return a langchain Tool that calls rag_tool.ask and returns answer text.

    This helper tries to construct a Tool compatible with different LangChain versions.
    """
    description = description or (
        "Use this tool to answer user queries by retrieving relevant documents from the Qdrant vector store "
        "and producing a concise answer. Input: a natural language question. Output: short answer + citations."
    )

    def _call(q: str) -> str:
        out = rag_tool.ask(q, k=3)
        answer = out.get("result") or out.get("answer") or ""
        sources = out.get("source_documents") or []
        cites = "\nSources:\n" + "\n".join([str(d.metadata.get("source") or d.metadata.get("id") or "<unknown>") for d in sources])
        return (answer or "") + cites

    if Tool is not None:
        try:
            return Tool.from_function(fn=_call, name=name, description=description)
        except Exception:
            # fallback for older Tool signature
            return Tool(name=name, func=_call, description=description)
    else:
        # If Tool class not available, return plain callable and metadata
        return {"name": name, "func": _call, "description": description}


# ------------------------- Usage example -------------------------
# Example (do not run at import time):
#
# from rag_tool import RAGTool, make_langchain_tool
#
# rag = RAGTool(collection_name="resume_documents")
# print(rag.retrieve("experience in data science", k=5)[0].page_content)
# out = rag.ask("Berikan ringkasan pengalaman kandidat yang relevan dengan data science", k=3)
# print(out['result'])
#
# # Create a Tool to attach to an Agent
# tool = make_langchain_tool(rag, name="resume_rag", description="Answer questions using the resume collection")
#
# # Example agent usage differs between LangChain versions; if using agents:
# # from langchain.agents import initialize_agent, AgentType
# # agent = initialize_agent([tool], llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
# # agent.run("Siapa kandidat yang punya pengalaman machine learning?")

