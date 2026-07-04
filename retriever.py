import os
import logging
import uuid
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_compressors import FlashrankRerank
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import ContextualCompressionRetriever, EnsembleRetriever
import config

logger = logging.getLogger("inquira_rag")

HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    try:
        from langchain_huggingface import HuggingFaceEndpointEmbeddings
        _embeddings = HuggingFaceEndpointEmbeddings(
            model=f"https://api-inference.huggingface.co/pipeline/feature-extraction/{config.EMBEDDING_MODEL}",
            huggingfacehub_api_token=HF_TOKEN,
        )
        logger.info("Using HuggingFace Endpoint Embeddings for zero-RAM embeddings")
    except Exception as e:
        logger.warning(f"Falling back to local embeddings: {e}")
        _embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
else:
    _embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)

_pc = Pinecone(api_key=config.PINECONE_API_KEY)
_compressor = FlashrankRerank(model="ms-marco-TinyBERT-L-2-v2", top_n=5)

_index_ready = False


def _ensure_index_exists():
    global _index_ready
    if _index_ready:
        return
    try:
        existing = _pc.list_indexes().names()
        if config.INDEX_NAME not in existing:
            _pc.create_index(
                name=config.INDEX_NAME,
                dimension=384,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            logger.info(f"Pinecone index '{config.INDEX_NAME}' created")
        else:
            logger.info(f"Pinecone index '{config.INDEX_NAME}' already exists")
        _index_ready = True
    except Exception as e:
        logger.error(f"Failed to verify/create Pinecone index: {e}")
        raise RuntimeError("Vector database is not available right now") from e


def get_hybrid_retriever(chunks, video_id: str):
    if not chunks:
        raise ValueError("No chunks provided to build a retriever from")

    _ensure_index_exists()

    try:
        chunk_ids = [
            str(uuid.uuid5(uuid.NAMESPACE_URL, f"{video_id}-{i}"))
            for i in range(len(chunks))
        ]

        vector_store = PineconeVectorStore.from_documents(
            documents=chunks,
            embedding=_embeddings,
            index_name=config.INDEX_NAME,
            namespace=video_id,
            ids=chunk_ids,
        )
        logger.info(f"Embeddings stored in Pinecone under namespace '{video_id}'")
    except Exception as e:
        logger.error(f"Failed to store embeddings for video {video_id}: {e}")
        raise RuntimeError("Failed to index this video's transcript") from e

    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 15

    vector_retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 15, "namespace": video_id},
    )

    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.4, 0.6],
    )

    retriever = ContextualCompressionRetriever(
        base_compressor=_compressor,
        base_retriever=hybrid_retriever,
    )

    return retriever