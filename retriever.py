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

# --------------------------------------------------------------------------
# Load the embedding model and Pinecone client ONCE at import time, not on
# every /load call. HuggingFaceEmbeddings loading a transformer model from
# disk/network on every single request would be slow and wasteful.
# --------------------------------------------------------------------------
_embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
_pc = Pinecone(api_key=config.PINECONE_API_KEY)

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
    """
    Builds a hybrid (BM25 + vector) retriever scoped to a single video.

    CRITICAL: every video's vectors are stored under a Pinecone *namespace*
    equal to its video_id. Without this, all videos would share one flat
    index and a question about Video A could retrieve chunks from Video B.
    Chunk IDs are also deterministic (derived from video_id + position), so
    reloading the same video overwrites its old vectors instead of creating
    duplicates.
    """
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
            namespace=video_id,   # isolates this video's vectors from all others
            ids=chunk_ids,        # reloading the same video overwrites, not duplicates
        )
        logger.info(f"Embeddings stored in Pinecone under namespace '{video_id}'")
    except Exception as e:
        logger.error(f"Failed to store embeddings for video {video_id}: {e}")
        raise RuntimeError("Failed to index this video's transcript") from e

    # Hybrid retriever (BM25 keyword + Pinecone vector), both scoped to this video only
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

    # Reranking (FlashRank)
    compressor = FlashrankRerank(top_n=5)
    retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=hybrid_retriever,
    )

    return retriever