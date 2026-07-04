import os
import logging
import threading

import config
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.documents import Document
from memory import get_chat_history, format_chat_history

logger = logging.getLogger("inquira_rag")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", getattr(config, "GROQ_API_KEY", None))
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY environment variable must be set.")


class ResilientGroqPool:
    def __init__(self, models_list, temperature=0.3):
        if not models_list:
            raise ValueError("models_list cannot be empty")
        self.models_list = models_list
        self.temperature = temperature
        self.llms = [
            ChatGroq(
                groq_api_key=GROQ_API_KEY,
                model=m,
                temperature=temperature,
                max_retries=1,
                timeout=30,
            )
            for m in models_list
        ]
        self.lock = threading.Lock()
        self.index = 0

    def get_runnable(self):
        with self.lock:
            idx = self.index
            self.index = (self.index + 1) % len(self.llms)

        ordered_llms = self.llms[idx:] + self.llms[:idx]
        return ordered_llms[0].with_fallbacks(ordered_llms[1:])

    def invoke(self, input_data, **kwargs):
        runnable = self.get_runnable()
        try:
            return runnable.invoke(input_data, **kwargs)
        except Exception as e:
            logger.error(f"All Groq models failed: {e}")
            raise


resilient_pool = ResilientGroqPool(getattr(config, "POWERFUL_MODELS", [config.LLM_MODEL]), temperature=0.3)
llm = RunnableLambda(resilient_pool.invoke)

query_rewrite_prompt = PromptTemplate(
    template="""
    You are an expert query rewriter for a YouTube video RAG system.
    Your job is to resolve pronouns and follow-up references using the Recent Chat History.

    Recent Chat History:
    {chat_history}

    Original query: {question}

    Instructions:
    1. Look at the Recent Chat History. If the Original query contains pronouns or references in English or Roman Urdu/Hindi (e.g., "iska", "isko", "it", "this", "usne", "what does it stand for?"), replace those pronouns with the actual topic/noun mentioned in the chat history (e.g., "RAG", "LangChain").
    2. Rewrite the query into a clear, standalone, detailed search query in English.
    3. Return ONLY the rewritten standalone query, nothing else.
    """,
    input_variables=["chat_history", "question"]
)
query_rewriter = query_rewrite_prompt | llm | StrOutputParser()


def compress_docs(question: str, retrieved_docs: list) -> list:
    if not retrieved_docs:
        return []

    all_content = "\n\n---\n\n".join(doc.page_content for doc in retrieved_docs)

    compression_prompt = f"""
    You are a helpful assistant that extracts relevant information.
    Question: "{question}"
    From the text below, extract ONLY sentences relevant to the question.
    Be LENIENT - if even partially related, include it.
    Only respond with "NOT RELEVANT" if absolutely nothing is related.
    Do not add explanation, just return the relevant text.
    Text:
    {all_content}
    """
    try:
        response = llm.invoke(compression_prompt)
    except Exception as e:
        logger.error(f"compress_docs LLM call failed: {e}")
        return retrieved_docs

    if "NOT RELEVANT" in response.content.upper():
        logger.info("Nothing relevant found in compression step, using original docs")
        return retrieved_docs
    return [Document(page_content=response.content)]


def rewrite_retrieve_compress(question: str, retriever, chat_history_str: str = "") -> str:
    try:
        rewritten = query_rewriter.invoke({"question": question, "chat_history": chat_history_str})
    except Exception as e:
        logger.error(f"Query rewrite failed, falling back to original question: {e}")
        rewritten = question

    logger.info(f"Original: {question} | Rewritten: {rewritten}")

    try:
        docs = retriever.invoke(rewritten)
    except Exception as e:
        logger.error(f"Retriever failed: {e}")
        raise RuntimeError("Failed to search the video transcript") from e

    compressed_docs = compress_docs(question, docs)
    logger.info(f"Chunks retrieved: {len(docs)} | After compression: {len(compressed_docs)}")
    return "\n\n".join(doc.page_content for doc in compressed_docs)


prompt = PromptTemplate(
    template="""
    You are a helpful assistant.
    Answer directly from the provided transcript context and chat history.
    If the user asks for the full form or definition of standard technical terms/abbreviations mentioned in the chat or context (e.g., RAG, LLM, API), you may provide the standard definition even if not explicitly spelled out in the video.
    Otherwise, if the context is insufficient for video facts, just say you don't know.
    ALWAYS answer in English.

    Recent Chat History:
    {chat_history}

    {context}
    Question: {question}
    """,
    input_variables=["chat_history", "context", "question"]
)

main_chain = prompt | llm | StrOutputParser()


grounding_prompt = PromptTemplate(
    template="""
    You are a fact-checking assistant.
    Check if the following answer is supported by the given context.
    ALWAYS respond in English.

    Context: {context}
    Answer to verify: {answer}

    Instructions:
    - If supported by context or if explaining a standard technical abbreviation (like RAG = Retrieval-Augmented Generation), respond with: "GROUNDED: " followed by the answer in English.
    - If NOT supported, respond with: "GROUNDED: " followed by a corrected English answer from context.
    - If no relevant info: "GROUNDED: I don't have enough information from the video."
    """,
    input_variables=["context", "answer"]
)


def grounded_chain(question: str, retriever, session_id: str = "user_session_1") -> str:
    if not question or not question.strip():
        raise ValueError("Question cannot be empty")

    history = get_chat_history(session_id)
    chat_history_str = format_chat_history(history.messages)

    context = rewrite_retrieve_compress(question, retriever, chat_history_str)

    if not context.strip():
        result_content = "I don't have enough information from the video to answer that."
        history.add_user_message(question)
        history.add_ai_message(result_content)
        return result_content

    try:
        raw_answer = main_chain.invoke({"chat_history": chat_history_str, "context": context, "question": question})
        logger.info(f"Raw answer generated ({len(raw_answer)} chars)")

        grounding_check = grounding_prompt.invoke({"context": context, "answer": raw_answer})
        grounded_answer = llm.invoke(grounding_check)
        result_content = grounded_answer.content
    except Exception as e:
        logger.error(f"grounded_chain generation failed for session {session_id}: {e}")
        raise RuntimeError("Failed to generate an answer") from e

    history.add_user_message(question)
    history.add_ai_message(result_content)

    return result_content
if __name__ == "__main__":
    from youtube_loader import load_youtube_chunks
    from retriever import get_hybrid_retriever
    logging.basicConfig(level=logging.INFO)
    test_video_id = os.getenv("TEST_VIDEO_ID", getattr(config, "VIDEO_ID", None))
    if not test_video_id:
        print("Set TEST_VIDEO_ID env var (or config.VIDEO_ID) to test locally.")
        raise SystemExit(1)
    print(f"\nLoading video {test_video_id}...")
    test_chunks = load_youtube_chunks(test_video_id)
    test_retriever = get_hybrid_retriever(test_chunks)
    print("\nYouTube RAG System Ready!")
    print("Type 'exit' to quit\n")
    while True:
        try:
            question = input("Apna sawal poochein: ")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break
        if question.strip().lower() == "exit":
            break
        result = grounded_chain(question, test_retriever)
        print(f"\nFinal Answer:\n{result}\n")
        print("-" * 50)