import os
import logging
import http.cookiejar
import requests
from langchain_core.documents import Document
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound

logger = logging.getLogger("inquira_rag")


def format_timestamp(seconds):
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"


def load_youtube_chunks(video_id: str):
    if not video_id or not video_id.strip():
        raise ValueError("video_id cannot be empty")

    if not os.path.exists("cookies.txt") and os.getenv("YOUTUBE_COOKIES"):
        try:
            with open("cookies.txt", "w", encoding="utf-8") as f:
                f.write(os.getenv("YOUTUBE_COOKIES"))
            logger.info("Generated cookies.txt from YOUTUBE_COOKIES environment variable")
        except Exception as e:
            logger.warning(f"Could not generate cookies.txt from env: {e}")

    fetched_transcript = None

    proxy_url = os.getenv("YOUTUBE_PROXY") or os.getenv("HTTPS_PROXY")
    proxies = {"http": proxy_url, "https": proxy_url} if proxy_url else None
    if proxy_url:
        logger.info("Using HTTP proxy to bypass YouTube IP block")

    try:
        if hasattr(YouTubeTranscriptApi, "list_transcripts"):
            cookies_path = "cookies.txt" if os.path.exists("cookies.txt") else None
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id, cookies=cookies_path, proxies=proxies)
        else:
            session = requests.Session()
            if proxies:
                session.proxies.update(proxies)
            if os.path.exists("cookies.txt"):
                try:
                    cj = http.cookiejar.MozillaCookieJar("cookies.txt")
                    cj.load(ignore_discard=True, ignore_expires=True)
                    session.cookies = cj
                except Exception as e:
                    logger.warning(f"Could not load cookies.txt: {e}")
            yta = YouTubeTranscriptApi(http_client=session)
            transcript_list = yta.list(video_id)

        try:
            fetched_transcript = transcript_list.find_transcript(["en"]).fetch()
            logger.info(f"English transcript fetched for {video_id}")
        except NoTranscriptFound:
            available_transcript = next(iter(transcript_list))
            logger.info(f"No English transcript for {video_id}, found: {available_transcript.language}")
            if available_transcript.is_translatable:
                fetched_transcript = available_transcript.translate("en").fetch()
                logger.info("Translated to English successfully")
            else:
                fetched_transcript = available_transcript.fetch()
                logger.info("Translation unavailable, using original language")

    except Exception as e:
        logger.error(f"Error fetching transcript for {video_id}: {e}")
        raise RuntimeError(f"Could not fetch a transcript for video '{video_id}'") from e

    if not fetched_transcript:
        raise RuntimeError(f"No transcript available for video '{video_id}'")

    chunks = []
    current_text = []
    chunk_start = 0.0

    for i, snippet in enumerate(fetched_transcript):
        text = snippet.text if hasattr(snippet, "text") else snippet["text"]
        start = snippet.start if hasattr(snippet, "start") else snippet["start"]
        duration = snippet.duration if hasattr(snippet, "duration") else snippet.get("duration", 0)

        if not current_text:
            chunk_start = start

        current_text.append(text)
        current_chunk_text = " ".join(current_text)

        if len(current_chunk_text) >= 1000 or i == len(fetched_transcript) - 1:
            chunk_end = start + duration
            start_str = format_timestamp(chunk_start)
            end_str = format_timestamp(chunk_end)

            chunks.append(Document(
                page_content=f"[{start_str} - {end_str}] {current_chunk_text}",
                metadata={
                    "start": float(chunk_start),
                    "end": float(chunk_end),
                    "start_str": start_str,
                    "end_str": end_str,
                    "video_id": video_id,
                },
            ))
            current_text = []
            chunk_start = 0.0

    if not chunks:
        raise RuntimeError(f"Transcript for video '{video_id}' produced no usable chunks")

    logger.info(f"Total chunks with timestamps for {video_id}: {len(chunks)}")
    return chunks