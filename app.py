from __future__ import annotations
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List
from openai import OpenAI
import streamlit as st

# ─────────────────────────────── Logging ────────────────────────────────
ROOT = Path(__file__).parent
logging.basicConfig(
    filename=ROOT / "streamlit.log",
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# ─────────────────────── Retrieval back-ends (imports) ───────────────────
try:
    from retrieval_faiss import search_faiss
    from retrieval_lexical import search_lexical
    from retrieval_pgvector import search_pgvector
except ImportError as e:
    logger.error(f"Failed to import retrieval modules: {e}")
    st.error("Required backend modules are missing.")
    st.stop()

# ───────────────────────────── OpenRouter client ─────────────────────────
OPENROUTER_API_KEY = "redacted"
try:
    openrouter_client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )
    logger.info("Initialized OpenRouter client")
except Exception as e:
    logger.error(f"Failed to initialize OpenRouter client: {e}")
    st.error("Failed to connect to OpenRouter API.")
    st.stop()

# ───────────────────────────── Load transcripts ──────────────────────────
TRANSCRIPTS_PATH = ROOT / "text_data.json"
if not TRANSCRIPTS_PATH.exists():
    logger.error(f"Transcripts file {TRANSCRIPTS_PATH} not found")
    st.error(f"Transcripts file {TRANSCRIPTS_PATH} not found")
    transcripts = []
else:
    try:
        with TRANSCRIPTS_PATH.open() as f:
            transcripts = json.load(f)
        logger.info(f"Loaded {len(transcripts)} transcript segments")
    except Exception as e:
        logger.error(f"Failed to load transcripts: {e}")
        st.error(f"Failed to load transcripts: {e}")
        transcripts = []

# ───────────────────────────── Streamlit config ─────────────────────────
st.set_page_config(page_title="Multimodal RAG for Video QA", layout="wide")
st.title("Multimodal RAG for Video QA")
st.markdown("Evaluate retrieval approaches for *Combinatorial Reconfiguration* video")

# ─────────────────────────── Helper functions ───────────────────────────
def _json_or_empty(path: Path) -> Any:
    """Safely load a JSON file; return empty list/dict on failure."""
    if not path.exists():
        logger.warning("File %s not found", path)
        return [] if path.suffix == ".json" else {}
    try:
        with path.open() as f:
            return json.load(f)
    except Exception as err:
        logger.error("Failed to load %s – %s", path, err, exc_info=True)
        return []

def format_seconds_to_hhmmss(seconds: float) -> str:
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02}:{m:02}:{s:02}"

def query_qwen_cag(query: str, transcripts: List[Dict]) -> Dict[str, Any]:
    """Query Qwen3 235B (CAG) with transcripts as context."""
    try:
        transcript_text = "\n".join(
            f"[t={entry['start_time']:.1f}s] {entry['text']}" for entry in transcripts
        )
        prompt = f""" You are a Cache Augmented Generational (CAG) assistant system for a video on Combinatorial Reconfiguration. Your role is to answer questions accurately based solely on the provided transcript. Each segment includes a timestamp (e.g., [t=30.0s]) and text. Ground your answers in the transcript, citing the relevant timestamp(s). If the answer is not in the transcript, respond with "Unanswerable". Do not generate or infer information beyond the transcript.

**Transcript**: {transcript_text}

**Question**: {query}

**Instructions**:
- Provide a concise answer grounded in the transcript.
- Include the relevant timestamp(s) (e.g., [t=30.0s]).
- If unanswerable, state "Unanswerable".
- Format the response as JSON: {{"answer": str, "timestamp": float or null}}.
"""
        completion = openrouter_client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "http://localhost:8501",
                "X-Title": "Multimodal RAG for Video QA",
            },
            extra_body={},
            model="qwen/qwen3-235b-a22b",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": query},
            ],
            response_format={"type": "json_object"},
        )
        response = json.loads(completion.choices[0].message.content)
        logger.info(f"Qwen3 response for query '{query}': {response}")
        return response
    except Exception as e:
        logger.error(f"Qwen3 query failed for '{query}': {e}")
        return {"answer": f"Error: {e}", "timestamp": None}

# ─────────────────────────────── Data load ──────────────────────────────
@st.cache_data(show_spinner=True)
def load_results_and_benchmarks() -> Dict[str, Dict[str, Any]]:
    approaches: Dict[str, Dict[str, str]] = {
        "FAISS": {
            "results": "faiss_results.json",
            "benchmark": "faiss_benchmark.json",
        },
        "pgvector IVFFlat": {
            "results": "pgvector_ivfflat_results.json",
            "benchmark": "pgvector_ivfflat_benchmark.json",
        },
        "pgvector HNSW": {
            "results": "pgvector_hnsw_results.json",
            "benchmark": "pgvector_hnsw_benchmark.json",
        },
        "TF-IDF": {
            "results": "tfidf_results.json",
            "benchmark": "tfidf_benchmark.json",
        },
        "BM25": {
            "results": "bm25_results.json",
            "benchmark": "bm25_benchmark.json",
        },
    }
    data: Dict[str, Dict[str, Any]] = {}
    for name, files in approaches.items():
        data[name] = {
            key: _json_or_empty(ROOT / filename) for key, filename in files.items()
        }
    logger.info("Loaded results & benchmarks for %d approaches", len(data))
    return data

data = load_results_and_benchmarks()

# ─────────────────────────── Video section ──────────────────────────────
VIDEO_PATH = ROOT / "input" / "Video.mp4"
if not VIDEO_PATH.exists():
    st.error(f"Video file {VIDEO_PATH} not found")
    logger.error("Video file %s not found", VIDEO_PATH)
else:
    st.subheader("Video Player")
    st.video(str(VIDEO_PATH))

seek_script = """<script>
window.seekTo = function(t) {
    const trySeek = () => {
        const v = document.querySelector('video');
        if (v) {
            v.currentTime = t;
            v.play();
            console.log('Successfully sought to ' + t + 's');
            return true;
        }
        return false;
    };
    if (trySeek()) return;
    const observer = new MutationObserver((mutations, obs) => {
        if (trySeek()) {
            obs.disconnect();
        }
    });
    observer.observe(document.body, { childList: true, subtree: true });
    setTimeout(() => observer.disconnect(), 5000);
};
window.addEventListener('streamlitSeek', function(e) {
    const timestamp = e.detail.timestamp;
    if (timestamp >= 0) {
        window.seekTo(timestamp);
    }
});
console.log('Seek script initialized');
</script>"""
st.components.v1.html(seek_script, height=0)

if "seek_timestamp" not in st.session_state:
    st.session_state.seek_timestamp = None

if st.session_state.seek_timestamp is not None:
    timestamp = st.session_state.seek_timestamp
    st.components.v1.html(
        f"<script>window.dispatchEvent(new CustomEvent('streamlitSeek', {{ detail: {{ timestamp: {timestamp} }} }}));</script>",
        height=0
    )
    logger.info(f"Triggered seek to {timestamp}s")
    st.session_state.seek_timestamp = None

# ──────────────────────── High-level page routing ───────────────────────
if "page_index" not in st.session_state:
    st.session_state.page_index = 0

PAGES: List[str] = list(data.keys()) + ["Chat Interface"]
prev_col, _, next_col = st.columns([1, 8, 1])
with prev_col:
    if st.button("← Previous", disabled=st.session_state.page_index == 0):
        st.session_state.page_index -= 1
with next_col:
    if st.button("Next →", disabled=st.session_state.page_index >= len(PAGES) - 1):
        st.session_state.page_index += 1

current_page = PAGES[st.session_state.page_index]

# ─────────────────────────── Results pages ──────────────────────────────
if current_page != "Chat Interface":
    approach_data = data[current_page]
    st.header(current_page)

    benchmark = approach_data.get("benchmark") or {}
    if benchmark:
        st.subheader("Benchmark Results")
        st.metric("Top-1 Accuracy", f"{benchmark['top_1_percent']:.2f}%")
        st.metric("Top-3 Accuracy", f"{benchmark['top_3_percent']:.2f}%")
        st.metric("Top-5 Accuracy", f"{benchmark['top_5_percent']:.2f}%")
        st.metric("Rejection Rate", f"{benchmark['rejection_percent']:.2f}%")

    st.subheader("Gold Test Set Results")
    results = approach_data.get("results") or []
    if not results:
        st.info("No results available for this approach yet.")
    else:
        st.markdown("Click a timestamp to seek the video.")
        for item in results:
            query = item["query"]
            if item["answerable"]:
                start_time = item["result"]["data"]["start_time"]
                snippet = item["result"]["data"]["text"]
                score = item["result"]["score"]
                hhmmss = format_seconds_to_hhmmss(start_time)
                label = f"{query} | [{hhmmss}] {snippet} (Score: {score:.2f})"
                if st.button(label, key=f"{current_page}_{query}_{start_time}"):
                    st.session_state.seek_timestamp = start_time
                    st.rerun()
                    logger.info(f"Clicked seek button for {query} at {start_time}s")
            else:
                st.write(f"{query} | **Unanswerable**")

# ───────────────────────────── Chat page ────────────────────────────────
else:
    st.header("Chat Interface")
    method = st.selectbox(
        "Select Retrieval Method",
        ["FAISS", "pgvector IVFFlat", "pgvector HNSW", "TF-IDF", "BM25", "Qwen3 235B (CAG)"],
        key="retrieval_method",
    )
    query = st.chat_input("Ask a question about the video")
    if query:
        st.chat_message("user").markdown(query)
        try:
            if method == "Qwen3 235B (CAG)":
                response = query_qwen_cag(query, transcripts)
                with st.chat_message("assistant"):
                    answer = response.get("answer", "Error: No response")
                    t0 = response.get("timestamp")
                    if t0 is not None and answer != "Unanswerable":
                        hhmmss = format_seconds_to_hhmmss(t0)
                        st.markdown(f"**Answer @ {hhmmss}:** {answer}")
                        if st.button("Go to timestamp", key=f"chat_{t0}"):
                            st.session_state.seek_timestamp = t0
                            st.rerun()
                            logger.info(f"Clicked chat seek button for query '{query}' at {t0}s")
                    else:
                        st.markdown(f"**Answer:** {answer}")
            else:
                if method == "FAISS":
                    results = search_faiss(query, k=1)
                elif method.startswith("pgvector"):
                    index_type = "ivfflat" if "IVFFlat" in method else "hnsw"
                    results = search_pgvector(query, k=1, index_type=index_type)
                elif method == "TF-IDF":
                    results = search_lexical(query, k=1, method="tfidf")
                else:
                    results = search_lexical(query, k=1, method="bm25")

                with st.chat_message("assistant"):
                    threshold = 0.7 if method.startswith(("FAISS", "pgvector")) else 0.5
                    if results and results[0]["score"] > threshold:
                        top = results[0]
                        t0 = top["data"]["start_time"]
                        snippet = top["data"]["text"]
                        snippet = snippet[:500] + "…" if len(snippet) > 500 else snippet
                        hhmmss = format_seconds_to_hhmmss(t0)
                        st.markdown(f"**Answer @ {hhmmss}:** {snippet}")
                        if st.button("Go to timestamp", key=f"chat_{t0}"):
                            st.session_state.seek_timestamp = t0
                            st.rerun()
                            logger.info(f"Clicked chat seek button for query '{query}' at {t0}s")
                    else:
                        st.markdown("❌ Sorry, the answer does not appear in the video.")
        except Exception as exc:
            logger.error("Chat query failed: %s", exc, exc_info=True)
            st.error(f"Error: {exc}")

# ──────────────────────────── House-keeping ─────────────────────────────
logger.info("Displayed page: %s", current_page)
