# Multimodal RAG for Video QA – Setup & Usage Guide

## 1. Project Overview

This repository turns a **plain lecture video** into a fully‑searchable, multimodal knowledge base and interactive QA demo built with **Streamlit**.
A single script‑driven pipeline

1. **Pre‑processes** the audio (16 kHz),
2. Generates an **ASR transcript** (NVIDIA *Parakeet*),
3. **Extracts key‑frames** from the video,
4. Creates **text + image embeddings** (MiniLM + CLIP),
5. Indexes them with **FAISS** *and* **pgvector** inside PostgreSQL, alongside **TF‑IDF/BM25** lexical indices, and finally
6. Launches a **Streamlit RAG app** for natural‑language video search & evaluation.

Everything runs **offline on CPU** by default; a CUDA‑capable GPU simply speeds things up.

---

## 2. Prerequisites

| Requirement               | Tested Version     | Notes                                                                                                         |
| ------------------------- | ------------------ | ------------------------------------------------------------------------------------------------------------- |
| **Python**                | 3.10 ≥ 3.12        | Use `pyenv` or `conda` if you have multiple versions.                                                         |
| **WSL 2**                 | Ubuntu 22.04       | Many CV & audio libs install more smoothly in WSL; native Linux/macOS works, Windows CMD **not recommended**. |
| **FFmpeg**                | ≥ 5.1              | For resampling audio.                                                                                         |
| **PostgreSQL**            |  15+               | Must be compiled with `--with-python` (standard in all official binaries).                                    |
| **pgvector extension**    |  0.5.1+            | `CREATE EXTENSION vector;` inside each DB.                                                                    |
| **CUDA 12.xx** (optional) | Tested on RTX 2070 | *Parakeet* auto‑detects GPU; falls back to CPU otherwise.                                                     |

> **Tip** If you are on Windows, install [**WSL 2**](https://learn.microsoft.com/windows/wsl/install) and pick *Ubuntu 22.04*; the scripts assume a Linux‑like path layout.

---

## 3. Repository Layout

```text
.
├── input/
│   ├── Video.mp4          # The original video lecture
│   └── Audio_16khz.wav    # (generated) 16 kHz mono WAV
├── keyframes/             # (generated) extracted frames
├── transcripts.json       # (generated) ASR output
├── text_embeddings.npy    # (generated)
├── image_embeddings.npy   # (generated)
├── text_data.json         # (generated) embeddings + metadata
├── image_data.json        # (generated)
├── requirements-transcribe.txt
├── transcribe.py
├── extract.py
├── embed.py
├── retrieval_faiss.py
├── retrieval_pgvector.py
├── retrieval_lexical.py
└── app.py                 # Streamlit front‑end
```

---

## 4. Quick‑start (copy‑paste friendly)

```bash
# 1️⃣  Clone & enter
$ git clone https://github.com/your‑org/video‑rag.git && cd video‑rag

# 2️⃣  Python env
$ python -m venv .venv && source .venv/bin/activate
$ pip install --upgrade pip
$ pip install -r requirements-transcribe.txt   # heavy ASR deps first
$ pip install -r requirements.txt              # the rest (faiss‑cpu, pgvector‑psycopg, streamlit…)  

# 3️⃣  System deps (Ubuntu / WSL)
$ sudo apt update && sudo apt install ffmpeg libgl1‑mesa‑glx postgresql‑15 postgresql‑contrib
$ sudo -u postgres psql -c "CREATE EXTENSION IF NOT EXISTS vector;"

# 4️⃣  Resample audio → 16 kHz mono WAV
$ ffmpeg -i input/Video.mp4 -ar 16000 -ac 1 input/Audio_16khz.wav

# 5️⃣  Automatic pipeline (may take a while!)
$ python transcribe.py        # ASR → transcripts.json
$ python extract.py           # keyframes → keyframes_mapped.json
$ python embed.py             # embeddings → *.npy / *.json
$ python retrieval_faiss.py   # builds FAISS index + benchmarks
$ python retrieval_pgvector.py # creates DB, inserts vectors, builds indexes
$ python retrieval_lexical.py # TF‑IDF & BM25 indices + benchmarks

# 6️⃣  Launch demo
$ export OPENROUTER_API_KEY=sk‑…  # required for Chat tab
$ streamlit run app.py            # ⬅ open http://localhost:8501 in your browser
```

---

## 5. Detailed Pipeline & Scripts

### 5.1 Audio Pre‑processing

Convert the video’s audio track to **16 kHz 16‑bit PCM mono** – the required input for *Parakeet*.

```bash
ffmpeg -i input/Video.mp4 -ar 16000 -ac 1 input/Audio_16khz.wav
```

`ffmpeg` writes the file in‑place; no edits to the scripts are needed.

### 5.2 `transcribe.py` – Speech‑to‑Text

* Loads **nvidia/parakeet‑tdt‑0.6b‑v2** (automatic mixed precision on GPU).
* Uses **WebRTC‑VAD** to skip silence.
* Streams 10‑second segments and writes an array of `{start_time, end_time, text}` to `transcripts.json`.
* Output is always **lower‑cased** to simplify lexical matching.

**Dependencies** are heavy (PyTorch, NeMo, CuPy if CUDA). They are isolated in `requirements‑transcribe.txt` so you can pre‑install on GPU machines only.

### 5.3 `extract.py` – Key‑frame Detection

* Reads `input/Video.mp4` with OpenCV (CPU‑only).
* Saves JPEGs whenever the grayscale histogram correlation drops below `0.6` **or** every 60 s as a fallback.
* Produces:

  * `keyframes/*.jpg` – the actual images
  * `keyframes/keyframes.json` – timestamps & paths
  * `keyframes/keyframes_mapped.json` – same, but with the transcript text that overlaps each frame

Adjust `threshold`, `min_interval`, or `fallback_interval` at the top if your video is noisier/calmer.

### 5.4 `embed.py` – Text & Image Embeddings

* **Text**: `all‑MiniLM‑L6‑v2` → 384‑D vectors
* **Image**: `openai/clip‑vit‑base‑patch32` → 512‑D vectors
  (runs on CPU by default; change `device` to `cuda` if available)
* Writes:

  * `text_embeddings.npy` / `image_embeddings.npy`
  * `text_data.json` / `image_data.json` – vectors + metadata so they can be re‑inserted into pgvector later.

### 5.5 `retrieval_faiss.py`

* Normalises vectors to unit length (cosine ≈ inner product).
* Builds two **in‑memory FAISS `IndexFlatIP`** indices (text & image).
* Runs an evaluation over `gold_test_set.json` and writes precision metrics to disk.

### 5.6 `retrieval_pgvector.py`

1. Connects to PostgreSQL with the hard‑coded *local* super‑user:

   ```python
   psycopg2.connect(dbname="postgres", user="postgres", password="postgres123", host="localhost", port="5432")
   ```

   Change **any** of these and re‑run the script *or* export ENV vars and read them with `os.getenv`.
2. (Re)creates a fresh `rag` database and enables `CREATE EXTENSION vector;`.
3. Inserts every row from `text_data.json` and `image_data.json`.
4. Creates **IVFFlat** *and* **HNSW** cosine indexes so you can benchmark both:

   ```sql
   CREATE INDEX text_ivfflat ON text_embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
   CREATE INDEX text_hnsw    ON text_embeddings USING hnsw    (embedding vector_cosine_ops) WITH (m = 24, ef_construction = 100);
   ```
5. Runs the same gold‑test evaluation and saves `{pgvector_ivfflat|pgvector_hnsw}_{results|benchmark}.json`.

> **ℹ WSL quirk** `localhost` in WSL is **inside** Linux, not Windows. If your DB runs natively on Windows, point `host` to Windows’ IP (e.g. `172.x.x.x`).

### 5.7 `retrieval_lexical.py`

* Builds **TF‑IDF** and **BM25** bag‑of‑words matrices over the transcript.
* Stores them as sparse SciPy matrices in memory only (tiny) and writes benchmarks.

### 5.8 `app.py` (Streamlit)

* Loads **all** benchmark/result files lazily (`@st.cache_data`).
* Supports five retrieval back‑ends + one LLM (**Qwen3 235B CAG** via [OpenRouter](https://openrouter.ai)).
* Presents a multipage UI:

  * Quantitative metrics per method
  * Gold test answers with *seek* buttons – jumps straight to the correct moment in the video
  * Chat‑style QA with citation & timestamp

Environment‑variables used:

```bash
OPENROUTER_API_KEY   # required only for "Qwen3 235B (CAG)" mode
```

---

## 6. Customisation & Tips

* **Use a GPU?** Comment `device = "cpu"` lines in `embed.py` & `retrieval_pgvector.py` and let PyTorch auto‑select CUDA.
* **Transcript language** – `transcribe.py` currently forces lower‑case English. Adjust `hypothesis.text.lower()` if you need casing.
* **Index granularity** – Increase `segment_length` in `transcribe.py` for coarser speech chunks (faster) or lower it for fine‑grained retrieval.
* **Database reuse** – If you want to keep your vectors, comment out the *DROP DATABASE* section in `retrieval_pgvector.py`.

---

## 7. Troubleshooting

| Symptom                                                            | Likely Cause                                | Fix                                                                                                          |
| ------------------------------------------------------------------ | ------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| `torch.cuda.OutOfMemoryError` during ASR                           | GPU too small (Parakeet >6 GB)              | Run on CPU (`CUDA_VISIBLE_DEVICES="" python transcribe.py`) or use a smaller model.                          |
| `psycopg2.OperationalError: FATAL: password authentication failed` | Wrong credentials                           | Edit the *connect\_to\_db* block in `retrieval_pgvector.py`.                                                 |
| Streamlit shows *Required backend modules are missing*             | One of the retrieval scripts failed earlier | Re‑run the failed script and ensure the corresponding `*_results.json` exists.                               |
| Key‑frames are all black                                           | OpenCV fails on certain codecs              | Transcode the video first: `ffmpeg -i input/Video.mp4 -c:v libx264 -crf 23 -c:a copy input/Video_reenc.mp4`. |

---
