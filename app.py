import streamlit as st
import json
from retrieval_pgvector import search_pgvector
from retrieval_faiss import search_faiss
from retrieval_lexical import search_lexical
import base64

st.title("Multimodal RAG for Video QA")
st.markdown("Query the video: [Combinatorial Reconfiguration](https://www.youtube.com/watch?v=dARr3lGKwk8)")

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Query input
query = st.chat_input("Ask a question about the video")
method = st.selectbox("Retrieval Method", ["pgvector_ivfflat", "pgvector_hnsw", "faiss", "tfidf", "bm25"])

if query:
    with st.chat_message("user"):
        st.markdown(query)
    st.session_state.messages.append({"role": "user", "content": query})

    # Perform retrieval
    try:
        if method == "pgvector_ivfflat":
            results = search_pgvector(query, k=3, index_type="ivfflat")
        elif method == "pgvector_hnsw":
            results = search_pgvector(query, k=3, index_type="hnsw")
        elif method == "faiss":
            results = search_faiss(query, k=3)
        elif method == "tfidf":
            results = search_lexical(query, k=3, method="tfidf")
        else:
            results = search_lexical(query, k=3, method="bm25")

        # Display results
        with st.chat_message("assistant"):
            if results and results[0]["score"] > 0.7:  # Threshold for relevance
                top_result = results[0]["data"]
                start_time = top_result["start_time"]
                text = top_result["text"]
                st.markdown(f"**Answer found at {start_time}s**: {text}")
                # Embed YouTube video with timestamp
                youtube_url = f"https://www.youtube.com/embed/dARr3lGKwk8?start={int(start_time)}"
                st.video(youtube_url)
            else:
                st.markdown("Sorry, the answer is not present in the video. Please ask another question.")
            st.session_state.messages.append({"role": "assistant", "content": st.session_state.messages[-1]["content"]})
    except Exception as e:
        st.error(f"Error: {e}")
        logging.error(f"Streamlit query failed: {e}")

# Save chat history
with open("chat_history.json", "w") as f:
    json.dump(st.session_state.messages, f, indent=2)