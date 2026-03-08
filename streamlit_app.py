import streamlit as st
import requests
import time
import pandas as pd
import plotly.express as px
import pickle
import numpy as np

API_URL = "http://localhost:8000/query"

st.set_page_config(
    page_title="Semantic Search AI",
    page_icon="🔎",
    layout="wide"
)

# ------------------------------------------------
# CLUSTER FUNCTION
# ------------------------------------------------

@st.cache_data
def load_cluster_map():

    try:
        with open("data/umap_embeddings.pkl", "rb") as f:
            embedding_2d = pickle.load(f)

        with open("data/membership_matrix.pkl", "rb") as f:
            membership = pickle.load(f)

    except:
        st.error("Cluster files not found in /data folder")
        st.stop()

    embedding_2d = np.array(embedding_2d)
    membership = np.array(membership)

    # IMPORTANT FIX
    clusters = np.argmax(membership, axis=0)

    return embedding_2d, clusters


# ------------------------------------------------
# SESSION STATE
# ------------------------------------------------

if "history" not in st.session_state:
    st.session_state.history = []

if "latency" not in st.session_state:
    st.session_state.latency = []

if "page" not in st.session_state:
    st.session_state.page = "Home"

if "query" not in st.session_state:
    st.session_state.query = ""

# ------------------------------------------------
# THEME
# ------------------------------------------------

dark_mode = st.sidebar.toggle("Dark Mode")

if dark_mode:
    bg = "#0f172a"
    text = "#f1f5f9"
    card = "#1e293b"
    border = "#334155"
else:
    bg = "#f8fafc"
    text = "#0f172a"
    card = "#ffffff"
    border = "#e2e8f0"

# ------------------------------------------------
# CSS
# ------------------------------------------------

st.markdown(f"""
<style>

.stApp {{
background:{bg};
color:{text};
font-family: Inter, sans-serif;
}}

.result-card {{
background:{card};
border-radius:12px;
padding:18px;
margin-bottom:15px;
border:1px solid {border};
box-shadow:0 3px 12px rgba(0,0,0,0.05);
}}

.stButton > button {{
background:#2563eb;
color:white;
border-radius:8px;
border:none;
padding:8px 16px;
}}

mark {{
background:#fde68a;
padding:2px 6px;
border-radius:4px;
}}

</style>
""", unsafe_allow_html=True)

# ------------------------------------------------
# NAVBAR
# ------------------------------------------------

col1,col2,col3,col4,col5,col6 = st.columns([2,1,1,1,1,1])

with col1:
    st.markdown("### SemanticSearch AI")

with col2:
    if st.button("Home"):
        st.session_state.page = "Home"

with col3:
    if st.button("Features"):
        st.session_state.page = "Features"

with col4:
    if st.button("Docs"):
        st.session_state.page = "Docs"

with col5:
    if st.button("Clusters"):
        st.session_state.page = "Clusters"

with col6:
    if st.button("About"):
        st.session_state.page = "About"

# ------------------------------------------------
# SIDEBAR HISTORY
# ------------------------------------------------

st.sidebar.title("Search History")

if st.session_state.history:

    for q in reversed(st.session_state.history[-5:]):
        if st.sidebar.button(q):
            st.session_state.query = q
            st.session_state.page = "Home"

else:
    st.sidebar.write("No searches yet")

# ------------------------------------------------
# HOME PAGE
# ------------------------------------------------

if st.session_state.page == "Home":

    col1, col2 = st.columns([2,1])

    with col1:

        st.markdown("""
        # AI Semantic Search

        Vector Search • Hybrid Retrieval • Semantic Cache

        Intelligent search powered by embeddings.
        """)

        st.markdown("### Try Example Searches")

        samples = [
            "satellite launches",
            "space exploration missions",
            "NASA mars rover discoveries",
            "rocket propulsion systems"
        ]

        cols = st.columns(4)

        query = st.session_state.query
        search = False

        for i,q in enumerate(samples):
            if cols[i].button(q):
                st.session_state.query = q
                query = q
                search = True

        query = st.text_input(
            "Search knowledge",
            value=query,
            placeholder="Ask anything..."
        )

        search = st.button("Search") or search

    with col2:
        st.image(
            "https://cdn-icons-png.flaticon.com/512/4140/4140047.png",
            width=320
        )

    # ------------------------------------------------
    # SEARCH
    # ------------------------------------------------

    if search and query:

        if query not in st.session_state.history:
            st.session_state.history.append(query)

        with st.spinner("Searching semantic index..."):

            start = time.time()

            try:
                response = requests.post(API_URL, json={"query":query}, timeout=30)
                data = response.json()
            except:
                st.error("FastAPI server is not running")
                st.stop()

            latency = round((time.time()-start)*1000,2)

        st.session_state.latency.append(latency)

        if data.get("cache_hit"):
            st.success("⚡ Cache Hit")
        else:
            st.warning("Cache Miss")

        st.write(f"Latency: {latency} ms")

        results = data.get("results",[])

        st.markdown("## Results")

        for i,r in enumerate(results):

            text = r["text"] if isinstance(r, dict) else str(r)

            if query.lower() in text.lower():
                highlighted = text.replace(query, f"<mark>{query}</mark>")
            else:
                highlighted = text

            st.markdown(
            f"""
            <div class="result-card">
            {highlighted}
            </div>
            """,
            unsafe_allow_html=True
            )

            if st.button("Copy", key=f"copy{i}"):
                st.toast("Copied to clipboard")

    # ------------------------------------------------
    # LATENCY GRAPH
    # ------------------------------------------------

    if len(st.session_state.latency) > 1:

        df = pd.DataFrame({
            "Search Number": list(range(len(st.session_state.latency))),
            "Latency": st.session_state.latency
        })

        fig = px.line(
            df,
            x="Search Number",
            y="Latency",
            title="Search Latency Over Time"
        )

        st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------
# CLUSTERS PAGE
# ------------------------------------------------

elif st.session_state.page == "Clusters":

    st.header("Document Cluster Visualization")

    with st.spinner("Loading cluster map..."):

        embedding_2d, clusters = load_cluster_map()

        fig = px.scatter(
            x=embedding_2d[:,0],
            y=embedding_2d[:,1],
            color=clusters.astype(str),
            title="Document Clusters (UMAP)"
        )

        st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------
# FEATURES PAGE
# ------------------------------------------------

elif st.session_state.page == "Features":

    st.header("Features")

    st.markdown("""
🔎 Semantic Vector Search  
⚡ FAISS Similarity Retrieval  
💾 Semantic Cache  
📊 Search Analytics  
📚 Query History  
📈 Cluster Visualization
""")

# ------------------------------------------------
# DOCS PAGE
# ------------------------------------------------

elif st.session_state.page == "Docs":

    st.header("System Architecture")

    st.markdown("""
User Query  
↓  
Embedding Model  
↓  
Vector Database (FAISS)  
↓  
Semantic Cache  
↓  
Re-ranked Results
""")

# ------------------------------------------------
# ABOUT PAGE
# ------------------------------------------------

elif st.session_state.page == "About":

    col1,col2 = st.columns([1,2])

    with col1:
        st.image(
            "https://cdn-icons-png.flaticon.com/512/6997/6997662.png",
            width=150
        )

    with col2:
        st.markdown("""
## Bhumistha Sahoo

AI / Machine Learning Engineer

This project demonstrates a **Semantic Search Engine** powered by:

- Transformer embeddings
- FAISS vector search
- Semantic caching
- Hybrid retrieval

Technologies used:

- Python
- FastAPI
- Streamlit
- Docker
""")

# ------------------------------------------------
# FOOTER
# ------------------------------------------------

st.markdown("---")

st.markdown("""
<center>

Built by **Bhumistha Sahoo**

AI / ML Engineer  
Semantic Search • Intelligent Systems

</center>
""", unsafe_allow_html=True)