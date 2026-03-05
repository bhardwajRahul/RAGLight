import os
import requests
import streamlit as st

API_URL = os.environ.get("RAGLIGHT_API_URL", "http://localhost:8000")
# NOUVEAU: Timeout configurable (par défaut à 300 secondes, soit 5 minutes)
API_TIMEOUT = int(os.environ.get("RAGLIGHT_API_TIMEOUT", 300))

st.set_page_config(
    page_title="RAGLight",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
#MainMenu, footer, header { visibility: hidden; }

section[data-testid="stSidebar"] {
    background-color: #0f172a;
    border-right: 1px solid #1e293b;
}
section[data-testid="stSidebar"] * { color: #cbd5e1; }
section[data-testid="stSidebar"] h1 { color: #f1f5f9 !important; font-size: 1.3rem !important; }
section[data-testid="stSidebar"] .stButton button {
    background: #1e293b;
    color: #e2e8f0;
    border: 1px solid #334155;
    border-radius: 8px;
}
section[data-testid="stSidebar"] .stButton button:hover {
    background: #334155;
    border-color: #475569;
}
section[data-testid="stSidebar"] hr { border-color: #1e293b; }

.source-pill {
    display: inline-block;
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.78rem;
    margin: 3px 2px;
    color: #94a3b8;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_data(ttl=10)
def api_health() -> bool:
    try:
        return requests.get(f"{API_URL}/health", timeout=3).status_code == 200
    except Exception:
        return False

@st.cache_data(ttl=30)
def fetch_collections() -> list:
    try:
        r = requests.get(f"{API_URL}/collections", timeout=5)
        r.raise_for_status()
        return r.json().get("collections", [])
    except Exception:
        return []

# NOUVEAU: Fonction générateur pour le streaming de la réponse
def stream_response(prompt: str):
    """
    Exemple de fonction pour gérer le streaming si le backend le supporte.
    Attend une route d'API qui renvoie du Server-Sent Events (SSE) ou des chunks.
    """
    try:
        # Note: Adapte le endpoint si nécessaire (ex: /generate_stream)
        with requests.post(f"{API_URL}/generate", json={"question": prompt}, stream=True, timeout=API_TIMEOUT) as r:
            r.raise_for_status()
            for chunk in r.iter_content(chunk_size=1024, decode_unicode=True):
                if chunk:
                    yield chunk
    except Exception as e:
        yield f"⚠️ Erreur: {str(e)}"

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("# ⚡ RAGLight")

    healthy = api_health()
    st.caption(f"{'🟢' if healthy else '🔴'} {'Connected' if healthy else 'Unreachable'} · {API_URL}")

    st.divider()

    st.markdown("**📥 Import knowledge**")

    with st.expander("Upload files", expanded=True):
        uploaded = st.file_uploader("files", accept_multiple_files=True, label_visibility="collapsed")
        if st.button("Ingest files", use_container_width=True, disabled=not uploaded):
            with st.spinner("Ingesting…"):
                # ATTENTION: f.getvalue() met tout en RAM. OK pour les petits fichiers.
                files = [("files", (f.name, f.getvalue())) for f in uploaded]
                try:
                    r = requests.post(f"{API_URL}/ingest/upload", files=files, timeout=300)
                    r.raise_for_status()
                    st.success(r.json().get("message", "Files ingested successfully"))
                    fetch_collections.clear()
                except Exception as e:
                    st.error(str(e))

    with st.expander("Add a directory", expanded=False):
        st.caption("Path on the machine running the server")
        dir_path = st.text_input("dir", placeholder="/path/to/my/docs", label_visibility="collapsed")
        if st.button("Ingest directory", use_container_width=True, disabled=not dir_path):
            with st.spinner("Ingesting…"):
                try:
                    r = requests.post(f"{API_URL}/ingest", json={"data_path": dir_path}, timeout=300)
                    r.raise_for_status()
                    st.success(r.json().get("message", "Directory ingested successfully"))
                    fetch_collections.clear()
                except Exception as e:
                    st.error(str(e))

    st.divider()

    col_title, col_refresh = st.columns([4, 1])
    with col_title:
        st.markdown("**📚 Knowledge sources**")
    with col_refresh:
        if st.button("↺", help="Refresh sources"):
            fetch_collections.clear()
            st.rerun()

    collections = fetch_collections()
    if collections:
        pills = "".join(f'<span class="source-pill">{c}</span>' for c in collections)
        st.markdown(pills, unsafe_allow_html=True)
    else:
        st.caption("No collections yet.")

    st.divider()

    if st.button("🗑 Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ── Chat ──────────────────────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("""
    <div style="text-align:center; padding-top:12vh; color:#6b7280">
        <div style="font-size:2.5rem">⚡</div>
        <div style="font-size:1.8rem; font-weight:700; color:#111827; margin:0.5rem 0">
            What can I help you with?
        </div>
        <div style="font-size:1rem">
            Ask anything about your documents,<br>or import knowledge from the sidebar.
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

if prompt := st.chat_input("Message RAGLight…", disabled=not healthy):
    # 1. Ajouter le message utilisateur à l'historique et l'afficher
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Générer et afficher la réponse de l'assistant
    with st.chat_message("assistant"):
        # OPTION A: Rendu avec Streaming (recommandé si ton API le supporte)
        # full_response = st.write_stream(stream_response(prompt))
        
        # OPTION B: Ton code d'origine (Bloquant) - Actif par défaut ici
        with st.spinner("Thinking..."):
            try:
                # Utilisation de la nouvelle variable API_TIMEOUT
                r = requests.post(f"{API_URL}/generate", json={"question": prompt}, timeout=API_TIMEOUT)
                r.raise_for_status()
                full_response = r.json()["answer"]
            except Exception as e:
                full_response = f"⚠️ {e}"
        
        st.markdown(full_response)

    # 3. Sauvegarder la réponse dans l'historique
    st.session_state.messages.append({"role": "assistant", "content": full_response})