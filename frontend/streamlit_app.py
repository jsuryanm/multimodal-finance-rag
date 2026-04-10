from __future__ import annotations

import requests
import streamlit as st

API_URL = "http://localhost:8000"

ROUTE_BADGES = {
    "summary":     ("📄", "summary",    "#1f6feb"),
    "chart":       ("🖼",  "chart",      "#8957e5"),
    "comparision": ("⚖️", "comparision","#f0883e"),
    "stock_price": ("📈", "stock_price","#3fb950"),
}

# session state defaults 
def _init_state():
    defaults = {
        "session_id": None,
        "session_id_b": None,
        "uploaded_filename_a": None,
        "uploaded_filename_b": None,
        "messages": [],
        "page_number": 1,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


#  backend health check 
def _check_backend() -> bool:
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        return r.status_code == 200
    except requests.exceptions.ConnectionError:
        return False


# ── sidebar ───────────────────────────────────────────────────────────────────
def _render_sidebar():
    with st.sidebar:
        st.title("📊 Finance RAG")
        st.divider()

        # Company A upload
        st.markdown("**Company A**")
        file_a = st.file_uploader(
            "Upload annual report PDF",
            type="pdf",
            key="uploader_a",
            label_visibility="collapsed",
        )
        if file_a and file_a.name != st.session_state.uploaded_filename_a:
            _upload_file(file_a, slot="a")

        if st.session_state.session_id:
            sid = st.session_state.session_id
            st.caption(f"Session: `{sid[:8]}…{sid[-4:]}`")

        st.divider()

        # Company B upload
        st.markdown("**Company B** *(optional — enables comparison)*")
        file_b = st.file_uploader(
            "Upload second PDF",
            type="pdf",
            key="uploader_b",
            label_visibility="collapsed",
        )
        if file_b and file_b.name != st.session_state.uploaded_filename_b:
            _upload_file(file_b, slot="b")

        if st.session_state.session_id_b:
            sid = st.session_state.session_id_b
            st.caption(f"Session: `{sid[:8]}…{sid[-4:]}`")

        st.divider()

        # Chart page number
        st.markdown("**Chart page**")
        st.session_state.page_number = st.number_input(
            "Page number for chart analysis",
            min_value=1,
            value=st.session_state.page_number,
            label_visibility="collapsed",
        )

        st.divider()

        if st.button("🗑 Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


# upload helper 
def _upload_file(file, slot: str):
    with st.sidebar:
        with st.spinner(f"Indexing {file.name}…"):
            try:
                response = requests.post(
                    f"{API_URL}/upload",
                    files={"file": (file.name, file.getvalue(), "application/pdf")},
                    timeout=300,
                )
                response.raise_for_status()
                data = response.json()
                if slot == "a":
                    st.session_state.session_id = data["session_id"]
                    st.session_state.uploaded_filename_a = file.name
                else:
                    st.session_state.session_id_b = data["session_id"]
                    st.session_state.uploaded_filename_b = file.name
                st.success(f"✓ {file.name} — {data['chunks']} chunks, {data['pages']} pages")
            except requests.exceptions.HTTPError as e:
                try:
                    detail = e.response.json().get("detail", str(e))
                except Exception:
                    detail = str(e)
                st.error(f"Upload failed: {detail}")
            except requests.exceptions.ReadTimeout:
                st.error(
                    "Upload timed out — the PDF is large and indexing is still running. "
                    "Try uploading a smaller file, or wait and refresh the page."
                )
            except requests.exceptions.ConnectionError:
                st.error("Cannot reach backend. Is it running?")


# ── chat history ──────────────────────────────────────────────────────────────
def _render_chat():
    for msg in st.session_state.messages:
        role = msg["role"]
        with st.chat_message(role):
            if role == "assistant" and msg.get("route"):
                route = msg["route"]
                emoji, label, color = ROUTE_BADGES.get(route, ("🤖", route, "#8b949e"))
                st.markdown(
                    f'<span style="font-size:11px;background:#21262d;'
                    f'border:1px solid #30363d;border-radius:10px;'
                    f'padding:1px 8px;color:{color};">{emoji} {label}</span>',
                    unsafe_allow_html=True,
                )
            st.markdown(msg["content"])


# ── streaming chat ────────────────────────────────────────────────────────────
def _stream_answer(question: str):
    payload = {
        "session_id": st.session_state.session_id,
        "session_id_b": st.session_state.session_id_b,
        "question": question,
        "page_number": st.session_state.page_number,
    }

    route = "summary"

    def token_generator():
        nonlocal route
        with requests.post(
            f"{API_URL}/chat/stream", json=payload, stream=True, timeout=120
        ) as response:
            for raw_line in response.iter_lines():
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                if data.startswith("[ERROR]"):
                    yield f"⚠️ {data[7:].strip()}"
                    break
                if data.startswith("[ROUTE:"):
                    route = data[7:-1]
                    continue
                yield data

    with st.chat_message("assistant"):
        full_response = st.write_stream(token_generator())

    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response or "",
        "route": route,
    })


# main 
def main():
    st.set_page_config(page_title="Finance RAG", page_icon="📊", layout="wide")
    _init_state()

    if not _check_backend():
        st.error(
            "⚠️ Backend not reachable at `http://localhost:8000` — "
            "run `uv run uvicorn backend.app:app --reload --port 8000` first."
        )
        return

    _render_sidebar()
    _render_chat()

    if not st.session_state.session_id:
        st.info("Upload a PDF in the sidebar to start asking questions.")
        return

    if question := st.chat_input("Ask about the annual report…"):
        st.session_state.messages.append({"role": "user", "content": question, "route": None})
        with st.chat_message("user"):
            st.markdown(question)
        try:
            _stream_answer(question)
        except requests.exceptions.ConnectionError:
            st.error("Lost connection to backend.")
        st.rerun()


if __name__ == "__main__":
    main()
