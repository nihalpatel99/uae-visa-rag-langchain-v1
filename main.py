import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import streamlit as st
from langchain.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.agents import create_agent

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="UAE Visa Assistant",
    page_icon="🇦🇪",
    layout="centered",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Sora', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0a0a0f 0%, #0d1117 50%, #0a0f1a 100%);
    color: #e8eaf0;
}

#MainMenu, footer, header { visibility: hidden; }

.title-block {
    text-align: center;
    padding: 2.5rem 0 1.5rem;
}
.title-block h1 {
    font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(90deg, #c9a84c, #f0d080, #c9a84c);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.5px;
    margin: 0;
}
.title-block p {
    color: #6b7280;
    font-size: 0.9rem;
    margin-top: 0.4rem;
    font-weight: 300;
}

[data-testid="stChatMessage"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(201,168,76,0.12) !important;
    border-radius: 12px !important;
    margin-bottom: 0.75rem !important;
    padding: 0.9rem 1.1rem !important;
}

[data-testid="stChatMessage"][data-testid*="user"] {
    border-color: rgba(201,168,76,0.25) !important;
    background: rgba(201,168,76,0.05) !important;
}

.step-box {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 8px;
    padding: 0.6rem 0.9rem;
    margin-top: 0.5rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: #6b7280;
    white-space: pre-wrap;
    word-break: break-word;
}

[data-testid="stChatInputContainer"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(201,168,76,0.2) !important;
    border-radius: 12px !important;
}

[data-testid="stChatInputContainer"] textarea {
    color: #e8eaf0 !important;
    font-family: 'Sora', sans-serif !important;
    font-size: 0.92rem !important;
}

[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.02) !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
}

.sidebar-section {
    background: rgba(201,168,76,0.07);
    border: 1px solid rgba(201,168,76,0.15);
    border-radius: 10px;
    padding: 1rem;
    margin-bottom: 1rem;
    font-size: 0.82rem;
    color: #9ca3af;
    line-height: 1.6;
}

.sidebar-section strong {
    color: #c9a84c;
    display: block;
    margin-bottom: 0.4rem;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.status-dot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    margin-right: 6px;
    vertical-align: middle;
}
.dot-green { background: #22c55e; box-shadow: 0 0 6px #22c55e88; }
.dot-red   { background: #ef4444; box-shadow: 0 0 6px #ef444488; }
</style>
""", unsafe_allow_html=True)



if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent" not in st.session_state:
    st.session_state.agent = None
if "agent_error" not in st.session_state:
    st.session_state.agent_error = None



with st.sidebar:
   

    api_key     = st.text_input("OpenAI API Key", type="password", placeholder="sk-…")
    chat_model  = st.selectbox("Chat Model", ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"])
    embed_model = st.selectbox("Embedding Model", ["text-embedding-3-small"])
    persist_dir = st.text_input("Chroma Directory", value="uae_visa_index")

    if st.button("Initialize Agent", use_container_width=True):
        if not api_key:
            st.error("Please enter your OpenAI API key.")
        else:
            with st.spinner("Loading models & vector store…"):
                try:
                    embeddings  = OpenAIEmbeddings(model=embed_model, dimensions=1024, openai_api_key=api_key)
                    vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)

                    @tool(response_format="content_and_artifact")
                    def retrieve_context(query: str):
                        """Retrieve information to help answer a query about UAE visas."""
                        retrieved_docs = vectorstore.similarity_search(query, k=2)
                        serialized = "\n\n".join(
                            f"Source: {doc.metadata}\nContent: {doc.page_content}"
                            for doc in retrieved_docs
                        )
                        return serialized, retrieved_docs

                    model  = ChatOpenAI(model=chat_model, openai_api_key=api_key)
                    tools  = [retrieve_context]
                    prompt = (
                        "You have access to a tool that retrieves context regarding UAE visa queries "
                        "for residents, investors and visitors. Use the tools to help answer user queries."
                    )
                    st.session_state.agent       = create_agent(model, tools, system_prompt=prompt)
                    st.session_state.agent_error = None
                    st.success("Agent ready!")
                except Exception as e:
                    st.session_state.agent_error = str(e)
                    st.error(f"Failed: {e}")

    st.markdown("---")

    if st.session_state.agent:
        st.markdown('<span class="status-dot dot-green"></span> Agent online', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-dot dot-red"></span> Agent offline', unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""
    <div class="sidebar-section">
        <strong>Sample Questions</strong>
        • What visa do I need to work in Dubai?<br>
        • How long can tourists stay in the UAE?<br>
        • Can investors get long-term residency?<br>
        • What documents are required for a family visa?
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-section">
        <strong>About</strong>
        Powered by OpenAI and a Chroma vector store indexed from UAE visa documents.
    </div>
    """, unsafe_allow_html=True)

    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ── Main area ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="title-block">
    <h1>🇦🇪 UAE Visa Assistant</h1>
    <p>Ask anything about UAE visas — for residents, investors & visitors</p>
</div>
""", unsafe_allow_html=True)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("steps"):
            with st.expander("🔍 Agent reasoning", expanded=False):
                st.markdown(f'<div class="step-box">{msg["steps"]}</div>', unsafe_allow_html=True)

if prompt := st.chat_input("Type your visa question…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if not st.session_state.agent:
            st.warning("Agent not initialized. Configure it in the sidebar and click **Initialize Agent**.")
        else:
            with st.spinner("Thinking…"):
                try:
                    steps_log   = []
                    final_reply = ""

                    for step in st.session_state.agent.stream(
                        {"messages": [{"role": "user", "content": prompt}]},
                        stream_mode="values",
                    ):
                        last_msg = step["messages"][-1]
                        if hasattr(last_msg, "type") and last_msg.type != "ai":
                            steps_log.append(str(last_msg))
                        final_reply = last_msg

                    if hasattr(final_reply, "content"):
                        answer = final_reply.content
                    else:
                        answer = str(final_reply)

                    steps_text = "\n\n".join(steps_log) if steps_log else ""

                    st.markdown(answer)
                    if steps_text:
                        with st.expander("🔍 Agent reasoning", expanded=False):
                            st.markdown(f'<div class="step-box">{steps_text}</div>', unsafe_allow_html=True)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "steps": steps_text,
                    })

                except Exception as e:
                    err = f"❌ Error: {e}"
                    st.error(err)
                    st.session_state.messages.append({"role": "assistant", "content": err})