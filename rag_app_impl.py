import os
import io
import re
import zipfile
from typing import List, Tuple
from pathlib import Path
from xml.etree import ElementTree as ET

import streamlit as st
from dotenv import load_dotenv

from pdfminer.high_level import extract_text

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS


# ----------------------------
# Page & Theme
# ----------------------------
st.set_page_config(
    page_title="Contract Policy RAG Chatbot",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
  :root {
    --ink: #e2e8f0;
    --muted: #94a3b8;
    --line: #1e293b;
    --surface: rgba(20, 30, 52, 0.78);
    --surface-strong: #0f172a;
    --panel: #131c33;
    --panel-2: #182441;
    --brand: #3b82f6;
    --brand-deep: #93c5fd;
    --accent: #60a5fa;
    --accent-soft: #93c5fd;
    --warning: #fbbf24;
    --shadow: 0 18px 48px rgba(0, 0, 0, 0.55);
  }

  .stApp {
    color: var(--ink);
    background:
      radial-gradient(circle at 8% 8%, rgba(59, 130, 246, 0.22), transparent 32%),
      radial-gradient(circle at 92% 0%, rgba(96, 165, 250, 0.16), transparent 30%),
      linear-gradient(180deg, #070b1a 0%, #0d1428 48%, #08101f 100%);
  }

  /* Hide the default left sidebar (settings now live in a right column) */
  section[data-testid="stSidebar"] {
    display: none !important;
  }
  [data-testid="stSidebarCollapsedControl"] {
    display: none !important;
  }
  div[data-testid="stAppViewContainer"] > section.main {
    margin-left: 0 !important;
  }

  .block-container {
    padding-top: 1.25rem;
    max-width: 1280px;
  }

  .hero {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 28px;
    padding: 30px 34px;
    margin-bottom: 26px;
    border: 1px solid rgba(96, 165, 250, 0.22);
    border-radius: 14px;
    background:
      linear-gradient(135deg, rgba(15, 23, 42, 0.92), rgba(15, 23, 42, 0.72)),
      linear-gradient(135deg, rgba(59, 130, 246, 0.35), rgba(96, 165, 250, 0.18));
    box-shadow: var(--shadow);
  }

  .hero h1 {
    margin: 0;
    color: #f1f5f9;
    font-size: clamp(2rem, 4.2vw, 4.4rem);
    line-height: 1.0;
    font-weight: 800;
    letter-spacing: -0.01em;
  }

  .hero p {
    max-width: 720px;
    margin: 14px 0 0;
    color: var(--muted);
    font-size: 1rem;
    line-height: 1.6;
  }

  .hero-badges {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 18px;
  }

  .pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 12px;
    border: 1px solid rgba(96, 165, 250, 0.35);
    border-radius: 999px;
    background: rgba(59, 130, 246, 0.18);
    color: #bfdbfe;
    font-size: 0.78rem;
    font-weight: 700;
  }

  .hero-side {
    min-width: 220px;
    padding: 16px 18px;
    border: 1px solid rgba(96, 165, 250, 0.28);
    border-radius: 12px;
    background: linear-gradient(160deg, rgba(30, 41, 59, 0.88), rgba(30, 58, 138, 0.45));
  }

  .hero-side strong {
    display: block;
    color: #dbeafe;
    font-size: 1.8rem;
    line-height: 1;
    letter-spacing: 0.02em;
  }

  .hero-side span {
    display: block;
    margin-top: 6px;
    color: var(--muted);
    font-size: 0.82rem;
  }

  .panel {
    padding: 18px;
    border: 1px solid rgba(96, 165, 250, 0.18);
    border-radius: 12px;
    background: var(--surface);
    box-shadow: 0 10px 32px rgba(0, 0, 0, 0.45);
  }

  .callout {
    padding: 16px 18px;
    border: 1px solid rgba(96, 165, 250, 0.28);
    border-left: 4px solid var(--accent);
    border-radius: 10px;
    background: rgba(30, 58, 138, 0.30);
    color: #cbd5e1;
    line-height: 1.55;
  }

  .settings-panel {
    padding: 20px 18px 4px;
    border: 1px solid rgba(96, 165, 250, 0.22);
    border-radius: 14px;
    background: linear-gradient(180deg, rgba(20, 30, 52, 0.82), rgba(15, 23, 42, 0.92));
    box-shadow: 0 16px 38px rgba(0, 0, 0, 0.45);
    position: sticky;
    top: 1rem;
  }

  .settings-panel h3 {
    color: #dbeafe !important;
    margin: 0 0 4px 0 !important;
    font-size: 1.05rem !important;
    letter-spacing: 0.02em;
  }

  .settings-panel .settings-sub {
    color: #93a4c1 !important;
    font-size: 0.82rem;
    margin-bottom: 8px;
  }

  .settings-divider {
    height: 1px;
    background: rgba(96, 165, 250, 0.18);
    margin: 14px 0 10px;
  }

  .hint {
    color: var(--muted);
    font-size: 0.9rem;
    line-height: 1.5;
  }

  .section-title {
    display: flex;
    align-items: baseline;
    gap: 10px;
    margin: 26px 0 12px;
    white-space: nowrap;
  }

  .section-title span:first-child {
    display: inline;
    width: auto;
    height: auto;
    border-radius: 0;
    background: none;
    box-shadow: none;
    padding: 0;
    color: #60a5fa;
    font-size: 1.18rem;
    font-weight: 800;
    line-height: 1.2;
  }

  .section-title h2 {
    margin: 0;
    color: #dbeafe;
    font-size: 1.18rem;
    font-weight: 700;
    letter-spacing: 0;
    line-height: 1.2;
  }

  .status-ready, .status-waiting {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    padding: 14px 16px;
    border-radius: 10px;
    border: 1px solid rgba(96, 165, 250, 0.20);
    background: rgba(20, 30, 52, 0.75);
  }

  .status-ready strong { color: #93c5fd; }
  .status-waiting strong { color: var(--warning); }
  .status-ready span, .status-waiting span { color: var(--muted); }

  .stButton > button {
    border-radius: 10px;
    border: 1px solid rgba(96, 165, 250, 0.28);
    background: rgba(30, 41, 59, 0.85);
    color: #dbeafe;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.35);
    transition: all 140ms ease;
  }

  .stButton > button:hover {
    border-color: rgba(96, 165, 250, 0.65);
    color: #ffffff;
    background: rgba(37, 99, 235, 0.32);
    transform: translateY(-1px);
    box-shadow: 0 8px 20px rgba(0, 0, 0, 0.5);
  }

  div[data-testid="stMetric"] {
    padding: 14px 16px;
    border: 1px solid rgba(96, 165, 250, 0.20);
    border-radius: 10px;
    background: rgba(20, 30, 52, 0.80);
  }

  div[data-testid="stChatMessage"] {
    border-radius: 10px;
    border: 1px solid rgba(96, 165, 250, 0.18);
    background: rgba(20, 30, 52, 0.78);
  }

  div[data-testid="stChatMessage"] p,
  div[data-testid="stChatMessage"] li,
  div[data-testid="stChatMessage"] span {
    color: var(--ink) !important;
  }

  .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {
    color: #e2e8f0;
  }

  .stApp p, .stApp li, .stApp label, .stApp span {
    color: var(--ink);
  }

  .stApp [data-testid="stCaptionContainer"],
  .stApp small {
    color: var(--muted) !important;
  }

  .stApp [data-testid="stMarkdownContainer"] p,
  .stApp [data-testid="stMarkdownContainer"] li {
    color: var(--ink);
  }

  .stApp input, .stApp textarea, .stApp select,
  div[data-baseweb="input"] input,
  div[data-baseweb="select"] div,
  div[data-baseweb="textarea"] textarea {
    color: #e2e8f0 !important;
    background: rgba(15, 23, 42, 0.85) !important;
  }

  div[data-baseweb="input"], div[data-baseweb="select"] > div, div[data-baseweb="textarea"] {
    background: rgba(15, 23, 42, 0.85) !important;
    border-color: rgba(96, 165, 250, 0.28) !important;
  }

  /* File uploader (drag-drop area) */
  [data-testid="stFileUploader"] section {
    background: rgba(20, 30, 52, 0.65) !important;
    border: 1px dashed rgba(96, 165, 250, 0.35) !important;
    border-radius: 12px !important;
  }
  [data-testid="stFileUploader"] section * {
    color: var(--ink) !important;
  }
  [data-testid="stFileUploader"] button {
    background: rgba(37, 99, 235, 0.35) !important;
    color: #dbeafe !important;
    border: 1px solid rgba(96, 165, 250, 0.35) !important;
  }

  div[data-testid="stMetricValue"] {
    color: #dbeafe !important;
  }

  div[data-testid="stMetricLabel"] {
    color: var(--muted) !important;
  }

  div[data-baseweb="slider"] [role="slider"] {
    background: var(--brand) !important;
    border-color: var(--accent) !important;
  }

  div[data-baseweb="slider"] div[data-testid="stTickBar"] * {
    color: var(--muted) !important;
  }

  .stChatInput textarea, .stChatInput input {
    color: #e2e8f0 !important;
    background: rgba(15, 23, 42, 0.92) !important;
  }
  [data-testid="stChatInput"] {
    background: rgba(15, 23, 42, 0.92) !important;
    border: 1px solid rgba(96, 165, 250, 0.30) !important;
    border-radius: 10px;
  }

  /* Expander */
  [data-testid="stExpander"] {
    background: rgba(20, 30, 52, 0.70) !important;
    border: 1px solid rgba(96, 165, 250, 0.18) !important;
    border-radius: 10px;
  }
  [data-testid="stExpander"] summary, [data-testid="stExpander"] p {
    color: var(--ink) !important;
  }

  /* Alerts */
  [data-testid="stAlert"] {
    background: rgba(20, 30, 52, 0.78) !important;
    color: var(--ink) !important;
    border: 1px solid rgba(96, 165, 250, 0.22) !important;
    border-radius: 10px;
  }

  code, pre {
    background: rgba(15, 23, 42, 0.92) !important;
    color: #93c5fd !important;
    border-radius: 6px;
  }

  .footer {
    text-align: center;
    color: var(--muted);
    margin: 24px 0 8px;
    font-size: 0.86rem;
  }

  @media (max-width: 760px) {
    .hero {
      display: block;
      padding: 22px;
    }
    .hero-side {
      margin-top: 18px;
      min-width: 0;
    }
  }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="hero">
  <div>
    <h1>Contract Policy RAG</h1>
    <p>업로드한 계약업무지침만 근거로 조항, 리스크, 승인 절차를 빠르게 확인하는 업무형 AI 챗봇입니다.</p>
    <div class="hero-badges">
      <span class="pill">Document-grounded</span>
      <span class="pill">Source-aware</span>
      <span class="pill">FAISS Retrieval</span>
      <span class="pill">OpenAI</span>
    </div>
  </div>
  <div class="hero-side">
    <strong>RAG</strong>
    <span>계약 문서 검색과 답변 생성을 한 화면에서 관리</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)


# ----------------------------
# Helpers
# ----------------------------
def load_api_key() -> str:
    load_dotenv(override=False)
    api_key = os.getenv("OPENAI_API_KEY", "")
    if "openai_api_key" in st.session_state and st.session_state.openai_api_key:
        api_key = st.session_state.openai_api_key
    return api_key


def load_deploy_link() -> str:
    """Load deployed app URL from env or DEPLOY_LINK.txt at repo root."""
    link = os.getenv("STREAMLIT_DEPLOY_URL") or os.getenv("DEPLOY_URL") or ""
    if link:
        return link.strip()
    try:
        path = Path(__file__).resolve().parent.parent / "DEPLOY_LINK.txt"
        if path.exists():
            content = path.read_text(encoding="utf-8").strip()
            return content
    except Exception:
        pass
    return ""


def extract_docx_text(raw_bytes: bytes) -> str:
    """Extract readable text from a DOCX file using only the standard library."""
    with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
        xml_bytes = zf.read("word/document.xml")

    root = ET.fromstring(xml_bytes)
    namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    paragraphs = []
    for paragraph in root.findall(".//w:p", namespace):
        runs = [node.text or "" for node in paragraph.findall(".//w:t", namespace)]
        if runs:
            paragraphs.append("".join(runs))
    return "\n".join(paragraphs)


def extract_legacy_doc_text(raw_bytes: bytes) -> str:
    """
    Best-effort extraction for old OLE .doc files.
    This handles the provided contract guideline file, whose Korean text is visible as UTF-16LE.
    """
    decoded = raw_bytes.decode("utf-16le", errors="ignore")
    candidates = re.findall(r"[가-힣A-Za-z0-9()\[\]{}.,;:!?%/ㆍ·\-~\s]{8,}", decoded)
    text = "\n".join(part.strip() for part in candidates if part.strip())
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def read_uploaded_file(file) -> Tuple[str, str]:
    """
    Returns (text, source_name)
    Supports PDF, TXT, DOCX, and best-effort legacy DOC.
    """
    filename = file.name
    name_lower = filename.lower()
    raw_bytes = file.read()

    if name_lower.endswith(".pdf"):
        try:
            text = extract_text(io.BytesIO(raw_bytes))
        except Exception as e:
            raise RuntimeError(f"PDF 파싱 실패: {e}")
    elif name_lower.endswith(".txt"):
        try:
            text = raw_bytes.decode("utf-8", errors="ignore")
        except Exception:
            text = raw_bytes.decode("cp949", errors="ignore")
    elif name_lower.endswith(".docx"):
        try:
            text = extract_docx_text(raw_bytes)
        except Exception as e:
            raise RuntimeError(f"DOCX 파싱 실패: {e}")
    elif name_lower.endswith(".doc"):
        text = extract_legacy_doc_text(raw_bytes)
        if len(text) < 200:
            raise RuntimeError(
                "DOC 파일에서 충분한 텍스트를 추출하지 못했습니다. PDF 또는 TXT로 변환 후 업로드하세요."
            )
    else:
        raise RuntimeError("지원하지 않는 파일 형식입니다. PDF, TXT, DOC, DOCX 파일을 업로드하세요.")

    text = text.strip()
    if not text:
        raise RuntimeError("파일에서 텍스트를 추출하지 못했습니다.")
    return text, filename


def chunk_documents(texts_with_sources: List[Tuple[str, str]], chunk_size: int, chunk_overlap: int) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    docs: List[Document] = []
    for text, source in texts_with_sources:
        for chunk_number, chunk in enumerate(splitter.split_text(text), start=1):
            docs.append(Document(page_content=chunk, metadata={"source": source, "chunk_number": chunk_number}))
    return docs


def build_vectorstore(docs: List[Document], api_key: str) -> FAISS:
    embeddings = OpenAIEmbeddings(
        api_key=api_key,
        model="text-embedding-3-small",
        tiktoken_enabled=False,
        check_embedding_ctx_length=False,
    )
    vs = FAISS.from_documents(docs, embeddings)
    return vs


def format_context(docs: List[Document]) -> str:
    blocks = []
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source", "unknown")
        chunk_number = d.metadata.get("chunk_number", i)
        blocks.append(f"[문서 {i}] (출처: {src}, 청크: {chunk_number})\n{d.page_content}")
    return "\n\n".join(blocks)


def translate_to_korean_if_needed(query: str, api_key: str, model: str) -> str:
    """
    Contract guidelines are in Korean, but the UI/sample questions are in English.
    If the query has no Korean characters, translate it to Korean so the embedding
    query matches the Korean document chunks. Korean queries pass through unchanged.
    """
    if re.search(r"[가-힣]", query):
        return query
    llm = ChatOpenAI(model=model, temperature=0, api_key=api_key)
    translation_prompt = (
        "Translate the following question into natural Korean as used in Korean "
        "business/contract (계약업무) context. Preserve the meaning and use the "
        "Korean terminology a procurement or contract officer would use "
        "(e.g., '계약서', '수의계약', '계약보증금', '지체상금', '대손'). "
        "Output ONLY the Korean translation, with no quotes or extra commentary.\n\n"
        f"Question: {query}\n\n한국어 번역:"
    )
    res = llm.invoke([{"role": "user", "content": translation_prompt}])
    translated = res.content if hasattr(res, "content") else str(res)
    return translated.strip()


def generate_answer(
    query: str,
    retrieved_docs: List[Document],
    api_key: str,
    model: str,
    temperature: float,
    question_type: str,
) -> str:
    context_text = format_context(retrieved_docs)
    system_prompt = """You are a contract policy assistant that answers based only on the uploaded “Contract Guidelines” document.

Follow these rules strictly:
1) Answer only from the uploaded document.
2) Do not guess or use outside knowledge.
3) For judgment questions, clearly state one of: “Allowed”, “Not allowed”, or “Conditionally allowed”.
4) Always mention the relevant article number if available, such as Article 7, Article 17, Article 26, etc.
5) If the answer involves risk, exception, approval, guarantee, retroactive contract, private contract, delayed performance, or bad debt, clearly add a warning.
6) If the document does not contain enough information, say that the document does not provide sufficient grounds.
7) Keep the answer practical and business-friendly."""

    user_prompt = (
        "Question Type: " + question_type + "\n\n"
        "Question:\n" + query + "\n\n"
        "Use only the retrieved Contract Guidelines context below. Answer in Korean unless the user asks otherwise.\n"
        "Force the response to follow this exact structure:\n\n"
        "[Decision]\n"
        "- Allowed / Not allowed / Conditionally allowed / Not applicable\n\n"
        "[Relevant Article]\n"
        "- Article number and short summary\n\n"
        "[Explanation]\n"
        "- Practical interpretation based on the document\n\n"
        "[Required Action]\n"
        "- What the user should prepare, check, or approve\n\n"
        "[Warning]\n"
        "- Risk, exception, or missing information\n\n"
        "Retrieved Contract Guidelines context:\n\n" + context_text
    )

    llm = ChatOpenAI(model=model, temperature=temperature, api_key=api_key)
    res = llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ])
    return res.content if hasattr(res, "content") else str(res)


# ----------------------------
# Layout: main on left, settings panel on right
# ----------------------------
main_col, right_col = st.columns([3, 1], gap="large")

with right_col:
    st.markdown('<div class="settings-panel">', unsafe_allow_html=True)
    st.markdown("### ⚙ Settings")
    st.markdown('<div class="settings-sub">검색 품질과 응답 스타일을 조정합니다.</div>', unsafe_allow_html=True)

    api_key_input = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="sk-...",
        help="환경변수 OPENAI_API_KEY 또는 여기 입력 (저장 안 됨)",
    )
    if api_key_input:
        st.session_state.openai_api_key = api_key_input

    st.markdown('<div class="settings-divider"></div>', unsafe_allow_html=True)
    st.markdown("**Response**")
    model = st.selectbox("모델", ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"], index=0)
    temperature = st.slider("창의성 (temperature)", 0.0, 1.0, 0.2, 0.1)
    question_type = st.selectbox(
        "질문 유형",
        ["Auto", "Definition", "Judgment", "Procedure", "Risk Check", "Summary"],
        index=0,
    )

    st.markdown('<div class="settings-divider"></div>', unsafe_allow_html=True)
    st.markdown("**Retrieval**")
    chunk_size = st.slider("청크 크기", 200, 2000, 700, 50)
    chunk_overlap = st.slider("청크 중첩", 0, 400, 150, 10)
    top_k = st.slider("검색 문서 수 (k)", 1, 10, 5, 1)

    st.markdown('<div class="settings-divider"></div>', unsafe_allow_html=True)
    clear_btn = st.button("Reset index and chat", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ----------------------------
# Session State
# ----------------------------
if clear_btn:
    for key in ["vectorstore", "docs_info", "messages"]:
        if key in st.session_state:
            del st.session_state[key]

if "messages" not in st.session_state:
    st.session_state.messages = []  # list of (role, content)


api_key = load_api_key()

with main_col:
    # ----------------------------
    # Upload & Index Build
    # ----------------------------
    st.markdown(
        '<div class="section-title"><span>1</span><h2>문서 업로드</h2></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="callout">Contract Guidelines 문서를 업로드하면 계약업무지침 전용 벡터 인덱스를 생성합니다. PDF, TXT, DOCX를 권장하며, 구형 DOC는 문서 구조에 따라 추출 품질이 달라질 수 있습니다.</div>',
        unsafe_allow_html=True,
    )

    uploads = st.file_uploader(
        "문서 선택",
        type=["pdf", "txt", "doc", "docx"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    build_col1, build_col2 = st.columns([1, 3])
    with build_col1:
        build_clicked = st.button("Build index", use_container_width=True)
    with build_col2:
        st.markdown("<span class=hint>업로드 후 인덱스를 생성하면 챗봇이 해당 문서를 근거로 답변합니다.</span>", unsafe_allow_html=True)

    if build_clicked:
        if not api_key:
            st.error("OpenAI API Key가 필요합니다. 오른쪽 설정 패널에 입력하거나 환경변수를 설정하세요.")
        elif not uploads:
            st.error("최소 1개 이상의 파일을 업로드하세요.")
        else:
            try:
                texts_with_sources: List[Tuple[str, str]] = []
                for f in uploads:
                    text, src = read_uploaded_file(f)
                    texts_with_sources.append((text, src))

                docs = chunk_documents(texts_with_sources, chunk_size, chunk_overlap)
                vs = build_vectorstore(docs, api_key)
                st.session_state.vectorstore = vs
                st.session_state.docs_info = {
                    "num_files": len(uploads),
                    "num_chunks": len(docs),
                    "files": [f.name for f in uploads],
                }
                st.success("인덱스 생성 완료! 계약업무지침 기반 질문을 입력하세요.")
            except Exception as e:
                st.error(f"인덱스 생성 실패: {e}")


    # ----------------------------
    # Recommended Questions
    # ----------------------------
    st.markdown(
        '<div class="section-title"><span>2</span><h2>추천 질문</h2></div>',
        unsafe_allow_html=True,
    )
    demo_questions = [
        "계약서 작성을 생략할 수 있는 경우는 언제인가요?",
        "수의계약은 어떤 경우에 허용되나요?",
        "계약보증금 면제 조건은 무엇인가요?",
        "지체상금률은 얼마인가요?",
        "91일 경과 후 대손이 발생한 경우 어떻게 처리해야 하나요?",
        "1천만 원 이상 계약에서 선금 지급이 가능한가요?",
    ]
    question_cols = st.columns(2)
    for i, sample_question in enumerate(demo_questions):
        with question_cols[i % 2]:
            if st.button(sample_question, key=f"demo_question_{i}", use_container_width=True):
                st.session_state.pending_question = sample_question


    # ----------------------------
    # Index Status
    # ----------------------------
    st.markdown(
        '<div class="section-title"><span>3</span><h2>인덱스 상태</h2></div>',
        unsafe_allow_html=True,
    )
    status_container = st.container()
    with status_container:
        if "vectorstore" in st.session_state:
            info = st.session_state.get("docs_info", {})
            st.markdown(
                '<div class="status-ready"><strong>Ready</strong><span>문서 검색 인덱스가 준비되었습니다.</span></div>',
                unsafe_allow_html=True,
            )
            metric_cols = st.columns(3)
            metric_cols[0].metric("Files", info.get("num_files", 0))
            metric_cols[1].metric("Chunks", info.get("num_chunks", 0))
            metric_cols[2].metric("Top-k", top_k)
            if info.get("files"):
                st.caption("업로드 파일: " + ", ".join(info["files"]))
        else:
            st.markdown(
                '<div class="status-waiting"><strong>Waiting</strong><span>아직 인덱스가 생성되지 않았습니다.</span></div>',
                unsafe_allow_html=True,
            )


    # ----------------------------
    # Chat Interface
    # ----------------------------
    st.markdown(
        '<div class="section-title"><span>4</span><h2>챗봇</h2></div>',
        unsafe_allow_html=True,
    )
    chat_container = st.container()

    with chat_container:
        for role, content in st.session_state.messages:
            with st.chat_message(role):
                st.markdown(content)

        pending_question = st.session_state.pop("pending_question", None)
        typed_question = st.chat_input("계약업무지침에 대해 질문하세요…")
        user_input = pending_question or typed_question

        if user_input:
            if "vectorstore" not in st.session_state:
                st.error("먼저 파일을 업로드하고 인덱스를 생성하세요.")
            elif not api_key:
                st.error("OpenAI API Key가 필요합니다.")
            else:
                st.session_state.messages.append(("user", user_input))
                with st.chat_message("user"):
                    st.markdown(user_input)

                try:
                    with st.spinner("질문을 한국어로 변환 중…"):
                        korean_query = translate_to_korean_if_needed(user_input, api_key, model)

                    retriever = st.session_state.vectorstore.as_retriever(
                        search_type="similarity",
                        search_kwargs={"k": top_k},
                    )
                    # LangChain retrievers use .invoke() in recent versions
                    retrieved_docs: List[Document] = retriever.invoke(korean_query)

                    with st.spinner("답변 생성 중…"):
                        answer = generate_answer(korean_query, retrieved_docs, api_key, model, temperature, question_type)

                    with st.chat_message("assistant"):
                        st.caption("계약업무지침 기반 답변입니다")
                        if korean_query != user_input:
                            st.caption(f"검색에 사용된 한국어 질의: {korean_query}")
                        st.markdown(answer)
                        with st.expander("참조 조항 보기"):
                            for i, d in enumerate(retrieved_docs, start=1):
                                src = d.metadata.get("source", "unknown")
                                chunk_number = d.metadata.get("chunk_number", i)
                                preview = d.page_content[:1000] + ("…" if len(d.page_content) > 1000 else "")
                                st.markdown(f"**참조 {i}**")
                                st.markdown(f"- Source file name: `{src}`")
                                st.markdown(f"- Retrieved chunk number: `{chunk_number}`")
                                st.write(preview)

                    st.session_state.messages.append(("assistant", answer))
                except Exception as e:
                    err_msg = f"오류가 발생했습니다: {e}"
                    with st.chat_message("assistant"):
                        st.error(err_msg)
                    st.session_state.messages.append(("assistant", err_msg))


st.markdown("---")
st.markdown(
    "<div class=footer>© Contract Guidelines RAG Chatbot · Streamlit · LangChain · FAISS</div>",
    unsafe_allow_html=True,
)


