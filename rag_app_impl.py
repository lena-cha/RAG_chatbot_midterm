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
    page_icon="📄",
    layout="wide",
)

st.markdown(
    """
<style>
  .main { background: #f6fbf9; }
  .app-title {
    background: linear-gradient(90deg, #00704a, #2d8659);
    color: white; padding: 16px 24px; border-radius: 8px; font-weight: 700;
    display:flex; align-items:center; gap:10px; margin-bottom:16px;
  }
  .pill { display:inline-block; padding:2px 8px; border-radius:999px; background:#e8f5f0; color:#00704a; font-size:12px; margin-left:8px;}
  .card { background:white; padding:16px; border-radius:8px; border-left:4px solid #00704a; box-shadow:0 2px 6px rgba(0,0,0,0.06); color:#00704a; }
  .hint { color:#2d8659; font-size:13px; }
  .footer { text-align:center; color:#2d8659; margin-top:24px; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="app-title">
  <span style="font-size:26px">Contract Policy RAG Chatbot</span>
  <span class="pill">Contract Guidelines · RAG · FAISS · OpenAI</span>
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
# Sidebar Controls
# ----------------------------
with st.sidebar:
    st.markdown("### ⚙️ 설정")
    api_key_input = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="sk-...",
        help="환경변수 OPENAI_API_KEY 또는 여기 입력 (저장 안 됨)",
    )
    if api_key_input:
        st.session_state.openai_api_key = api_key_input

    model = st.selectbox("모델", ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"], index=0)
    temperature = st.slider("창의성 (temperature)", 0.0, 1.0, 0.2, 0.1)
    question_type = st.selectbox(
        "질문 유형",
        ["Auto", "Definition", "Judgment", "Procedure", "Risk Check", "Summary"],
        index=0,
    )
    chunk_size = st.slider("청크 크기", 200, 2000, 700, 50)
    chunk_overlap = st.slider("청크 중첩", 0, 400, 150, 10)
    top_k = st.slider("검색 문서 수 (k)", 1, 10, 5, 1)

    st.markdown("---")
    clear_btn = st.button("🗑️ 인덱스/대화 초기화")


# ----------------------------
# Session State
# ----------------------------
if clear_btn:
    for key in ["vectorstore", "docs_info", "messages"]:
        if key in st.session_state:
            del st.session_state[key]

if "messages" not in st.session_state:
    st.session_state.messages = []  # list of (role, content)


# ----------------------------
# Upload & Index Build
# ----------------------------
st.markdown("#### 1) 문서 업로드")
st.markdown(
    '<div class="card">Contract Guidelines 문서를 업로드하면 계약업무지침 전용 벡터 인덱스를 생성합니다.</div>',
    unsafe_allow_html=True,
)

uploads = st.file_uploader(
    "문서 업로드 (다중 선택 가능)",
    type=["pdf", "txt", "doc", "docx"],
    accept_multiple_files=True,
)
st.caption("PDF/TXT/DOCX를 권장합니다. 구형 DOC 파일은 텍스트 추출 품질이 문서 구조에 따라 달라질 수 있습니다.")

api_key = load_api_key()

build_col1, build_col2 = st.columns([1, 3])
with build_col1:
    build_clicked = st.button("🧠 인덱스 생성/갱신")
with build_col2:
    st.markdown("<span class=hint>업로드 후 인덱스 생성 버튼을 눌러주세요.</span>", unsafe_allow_html=True)

if build_clicked:
    if not api_key:
        st.error("OpenAI API Key가 필요합니다. 사이드바에 입력하거나 환경변수 설정하세요.")
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
st.markdown("#### 추천 질문")
demo_questions = [
    "When can a contract document be omitted?",
    "When is a private contract allowed?",
    "What are the conditions for guarantee deposit exemption?",
    "What is the penalty rate for delayed performance?",
    "What should be done when bad debt occurs after 91 days?",
    "Can advance payment be made for a 10 million KRW or higher contract?",
]
question_cols = st.columns(2)
for i, sample_question in enumerate(demo_questions):
    with question_cols[i % 2]:
        if st.button(sample_question, key=f"demo_question_{i}", use_container_width=True):
            st.session_state.pending_question = sample_question


# ----------------------------
# Index Status
# ----------------------------
st.markdown("#### 2) 인덱스 상태")
status_container = st.container()
with status_container:
    if "vectorstore" in st.session_state:
        info = st.session_state.get("docs_info", {})
        st.markdown(
            f"- 인덱스 상태: ✅ 준비됨  "+
            f"- 파일 수: {info.get('num_files', 0)}  "+
            f"- 청크 수: {info.get('num_chunks', 0)}"
        )
        if info.get("files"):
            st.caption("업로드 파일: " + ", ".join(info["files"]))
    else:
        st.markdown("- 인덱스 상태: ⏳ 아직 생성되지 않음")


# ----------------------------
# Chat Interface
# ----------------------------
st.markdown("#### 3) 챗봇")
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
                retriever = st.session_state.vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": top_k},
                )
                # LangChain retrievers use .invoke() in recent versions
                retrieved_docs: List[Document] = retriever.invoke(user_input)

                with st.spinner("답변 생성 중…"):
                    answer = generate_answer(user_input, retrieved_docs, api_key, model, temperature, question_type)

                with st.chat_message("assistant"):
                    st.caption("계약업무지침 기반 답변입니다")
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


