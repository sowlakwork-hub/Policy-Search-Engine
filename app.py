import re
import hashlib
from io import BytesIO

import streamlit as st
from pypdf import PdfReader
from docx import Document
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------
# PAGE SETTINGS
# -----------------------------
st.set_page_config(
    page_title="Policy AI Chatbot",
    page_icon="🤖",
    layout="centered",
)

MODEL_NAME = "all-MiniLM-L6-v2"


# -----------------------------
# TEXT HELPERS
# -----------------------------
def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def split_long_text(text: str, max_words: int = 120, overlap: int = 20):
    words = clean_text(text).split()
    if not words:
        return []

    chunks = []
    start = 0

    while start < len(words):
        end = min(start + max_words, len(words))
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(words):
            break

        start = max(0, end - overlap)

    return chunks


def split_into_sections(text: str, max_words: int = 120):
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    raw_sections = re.split(r"\n\s*\n", text)

    sections = []
    for part in raw_sections:
        part = clean_text(part)
        if not part:
            continue

        words = part.split()
        if len(words) <= max_words:
            sections.append(part)
        else:
            sections.extend(split_long_text(part, max_words=max_words, overlap=20))

    if not sections and clean_text(text):
        sections = split_long_text(text, max_words=max_words, overlap=20)

    return sections


def first_two_sentences(text: str) -> str:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if not sentences:
        return text.strip()
    return " ".join(sentences[:2]).strip()


# -----------------------------
# FILE READING
# -----------------------------
def extract_text_from_pdf(file_bytes: bytes) -> str:
    reader = PdfReader(BytesIO(file_bytes))
    parts = []

    for page in reader.pages:
        page_text = page.extract_text() or ""
        if page_text.strip():
            parts.append(page_text)

    return "\n\n".join(parts)


def extract_text_from_docx(file_bytes: bytes) -> str:
    doc = Document(BytesIO(file_bytes))
    parts = []

    for para in doc.paragraphs:
        if para.text.strip():
            parts.append(para.text)

    return "\n".join(parts)


def extract_text_from_txt(file_bytes: bytes) -> str:
    return file_bytes.decode("utf-8", errors="ignore")


def extract_text_from_file(uploaded_file) -> str:
    name = uploaded_file.name.lower()
    file_bytes = uploaded_file.getvalue()

    if name.endswith(".pdf"):
        return extract_text_from_pdf(file_bytes)
    elif name.endswith(".docx"):
        return extract_text_from_docx(file_bytes)
    elif name.endswith(".txt"):
        return extract_text_from_txt(file_bytes)
    else:
        return ""


# -----------------------------
# MODEL
# -----------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer(MODEL_NAME)


# -----------------------------
# KNOWLEDGE BASE
# -----------------------------
def build_knowledge_base(file_text: str):
    sections = split_into_sections(file_text, max_words=120)

    if not sections:
        return [], None, "No readable text was found in the uploaded file."

    model = load_model()
    embeddings = model.encode(sections, normalize_embeddings=True)

    return sections, embeddings, None


# -----------------------------
# QUERY ENHANCEMENT
# -----------------------------
def expand_query(question: str) -> str:
    q = question.lower().strip()

    semantic_map = {
        "vacation": "annual leave holiday leave days off employee leave",
        "leave": "annual leave vacation holiday leave days off employee leave",
        "picnic": "travel trip business travel official travel reimbursement",
        "trip": "travel reimbursement business travel official travel",
        "travel": "travel reimbursement business travel official travel",
        "remote work": "work from home wfh remote",
        "wfh": "work from home remote work",
        "insurance": "medical insurance health coverage reimbursement",
        "medical": "medical reimbursement health insurance hospital",
        "attendance": "office timing attendance check in late arrival",
        "salary": "payroll compensation employee payment",
        "security": "password confidential data privacy",
        "training": "training learning development course",
        "laptop": "device laptop usage software policy",
    }

    # Match longer phrases first
    for key in sorted(semantic_map.keys(), key=len, reverse=True):
        if key in q:
            q += " " + semantic_map[key]

    return q


# -----------------------------
# NORMAL CHAT REPLIES
# -----------------------------
def smart_reply(question: str):
    q = question.lower().strip()

    if any(word in q for word in ["hi", "hello", "hey"]):
        return "Hello 👋\n\nHow can I help you today?"

    if any(word in q for word in ["thanks", "thank you"]):
        return "You're welcome 😊"

    if any(word in q for word in ["bye", "goodbye"]):
        return "Goodbye 👋"

    return None


# -----------------------------
# SEARCH
# -----------------------------
def search_answer(question, sections, embeddings, threshold=0.12, top_k=3):
    normal_reply = smart_reply(question)
    if normal_reply:
        return normal_reply, []

    model = load_model()
    expanded_question = expand_query(question)

    question_embedding = model.encode(
        [expanded_question],
        normalize_embeddings=True
    )

    similarities = cosine_similarity(question_embedding, embeddings)[0]
    ranked_indices = similarities.argsort()[::-1]

    if len(ranked_indices) == 0:
        return "I could not find relevant information in the document.", []

    best_index = ranked_indices[0]
    best_score = similarities[best_index]

    top_indices = ranked_indices[: min(top_k, len(ranked_indices))]
    top_matches = [(sections[i], float(similarities[i])) for i in top_indices]

    if best_score < threshold:
        return "I could not find relevant information in the document.", top_matches

    best_section = sections[best_index]
    short_answer = first_two_sentences(best_section)

    return f"According to company policy, {short_answer}", top_matches


# -----------------------------
# UI
# -----------------------------
st.title("🤖 Policy AI Chatbot")
st.write("Upload a PDF, DOCX, or TXT policy file and ask questions in natural language.")

with st.sidebar:
    st.header("Settings")
    threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.12, 0.01)
    top_k = st.slider("Top results", 1, 5, 3)
    st.caption("Lower threshold = more answers. Higher threshold = stricter filtering.")

uploaded_file = st.file_uploader(
    "Upload policy file",
    type=["pdf", "docx", "txt"]
)

file_text = None

if uploaded_file is not None:
    file_text = extract_text_from_file(uploaded_file)

if file_text:
    file_hash = hashlib.md5(file_text.encode("utf-8", errors="ignore")).hexdigest()

    if (
        "file_hash" not in st.session_state
        or st.session_state.file_hash != file_hash
    ):
        with st.spinner("Preparing AI knowledge base..."):
            sections, embeddings, error = build_knowledge_base(file_text)

            st.session_state.file_hash = file_hash
            st.session_state.sections = sections
            st.session_state.embeddings = embeddings
            st.session_state.error = error
            st.session_state.messages = [
                {
                    "role": "assistant",
                    "content": "Hello 👋\n\nPDF/DOCX/TXT uploaded successfully. Ask me anything about the policy."
                }
            ]

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if st.session_state.get("error"):
        st.error(st.session_state["error"])
        st.stop()

    st.success("File uploaded successfully ✅")
    st.caption(f"Searchable sections created: {len(st.session_state.sections)}")

    with st.expander("Show extracted policy sections"):
        for i, sec in enumerate(st.session_state.sections, start=1):
            st.write(f"**Section {i}**")
            st.write(sec)
            st.write("---")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    question = st.chat_input("Type your question...")

    if question:
        st.session_state.messages.append(
            {"role": "user", "content": question}
        )

        with st.chat_message("user"):
            st.write(question)

        answer, top_matches = search_answer(
            question,
            st.session_state.sections,
            st.session_state.embeddings,
            threshold=threshold,
            top_k=top_k,
        )

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )

        with st.chat_message("assistant"):
            st.write(answer)

            with st.expander("Show top matches"):
                for idx, (text, score) in enumerate(top_matches, start=1):
                    st.write(f"**{idx}. Score:** {score:.2f}")
                    st.write(text)
                    st.write("---")
else:
    st.info("Please upload a PDF, DOCX, or TXT file to continue.")