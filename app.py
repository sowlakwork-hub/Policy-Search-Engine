import os
import re
import hashlib
from io import BytesIO
from difflib import get_close_matches

import streamlit as st
from pypdf import PdfReader

# IMPORTANT (Windows): import sentence_transformers (which loads torch)
# BEFORE scikit-learn, otherwise torch's DLLs can fail to load (WinError 1114).
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_THRESHOLD = 0.30   # tune with the sidebar slider
TOP_K = 3


# -----------------------------------
# LOAD AI MODEL (once per server run)
# -----------------------------------

@st.cache_resource(show_spinner="Loading AI model (first run is slow)...")
def load_model():
    return SentenceTransformer(MODEL_NAME)


# -----------------------------------
# READ PDF
# -----------------------------------

def extract_text_from_pdf(pdf_bytes):
    reader = PdfReader(BytesIO(pdf_bytes))
    pages = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            pages.append(page_text)
    return "\n".join(pages)


# -----------------------------------
# TEXT CLEANING
# -----------------------------------

def clean_line(line):
    # pypdf often turns bullets into control characters such as \x7f
    line = re.sub(r"[\x00-\x1f\x7f]", " ", line)
    return re.sub(r"\s+", " ", line).strip()


HEADING = re.compile(r"^\d+[.)]\s+(.+)$")


def looks_like_bullet(stripped):
    first = stripped[0]
    return (not first.isalnum()) and first not in "(\"'“‘"


# -----------------------------------
# SPLIT INTO MEANINGFUL UNITS
# -----------------------------------
# Policy PDFs are structured: numbered headings with bullet points.
# Each bullet becomes one searchable unit that remembers its heading.
# If the PDF has no such structure, we fall back to sentence windows.

def split_structured(text):
    units = []
    section = "General"
    last_was_bullet = False

    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped:
            continue

        bullet = looks_like_bullet(stripped)
        line = clean_line(stripped[1:] if bullet else stripped)
        if not line:
            continue

        heading = HEADING.match(line) if not bullet else None

        if heading:
            section = heading.group(1).strip()
            last_was_bullet = False
        elif bullet:
            units.append({"section": section, "text": line})
            last_was_bullet = True
        elif last_was_bullet and units and not units[-1]["text"].endswith((".", "!", "?", ":")):
            # wrapped line that belongs to the previous bullet
            units[-1]["text"] += " " + line

    return units


def split_sentence_windows(text, window=2):
    text = clean_line(text.replace("\n", " "))
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    units = []
    step = max(1, window - 1)
    for start in range(0, len(sentences), step):
        chunk = " ".join(sentences[start:start + window])
        units.append({"section": "Document", "text": chunk})
        if start + window >= len(sentences):
            break
    return units


def split_into_units(text):
    units = split_structured(text)
    if len(units) < 3:
        units = split_sentence_windows(text)
    for u in units:
        if u["section"] in ("Document", "General"):
            u["embed_text"] = u["text"]
        else:
            u["embed_text"] = u["section"] + ". " + u["text"]
    return units


# -----------------------------------
# BUILD KNOWLEDGE BASE (cached per PDF)
# -----------------------------------

@st.cache_data(show_spinner="Reading the PDF and building the knowledge base...")
def build_knowledge_base(pdf_bytes):
    text = extract_text_from_pdf(pdf_bytes)
    units = split_into_units(text)
    if not units:
        return [], None, []

    model = load_model()
    embeddings = model.encode(
        [u["embed_text"] for u in units],
        normalize_embeddings=True,
        batch_size=64,
    )
    vocab = sorted(set(re.findall(r"[a-z']+", text.lower())))
    return units, embeddings, vocab


# -----------------------------------
# SMALL TALK (whole-word matching)
# -----------------------------------

GREETINGS = {"hi", "hii", "hello", "hey", "hola"}


def smart_reply(question):
    tokens = re.findall(r"[a-z']+", question.lower())
    if not tokens or len(tokens) > 4:
        return None
    phrase = " ".join(tokens)
    if phrase in {"how are you", "how r u", "how are u", "how do you do",
                  "whats up", "what's up", "how you doing"}:
        return "I'm doing great, thanks for asking 😊\n\nAsk me anything about the policy."
    if phrase in {"who are you", "what can you do", "what do you do", "help"}:
        return ("I'm a Policy AI Chatbot 🤖\n\nI search the PDF you uploaded by "
                "meaning, not just keywords, and show the closest policy points.")
    if tokens[0] in GREETINGS or " ".join(tokens[:2]) in {
        "good morning", "good afternoon", "good evening"
    }:
        return "Hello 👋\n\nHow can I help you today?"
    if "thanks" in tokens or "thank" in tokens:
        return "You're welcome 😊"
    if "bye" in tokens or "goodbye" in tokens:
        return "Goodbye 👋"
    return None


# -----------------------------------
# SPELLING FIX (typos only, not meaning)
# -----------------------------------

COMMON_TERMS = [
    "vacation", "holiday", "holidays", "insurance", "salary", "payroll",
    "reimbursement", "attendance", "remote", "laptop", "internet",
    "password", "security", "meeting", "performance", "medical",
    "emergency", "timing", "approval", "resign", "notice", "bonus",
    "overtime", "travel", "allowance", "manager", "office", "leave",
    "policy", "employee", "dress", "download", "software", "badge",
]


def fix_typos(question, vocab):
    known = set(vocab) | set(COMMON_TERMS)
    candidates = list(known)
    fixed = []
    for word in question.split():
        core = re.sub(r"[^A-Za-z']", "", word).lower()
        if len(core) >= 5 and core not in known:
            match = get_close_matches(core, candidates, n=1, cutoff=0.85)
            if match:
                fixed.append(match[0])
                continue
        fixed.append(word)
    return " ".join(fixed)


# -----------------------------------
# SEMANTIC SEARCH
# -----------------------------------

def semantic_search(question, units, embeddings, top_k=TOP_K):
    model = load_model()

    # Very short questions (e.g. "holiday") give the model little context,
    # so also ask it in full-sentence form and keep the better score.
    queries = [question]
    if len(question.split()) <= 3:
        queries.append("What is the company policy about " + question + "?")

    q_emb = model.encode(queries, normalize_embeddings=True)
    sims = cosine_similarity(q_emb, embeddings).max(axis=0)
    order = sims.argsort()[::-1][:top_k]
    return [(units[i], float(sims[i])) for i in order]


def build_answer(results, threshold):
    best_unit, best_score = results[0]

    if best_score < threshold:
        return "I could not find relevant information in the PDF."

    answer = (
        "According to company policy — **"
        + best_unit["section"] + "**: " + best_unit["text"]
    )

    related = [
        (u, s) for u, s in results[1:]
        if s >= threshold and s >= best_score - 0.10
    ]
    if related:
        answer += "\n\n*Also related:*"
        for u, _ in related:
            answer += "\n- **" + u["section"] + "**: " + u["text"]

    return answer


def answer_question(question, units, embeddings, vocab, threshold):
    """Returns (answer_text, details_for_expander)."""
    small_talk = smart_reply(question)
    if small_talk:
        return small_talk, None

    corrected = fix_typos(question, vocab)
    results = semantic_search(corrected, units, embeddings)
    answer = build_answer(results, threshold)

    details = []
    if corrected.lower() != question.lower():
        details.append("Spelling corrected to: *" + corrected + "*")
    for u, s in results:
        details.append(
            "- score **" + format(s, ".2f") + "** — " + u["section"] + ": " + u["text"]
        )
    details.append("Threshold: " + format(threshold, ".2f"))
    return answer, details


# -----------------------------------
# STREAMLIT APP
# -----------------------------------

def render_message(message):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("details"):
            with st.expander("How the AI matched this"):
                for line in message["details"]:
                    st.markdown(line)


def main():
    st.set_page_config(
        page_title="Policy AI Chatbot",
        page_icon="🤖",
        layout="centered",
    )

    st.title("🤖 Policy AI Chatbot")
    st.write(
        "Upload your company policy PDF and ask questions using "
        "Semantic AI Search."
    )

    with st.sidebar:
        st.header("Settings")
        threshold = st.slider(
            "Match threshold",
            min_value=0.10,
            max_value=0.60,
            value=DEFAULT_THRESHOLD,
            step=0.01,
            help="Higher = stricter (more 'not found'). Lower = more answers.",
        )

    # Load the model right away so the delay happens once, with a spinner
    load_model()

    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

    if uploaded_file is None:
        st.info("Please upload a PDF file.")
        return

    pdf_bytes = uploaded_file.getvalue()
    file_hash = hashlib.md5(pdf_bytes).hexdigest()

    units, embeddings, vocab = build_knowledge_base(pdf_bytes)

    if not units:
        st.error(
            "No readable text found in this PDF. "
            "It may be a scanned image, which this app cannot read."
        )
        return

    if st.session_state.get("file_hash") != file_hash:
        st.session_state.file_hash = file_hash
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Hello 👋\n\nPDF uploaded successfully. I found "
                    + str(len(units))
                    + " policy points. Ask me anything about the policy."
                ),
            }
        ]

    for message in st.session_state.messages:
        render_message(message)

    question = st.chat_input("Type your question...")

    if question:
        user_msg = {"role": "user", "content": question}
        st.session_state.messages.append(user_msg)
        render_message(user_msg)

        answer, details = answer_question(
            question, units, embeddings, vocab, threshold
        )

        bot_msg = {"role": "assistant", "content": answer, "details": details}
        st.session_state.messages.append(bot_msg)
        render_message(bot_msg)


if __name__ == "__main__":
    main()
