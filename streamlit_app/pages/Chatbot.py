import html
import re
from typing import Optional, Tuple, Any, Dict, List

import streamlit as st

from api_client import predict


CHATBOT_MODEL_ID = "chatbot_rag_conversational_v3"
TRANSLATOR_MODEL_ID = "translator_marian_v2"


# -----------------------------
# Page config + styling
# -----------------------------
st.set_page_config(
    page_title="Healthcare Chatbot",
    page_icon="🩺",
    layout="wide",
)

st.markdown(
    """
    <style>
    .page-subtitle {
        color: #6b7280;
        margin-top: -8px;
        margin-bottom: 18px;
        font-size: 0.98rem;
    }
    .soft-card {
        background: rgba(120, 120, 120, 0.08);
        padding: 14px 16px;
        border-radius: 16px;
        border: 1px solid rgba(120, 120, 120, 0.12);
        margin-bottom: 12px;
    }
    .mini-card {
        background: rgba(120, 120, 120, 0.06);
        padding: 10px 12px;
        border-radius: 14px;
        border: 1px solid rgba(120, 120, 120, 0.10);
        margin-bottom: 8px;
    }
    .muted {
        color: #6b7280;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🩺 Healthcare Chatbot")
st.markdown(
    "<div class='page-subtitle'>Ask symptom-based healthcare questions and view supporting evidence.</div>",
    unsafe_allow_html=True,
)


# -----------------------------
# Language detection
# -----------------------------
TAMIL_RE = re.compile(r"[\u0B80-\u0BFF]")
HINDI_RE = re.compile(r"[\u0900-\u097F]")
MALAYALAM_RE = re.compile(r"[\u0D00-\u0D7F]")
TELUGU_RE = re.compile(r"[\u0C00-\u0C7F]")

SUPPORTED_UI_LANGS = {
    "en": "English",
    "ta": "Tamil",
    "hi": "Hindi",
    "ml": "Malayalam",
    "te": "Telugu",
}


def detect_lang(text: str, default: str = "en") -> str:
    t = (text or "").strip()
    if not t:
        return default

    if TAMIL_RE.search(t):
        return "ta"
    if HINDI_RE.search(t):
        return "hi"
    if MALAYALAM_RE.search(t):
        return "ml"
    if TELUGU_RE.search(t):
        return "te"

    try:
        from langdetect import detect as _detect
        detected = (_detect(t) or default).lower()
        return detected if detected in SUPPORTED_UI_LANGS else default
    except Exception:
        return default


def normalize_lang(lang: Optional[str], default: str = "en") -> str:
    s = (lang or "").strip().lower()
    mapping = {
        "english": "en",
        "eng": "en",
        "tamil": "ta",
        "tam": "ta",
        "hindi": "hi",
        "hin": "hi",
        "malayalam": "ml",
        "mal": "ml",
        "telugu": "te",
        "tel": "te",
    }
    s = mapping.get(s, s)
    return s if s in SUPPORTED_UI_LANGS else default


# -----------------------------
# Session state
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_hits" not in st.session_state:
    st.session_state.last_hits = []

if "last_debug" not in st.session_state:
    st.session_state.last_debug = {}


# -----------------------------
# Response normalization
# -----------------------------
def _extract_output_and_meta(resp: Any) -> Tuple[Any, Dict[str, Any]]:
    if isinstance(resp, dict):
        if "output" in resp:
            return resp.get("output"), resp.get("meta", {}) or {}
        return resp, resp.get("meta", {}) or {}
    return resp, {}


def _extract_translation_text(resp: Any) -> str:
    out, _ = _extract_output_and_meta(resp)

    if isinstance(out, dict):
        if "output" in out:
            return html.unescape(str(out["output"]))
        if "translated_text" in out:
            return html.unescape(str(out["translated_text"]))
        if "text" in out:
            return html.unescape(str(out["text"]))

    return html.unescape(str(out))


def _extract_chatbot_answer(resp: Any) -> Tuple[str, str, List[Dict[str, Any]], Dict[str, Any]]:
    out, meta = _extract_output_and_meta(resp)

    if isinstance(out, dict):
        answer = str(out.get("answer", "") or "").strip()
        answer_en = str(out.get("answer_en", answer) or "").strip()
        hits = out.get("hits", []) or []
        return answer, answer_en, hits, meta

    if isinstance(resp, dict):
        answer = str(resp.get("answer", "") or "").strip()
        answer_en = str(resp.get("answer_en", answer) or "").strip()
        hits = resp.get("hits", []) or []
        if answer:
            return answer, answer_en, hits, meta

    answer = str(out or "").strip()
    return answer, answer, [], meta


def _best_evidence_summary(hits: List[Dict[str, Any]]) -> str:
    if not hits:
        return ""

    first = hits[0] or {}

    if first.get("answer_conversational"):
        return str(first["answer_conversational"]).strip()

    if first.get("guidance"):
        return str(first["guidance"]).strip()

    if first.get("doc_text"):
        return str(first["doc_text"]).strip()

    return ""


def _looks_like_raw_dump(text: str) -> bool:
    if not text:
        return False

    t = text.strip().lower()
    return (
        t.startswith("diagnosis:")
        or ("guidance:" in t and "plan:" in t)
        or ("notes:" in t and "summary:" in t)
    )


# -----------------------------
# Translation
# -----------------------------
def translate_backend_model(
    text: str,
    target_lang: str,
    source_lang: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    meta: Dict[str, Any] = {
        "used": None,
        "source_lang": source_lang,
        "target_lang": target_lang,
    }

    if not text or not str(text).strip():
        meta["used"] = "skipped_empty"
        return text, meta

    if not target_lang:
        meta["used"] = "skipped_no_target"
        return text, meta

    if source_lang and target_lang == source_lang:
        meta["used"] = "skipped_same_lang"
        return text, meta

    payload = {
        "text": str(text),
        "source_lang": normalize_lang(source_lang, "en") if source_lang not in (None, "", "auto") else "en",
        "target_lang": normalize_lang(target_lang, "en"),
    }

    try:
        resp = predict(TRANSLATOR_MODEL_ID, payload)
        out_str = _extract_translation_text(resp)
        meta["used"] = "backend_model"
        meta["payload_used"] = payload
        return out_str, meta
    except Exception as e:
        meta["used"] = "backend_failed"
        meta["error"] = str(e)
        return str(text), meta


def translate_text(
    text: str,
    target_lang: str,
    source_lang: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    return translate_backend_model(text=text, target_lang=target_lang, source_lang=source_lang)


# -----------------------------
# Conversation helpers
# -----------------------------
def build_history_text(messages: List[Dict[str, str]], max_turns: int = 6) -> str:
    if not messages:
        return ""

    trimmed = messages[-max_turns:]
    lines: List[str] = []

    for m in trimmed:
        role = m.get("role", "")
        content = (m.get("content", "") or "").strip()
        if not content:
            continue
        if role == "user":
            lines.append(f"user: {content}")
        elif role == "assistant":
            lines.append(f"assistant: {content}")

    return " ".join(lines).strip()


def add_message(role: str, content: str) -> None:
    st.session_state.messages.append({"role": role, "content": content})


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.subheader("Settings")

    output_lang = st.selectbox(
        "Answer language",
        options=list(SUPPORTED_UI_LANGS.keys()),
        format_func=lambda x: SUPPORTED_UI_LANGS[x],
        index=1,
    )

    top_k = st.slider("Evidence to retrieve", 1, 10, 5)
    show_debug = st.toggle("Show debug details", value=False)

    st.markdown("---")

    st.markdown(
        f"""
        <div class="mini-card">
            <strong>Chatbot model</strong><br>{CHATBOT_MODEL_ID}
        </div>
        <div class="mini-card">
            <strong>Translator model</strong><br>{TRANSLATOR_MODEL_ID}
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_hits = []
        st.session_state.last_debug = {}
        st.rerun()


# -----------------------------
# Small top status area
# -----------------------------
left_col, right_col = st.columns([2, 1])

with left_col:
    st.markdown(
        """
        <div class="soft-card">
            Ask about symptoms, medication guidance, warning signs, or follow-up care.
        </div>
        """,
        unsafe_allow_html=True,
    )

with right_col:
    st.markdown(
        f"""
        <div class="soft-card">
            <strong>Output:</strong> {SUPPORTED_UI_LANGS[output_lang]}
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------
# Existing chat
# -----------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])


# -----------------------------
# Chat input
# -----------------------------
question = st.chat_input("Type your healthcare question here...")

if question:
    add_message("user", question)

    with st.chat_message("user"):
        st.write(question)

    detected = detect_lang(question, default="en")
    history_text = build_history_text(st.session_state.messages[:-1], max_turns=6)

    # Translate incoming query only if needed
    q_en = question
    qmeta = {"used": "skipped"}
    if detected != "en":
        q_en, qmeta = translate_text(
            text=question,
            target_lang="en",
            source_lang=detected,
        )

    # Call chatbot in English
    try:
        resp = predict(
            CHATBOT_MODEL_ID,
            {
                "query": q_en,
                "history": history_text,
                "target_lang": "en",
                "top_k": top_k,
            },
            top_k=top_k,
        )
    except Exception as e:
        with st.chat_message("assistant"):
            st.error(f"Chatbot failed: {e}")
        st.stop()

    answer_en, answer_debug_en, hits, chatbot_meta = _extract_chatbot_answer(resp)

    # Prefer cleaner evidence-backed answer instead of raw clinical dump
    evidence_summary = _best_evidence_summary(hits)

    if (_looks_like_raw_dump(answer_en) or not answer_en) and evidence_summary:
        answer_en = evidence_summary
        answer_debug_en = evidence_summary

    if not answer_en:
        answer_en = (
            "I could not generate a reliable answer from the available context. "
            "Please share more details such as symptoms, duration, severity, and any existing conditions."
        )
        answer_debug_en = answer_en

    # Translate final answer only if output language is non-English
    final_answer = answer_en
    ameta = {"used": "skipped"}

    if output_lang != "en":
        final_answer, ameta = translate_text(
            text=answer_en,
            target_lang=output_lang,
            source_lang="en",
        )

    translation_failed = ameta.get("used") == "backend_failed"
    if translation_failed:
        final_answer = answer_en

    add_message("assistant", final_answer)

    st.session_state.last_hits = hits
    st.session_state.last_debug = {
        "detected_input_language": detected,
        "requested_output_language": output_lang,
        "translated_query_english": q_en,
        "query_translation_meta": qmeta,
        "answer_translation_meta": ameta,
        "chatbot_meta": chatbot_meta,
        "english_answer": answer_debug_en,
    }

    with st.chat_message("assistant"):
        if translation_failed:
            st.warning("Translation failed, so the answer is shown in English.")

        st.write(final_answer)

        evidence_tab, debug_tab = st.tabs(["Evidence", "Debug"])

        with evidence_tab:
            if hits:
                for i, h in enumerate(hits[:top_k], start=1):
                    score = ""
                    if isinstance(h, dict) and "score" in h:
                        try:
                            score = f" (score={float(h['score']):.4f})"
                        except Exception:
                            score = ""

                    title = f"Evidence #{i}{score}"
                    if isinstance(h, dict) and h.get("diagnosis"):
                        title += f" — {h['diagnosis']}"

                    with st.expander(title, expanded=(i == 1)):
                        if isinstance(h, dict):
                            if h.get("answer_conversational"):
                                st.markdown("**Suggested answer**")
                                st.write(h["answer_conversational"])

                            cols = st.columns(2)
                            with cols[0]:
                                if h.get("diagnosis"):
                                    st.markdown(f"**Diagnosis:** {h['diagnosis']}")
                                if h.get("plan"):
                                    st.markdown(f"**Plan:** {h['plan']}")
                            with cols[1]:
                                if h.get("score") is not None:
                                    st.markdown(f"**Score:** {float(h['score']):.4f}")

                            st.markdown("**Full evidence**")
                            st.json(h)
                        else:
                            st.write(h)
            else:
                st.info("No evidence returned.")

        with debug_tab:
            if show_debug:
                st.write(f"**Detected input language:** `{detected}`")
                st.write(f"**Requested output language:** `{output_lang}`")

                if detected != "en":
                    st.write("**Translated query used for retrieval (English):**")
                    st.code(q_en)

                st.write("**English answer from chatbot:**")
                st.write(answer_debug_en)

                st.write("**Query translation meta:**")
                st.json(qmeta)

                st.write("**Answer translation meta:**")
                st.json(ameta)

                st.write("**Chatbot meta:**")
                st.json(chatbot_meta)
            else:
                st.info("Enable 'Show debug details' from the sidebar to view debug information.")