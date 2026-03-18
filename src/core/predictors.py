from __future__ import annotations

from typing import Any, Tuple, Optional
import ast
import base64
import io
import os
import re
import html
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.registry import ModelSpec
from src.core import loaders
from src.core.marian_translator import MarianTranslator
from src.core.cnn_xray_model import XrayMultiLabelClassifier


# -----------------------------
# Path helpers (fix artifacts/artifacts)
# -----------------------------
def _resolve_artifact_path(p: str | Path) -> str:
    """
    Normalizes model/meta paths to avoid common issues like:
      /.../artifacts/artifacts/...
    Works for both absolute + relative paths.
    """
    s = str(p)

    s = s.replace("/artifacts/artifacts/", "/artifacts/")
    s = s.replace("\\artifacts\\artifacts\\", "\\artifacts\\")
    s = s.replace("/artifacts\\artifacts/", "/artifacts/")
    s = s.replace("\\artifacts/artifacts\\", "\\artifacts\\")

    return s


def _path_exists(p: str | Path) -> bool:
    try:
        return Path(str(p)).exists()
    except Exception:
        return False


def _load_pickle(path: str | Path):
    with open(_resolve_artifact_path(path), "rb") as f:
        return pickle.load(f)


# -----------------------------
# Torch helpers
# -----------------------------
def _get_torch():
    import torch
    import torch.nn as nn
    return torch, nn


# -----------------------------
# Torch: minimal TS model builder
# -----------------------------
class TSRegressor:
    """
    Minimal RNN/LSTM regressor wrapper.
    Built from arch.json fields.

    Expected input: [B, T, F]
    Output: [B, 1] or [B, out_size]
    """

    def __init__(self, arch: dict[str, Any], force_type: str | None = None):
        torch, nn = _get_torch()

        rnn_type = (force_type or str(arch.get("type", arch.get("rnn_type", "lstm")))).lower()

        input_size = int(arch.get("input_size", arch.get("n_features", arch.get("in_features", 4))))
        hidden_size = int(arch.get("hidden_size", 64))
        num_layers = int(arch.get("num_layers", 1))
        dropout = float(arch.get("dropout", 0.0))
        bidirectional = bool(arch.get("bidirectional", False))
        out_size = int(arch.get("output_size", arch.get("out_features", 1)))

        class _TS(nn.Module):
            def __init__(self):
                super().__init__()
                mult = 2 if bidirectional else 1

                if rnn_type == "rnn":
                    self.rnn = nn.RNN(
                        input_size=input_size,
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        dropout=dropout if num_layers > 1 else 0.0,
                        batch_first=True,
                        bidirectional=bidirectional,
                    )
                else:
                    self.rnn = nn.LSTM(
                        input_size=input_size,
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        dropout=dropout if num_layers > 1 else 0.0,
                        batch_first=True,
                        bidirectional=bidirectional,
                    )

                self.head = nn.Linear(hidden_size * mult, out_size)

            def forward(self, x):
                out, _ = self.rnn(x)
                last = out[:, -1, :]
                y = self.head(last)
                return y

        self.model = _TS()

    def load_state_dict(self, state: dict, strict: bool = True):
        self.model.load_state_dict(state, strict=strict)

    def eval(self):
        self.model.eval()
        return self

    def __call__(self, x):
        return self.model(x)


# -----------------------------
# Chatbot: attention seq2seq
# -----------------------------
class ChatEncoder:
    def __new__(cls, vocab_size: int, embed_dim: int, hidden_dim: int, pad_idx: int, num_layers: int = 1, dropout: float = 0.0):
        torch, nn = _get_torch()

        class _Encoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
                self.dropout = nn.Dropout(dropout)
                self.lstm = nn.LSTM(
                    embed_dim,
                    hidden_dim,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=dropout if num_layers > 1 else 0.0,
                )

            def forward(self, src):
                embedded = self.dropout(self.embedding(src))
                outputs, (hidden, cell) = self.lstm(embedded)
                return outputs, hidden, cell

        return _Encoder()


class BahdanauAttention:
    def __new__(cls, hidden_dim: int):
        torch, nn = _get_torch()

        class _Attention(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
                self.v = nn.Linear(hidden_dim, 1, bias=False)

            def forward(self, hidden, encoder_outputs):
                src_len = encoder_outputs.shape[1]
                hidden_last = hidden[-1].unsqueeze(1).repeat(1, src_len, 1)
                energy = torch.tanh(self.attn(torch.cat((hidden_last, encoder_outputs), dim=2)))
                attention = self.v(energy).squeeze(2)
                return torch.softmax(attention, dim=1)

        return _Attention()


class ChatDecoder:
    def __new__(cls, vocab_size: int, embed_dim: int, hidden_dim: int, pad_idx: int, attention, num_layers: int = 1, dropout: float = 0.0):
        torch, nn = _get_torch()

        class _Decoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.attention = attention
                self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
                self.dropout = nn.Dropout(dropout)
                self.lstm = nn.LSTM(
                    embed_dim + hidden_dim,
                    hidden_dim,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=dropout if num_layers > 1 else 0.0,
                )
                self.fc = nn.Linear(hidden_dim * 2 + embed_dim, vocab_size)

            def forward(self, input_token, hidden, cell, encoder_outputs):
                input_token = input_token.unsqueeze(1)
                embedded = self.dropout(self.embedding(input_token))
                attn_weights = self.attention(hidden, encoder_outputs).unsqueeze(1)
                context = torch.bmm(attn_weights, encoder_outputs)
                lstm_input = torch.cat((embedded, context), dim=2)
                output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
                prediction = self.fc(torch.cat((output.squeeze(1), context.squeeze(1), embedded.squeeze(1)), dim=1))
                return prediction, hidden, cell, attn_weights

        return _Decoder()


class AttentionSeq2Seq:
    def __new__(cls, encoder, decoder, sos_idx: int, eos_idx: int, device: str):
        torch, nn = _get_torch()

        class _Seq2Seq(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = encoder
                self.decoder = decoder
                self.sos_idx = sos_idx
                self.eos_idx = eos_idx
                self.device = device

            def forward(self, src, trg=None, teacher_forcing_ratio=0.0, max_len=60):
                encoder_outputs, hidden, cell = self.encoder(src)
                batch_size = src.shape[0]

                if trg is not None:
                    trg_len = trg.shape[1]
                    vocab_size = self.decoder.fc.out_features
                    outputs = torch.zeros(batch_size, trg_len, vocab_size).to(src.device)
                    input_token = trg[:, 0]

                    for t in range(1, trg_len):
                        output, hidden, cell, _ = self.decoder(input_token, hidden, cell, encoder_outputs)
                        outputs[:, t, :] = output
                        teacher_force = np.random.rand() < teacher_forcing_ratio
                        top1 = output.argmax(1)
                        input_token = trg[:, t] if teacher_force else top1
                    return outputs

                input_token = torch.full((batch_size,), self.sos_idx, dtype=torch.long, device=src.device)
                generated = []

                for _ in range(max_len):
                    output, hidden, cell, _ = self.decoder(input_token, hidden, cell, encoder_outputs)
                    pred = output.argmax(1)
                    generated.append(pred)
                    input_token = pred

                return torch.stack(generated, dim=1)

        return _Seq2Seq()


# -----------------------------
# Imaging helpers
# -----------------------------
def _decode_image_b64(image_b64: str):
    from PIL import Image

    if not image_b64:
        raise ValueError("Empty image_b64")

    s = image_b64.strip()
    if s.lower().startswith("data:") and "," in s:
        s = s.split(",", 1)[1]

    raw = base64.b64decode(s)
    img = Image.open(io.BytesIO(raw))
    return img


def _get_image_bytes_from_payload(payload: dict[str, Any], field: str = "image_b64") -> bytes:
    """
    Supports:
    - image_b64: base64 string
    - image_bytes: raw bytes
    """
    image_bytes = payload.get("image_bytes")
    if image_bytes is not None:
        if isinstance(image_bytes, bytes):
            return image_bytes
        raise ValueError("'image_bytes' must be bytes")

    image_b64 = payload.get(field)
    if not image_b64:
        raise ValueError(f"Missing '{field}' in payload")

    s = str(image_b64).strip()
    if s.lower().startswith("data:") and "," in s:
        s = s.split(",", 1)[1]

    return base64.b64decode(s)


def _build_resnet18(num_classes: int):
    import torch.nn as nn
    from torchvision import models

    m = models.resnet18(weights=None)
    in_features = m.fc.in_features
    m.fc = nn.Linear(in_features, num_classes)
    m.eval()
    return m


def _extract_state_dict(obj: Any) -> Any:
    if isinstance(obj, dict):
        for k in ["state_dict", "model_state_dict", "model", "net", "weights"]:
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
    return obj


def _strip_prefixes(sd: dict, prefixes: tuple[str, ...] = ("module.", "model.", "net.")) -> dict:
    if not isinstance(sd, dict) or not sd:
        return sd

    def strip_once(k: str) -> str:
        for p in prefixes:
            if k.startswith(p):
                return k[len(p):]
        return k

    if any(isinstance(k, str) and any(k.startswith(p) for p in prefixes) for k in sd.keys()):
        return {strip_once(k): v for k, v in sd.items()}
    return sd


def _imaging_preprocess(img, cfg: dict[str, Any]):
    from torchvision import transforms

    image_size = int(cfg.get("image_size", 224))
    input_mode = str(cfg.get("input_mode", "RGB")).upper()
    do_norm = bool(cfg.get("normalize", False))
    mean = cfg.get("mean")
    std = cfg.get("std")

    if input_mode == "RGB":
        img = img.convert("RGB")
    else:
        img = img.convert("L")

    tfs = [transforms.Resize((image_size, image_size)), transforms.ToTensor()]

    if do_norm:
        if mean is None or std is None:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
        tfs.append(transforms.Normalize(mean=mean, std=std))

    tfm = transforms.Compose(tfs)
    x = tfm(img).unsqueeze(0)
    return x


def _imaging_postprocess(logits, label_map: dict[str, str]):
    torch, _ = _get_torch()

    if not isinstance(logits, torch.Tensor):
        logits = torch.tensor(logits)

    if logits.ndim == 2 and logits.shape[1] >= 2:
        probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy().tolist()
        pred_idx = int(np.argmax(probs))
        pred_label = label_map.get(str(pred_idx), str(pred_idx))
        return pred_label, {"proba": probs, "pred_idx": pred_idx, "label_map": label_map}

    score = float(torch.sigmoid(logits.flatten()[0]).item())
    pred_idx = 1 if score >= 0.5 else 0
    pred_label = label_map.get(str(pred_idx), str(pred_idx))
    return pred_label, {"score": score, "pred_idx": pred_idx, "label_map": label_map}


# -----------------------------
# General helpers
# -----------------------------
def _to_jsonable(x: Any) -> Any:
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, pd.Timestamp):
        return x.isoformat()
    if isinstance(x, pd.Timedelta):
        return str(x)

    if isinstance(x, dict):
        return {k: _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]

    return x


def _load_feature_columns(spec: ModelSpec) -> list[str]:
    fc_path = spec.meta_paths.get("feature_columns")
    if fc_path:
        cols = loaders.load_json(_resolve_artifact_path(fc_path))
        if isinstance(cols, dict) and "columns" in cols:
            cols = cols["columns"]
        return list(cols) if isinstance(cols, (list, tuple)) else []

    cfg_path = spec.meta_paths.get("config")
    if cfg_path:
        cfg = loaders.load_json(_resolve_artifact_path(cfg_path))
        for key in [
            "kmeans_input_cols",
            "feature_columns",
            "columns",
            "features",
            "input_columns",
            "profile_numeric_cols",
        ]:
            if key in cfg and isinstance(cfg[key], list):
                return cfg[key]

    try:
        model = loaders.load_joblib(_resolve_artifact_path(spec.model_path))
        if hasattr(model, "feature_names_in_"):
            return list(model.feature_names_in_)
    except Exception:
        pass

    return []


def _tabular_to_df(features: dict[str, Any], feature_columns: list[str]) -> pd.DataFrame:
    row = {c: features.get(c, np.nan) for c in feature_columns}
    return pd.DataFrame([row], columns=feature_columns)


def _maybe_add_label_map(meta: dict[str, Any], spec: ModelSpec) -> None:
    lm_path = spec.meta_paths.get("label_map")
    if lm_path:
        meta["label_map"] = loaders.load_json(_resolve_artifact_path(lm_path))


def _extract_npz_matrix(npz_obj) -> Any:
    if hasattr(npz_obj, "files"):
        if "arr_0" in npz_obj.files:
            return npz_obj["arr_0"]
        if "matrix" in npz_obj.files:
            return npz_obj["matrix"]
        if len(npz_obj.files) > 0:
            return npz_obj[npz_obj.files[0]]
    raise ValueError("NPZ file contains no arrays.")


def _safe_parse_antecedent(a: Any) -> Optional[set]:
    if isinstance(a, (list, tuple, set)):
        return set(a)

    if isinstance(a, str):
        s = a.strip()
        try:
            val = ast.literal_eval(s)
            if isinstance(val, (list, tuple, set)):
                return set(val)
        except Exception:
            pass

        if "," in s:
            parts = [p.strip() for p in s.split(",") if p.strip()]
            return set(parts) if parts else None

        return {s} if s else None

    return None


def _patch_sklearn_lr(obj: Any) -> None:
    if obj is None:
        return

    try:
        name = obj.__class__.__name__
        if name in ("LogisticRegression", "LogisticRegressionCV") and not hasattr(obj, "multi_class"):
            setattr(obj, "multi_class", "auto")
    except Exception:
        pass

    if hasattr(obj, "steps") and isinstance(getattr(obj, "steps"), list):
        for _, step in obj.steps:
            _patch_sklearn_lr(step)

    for attr in ("estimator", "base_estimator", "classifier", "model"):
        if hasattr(obj, attr):
            try:
                _patch_sklearn_lr(getattr(obj, attr))
            except Exception:
                pass

    for attr in ("best_estimator_", "estimator_"):
        if hasattr(obj, attr):
            try:
                _patch_sklearn_lr(getattr(obj, attr))
            except Exception:
                pass


def _impute_tabular_soft(
    X_df: pd.DataFrame,
    feature_columns: list[str],
    imputed_cols: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    if not X_df.isna().any(axis=None):
        return X_df, imputed_cols

    missing_now = [c for c in feature_columns if pd.isna(X_df.loc[0, c])]
    imputed_cols = sorted(set(imputed_cols + missing_now))

    coerced = X_df.apply(pd.to_numeric, errors="coerce")

    numeric_like = [c for c in feature_columns if not pd.isna(coerced.loc[0, c])]
    numeric_dtype = X_df.select_dtypes(include=[np.number]).columns.tolist()
    for c in numeric_dtype:
        if c not in numeric_like:
            numeric_like.append(c)

    numeric_like = [c for c in feature_columns if c in set(numeric_like)]
    other_cols = [c for c in feature_columns if c not in set(numeric_like)]

    if numeric_like:
        X_df[numeric_like] = coerced[numeric_like].fillna(0.0)

    if other_cols:
        X_df[other_cols] = X_df[other_cols].astype("object").fillna("unknown")

    return X_df, imputed_cols


def _unescape_translate_text(s: Any) -> str:
    if s is None:
        return ""
    return html.unescape(str(s))


def _normalize_lang_code(lang: str | None) -> str:
    s = str(lang or "").strip().lower()
    mapping = {
        "english": "en",
        "eng": "en",
        "tamil": "ta",
        "tam": "ta",
        "malayalam": "ml",
        "mal": "ml",
        "telugu": "te",
        "tel": "te",
        "hindi": "hi",
        "hin": "hi",
    }
    return mapping.get(s, s or "en")


def _clean_chatbot_text(text: str) -> str:
    if not text:
        return ""

    text = str(text)
    text = re.sub(r"\b(\w+)([:.,!?]?)\s+\1\b", r"\1", text, flags=re.IGNORECASE)
    text = text.replace("to: to:", "to:")
    text = text.replace("follow follow", "follow")
    text = text.replace("plan: plan:", "plan:")
    text = text.replace("the the", "the")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _build_chatbot_doc_text(store: pd.DataFrame, cfg: dict[str, Any]) -> pd.Series:
    if "doc_text" in store.columns:
        return store["doc_text"].fillna("").astype(str)

    doc_cols = cfg.get("doc_cols", [])
    if doc_cols:
        parts = []
        for c in doc_cols:
            if c in store.columns:
                parts.append(store[c].fillna("").astype(str))
        if parts:
            combined = parts[0]
            for p in parts[1:]:
                combined = combined + " " + p
            return combined.str.strip()

    text_candidates = [
        "context",
        "retrieved_context",
        "reference_text",
        "target_text",
        "chatbot_reference_answer",
        "doctor_notes_text",
        "discharge_summary_text",
    ]
    available = [c for c in text_candidates if c in store.columns]
    if available:
        combined = store[available[0]].fillna("").astype(str)
        for c in available[1:]:
            combined = combined + " " + store[c].fillna("").astype(str)
        return combined.str.strip()

    return pd.Series([""] * len(store))


def _retrieve_chatbot_hits(spec: ModelSpec, query: str, top_k: int, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    from sklearn.metrics.pairwise import cosine_similarity
    from scipy import sparse

    tfidf_path = spec.meta_paths.get("tfidf")
    matrix_path = spec.meta_paths.get("matrix")
    store_path = spec.meta_paths.get("store")

    if not tfidf_path or not store_path:
        return []

    vectorizer = loaders.load_joblib(_resolve_artifact_path(tfidf_path))
    store: pd.DataFrame = loaders.load_parquet(_resolve_artifact_path(store_path))

    if matrix_path and _path_exists(matrix_path):
        try:
            doc_matrix = sparse.load_npz(_resolve_artifact_path(matrix_path))
        except Exception:
            mat_npz = loaders.load_npz(_resolve_artifact_path(matrix_path))
            doc_matrix = _extract_npz_matrix(mat_npz)
            if not sparse.issparse(doc_matrix):
                doc_matrix = sparse.csr_matrix(doc_matrix)
    else:
        doc_texts = _build_chatbot_doc_text(store, cfg).tolist()
        doc_matrix = vectorizer.transform(doc_texts)

    qv = vectorizer.transform([query])

    if doc_matrix.shape[0] != len(store) and doc_matrix.shape[1] == len(store):
        doc_matrix = doc_matrix.T

    sims = cosine_similarity(qv, doc_matrix).ravel()
    k = max(1, int(top_k))
    idx = np.argsort(-sims)[:k]

    results = []
    for i in idx:
        row = store.iloc[int(i)].to_dict()
        row["score"] = float(sims[int(i)])
        results.append(_to_jsonable(row))
    return results


def _format_chatbot_context(hits: list[dict[str, Any]], cfg: dict[str, Any]) -> str:
    if not hits:
        return ""

    chunks: list[str] = []
    doc_cols = cfg.get("doc_cols", [])
    ref_col = cfg.get("ref_col", "chatbot_reference_answer")

    for i, hit in enumerate(hits, start=1):
        parts: list[str] = []
        for c in doc_cols:
            if hit.get(c):
                parts.append(f"{c}: {hit[c]}")
        if hit.get(ref_col):
            parts.append(f"reference_answer: {hit[ref_col]}")
        if hit.get("doc_text"):
            parts.append(f"doc_text: {hit['doc_text']}")

        if not parts:
            parts.append(json.dumps(hit, ensure_ascii=False))

        chunks.append(f"[Context {i}]\n" + "\n".join(parts))

    return "\n\n".join(chunks)


def _chatbot_rule_based_answer(query: str, hits: list[dict[str, Any]], cfg: dict[str, Any]) -> str:
    if not hits:
        return "I could not find enough relevant context to answer that safely. Please consult a clinician."

    best = hits[0]
    ref_col = cfg.get("ref_col", "chatbot_reference_answer")
    answer = str(best.get(ref_col) or best.get("reference_text") or best.get("target_text") or "").strip()
    if answer:
        return _clean_chatbot_text(answer)

    if best.get("doc_text"):
        return _clean_chatbot_text(str(best["doc_text"]))

    return "I found related medical context, but I could not form a reliable answer. Please consult a clinician."


def _chatbot_vertex_answer(query: str, history: str, hits: list[dict[str, Any]], cfg: dict[str, Any]) -> str:
    rag_cfg = cfg.get("rag", {}) if isinstance(cfg, dict) else {}
    model_name = str(rag_cfg.get("llm_model", "gemini-1.5-flash"))
    temperature = float(rag_cfg.get("temperature", 0.2))
    max_output_tokens = int(rag_cfg.get("max_output_tokens", 512))

    context = _format_chatbot_context(hits, cfg)

    prompt = (
        "You are a helpful healthcare assistant.\n"
        "Use ONLY the provided context to answer.\n"
        "If the answer is uncertain, say so plainly.\n"
        "Do not invent diagnoses or medications.\n\n"
        f"Conversation history:\n{history or '(none)'}\n\n"
        f"User question:\n{query}\n\n"
        f"Retrieved context:\n{context}\n\n"
        "Answer clearly and briefly."
    )

    try:
        import vertexai
        from vertexai.generative_models import GenerativeModel, GenerationConfig

        project = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCP_PROJECT")
        location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
        if project:
            vertexai.init(project=project, location=location)

        model = GenerativeModel(model_name)
        resp = model.generate_content(
            prompt,
            generation_config=GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            ),
        )
        out = str(getattr(resp, "text", "") or "").strip()
        return _clean_chatbot_text(out)
    except Exception:
        return _chatbot_rule_based_answer(query, hits, cfg)


def _build_local_chatbot_model(vocab_size: int, cfg: dict[str, Any]):
    embed_dim = int(cfg.get("embed_dim", 128))
    hidden_dim = int(cfg.get("hidden_dim", 256))
    num_layers = int(cfg.get("num_layers", 1))
    dropout = float(cfg.get("dropout", 0.0))

    special_tokens = cfg.get("special_tokens", {})
    pad_token = special_tokens.get("pad_token", "<pad>")
    sos_token = special_tokens.get("sos_token", "<sos>")
    eos_token = special_tokens.get("eos_token", "<eos>")

    vocab = cfg["_runtime_vocab"]
    pad_idx = int(vocab.get(pad_token, 0))
    sos_idx = int(vocab.get(sos_token, 1))
    eos_idx = int(vocab.get(eos_token, 2))

    encoder = ChatEncoder(vocab_size, embed_dim, hidden_dim, pad_idx, num_layers=num_layers, dropout=dropout)
    attention = BahdanauAttention(hidden_dim)
    decoder = ChatDecoder(vocab_size, embed_dim, hidden_dim, pad_idx, attention, num_layers=num_layers, dropout=dropout)
    model = AttentionSeq2Seq(encoder, decoder, sos_idx, eos_idx, "cpu")
    return model, pad_idx, sos_idx, eos_idx


def _chatbot_encode_text(text: str, vocab: dict[str, int], cfg: dict[str, Any], max_len: int):
    special_tokens = cfg.get("special_tokens", {})
    eos_token = special_tokens.get("eos_token", "<eos>")
    unk_token = special_tokens.get("unk_token", "<unk>")
    unk_idx = int(vocab.get(unk_token, 3))
    eos_idx = int(vocab.get(eos_token, 2))

    tokens = str(text).strip().lower().split()
    ids = [vocab.get(tok, unk_idx) for tok in tokens] + [eos_idx]

    if len(ids) < max_len:
        pad_idx = int(vocab.get(special_tokens.get("pad_token", "<pad>"), 0))
        ids = ids + [pad_idx] * (max_len - len(ids))
    else:
        ids = ids[:max_len]
        ids[-1] = eos_idx

    return ids


def _chatbot_decode_ids(ids, idx2word, cfg: dict[str, Any]) -> str:
    special_tokens = cfg.get("special_tokens", {})
    stop_tokens = {
        special_tokens.get("pad_token", "<pad>"),
        special_tokens.get("sos_token", "<sos>"),
    }
    eos_token = special_tokens.get("eos_token", "<eos>")

    words = []
    for idx in ids:
        token = idx2word.get(int(idx), special_tokens.get("unk_token", "<unk>"))
        if token == eos_token:
            break
        if token not in stop_tokens:
            words.append(token)
    return " ".join(words)


def _local_chatbot_generate(spec: ModelSpec, query: str, history: str, cfg: dict[str, Any]) -> Optional[str]:
    try:
        torch, _ = _get_torch()
        vocab = _load_pickle(spec.meta_paths["vocab"])
        idx2word = _load_pickle(spec.meta_paths["idx2word"])

        if isinstance(idx2word, dict):
            idx2word = {int(k): v for k, v in idx2word.items()}

        cfg["_runtime_vocab"] = vocab
        model, _, _, _ = _build_local_chatbot_model(len(vocab), cfg)

        ckpt = torch.load(_resolve_artifact_path(spec.model_path), map_location="cpu")
        state = _extract_state_dict(ckpt)
        state = _strip_prefixes(state)
        model.load_state_dict(state, strict=False)
        model.eval()

        full_query = f"{history} {query}".strip() if history else query
        max_input_len = int(cfg.get("max_input_len", 60))
        max_output_len = int(cfg.get("max_output_len", 80))

        src = _chatbot_encode_text(full_query, vocab, cfg, max_input_len)
        src_tensor = torch.tensor(src, dtype=torch.long).unsqueeze(0)

        with torch.no_grad():
            generated = model(src_tensor, trg=None, teacher_forcing_ratio=0.0, max_len=max_output_len)

        if hasattr(generated, "detach"):
            generated_ids = generated[0].detach().cpu().numpy().tolist()
        else:
            generated_ids = list(generated[0])

        out = _chatbot_decode_ids(generated_ids, idx2word, cfg)
        out = _clean_chatbot_text(out)
        return out or None
    except Exception:
        return None


# -----------------------------
# Main predictor
# -----------------------------
def predict(
    spec: ModelSpec,
    payload: dict[str, Any],
    return_proba: bool = False,
    top_k: int = 5,
) -> Tuple[Any, dict[str, Any]]:
    kind = spec.kind

    # -----------------------------
    # SKLEARN
    # -----------------------------
    if kind == "sklearn":
        model_path = _resolve_artifact_path(spec.model_path)
        model = loaders.load_joblib(model_path)
        _patch_sklearn_lr(model)

        input_schema = spec.input_schema.get("type")

        if input_schema == "text":
            text_field = spec.input_schema.get("field", "text")
            text = payload.get(text_field, "")

            y = model.predict([text])[0]
            meta: dict[str, Any] = {}

            if return_proba and hasattr(model, "predict_proba"):
                meta["proba"] = model.predict_proba([text])[0].tolist()

            _maybe_add_label_map(meta, spec)
            return _to_jsonable(y), _to_jsonable(meta)

        feature_columns = _load_feature_columns(spec)
        if not feature_columns:
            raise ValueError(
                f"No feature columns found for model_id={spec.model_id}. "
                f"Add feature_columns.json or include columns in config."
            )

        features = payload.get(spec.input_schema.get("field", "features"), {}) or {}
        X_df = _tabular_to_df(features, feature_columns)

        imputed_cols: list[str] = []

        if spec.task == "clustering":
            X_df = X_df.apply(pd.to_numeric, errors="coerce")

            if X_df.isna().any(axis=None):
                imputed_cols = [c for c in feature_columns if pd.isna(X_df.loc[0, c])]

                fill_values = {}
                profiles_path = spec.meta_paths.get("profiles")
                if profiles_path:
                    try:
                        profiles_df = loaders.load_parquet(_resolve_artifact_path(profiles_path))
                        prof_num = profiles_df.reindex(columns=feature_columns).apply(pd.to_numeric, errors="coerce")
                        fill_values = prof_num.mean(numeric_only=True).to_dict()
                    except Exception:
                        fill_values = {}

                X_df = X_df.fillna(value=fill_values).fillna(0.0)

        if spec.task in ("los_regression", "risk_classification"):
            X_df, imputed_cols = _impute_tabular_soft(X_df, feature_columns, imputed_cols)

        y = model.predict(X_df)[0]
        meta: dict[str, Any] = {"feature_columns": feature_columns}

        if imputed_cols:
            meta["imputed_columns"] = imputed_cols

        if return_proba and hasattr(model, "predict_proba"):
            meta["proba"] = model.predict_proba(X_df)[0].tolist()

        _maybe_add_label_map(meta, spec)

        if spec.task == "clustering":
            profiles_path = spec.meta_paths.get("profiles")
            cfg_path = spec.meta_paths.get("config")

            if profiles_path:
                profiles_df = loaders.load_parquet(_resolve_artifact_path(profiles_path))

                cluster_id_col = "cluster_id"
                cluster_label_col = "cluster_label"
                if cfg_path:
                    cfg = loaders.load_json(_resolve_artifact_path(cfg_path))
                    cluster_id_col = cfg.get("cluster_id_col", cluster_id_col)
                    cluster_label_col = cfg.get("cluster_label_col", cluster_label_col)

                if cluster_id_col in profiles_df.columns:
                    match = profiles_df[profiles_df[cluster_id_col] == int(y)]
                    if len(match):
                        prof = _to_jsonable(match.iloc[0].to_dict())
                        meta["cluster_profile"] = prof
                        meta["cluster_label"] = prof.get(cluster_label_col)
                        meta["cluster_id_col"] = cluster_id_col
                        meta["cluster_label_col"] = cluster_label_col
                    else:
                        meta["cluster_profile"] = None
                else:
                    meta["cluster_profile_note"] = (
                        f"'{cluster_id_col}' not found in profiles. Columns: {list(profiles_df.columns)}"
                    )

        return _to_jsonable(y), _to_jsonable(meta)

    # -----------------------------
    # RULES
    # -----------------------------
    if kind == "rules":
        df: pd.DataFrame = loaders.load_parquet(_resolve_artifact_path(spec.model_path))

        items = payload.get(spec.input_schema.get("field", "items"), [])
        if isinstance(items, str):
            items = [x.strip() for x in items.split(",") if x.strip()]
        raw_items = [str(x).strip() for x in (items or []) if str(x).strip()]

        def parse_token_list(v: Any) -> set[str]:
            aset = _safe_parse_antecedent(v)
            if aset is not None:
                return {str(x).strip() for x in aset if str(x).strip()}
            if isinstance(v, str):
                parts = [p.strip() for p in v.split(",") if p.strip()]
                return set(parts)
            return set()

        if "antecedents" not in df.columns or "consequents" not in df.columns:
            raise ValueError("Association rules parquet must contain 'antecedents' and 'consequents' columns.")

        vocab: set[str] = set()
        for col in ["antecedents", "consequents"]:
            for v in df[col].tolist():
                vocab.update(parse_token_list(v))

        def _canon(s: str) -> str:
            return str(s).strip().lower().replace(" ", "_").replace("-", "_")

        if bool(payload.get("return_vocab")):
            return [], {"vocab": sorted(vocab), "vocab_size": len(vocab)}

        synonym_map = {"fever": "DX_R50.9", "r50.9": "DX_R50.9"}

        mapped: list[str] = []
        unknown: list[str] = []

        for it in raw_items:
            c = _canon(it)

            exact = next((v for v in vocab if v.lower() == c), None)
            if exact:
                mapped.append(exact)
                continue

            if c in synonym_map and synonym_map[c] in vocab:
                mapped.append(synonym_map[c])
                continue

            if re.fullmatch(r"[a-z]\d{2,3}(?:\.\d+)?", c):
                cand = "DX_" + c.upper()
                if cand in vocab:
                    mapped.append(cand)
                    continue

            found = False
            for pref in ["MED_", "COND_", "DX_"]:
                cand = pref + c
                if cand in vocab:
                    mapped.append(cand)
                    found = True
                    break
            if not found:
                unknown.append(it)

        items_set = set(mapped)

        ant_sets = df["antecedents"].apply(parse_token_list)
        con_sets = df["consequents"].apply(parse_token_list)

        cand = df[ant_sets.apply(lambda s: len(s) > 0 and s.issubset(items_set))].copy()
        mode_used = "strict"

        if cand.empty and items_set:
            ant_overlap = ant_sets.apply(lambda s: len(s.intersection(items_set)))
            con_overlap = con_sets.apply(lambda s: len(s.intersection(items_set)))
            rel_score = ant_overlap + con_overlap

            cand = df[rel_score >= 1].copy()
            cand["related_score"] = rel_score[rel_score >= 1].values
            cand["ant_overlap"] = ant_overlap[rel_score >= 1].values
            cand["con_overlap"] = con_overlap[rel_score >= 1].values
            mode_used = "related"

        if cand.empty:
            cand = df.copy()
            mode_used = "top_rules"

        sort_cols = [c for c in ["related_score", "lift", "confidence", "support"] if c in cand.columns]
        if sort_cols:
            cand = cand.sort_values(by=sort_cols, ascending=False)

        out = []
        for _, r in cand.head(10).iterrows():
            rec = r.to_dict()
            rec.pop("antecedents_str", None)
            rec.pop("consequents_str", None)
            rec["antecedents_list"] = sorted(parse_token_list(r["antecedents"]))
            rec["consequents_list"] = sorted(parse_token_list(r["consequents"]))
            out.append(_to_jsonable(rec))

        suggestions = {}
        if unknown:
            import difflib
            vocab_map = {v.lower(): v for v in vocab}
            vocab_lower = sorted(vocab_map.keys())
            suggestions = {
                u: [vocab_map[c] for c in difflib.get_close_matches(_canon(u), vocab_lower, n=5, cutoff=0.6)]
                for u in unknown
            }

        meta = {
            "input_raw": raw_items,
            "mapped_items": sorted(items_set),
            "unknown_items": unknown,
            "suggestions": suggestions,
            "mode_used": mode_used,
            "returned": len(out),
        }
        return out, meta

    # -----------------------------
    # RETRIEVAL
    # -----------------------------
    if kind == "retrieval":
        from sklearn.metrics.pairwise import cosine_similarity
        from scipy import sparse

        vectorizer = loaders.load_joblib(_resolve_artifact_path(spec.model_path))

        matrix_path = _resolve_artifact_path(spec.meta_paths["matrix"])
        try:
            doc_matrix = sparse.load_npz(matrix_path)
        except Exception:
            mat_npz = loaders.load_npz(matrix_path)
            doc_matrix = _extract_npz_matrix(mat_npz)
            if not sparse.issparse(doc_matrix):
                doc_matrix = sparse.csr_matrix(doc_matrix)

        store: pd.DataFrame = loaders.load_parquet(_resolve_artifact_path(spec.meta_paths["store"]))

        query = payload.get(spec.input_schema.get("field", "query"), "")
        qv = vectorizer.transform([query])

        if doc_matrix.shape[0] != len(store) and doc_matrix.shape[1] == len(store):
            doc_matrix = doc_matrix.T

        sims = cosine_similarity(qv, doc_matrix).ravel()
        k = max(1, int(top_k))
        idx = np.argsort(-sims)[:k]

        results = []
        for i in idx:
            row = store.iloc[int(i)].to_dict()
            row["score"] = float(sims[int(i)])
            results.append(row)

        answer = results[0] if results else None
        return _to_jsonable({"answer": answer, "hits": results}), {"top_k": k}

    # -----------------------------
    # TRANSLATION
    # -----------------------------
    if kind in {"translation", "hf_translation"}:
        text_field = spec.input_schema.get("field", "text")
        text = payload.get(text_field, "")

        source_lang = _normalize_lang_code(payload.get("source_lang", "auto"))
        target_lang = _normalize_lang_code(payload.get("target_lang", "en"))

        if not text or not str(text).strip():
            return "", {"detail": "Empty text"}

        if source_lang == target_lang:
            return str(text), {"provider": "noop", "source_lang": source_lang, "target_lang": target_lang}

        if kind == "hf_translation":
            translator = MarianTranslator(_resolve_artifact_path(spec.model_path))
            out = translator.translate(
                text=str(text),
                source_lang=source_lang,
                target_lang=target_lang,
            )
            return _unescape_translate_text(out), {
                "provider": "marianmt",
                "source_lang": source_lang,
                "target_lang": target_lang,
            }

        cfg: dict[str, Any] = {}
        cfg_path = _resolve_artifact_path(spec.model_path)
        if _path_exists(cfg_path):
            try:
                cfg = loaders.load_json(cfg_path)
            except Exception:
                cfg = {}

        provider = str(payload.get("provider", "") or "").strip().lower()
        if source_lang in (None, "", "auto"):
            source_lang = cfg.get("default_source_lang", "auto")
        if target_lang in (None, ""):
            target_lang = cfg.get("default_target_lang", "en")
        if not provider:
            provider = str(cfg.get("provider", "gcp")).strip().lower()

        if target_lang in (None, "", "auto"):
            return str(text), {"provider": "noop", "source_lang": source_lang, "target_lang": target_lang}
        if source_lang not in (None, "", "auto") and str(source_lang) == str(target_lang):
            return str(text), {"provider": "noop", "source_lang": source_lang, "target_lang": target_lang}

        if provider == "googletrans":
            try:
                from googletrans import Translator
                tr = Translator()
                res = tr.translate(
                    str(text),
                    src=None if source_lang in (None, "", "auto") else str(source_lang),
                    dest=str(target_lang),
                )
                return _unescape_translate_text(getattr(res, "text", text)), {
                    "provider": "googletrans",
                    "source_lang": source_lang,
                    "target_lang": target_lang,
                }
            except Exception as e:
                last_err = str(e)
        else:
            last_err = ""

        if provider == "gemini":
            try:
                import vertexai
                from vertexai.generative_models import GenerativeModel, GenerationConfig

                project = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCP_PROJECT")
                location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
                if project:
                    vertexai.init(project=project, location=location)

                model_name = str(cfg.get("gemini_model", "gemini-1.5-flash"))
                model = GenerativeModel(model_name)

                prompt = (
                    "You are a translation engine.\n"
                    f"Translate the following text.\n"
                    f"Source language: {source_lang}\n"
                    f"Target language: {target_lang}\n"
                    "Return ONLY the translated text, no explanations.\n\n"
                    f"TEXT:\n{text}"
                )

                resp = model.generate_content(
                    prompt,
                    generation_config=GenerationConfig(temperature=0.0, max_output_tokens=512),
                )
                out = str(getattr(resp, "text", "") or "").strip()
                if out:
                    return out, {
                        "provider": "gemini",
                        "source_lang": source_lang,
                        "target_lang": target_lang,
                        "model": model_name,
                    }
            except Exception as e:
                last_err = str(e)

        try:
            from google.cloud import translate_v2 as translate
            client = translate.Client()
            res = client.translate(
                str(text),
                target_language=str(target_lang),
                source_language=None if source_lang in (None, "", "auto") else str(source_lang),
            )
            return _unescape_translate_text(res.get("translatedText", text)), {
                "provider": "gcp_translate_v2",
                "source_lang": source_lang,
                "target_lang": target_lang,
            }
        except Exception as e:
            last_err = str(e)

        try:
            from google.cloud import translate

            project_id = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCP_PROJECT") or ""
            location = os.environ.get("GOOGLE_CLOUD_LOCATION", "global")
            if not project_id:
                return str(text), {
                    "provider": "fallback",
                    "reason": "Missing GOOGLE_CLOUD_PROJECT/GCP_PROJECT",
                    "source_lang": source_lang,
                    "target_lang": target_lang,
                    "prev_error": last_err,
                }

            client = translate.TranslationServiceClient()
            parent = f"projects/{project_id}/locations/{location}"
            resp = client.translate_text(
                request={
                    "parent": parent,
                    "contents": [str(text)],
                    "mime_type": "text/plain",
                    "source_language_code": "" if source_lang in (None, "", "auto") else str(source_lang),
                    "target_language_code": str(target_lang),
                }
            )
            if resp.translations:
                return _unescape_translate_text(resp.translations[0].translated_text), {
                    "provider": "gcp_translate_v3",
                    "source_lang": source_lang,
                    "target_lang": target_lang,
                }
        except Exception as e:
            last_err = str(e)

        return str(text), {
            "provider": "fallback",
            "reason": last_err or "Translation failed",
            "source_lang": source_lang,
            "target_lang": target_lang,
        }

    # -----------------------------
    # CNN XRAY MULTILABEL
    # -----------------------------
    if kind == "cnn_xray_multilabel":
        image_field = spec.input_schema.get("field", "image_b64")
        image_bytes = _get_image_bytes_from_payload(payload, field=image_field)

        resolved_model_path = Path(_resolve_artifact_path(spec.model_path))
        artifact_dir = str(resolved_model_path.parent)
        model_filename = resolved_model_path.name

        model = XrayMultiLabelClassifier(
            artifact_dir=artifact_dir,
            model_filename=model_filename,
        )

        result = model.predict(image_bytes)

        meta = {
            "task_type": result["task_type"],
            "threshold": result["threshold"],
            "probabilities": result["probabilities"],
            "top_scores": result["top_scores"],
        }

        metrics_path = spec.meta_paths.get("metrics")
        if metrics_path:
            try:
                metrics = loaders.load_json(_resolve_artifact_path(metrics_path))
                if isinstance(metrics, dict):
                    if "best_mean_auc" in metrics:
                        meta["best_mean_auc"] = metrics["best_mean_auc"]
                    if "per_class_auc" in metrics:
                        meta["per_class_auc"] = metrics["per_class_auc"]
            except Exception:
                pass

        return _to_jsonable(result["predicted_findings"]), _to_jsonable(meta)

    # -----------------------------
    # TORCH
    # -----------------------------
    if kind == "torch":
        if spec.task == "chatbot":
            cfg_path = spec.meta_paths.get("config")
            cfg = {}
            if cfg_path and _path_exists(cfg_path):
                cfg = loaders.load_json(_resolve_artifact_path(cfg_path))

            query = str(payload.get(spec.input_schema.get("field", "query"), "") or "").strip()
            history = str(payload.get("history", "") or "").strip()
            source_lang = _normalize_lang_code(payload.get("source_lang", "en"))
            target_lang = _normalize_lang_code(payload.get("target_lang", "en"))

            if not query:
                return "", {"detail": "Empty query"}

            if source_lang != "en":
                translator_spec = ModelSpec(
                    model_id="translator_marian_v2",
                    kind="hf_translation",
                    task="translation",
                    model_path=Path("artifacts/translator/translator_marian_v2/translator_config.json"),
                    meta_paths={"config": Path("artifacts/translator/translator_marian_v2/translator_config.json")},
                    input_schema={"type": "text", "field": "text"},
                )
                query_en, _ = predict(
                    translator_spec,
                    {
                        "text": query,
                        "source_lang": source_lang,
                        "target_lang": "en",
                    },
                )
                history_en = history
            else:
                query_en = query
                history_en = history

            hits = _retrieve_chatbot_hits(
                spec=spec,
                query=query_en,
                top_k=int(payload.get("top_k", cfg.get("top_k", top_k))),
                cfg=cfg,
            )

            local_answer = _local_chatbot_generate(spec, query_en, history_en, cfg)
            if local_answer:
                answer_en = local_answer
                provider = "local_seq2seq"
            else:
                llm_provider = str(cfg.get("rag", {}).get("llm_provider", "")).strip().lower()
                if llm_provider in {"vertexai_gemini", "gemini"}:
                    answer_en = _chatbot_vertex_answer(query_en, history_en, hits, cfg)
                    provider = "vertexai_gemini"
                else:
                    answer_en = _chatbot_rule_based_answer(query_en, hits, cfg)
                    provider = "retrieval_fallback"

            answer_en = _clean_chatbot_text(answer_en)

            final_answer = answer_en
            if target_lang != "en":
                translator_spec = ModelSpec(
                    model_id="translator_marian_v2",
                    kind="hf_translation",
                    task="translation",
                    model_path=Path("artifacts/translator/translator_marian_v2/translator_config.json"),
                    meta_paths={"config": Path("artifacts/translator/translator_marian_v2/translator_config.json")},
                    input_schema={"type": "text", "field": "text"},
                )
                final_answer, _ = predict(
                    translator_spec,
                    {
                        "text": answer_en,
                        "source_lang": "en",
                        "target_lang": target_lang,
                    },
                )

            return _to_jsonable({
                "answer": final_answer,
                "answer_en": answer_en,
                "hits": hits,
            }), {
                "provider": provider,
                "source_lang": source_lang,
                "target_lang": target_lang,
                "top_k": int(payload.get("top_k", cfg.get("top_k", top_k))),
            }

        mode, obj = loaders.load_torch_model(_resolve_artifact_path(spec.model_path), device="cpu")

        if mode == "torchscript":
            torch, _ = _get_torch()
            task = spec.task

            if task == "timeseries":
                seq = payload.get(spec.input_schema.get("field", "sequence"))
                if seq is None:
                    return {"detail": "Missing 'sequence' in payload"}, {"mode": "torchscript"}

                x = torch.tensor(seq, dtype=torch.float32)
                if x.dim() == 2:
                    x = x.unsqueeze(0)

                with torch.no_grad():
                    y = obj(x).detach().cpu().numpy()

                return float(np.ravel(y)[0]), {"shape_in": list(x.shape), "mode": "torchscript"}

            if task == "imaging":
                img_b64 = payload.get(spec.input_schema.get("field", "image_b64"))
                if not img_b64:
                    return {"detail": "Missing 'image_b64' in payload"}, {"mode": "torchscript"}

                cfg = loaders.load_json(_resolve_artifact_path(spec.meta_paths["config"])) if spec.meta_paths.get("config") else {}
                label_map = loaders.load_json(_resolve_artifact_path(spec.meta_paths["label_map"])) if spec.meta_paths.get("label_map") else {}

                img = _decode_image_b64(img_b64)
                x = _imaging_preprocess(img, cfg)

                with torch.no_grad():
                    logits = obj(x)

                pred_label, meta = _imaging_postprocess(logits, label_map)
                meta.update({"shape_in": list(x.shape), "mode": "torchscript"})
                return _to_jsonable(pred_label), _to_jsonable(meta)

            return {"detail": f"Torch model loaded for task={task} but predictor not implemented yet."}, {"mode": "torchscript"}

        if mode == "state_dict":
            torch, _ = _get_torch()
            task = spec.task

            if task == "timeseries":
                arch_path = spec.meta_paths.get("arch")
                if not arch_path:
                    raise ValueError("Timeseries state_dict needs meta_paths['arch'] (arch.json).")

                mid = str(spec.model_id).lower()
                force_type = "rnn" if "rnn" in mid else "lstm"

                arch = loaders.load_json(_resolve_artifact_path(arch_path))
                model = TSRegressor(arch, force_type=force_type)

                state = _extract_state_dict(obj)
                state = _strip_prefixes(state)

                try:
                    model.load_state_dict(state, strict=True)
                except Exception:
                    model.load_state_dict(state, strict=False)

                model.eval()

                seq = payload.get(spec.input_schema.get("field", "sequence"))
                if seq is None:
                    return {"detail": "Missing 'sequence' in payload"}, {"mode": "state_dict"}

                x = torch.tensor(seq, dtype=torch.float32)
                if x.dim() == 2:
                    x = x.unsqueeze(0)

                with torch.no_grad():
                    y = model(x).detach().cpu().numpy()

                return float(np.ravel(y)[0]), {"shape_in": list(x.shape), "mode": "state_dict", "arch_type": force_type}

            if task == "imaging":
                img_b64 = payload.get(spec.input_schema.get("field", "image_b64"))
                if not img_b64:
                    return {"detail": "Missing 'image_b64' in payload"}, {"mode": "state_dict"}

                cfg = loaders.load_json(_resolve_artifact_path(spec.meta_paths["config"])) if spec.meta_paths.get("config") else {}
                label_map = loaders.load_json(_resolve_artifact_path(spec.meta_paths["label_map"])) if spec.meta_paths.get("label_map") else {}

                arch_name = str(cfg.get("arch", "resnet18")).lower()
                num_classes = int(cfg.get("num_classes", 2))

                if arch_name != "resnet18":
                    raise ValueError(f"Unsupported imaging arch: {arch_name}. Only resnet18 implemented.")

                cnn = _build_resnet18(num_classes=num_classes)

                state = _extract_state_dict(obj)
                state = _strip_prefixes(state)
                cnn.load_state_dict(state, strict=True)
                cnn.eval()

                img = _decode_image_b64(img_b64)
                x = _imaging_preprocess(img, cfg)

                with torch.no_grad():
                    logits = cnn(x)

                pred_label, meta = _imaging_postprocess(logits, label_map)
                meta.update({"shape_in": list(x.shape), "mode": "state_dict"})
                return _to_jsonable(pred_label), _to_jsonable(meta)

            return {"detail": f"state_dict loading not implemented for task={task}."}, {"mode": "state_dict"}

        raise ValueError(f"Unsupported torch load mode: {mode}")

    raise ValueError(f"Unsupported kind: {kind}")