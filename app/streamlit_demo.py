"""
streamlit_demo.py — Vietnamese Toxic Comment Detection demo (PhoBERT-base-v2).

Run from the project root:
    streamlit run app/streamlit_demo.py

Standalone from the Week-3 LR app (``app/app.py``); only requires the
fine-tuned PhoBERT checkpoint under ``models/dl/phobert_best/`` (produced
by ``notebooks/04c_phobert.ipynb``).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Put BOTH the project root and this app/ folder on sys.path.
#
# `streamlit run app/streamlit_demo.py` puts the app/ folder on sys.path, so we
# import `predictor` as a sibling module (no `app.` prefix). The project root is
# added too so `predictor` can resolve `from src.preprocess import ...`.
_HERE = Path(__file__).resolve().parent          # .../app
_ROOT = _HERE.parent                              # project root
for _p in (str(_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import streamlit as st
import plotly.graph_objects as go

from predictor import PhoBERTPredictor   # sibling import (see note above)


# ──────────────────────────────────────────────────────────────────
# Debug logging — writes to app/demo_debug.log so we can see exactly
# what happens server-side (model load timing, any traceback) even when
# the browser only shows a generic "Connection error". Tail this file
# while reproducing: Get-Content app/demo_debug.log -Wait
# ──────────────────────────────────────────────────────────────────
_LOG_PATH = _HERE / "demo_debug.log"
_demo_logger = logging.getLogger("phobert_demo")
if not any(isinstance(h, logging.FileHandler) for h in _demo_logger.handlers):
    # Attach our own FileHandler directly to the logger (Streamlit already
    # owns the root logger, so logging.basicConfig would be a silent no-op).
    _fh = logging.FileHandler(_LOG_PATH, encoding="utf-8")
    _fh.setFormatter(logging.Formatter(
        "%(asctime)s  %(levelname)s  %(name)s  %(message)s"))
    _demo_logger.addHandler(_fh)
    _demo_logger.setLevel(logging.INFO)
    _demo_logger.propagate = False


# ──────────────────────────────────────────────────────────────────
# Page config + style constants
# ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Vietnamese Toxic Comment Detection",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="expanded",
)

LABEL_COLORS = {"CLEAN": "#4CAF50", "OFFENSIVE": "#FF9800", "HATE": "#F44336"}
LABEL_EMOJI  = {"CLEAN": "✅",      "OFFENSIVE": "⚠️",     "HATE": "🚫"}
LOW_CONF_THRESHOLD = 0.60          # below this → "uncertain" badge
HARD_TEXT_CHAR_LIMIT = 4000        # block obviously pathological inputs
# Note: the PhoBERT max_len ("truncated" notice + "Max len" metric) is read
# live from the loaded predictor (predictor.max_len) so it can never drift
# from the actual training/inference config.


# ──────────────────────────────────────────────────────────────────
# Cached model load (runs once per Streamlit process)
# ──────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading PhoBERT-base-v2 (one-time, ~10 s)…")
def get_predictor() -> PhoBERTPredictor:
    """Load the model once and reuse across reruns."""
    logging.getLogger("phobert_demo").info("get_predictor(): starting model load")
    try:
        p = PhoBERTPredictor()
    except Exception:
        logging.getLogger("phobert_demo").exception("get_predictor(): model load FAILED")
        raise
    logging.getLogger("phobert_demo").info("get_predictor(): model load succeeded")
    return p


# ──────────────────────────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────────────────────────
st.title("🛡️ Vietnamese Toxic Comment Detection")
st.markdown(
    "**PhoBERT-base-v2 fine-tuned on ViHSD** &nbsp;•&nbsp; "
    "Test macro-F1 = **0.6618** &nbsp;•&nbsp; "
    "3-class: CLEAN / OFFENSIVE / HATE"
)
st.caption(
    "Research prototype from a graduation thesis. Paste a Vietnamese comment "
    "below and click **Phân tích** to see the model's prediction, per-class "
    "probabilities, and the preprocessed text the model actually saw."
)
st.divider()


# ──────────────────────────────────────────────────────────────────
# Sidebar: examples + model info
# ──────────────────────────────────────────────────────────────────
EXAMPLES = {
    "✅ CLEAN — phàn nàn nhẹ":
        "Hôm nay trời mưa to quá, đi làm muộn mất rồi.",
    "⚠️ OFFENSIVE — chửi thề chung chung":
        "Đậu má cái xe này lại hỏng nữa rồi, bực thật sự.",
    "🚫 HATE — công kích có đối tượng":
        "Bọn này đáng chết, sống cũng vô dụng, không có ích gì cho xã hội.",
}

if "input_text" not in st.session_state:
    st.session_state["input_text"] = ""

with st.sidebar:
    st.header("📋 Ví dụ thử nghiệm")
    st.caption("Click một ví dụ để điền vào ô nhập:")
    for label, text in EXAMPLES.items():
        if st.button(label, width="stretch", key=f"ex_{label}"):
            st.session_state["input_text"] = text
            st.rerun()

    st.divider()
    st.header("ℹ️ Model info")
    try:
        info = get_predictor().info()
        st.markdown(f"""
- **Model**: PhoBERT-base-v2 (fine-tuned)
- **Device**: `{info['device']}`{'  •  fp16' if info['fp16'] else ''}
- **Params**: {info['param_count']:,}
- **Max len**: {info['max_len']} sub-word tokens
- **Labels**: {info['num_labels']}
""")
    except FileNotFoundError as e:
        st.error(f"⛔ Model not found.\n\n{e}")

    st.divider()
    st.markdown("**Dataset**: ViHSD (Luu et al., 2021)")
    st.markdown("_Graduation thesis demo — not for production moderation._")


# ──────────────────────────────────────────────────────────────────
# Input area
# ──────────────────────────────────────────────────────────────────
user_text = st.text_area(
    label="Bình luận tiếng Việt cần phân tích",
    value=st.session_state["input_text"],
    placeholder="VD: Hôm nay trời rất đẹp, tôi cảm thấy vui.",
    height=110,
    max_chars=HARD_TEXT_CHAR_LIMIT,
    key="text_area_input",
)
# Keep session_state in sync when the user types manually.
st.session_state["input_text"] = user_text

analyze = st.button("🔍 Phân tích / Analyze", type="primary", width="stretch")


# ──────────────────────────────────────────────────────────────────
# Helpers for output rendering
# ──────────────────────────────────────────────────────────────────
def render_prediction_badge(label_name: str, confidence: float) -> None:
    """Big coloured prediction header."""
    color = LABEL_COLORS[label_name]
    emoji = LABEL_EMOJI[label_name]
    st.markdown(
        f"""
        <div style="
            padding: 18px 22px;
            border-radius: 10px;
            background: {color};
            color: white;
            display: flex;
            align-items: center;
            gap: 14px;
        ">
            <span style="font-size: 38px;">{emoji}</span>
            <div>
                <div style="font-size: 13px; opacity: 0.85;">Predicted label</div>
                <div style="font-size: 26px; font-weight: 700;">{label_name}</div>
                <div style="font-size: 14px; opacity: 0.95;">
                    Confidence: {confidence:.1%}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_probability_chart(probs: dict) -> None:
    """Coloured horizontal-bar chart for the 3 class probabilities."""
    classes = ["CLEAN", "OFFENSIVE", "HATE"]
    values  = [probs[c] for c in classes]
    colors  = [LABEL_COLORS[c] for c in classes]
    fig = go.Figure(go.Bar(
        x=values, y=classes, orientation="h",
        marker_color=colors,
        text=[f"{v:.1%}" for v in values],
        textposition="outside",
        cliponaxis=False,
    ))
    fig.update_layout(
        xaxis=dict(range=[0, 1.0], tickformat=".0%", title="probability"),
        yaxis=dict(autorange="reversed"),                # CLEAN on top
        height=230, margin=dict(l=10, r=40, t=10, b=10),
        showlegend=False,
    )
    st.plotly_chart(fig, width="stretch")


# ──────────────────────────────────────────────────────────────────
# Inference + output
# ──────────────────────────────────────────────────────────────────
if analyze:
    txt = user_text.strip()
    if not txt:
        st.warning("⚠️ Vui lòng nhập một bình luận trước khi phân tích.")
        st.stop()

    # Heuristic non-Vietnamese sniff: % chars that are ASCII letters.
    ascii_letters = sum(1 for c in txt if c.isascii() and c.isalpha())
    total_letters = sum(1 for c in txt if c.isalpha())
    ascii_ratio = (ascii_letters / total_letters) if total_letters else 0.0
    if total_letters > 10 and ascii_ratio > 0.85:
        st.info(
            "ℹ️ Văn bản dường như không phải tiếng Việt. Mô hình vẫn chạy "
            "nhưng độ chính xác có thể giảm đáng kể (PhoBERT được huấn luyện "
            "trên dữ liệu tiếng Việt)."
        )

    try:
        with st.spinner("Đang chạy PhoBERT…"):
            predictor = get_predictor()
            out = predictor.predict(txt)
    except FileNotFoundError as e:
        st.error(f"⛔ Không tải được model.\n\n{e}")
        st.stop()
    except ValueError as e:
        st.warning(f"⚠️ {e}")
        st.stop()

    # ── Main result card
    render_prediction_badge(out["label_name"], out["confidence"])

    # ── Low-confidence flag
    if out["confidence"] < LOW_CONF_THRESHOLD:
        st.warning(
            f"🤔 **Low-confidence prediction** ({out['confidence']:.1%}). "
            "Model is uncertain — treat as borderline and prefer human review."
        )

    # ── Probability chart
    st.markdown("**Per-class probabilities**")
    render_probability_chart(out["probabilities"])

    # ── Diagnostics
    col1, col2, col3 = st.columns(3)
    col1.metric("⏱️ Inference", f"{out['inference_ms']:.1f} ms")
    col2.metric("🔤 PhoBERT tokens", f"{out['n_subword_tokens']}")
    col3.metric("📏 Max len", f"{predictor.max_len}")

    if out["truncated"]:
        st.info(
            f"✂️ Văn bản dài hơn {predictor.max_len} sub-word tokens — "
            "phần dư đã được PhoBERT cắt bớt khi suy luận."
        )

    # ── Preprocessed text (transparency)
    with st.expander("🔧 Xem văn bản sau khi xử lý (cleaned + word-segmented)"):
        st.markdown("**Original input:**")
        st.code(txt, language="text")
        st.markdown("**Cleaned + underthesea-segmented (input to PhoBERT):**")
        st.code(out["cleaned_text"], language="text")
        st.caption(
            "Pipeline: lowercase → remove URLs / emoji → collapse repeated chars "
            "→ teen-code normalisation → strip special chars → underthesea word segmentation. "
            "Underscores join multi-syllable words (e.g. `hôm_nay`)."
        )

    # ── Calibration disclaimer (always shown after a prediction)
    st.divider()
    st.markdown(
        ":warning: **Confidence scores may be overestimated.** "
        "The model is over-confident on test data (Expected Calibration "
        "Error ≈ 0.106). Wrong predictions can still report confidence "
        "above 0.9. This is a research prototype — not for production "
        "moderation without human review.",
        help="See Chapter 6 of the thesis: PhoBERT reliability diagram.",
    )


# ──────────────────────────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    """
<div style='font-size:12.5px; color:#666;'>
  PhoBERT-base-v2 fine-tuned on ViHSD (Luu et al., 2021).
  Test macro-F1 = 0.6618 &nbsp;•&nbsp; ECE = 0.106 &nbsp;•&nbsp;
  Trained on a single RTX 3060 (6 GB) with fp16.
  &nbsp;|&nbsp; Graduation thesis demo.
</div>
    """,
    unsafe_allow_html=True,
)