import streamlit as st
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.utils import load_model, predict_comment

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Vietnamese Toxic Comment Detector",
    page_icon="🛡️",
    layout="centered",
)

# ── Header ───────────────────────────────────────────────────
st.title("🛡️ Vietnamese Toxic Comment Detector")
st.markdown(
    "Nhập bình luận tiếng Việt để kiểm tra xem có chứa nội dung **độc hại** hay không."
)
st.divider()

# ── Model selection ───────────────────────────────────────────
model_choice = st.selectbox(
    "Chọn mô hình:",
    ["SVM + TF-IDF", "PhoBERT"],
    index=0,
)

# ── Input ─────────────────────────────────────────────────────
comment = st.text_area(
    "Nhập bình luận:",
    placeholder="Ví dụ: Tôi rất thích sản phẩm này!",
    height=150,
)

# ── Predict ───────────────────────────────────────────────────
if st.button("Phân tích", type="primary", use_container_width=True):
    if not comment.strip():
        st.warning("⚠️ Vui lòng nhập bình luận trước khi phân tích.")
    else:
        with st.spinner("Đang phân tích..."):
            model = load_model(model_choice)
            label, confidence = predict_comment(model, comment, model_choice)

        st.divider()
        if label == "toxic":
            st.error(f"🚨 **Độc hại** — Độ tin cậy: {confidence:.1%}")
        else:
            st.success(f"✅ **Bình thường** — Độ tin cậy: {confidence:.1%}")

# ── Footer ────────────────────────────────────────────────────
st.divider()
st.caption("Đồ án tốt nghiệp · Vietnamese Toxic Comment Detection · Data Science")
