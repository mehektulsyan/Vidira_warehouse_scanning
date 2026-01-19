import base64
import re
import cv2
import numpy as np
import pandas as pd
import requests
import streamlit as st
from datetime import datetime
from PIL import Image
from io import BytesIO

from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

st.set_page_config(page_title="Vidira Warehouse Scanner POC", layout="wide")
st.title("📦 Vidira Warehouse Scanner (Streamlit Cloud POC)")
st.caption("Live camera → Scan → QR (OpenCV) + OCR (Cloud API) → 2 scans merge into 1 row")


# ----------------------------
# Video capture
# ----------------------------
class FrameGrabber(VideoProcessorBase):
    def __init__(self):
        self.latest_bgr = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.latest_bgr = img
        return frame


# ----------------------------
# OCR via OCR.space
# Put your key in Streamlit Cloud secrets:
# OCR_SPACE_API_KEY="xxxxx"
# ----------------------------
def ocr_space_image_bytes(image_bytes: bytes) -> str:
    api_key = st.secrets.get("OCR_SPACE_API_KEY", "")
    if not api_key:
        st.error("Missing OCR_SPACE_API_KEY in Streamlit secrets.")
        st.stop()

    # OCR.space expects base64 or multipart. We'll do base64.
    b64 = base64.b64encode(image_bytes).decode("utf-8")

    resp = requests.post(
        "https://api.ocr.space/parse/image",
        data={
            "apikey": api_key,
            "base64Image": f"data:image/jpeg;base64,{b64}",
            "language": "eng",
            "isOverlayRequired": "false",
            # Helps with labels sometimes:
            "OCREngine": "2",
            "scale": "true",
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()

    if data.get("IsErroredOnProcessing"):
        msg = data.get("ErrorMessage") or data.get("ErrorDetails") or "OCR error"
        raise RuntimeError(str(msg))

    parsed = data.get("ParsedResults", [])
    if not parsed:
        return ""
    text = parsed[0].get("ParsedText", "") or ""
    text = text.replace("\x0c", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


# ----------------------------
# QR decode via OpenCV
# ----------------------------
def read_qr_opencv(bgr: np.ndarray):
    results = []
    try:
        qr = cv2.QRCodeDetector()
        data, points, _ = qr.detectAndDecode(bgr)
        if data:
            results.append({"type": "QRCODE", "data": data})
    except Exception:
        pass
    return results


# ----------------------------
# Basic preprocessing (makes OCR better)
# ----------------------------
def preprocess_for_ocr(bgr: np.ndarray) -> bytes:
    # Convert to grayscale + sharpen-ish
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    thr = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 5
    )
    # Encode to JPEG bytes
    ok, buf = cv2.imencode(".jpg", thr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise RuntimeError("Failed to encode image for OCR")
    return buf.tobytes()


# ----------------------------
# Classification + field extraction (your label types)
# ----------------------------
ORDER_HINTS = ["DELIVERY", "BOX NO", "DEALER", "TRPT", "CITY", "SAFEEXPRESS", "DATE"]
ITEM_HINTS  = ["MODEL", "STYLE", "COLOR", "QTY", "PRODUCT", "CODE", "MRP", "VEGA", "HELMET", "BAJAJ", "COMMODITY"]

def guess_label_type(text: str, qr_results):
    t = (text or "").upper()
    order_score = sum(1 for k in ORDER_HINTS if k in t)
    item_score  = sum(1 for k in ITEM_HINTS  if k in t)
    has_qr = any(r["type"] == "QRCODE" for r in (qr_results or []))

    if order_score > item_score:
        return "ORDER"
    if item_score > order_score:
        return "ITEM"
    return "ORDER" if has_qr else "ITEM"

def grab(text, pattern, flags=re.IGNORECASE):
    m = re.search(pattern, text, flags)
    return m.group(1).strip() if m else None

def extract_order(text: str):
    t = text.upper()
    return {
        "Delivery No": grab(text, r"DELIVERY\s*[:\-]?\s*([0-9A-Z\-\/]+)") or grab(text, r"DELIVERY\s+([0-9A-Z\-\/]+)"),
        "Box No": grab(text, r"BOX\s*NO\s*[:\-]?\s*([0-9A-Z\-\/]+)"),
        "Dealer": grab(text, r"DEALER\s*[:\-]?\s*([A-Z0-9 \-]+)"),
        "City": grab(text, r"CITY\s*[:\-]?\s*([A-Z0-9 \-]+)"),
        "Transporter": grab(text, r"TRPT\s*[:\-]?\s*([A-Z0-9 \-]+)") or ("SAFEEXPRESS" if "SAFEEXPRESS" in t else None),
        "Order Date": grab(text, r"DATE\s*[:\-]?\s*([0-9\/\.\-]+)"),
    }

def extract_vega(text: str):
    t = text.upper()
    return {
        "Brand": "VEGA" if "VEGA" in t else None,
        "Model": grab(text, r"MODEL\s*[:\-]\s*([A-Z0-9\-]+)"),
        "Style": grab(text, r"STYLE\s*[:\-]\s*([A-Z0-9 \-]+)"),
        "Color": grab(text, r"COLOR\s*[:\-]\s*([A-Z0-9 \-]+)"),
        "Qty": grab(text, r"QTY\s*[:\-]\s*([0-9]+)"),
        "Size": grab(text, r"SIZE\s*[:\-]\s*([A-Z0-9\(\) \-]+)"),
        "Item Code": grab(text, r"CODE\s*[:\-]\s*([A-Z0-9\-]+)"),
        "MRP": grab(text, r"MRP\s*[:\-]?\s*RS\.?\s*([0-9,\.]+)"),
        "MFG Date": grab(text, r"MFG\s*[:\-]\s*([0-9\/\-]+)"),
    }

def extract_bajaj(text: str):
    t = text.upper()
    return {
        "Brand": "BAJAJ" if "BAJAJ" in t else None,
        "Commodity": grab(text, r"COMMODITY\s*[:\-]\s*([A-Z0-9 \-\(\)\/]+)"),
        "MRP": grab(text, r"MRP\s*₹?\s*([0-9,\.]+)"),
        "Unit Sale Price": grab(text, r"UNIT SALE PRICE\s*₹?\s*([0-9,\.]+)"),
        "Manufactured On": grab(text, r"MANUFACTURED\s*(?:ON)?\s*[:\-]?\s*([0-9\-\/]+|[A-Z]{2,}-[0-9]{4})"),
    }

def extract_item(text: str):
    t = (text or "").upper()
    if "VEGA" in t or ("MODEL" in t and "STYLE" in t and "MRP" in t):
        return extract_vega(text)
    if "BAJAJ" in t or "COMMODITY" in t or "UNIT SALE PRICE" in t:
        return extract_bajaj(text)
    return {"Raw Item Text": (text or "")[:500]}


# ----------------------------
# Session state (2 scans per row)
# ----------------------------
if "pending_order" not in st.session_state:
    st.session_state.pending_order = None
if "pending_item" not in st.session_state:
    st.session_state.pending_item = None
if "rows" not in st.session_state:
    st.session_state.rows = []

def merge_if_ready():
    if st.session_state.pending_order and st.session_state.pending_item:
        merged = {
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            **st.session_state.pending_order,
            **st.session_state.pending_item,
        }
        st.session_state.rows.append(merged)
        st.session_state.pending_order = None
        st.session_state.pending_item = None
        st.success("✅ ORDER + ITEM merged into one row.")


# ----------------------------
# UI
# ----------------------------
left, right = st.columns([2, 1])

with left:
    st.info("On phone, allow camera. Fill frame with label, keep steady, press **Scan**.")
    webrtc_ctx = webrtc_streamer(
        key="scanner",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=FrameGrabber,
        media_stream_constraints={"video": {"facingMode": "environment"}, "audio": False},
        async_processing=True,
    )

with right:
    st.subheader("Controls")
    label_override = st.selectbox("Label type", ["AUTO", "ORDER", "ITEM"], index=0)
    scan_btn = st.button("✅ Scan current frame", use_container_width=True)

    st.markdown("### Pending (2-scan merge)")
    st.write("**Order label:**", "✅" if st.session_state.pending_order else "—")
    st.write("**Item label:**", "✅" if st.session_state.pending_item else "—")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("↩️ Reset pending", use_container_width=True):
            st.session_state.pending_order = None
            st.session_state.pending_item = None
            st.toast("Pending cleared.")
    with c2:
        if st.button("🧹 Clear rows", use_container_width=True):
            st.session_state.rows = []
            st.toast("All rows cleared.")

if scan_btn:
    if not webrtc_ctx.video_processor or webrtc_ctx.video_processor.latest_bgr is None:
        st.error("Camera not ready. Wait 1–2 seconds and try again.")
        st.stop()

    frame_bgr = webrtc_ctx.video_processor.latest_bgr

    # Show captured frame (optional, useful for debugging)
    st.subheader("Captured Frame")
    st.image(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)

    # QR decode
    qr_results = read_qr_opencv(frame_bgr)
    qr_text = " | ".join([f"{r['type']}:{r['data']}" for r in qr_results]) if qr_results else None

    # OCR via cloud
    try:
        ocr_bytes = preprocess_for_ocr(frame_bgr)
        text = ocr_space_image_bytes(ocr_bytes)
    except Exception as e:
        st.error(f"OCR failed: {e}")
        st.stop()

    if label_override == "AUTO":
        label_type = guess_label_type(text, qr_results)
    else:
        label_type = label_override

    if label_type == "ORDER":
        fields = extract_order(text)
        fields["Codes/QR"] = qr_text
        fields["Raw OCR"] = text[:800]
        st.session_state.pending_order = fields
        st.success("✅ Captured ORDER label.")
    else:
        fields = extract_item(text)
        fields["Codes/QR"] = qr_text
        fields["Raw OCR"] = text[:800]
        st.session_state.pending_item = fields
        st.success("✅ Captured ITEM label.")

    merge_if_ready()


st.divider()
st.subheader("📋 Merged Rows (Order + Item)")

if st.session_state.rows:
    df = pd.DataFrame(st.session_state.rows)
    st.dataframe(df, use_container_width=True)
    st.download_button("⬇️ Download CSV", df.to_csv(index=False).encode("utf-8"), "warehouse_scans.csv", "text/csv")
else:
    st.warning("No merged rows yet. Scan ORDER label then ITEM label (or vice versa).")
