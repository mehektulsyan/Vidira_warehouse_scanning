import re
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime

from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
from pyzbar.pyzbar import decode as zbar_decode
import pytesseract

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="Warehouse Label Scanner POC", layout="wide")
st.title("📦 Warehouse Label Scanner (POC)")
st.caption("Point camera at label → Scan → OCR + barcode/QR → 2 scans merge into 1 row (Order + Item).")

# ----------------------------
# Video frame capture
# ----------------------------
class FrameGrabber(VideoProcessorBase):
    def __init__(self):
        self.latest_bgr = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.latest_bgr = img
        return frame

# ----------------------------
# OCR + Barcode helpers
# ----------------------------
def preprocess_for_ocr(bgr: np.ndarray) -> np.ndarray:
    # Crop slightly to reduce background noise if label isn't full frame
    h, w = bgr.shape[:2]
    pad_h, pad_w = int(h * 0.03), int(w * 0.03)
    bgr = bgr[pad_h:h-pad_h, pad_w:w-pad_w]

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)

    thr = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 5
    )
    return thr

def run_ocr(image_for_ocr: np.ndarray) -> str:
    text = pytesseract.image_to_string(image_for_ocr)
    text = text.replace("\x0c", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text

def read_barcodes(bgr: np.ndarray):
    results = []
    for s in zbar_decode(bgr):
        results.append({"type": s.type, "data": s.data.decode("utf-8", errors="replace")})
    return results

# ----------------------------
# Label classification
# ----------------------------
ORDER_HINTS = [
    "DELIVERY", "BOX NO", "BOX", "DEALER", "TRPT", "CITY", "SAFEEXPRESS", "LR", "DATE:",
]
ITEM_HINTS = [
    "MODEL", "STYLE", "COLOR", "QTY", "PRODUCT", "CODE", "MRP", "VEGA", "HELMET",
    "BAJAJ AUTO", "GENUINE", "COMMODITY", "UNIT SALE PRICE", "MANUFACTURED"
]

def guess_label_type(text: str, barcodes) -> str:
    t = (text or "").upper()
    order_score = sum(1 for k in ORDER_HINTS if k in t)
    item_score  = sum(1 for k in ITEM_HINTS if k in t)

    # QR-heavy labels often are order/courier labels, but not always.
    has_qr = any(b["type"] in ["QRCODE", "QR_CODE"] for b in (barcodes or []))

    if order_score > item_score:
        return "ORDER"
    if item_score > order_score:
        return "ITEM"
    return "ORDER" if has_qr else "ITEM"

# ----------------------------
# Field extraction (templates)
# ----------------------------
def extract_vega_item(text: str):
    t = text.upper()

    def grab(pattern, flags=re.IGNORECASE):
        m = re.search(pattern, text, flags)
        return m.group(1).strip() if m else None

    # Vega format seen in your samples
    return {
        "Brand": "VEGA" if "VEGA" in t else None,
        "Model": grab(r"MODEL\s*[:\-]\s*([A-Z0-9\-]+)"),
        "Style": grab(r"STYLE\s*[:\-]\s*([A-Z0-9 \-]+)"),
        "Color": grab(r"COLOR\s*[:\-]\s*([A-Z0-9 \-]+)"),
        "Size": grab(r"SIZE\s*[:\-]\s*([A-Z0-9\(\) \-]+)"),
        "Qty": grab(r"QTY\s*[:\-]\s*([0-9]+)"),
        "Product": grab(r"PRODUCT\s*[:\-]\s*([A-Z0-9 \-]+)"),
        "Item Code": grab(r"CODE\s*[:\-]\s*([A-Z0-9\-]+)"),
        "MRP": grab(r"MRP\s*[:\-]?\s*RS\.?\s*([0-9,\.]+)"),
        "MFG Date": grab(r"MFG\s*[:\-]\s*([0-9\/\-]+)")
    }

def extract_bajaj_item(text: str):
    t = text.upper()

    def grab(pattern, flags=re.IGNORECASE):
        m = re.search(pattern, text, flags)
        return m.group(1).strip() if m else None

    return {
        "Brand": "BAJAJ" if "BAJAJ" in t else None,
        "Commodity": grab(r"COMMODITY\s*[:\-]\s*([A-Z0-9 \-\(\)\/]+)"),
        "MRP": grab(r"MRP\s*₹?\s*([0-9,\.]+)"),
        "Unit Sale Price": grab(r"UNIT SALE PRICE\s*₹?\s*([0-9,\.]+)"),
        "Manufactured On": grab(r"MANUFACTURED\s*(?:ON)?\s*[:\-]?\s*([0-9\-\/]+|[A-Z]{2,}-[0-9]{4})"),
        "Bajaj Ref/Part (raw)": grab(r"^\s*([A-Z0-9]{6,})\s*$", flags=re.MULTILINE),  # often like 56PF772V, JK122021 etc.
    }

def extract_order_label(text: str):
    t = text.upper()

    def grab(pattern, flags=re.IGNORECASE):
        m = re.search(pattern, text, flags)
        return m.group(1).strip() if m else None

    # Works for Safexpress-style + orange dealer labels shown
    delivery = grab(r"DELIVERY\s*[:\-]?\s*([0-9A-Z\-\/]+)") or grab(r"DELIVERY\s+([0-9A-Z\-\/]+)")
    box_no = grab(r"BOX\s*NO\s*[:\-]?\s*([0-9A-Z\-\/]+)")
    dealer = grab(r"DEALER\s*[:\-]?\s*([A-Z0-9 \-]+)")
    city = grab(r"CITY\s*[:\-]?\s*([A-Z0-9 \-]+)")
    trpt = grab(r"TRPT\s*[:\-]?\s*([A-Z0-9 \-]+)")
    date = grab(r"DATE\s*[:\-]?\s*([0-9\/\.\-]+)")

    # Sometimes there’s a prominent numeric like 057953 etc (wave/box label)
    wave = grab(r"\bWAVE\s*[:\-]?\s*([0-9]+)")

    return {
        "Delivery No": delivery,
        "Box No": box_no,
        "Dealer": dealer,
        "City": city,
        "Transporter": trpt or ("SAFEEXPRESS" if "SAFEEXPRESS" in t else None),
        "Order Date": date,
        "Wave": wave,
    }

def extract_fields(label_type: str, text: str):
    t = (text or "").upper()
    if label_type == "ORDER":
        return extract_order_label(text)

    # ITEM: choose best template
    if "VEGA" in t or "HELMET" in t or "MODEL" in t and "STYLE" in t and "MRP" in t:
        return extract_vega_item(text)

    if "BAJAJ" in t or "COMMODITY" in t or "UNIT SALE PRICE" in t:
        return extract_bajaj_item(text)

    # fallback: keep raw text only
    return {"Raw Text": text[:500]}

# ----------------------------
# Session state: pending scans + rows
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
        st.success("✅ Order + Item merged into one row and added to table.")

# ----------------------------
# UI: live camera
# ----------------------------
colA, colB = st.columns([2, 1])

with colA:
    st.info("Tip: Fill the frame with the label. Keep it flat, good lighting, steady focus. Then press **Scan**.")
    webrtc_ctx = webrtc_streamer(
        key="scanner",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=FrameGrabber,
        media_stream_constraints={"video": {"facingMode": "environment"}, "audio": False},
        async_processing=True,
    )

with colB:
    st.subheader("Scan Controls")

    label_override = st.selectbox(
        "Label type (auto-detect available)",
        ["AUTO", "ORDER", "ITEM"],
        index=0
    )

    scan_btn = st.button("✅ Scan current frame", use_container_width=True)

    st.markdown("### Pending (2-scan merge)")
    st.write("**Order label:**", "✅ captured" if st.session_state.pending_order else "—")
    st.write("**Item label:**", "✅ captured" if st.session_state.pending_item else "—")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("↩️ Reset pending", use_container_width=True):
            st.session_state.pending_order = None
            st.session_state.pending_item = None
            st.toast("Pending cleared.")
    with c2:
        if st.button("🧹 Clear all rows", use_container_width=True):
            st.session_state.rows = []
            st.toast("All rows cleared.")

# ----------------------------
# On Scan
# ----------------------------
if scan_btn:
    if not webrtc_ctx.video_processor:
        st.error("Camera not ready. Allow camera access and try again.")
        st.stop()

    frame_bgr = webrtc_ctx.video_processor.latest_bgr
    if frame_bgr is None:
        st.error("No frame captured yet. Wait 1–2 seconds and press Scan again.")
        st.stop()

    barcodes = read_barcodes(frame_bgr)
    ocr_img = preprocess_for_ocr(frame_bgr)
    text = run_ocr(ocr_img)

    # Determine label type
    if label_override == "AUTO":
        label_type = guess_label_type(text, barcodes)
    else:
        label_type = label_override

    fields = extract_fields(label_type, text)

    # Attach barcode payloads (useful)
    if barcodes:
        fields["Barcodes"] = " | ".join([f"{b['type']}:{b['data']}" for b in barcodes])
    else:
        fields["Barcodes"] = None

    # Keep raw OCR for debugging
    fields["Raw OCR"] = (text or "")[:800]

    # Store into pending
    if label_type == "ORDER":
        st.session_state.pending_order = fields
        st.success("✅ Captured ORDER label.")
    else:
        st.session_state.pending_item = fields
        st.success("✅ Captured ITEM label.")

    merge_if_ready()

# ----------------------------
# Display current state
# ----------------------------
st.divider()
st.subheader("📋 Merged Rows (Order + Item)")

if st.session_state.rows:
    df = pd.DataFrame(st.session_state.rows)
    st.dataframe(df, use_container_width=True)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download CSV", csv, "warehouse_scans.csv", "text/csv")
else:
    st.warning("No merged rows yet. Scan ORDER label then ITEM label (or vice versa).")