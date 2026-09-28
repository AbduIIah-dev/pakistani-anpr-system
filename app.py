import streamlit as st
import cv2
import tempfile
import os
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
from processor import ANPRProcessor

# 1. Page Configuration
st.set_page_config(
    page_title="VisionANPR - AI Portal",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 2. Modern Glassmorphic Dark SaaS UI CSS
CUSTOM_CSS = """
<style>
    /* Hide Default Streamlit Menu & Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Global App Background */
    .stApp {
        background-color: #0b0f19;
        color: #f1f5f9;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }

    /* Top Horizontal Navigation Container */
    .top-nav-container {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 10px 20px;
        margin-bottom: 24px;
    }

    /* ========================================================= */
    /* HIDE RADIO DOTS & TURN RADIO OPTIONS INTO INTERACTIVE BUTTONS */
    /* ========================================================= */
    div[data-testid="stRadio"] label > div:first-child {
        display: none !important; /* Hides radio circle completely */
    }

    div[data-testid="stRadio"] div[role="radiogroup"] {
        gap: 8px !important;
        display: flex !important;
        flex-direction: row !important;
        align-items: center !important;
    }

    div[data-testid="stRadio"] label {
        background: #0f172a !important;
        border: 1px solid #334155 !important;
        padding: 8px 18px !important;
        border-radius: 8px !important;
        margin: 0 !important;
        cursor: pointer !important;
        transition: all 0.25s ease-in-out !important;
        font-weight: 500 !important;
    }

    div[data-testid="stRadio"] label:hover {
        border-color: #38bdf8 !important;
        background: #1e293b !important;
        transform: translateY(-1px);
    }

    /* Active Selected Nav Button */
    div[data-testid="stRadio"] label:has(input:checked),
    div[data-testid="stRadio"] label[aria-checked="true"] {
        background: linear-gradient(135deg, #0284c7 0%, #2563eb 100%) !important;
        border-color: #38bdf8 !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 14px rgba(2, 132, 199, 0.4) !important;
    }

    div[data-testid="stRadio"] label p {
        color: inherit !important;
        font-size: 0.95rem !important;
    }
    /* ========================================================= */

    /* Hero Banner Section */
    .hero-container {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 32px;
        text-align: center;
        margin-bottom: 28px;
        box-shadow: 0 10px 30px -5px rgba(0, 0, 0, 0.5);
    }
    
    .hero-title {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
    }
    
    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.1rem;
        max-width: 800px;
        margin: 0 auto;
        line-height: 1.6;
    }

    /* Interactive Action Cards */
    .action-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 14px;
        padding: 24px;
        text-align: center;
        transition: transform 0.2s ease, border-color 0.2s ease;
        height: 100%;
    }
    
    .action-card:hover {
        border-color: #38bdf8;
        transform: translateY(-4px);
    }

    /* Number Plate Badge Display */
    .plate-badge {
        background: #0284c7;
        color: #ffffff;
        font-family: 'Courier New', Courier, monospace;
        font-weight: 800;
        padding: 6px 16px;
        border-radius: 8px;
        font-size: 1.3rem;
        letter-spacing: 2px;
        display: inline-block;
        box-shadow: 0 4px 12px rgba(2, 132, 199, 0.3);
    }

    /* Metric Summary Boxes */
    .metric-box {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }
    
    .status-badge-new {
        background-color: #10b981;
        color: white;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 600;
    }

    .status-badge-exist {
        background-color: #f59e0b;
        color: white;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 600;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# 3. Cache Engine Initialization
@st.cache_resource
def get_processor():
    return ANPRProcessor()

processor = get_processor()

# Session State Initializations
if "active_page" not in st.session_state:
    st.session_state.active_page = "🏠 Home"

if "detection_cards" not in st.session_state:
    st.session_state.detection_cards = []

# Helper: Fetch Records Directly from SQLite DB
# Helper: Fetch Records Directly from Database Handler
# Helper: Fetch Records Directly from Database Handler (Without Caching)
def fetch_all_db_records():
    try:
        rows = processor.db.fetch_all_plates()
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows, columns=['id', 'plate_number', 'image_path', 'confidence', 'timestamp'])
        return df
    except Exception as e:
        return pd.DataFrame()

# 4. TOP WEBSITE NAVBAR
nav_col1, nav_col2, nav_col3 = st.columns([1.5, 3.2, 1])

with nav_col1:
    st.markdown("### ⚡ **VisionANPR** Portal")

with nav_col2:
    selected_nav = st.radio(
        "Navbar Navigation",
        ["🏠 Home", "🎥 Video ANPR", "🖼️ Photo ANPR", "🗄️ Saved Database", "📘 Project Overview"],
        index=["🏠 Home", "🎥 Video ANPR", "🖼️ Photo ANPR", "🗄️ Saved Database", "📘 Project Overview"].index(st.session_state.active_page),
        horizontal=True,
        label_visibility="collapsed"
    )
    if selected_nav != st.session_state.active_page:
        st.session_state.active_page = selected_nav
        st.rerun()

with nav_col3:
    if st.button("🧹 Clear Live Memory", use_container_width=True):
        processor.reset_session()
        st.session_state.detection_cards = []
        st.toast("Live session cleared!", icon="✨")

st.markdown("---")


# ==========================================
# PAGE 1: INTERACTIVE LANDING DASHBOARD
# ==========================================
if st.session_state.active_page == "🏠 Home":
    
    # Hero Title & Project Wording
    st.markdown("""
    <div class="hero-container">
        <div class="hero-title">Automated License Plate Recognition (ANPR) System</div>
        <div class="hero-subtitle">
            Welcome to my AI Computer Vision project. This system is engineered to perform high-speed vehicle tracking, dynamic license plate localization, character extraction, and persistent database logging in real time.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Key Metrics Bar
    df_db = fetch_all_db_records()
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"<div class='metric-box'><p style='color:#94a3b8; margin:0;'>Saved Plates in DB</p><h3>{len(df_db)}</h3></div>", unsafe_allow_html=True)
    with m2:
        st.markdown(f"<div class='metric-box'><p style='color:#94a3b8; margin:0;'>Active Session Detections</p><h3>{len(st.session_state.detection_cards)}</h3></div>", unsafe_allow_html=True)
    with m3:
        st.markdown("<div class='metric-box'><p style='color:#94a3b8; margin:0;'>Detection Engine</p><h3>YOLOv11 Custom</h3></div>", unsafe_allow_html=True)
    with m4:
        st.markdown("<div class='metric-box'><p style='color:#94a3b8; margin:0;'>OCR Architecture</p><h3>PaddleOCR v2</h3></div>", unsafe_allow_html=True)

    st.markdown("<br><h3 style='text-align: center;'>Explore Portal Modules</h3><br>", unsafe_allow_html=True)

    # 4 Interactive Visual Feature Cards
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown("""
        <div class="action-card">
            <h3>🎥 Video ANPR</h3>
            <p style='color:#94a3b8; font-size: 0.9rem;'>Process live surveillance or traffic video feeds with frame stabilization.</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Open Video ANPR", use_container_width=True, key="btn_vid"):
            st.session_state.active_page = "🎥 Video ANPR"
            st.rerun()

    with c2:
        st.markdown("""
        <div class="action-card">
            <h3>🖼️ Photo ANPR</h3>
            <p style='color:#94a3b8; font-size: 0.9rem;'>Upload single high-resolution vehicle photos for instant plate extraction.</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Open Photo ANPR", use_container_width=True, key="btn_img"):
            st.session_state.active_page = "🖼️ Photo ANPR"
            st.rerun()

    with c3:
        st.markdown("""
        <div class="action-card">
            <h3>🗄️ Saved Database</h3>
            <p style='color:#94a3b8; font-size: 0.9rem;'>Access SQLite database logs, live search plate numbers, and export CSV reports.</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Open Saved DB", use_container_width=True, key="btn_db"):
            st.session_state.active_page = "🗄️ Saved Database"
            st.rerun()

    with c4:
        st.markdown("""
        <div class="action-card">
            <h3>📘 Project Overview</h3>
            <p style='color:#94a3b8; font-size: 0.9rem;'>Detailed technical breakdown of technologies used and challenges solved.</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("View Overview", use_container_width=True, key="btn_overview"):
            st.session_state.active_page = "📘 Project Overview"
            st.rerun()


# ==========================================
# PAGE 2: DEDICATED VIDEO ANPR
# ==========================================
elif st.session_state.active_page == "🎥 Video ANPR":
    st.subheader("🎥 Traffic & Surveillance Video ANPR")
    st.caption("Upload a video stream to run multi-vehicle tracking and automated plate recognition.")

    uploaded_video = st.file_uploader("Upload Traffic Video (.mp4, .avi, .mov)", type=["mp4", "avi", "mov"])

    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        cap = cv2.VideoCapture(tfile.name)

        col_feed, col_cards = st.columns([2.2, 1])

        with col_feed:
            st.markdown("##### 🔴 Live Annotated Processing Feed")
            video_placeholder = st.empty()

        with col_cards:
            st.markdown("##### 🚗 Detected License Plates")
            cards_placeholder = st.empty()

        def render_cards():
            with cards_placeholder.container():
                for det in reversed(st.session_state.detection_cards):
                    badge_class = "status-badge-new" if det['status'] == "New Entry Saved" else "status-badge-exist"
                    st.markdown(f"""
                    <div style="background:#1e293b; border:1px solid #334155; border-radius:10px; padding:14px; margin-bottom:10px;">
                        <div class="plate-badge">{det['plate_number']}</div>
                        <p style="margin-top:8px; margin-bottom:4px;">
                            <strong>DB Status:</strong> <span class="{badge_class}">{det['status']}</span>
                        </p>
                        <p style="margin-bottom:0px;">
                            <strong>Confidence:</strong> {det['confidence']*100:.1f}%
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    if det["crop"] is not None and det["crop"].size > 0:
                        st.image(det["crop"], channels="BGR", use_container_width=True)
                    st.divider()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            annotated_frame, detections = processor.process_frame(frame, is_video=True)
            video_placeholder.image(annotated_frame, channels="BGR", use_container_width=True)

            if detections:
                for det in detections:
                    if not any(c['plate_number'] == det['plate_number'] for c in st.session_state.detection_cards):
                        st.session_state.detection_cards.append(det)

            render_cards()

        cap.release()
        os.remove(tfile.name)
        st.success("Video processing complete!")


# ==========================================
# PAGE 3: DEDICATED PHOTO ANPR
# ==========================================
elif st.session_state.active_page == "🖼️ Photo ANPR":
    st.subheader("🖼️ High-Resolution Photo Recognition")
    st.caption("Upload static vehicle images for high-precision bounding box prediction.")

    uploaded_image = st.file_uploader("Upload Vehicle Photo (.jpg, .jpeg, .png)", type=["jpg", "jpeg", "png"])

    if uploaded_image:
        file_bytes = np.frombuffer(uploaded_image.read(), np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        col1, col2 = st.columns([2, 1])

        annotated_frame, detections = processor.process_frame(frame, is_video=False)

        with col1:
            st.markdown("##### Detected Bounding Box Output")
            st.image(annotated_frame, channels="BGR", width="stretch")

        with col2:
            st.markdown("##### Extracted Plate Details")
            if detections:
                # Har detected plate ke liye loop chalega taaki ek se zyada plates hon toh sab show hon
                for det in detections:
                    badge_class = "status-badge-new" if det['status'] == "New Entry Saved" else "status-badge-exist"
                    st.markdown(f"""
                    <div style="background:#1e293b; border:1px solid #334155; border-radius:10px; padding:16px; margin-bottom:10px;">
                        <div class="plate-badge">{det['plate_number']}</div>
                        <p style="margin-top:10px; margin-bottom:4px;">
                            <strong>DB Status:</strong> <span class="{badge_class}">{det['status']}</span>
                        </p>
                        <p style="margin-bottom:0px;">
                            <strong>YOLO Confidence:</strong> {det['confidence']*100:.1f}%
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    if det["crop"] is not None and det["crop"].size > 0:
                        st.image(det["crop"], channels="BGR", caption=f"Crop: {det['plate_number']}", width="stretch")
                    st.divider()
            else:
                st.warning("No valid license plate pattern detected in this image.")


# ==========================================
# PAGE 4: SAVED DATABASE & RECORDS
# ==========================================
elif st.session_state.active_page == "🗄️ Saved Database":
    st.subheader("🗄️ Saved License Plates Database")
    st.caption("All recognized license plates are stored directly in SQLite DB with timestamps and confidence scores.")

    df_records = fetch_all_db_records()

    if not df_records.empty:
        c_search, c_export, c_clear = st.columns([2.5, 1, 1])
        with c_search:
            search_query = st.text_input("🔍 Live Search by Plate Number:", "").strip().upper()
        with c_export:
            st.markdown("<br>", unsafe_allow_html=True)
            csv_data = df_records.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Export CSV",
                data=csv_data,
                file_name=f"ANPR_Database_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                width="stretch"
            )
        with c_clear:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🗑️ Clear DB", width="stretch"):
                processor.db.clear_database()
                st.toast("Database cleared successfully!", icon="⚠️")
                st.rerun()

        filtered_df = df_records
        if search_query:
            filtered_df = df_records[df_records['plate_number'].str.contains(search_query, na=False)]

        st.markdown(f"<p style='color:#94a3b8;'>Displaying <b>{len(filtered_df)}</b> records from SQLite database:</p>", unsafe_allow_html=True)

        for idx, row in filtered_df.iterrows():
            with st.container():
                col_img, col_info = st.columns([1, 4])
                with col_img:
                    if os.path.exists(str(row['image_path'])):
                        st.image(row['image_path'], width="stretch")
                    else:
                        st.info("Image not found")
                with col_info:
                    st.markdown(f"""
                    <div style="background:#1e293b; border:1px solid #334155; border-radius:10px; padding:16px; margin-bottom:12px;">
                        <div class="plate-badge">{row['plate_number']}</div>
                        <p style="margin-top:10px; margin-bottom:4px;">
                            <b>Confidence:</b> {float(row['confidence'])*100:.1f}% &nbsp;|&nbsp; 
                            <b>Timestamp:</b> {row['timestamp']}
                        </p>
                        <p style="color:#94a3b8; font-size:0.85rem; margin-bottom:0px;">
                            Saved Image Path: <code>{row['image_path']}</code>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
    else:
        st.info("No records currently found in database. Process images or videos to populate data.")


# ==========================================
# PAGE 5: PROFESSIONAL PROJECT OVERVIEW & SPECS
# ==========================================
elif st.session_state.active_page == "📘 Project Overview":
    st.subheader("📘 Professional Project Documentation & Technical Overview")
    
    st.markdown("""
    ---
    ### 📌 Project Wording & Description
    > **"This is my Automated License Plate Recognition (ANPR) project, in which advanced Deep Learning, Computer Vision algorithms, and dynamic OCR heuristics are combined to build a robust, production-grade vehicle monitoring system."**

    ---
    ### 🛠️ Key Technologies Used
    - **YOLOv11 (Ultralytics):** Fine-tuned deep learning model for real-time license plate detection.
    - **ByteTrack Algorithm:** Multi-object tracking framework to maintain persistent vehicle IDs across video frames.
    - **PaddleOCR Engine (v2.x):** Fast, accurate text extraction with custom image normalization heuristics.
    - **SQLite3 Database:** Light-weight relational database engine for storing detected plates, timestamps, and cropped image paths.
    - **OpenCV & NumPy:** High-performance spatial transformations, Lanczos-4 upscaling, and image filtering.
    - **Streamlit (Custom CSS):** Modern interactive web interface with glassmorphism visual layout.

    ---
    ### ⚡ Key Technical Challenges Solved

    #### 1. Character Misclassification ('M' vs 'H' Confusion)
    * **The Difficulty:** Aggressive binarization/thresholding degraded thin diagonal strokes in letters like **'M'**, causing OCR engines to misidentify them as **'H'** or **'N'** (e.g. predicting `H12` instead of `MH12`).
    * **The Engineering Fix:** Replaced harsh binarization with **Lanczos-4 High-Quality Interpolation**, **Min-Max Contrast Normalization**, and a **Fuzzy State-Code Matcher** to reconstruct valid prefix patterns.

    #### 2. Frame Edge Clipping & Sliced Plates
    * **The Difficulty:** Fast-moving vehicles entering or exiting video borders resulted in cut-off bounding boxes (e.g. losing the first or last character).
    * **The Engineering Fix:** Implemented **Dynamic 10% Bounding Box Crop Padding** and reduced edge rejection threshold to `margin=5`.

    #### 3. Temporal Frame Stabilization & Weighted Voting
    * **The Difficulty:** Single-frame lighting variations or camera blur caused fluctuating OCR predictions in consecutive frames.
    * **The Engineering Fix:** Engineered a **3-frame quality-weighted buffer** ($Quality = Area \times Sharpness$). The final string sequence is resolved using quality-weighted character voting before committing to SQLite DB.
    """, unsafe_allow_html=True)