import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from collections import defaultdict
from classifier import process_detection, generate_recommendation
from face_detector import run_face_detector
from face_detector import stop_all_streams, active_threads
import random
import subprocess
import csv
from db import get_all_detections, get_category_dwell
import time
import threading
import cv2
import numpy as np
from PIL import Image

st.set_page_config(layout="wide")

# ------------------------
# PRIVACY CONFIGURATION
# ------------------------
st.sidebar.header("2026 Compliance Settings")
anonymize_live = st.sidebar.checkbox("Enable Synthetic Face Mask", value=True, help="Replaces real faces with a non-existent AI proxy.")

# ------------------------
# STATE / DATA STRUCTURES
# ------------------------

def run_verifier():
    verifier_path = r"rust\dataset_verify\target\release\dataset_verify.exe"
    db_path = r"data\emotions.db"

    print("Verifier exists:", os.path.exists(verifier_path))
    print("DB exists:", os.path.exists(db_path))

    result = subprocess.run(
        [verifier_path, db_path],
        capture_output=True,
        text=True
    )
    return result.stdout

if "signal_counts" not in st.session_state:
    st.session_state.signal_counts = defaultdict(int)

if "category_scores" not in st.session_state:
    st.session_state.category_scores = defaultdict(int)

if "category_dwell" not in st.session_state:
    st.session_state.category_dwell = {}


# ------------------------
# SIMULATION (Temporary)
# ------------------------

CATEGORIES = ["DC", "Marvel", "Indie", "Manga"]
EMOTIONS = ["happy", "neutral", "surprise"]

def simulate_customer():
    emotion = random.choices(
        EMOTIONS,
        weights=[0.3, 0.5, 0.2]
    )[0]

    duration = random.uniform(1, 6)
    category = random.choice(CATEGORIES)
    revisit = random.random() < 0.2

    process_detection(emotion, duration, category,
              st.session_state.signal_counts,
              st.session_state.category_scores,
              revisit=revisit)
    
    return emotion, duration, category, revisit


# ------------------------
# VIDEO PIPELINE
# ------------------------

def process_video(uploaded_file):
    import tempfile
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")

    try:
        tfile.write(uploaded_file.read())
        tfile.flush()
        tfile.close()

        run_face_detector(video_path=tfile.name, video_name=uploaded_file.name)

        verifier_output = run_verifier()
        print(verifier_output)

        for emotion, category, duration, revisit in get_all_detections():
            process_detection(
                emotion,
                duration,
                category,
                st.session_state.signal_counts,
                st.session_state.category_scores,
                revisit=bool(revisit)
            )

        st.session_state.category_dwell = get_category_dwell()

    finally:
        os.unlink(tfile.name)


# ------------------------
# STREAMLIT UI
# ------------------------

st.title("Retail Emotion Analytics")

col1, col2 = st.columns(2)

with col1:
    if st.button("Run Analysis", key="run_btn"):
        with st.spinner("Analyzing customer behavior..."):
            num_customers = random.randint(5, 20)
            for _ in range(num_customers):
                simulate_customer()
        st.success(f"Analysis complete! Processed {num_customers} customers.")

with col2:
    if st.button("Reset Data", key="reset_btn"):
        st.session_state.signal_counts.clear()
        st.session_state.category_scores.clear()
        st.success("Data reset!")

uploaded_file = st.file_uploader("Upload a test video", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    st.video(uploaded_file)
    if st.button("Analyze Video"):
        with st.spinner("Processing video..."):
            process_video(uploaded_file)
        st.success("Video processed!")

# ------------------------
# LIVE STREAM DASHBOARD
# ------------------------

st.subheader("Live Analysis")

if "streaming" not in st.session_state:
    st.session_state.streaming = False
if "stream_start" not in st.session_state:
    st.session_state.stream_start = None

col3, col4 = st.columns(2)

with col3:
    if st.button("▶ Start", key="start_btn", disabled=st.session_state.streaming):
        st.session_state.streaming = True
        st.session_state.stream_start = time.time()
        st.session_state.signal_counts.clear()
        st.session_state.category_scores.clear()
        st.session_state.category_dwell = {}

        import sqlite3
        conn = sqlite3.connect("data/emotions.db")
        conn.execute("DELETE FROM detections")
        conn.execute("DELETE FROM sessions")
        conn.execute("DELETE FROM face_embeddings")
        conn.commit()
        conn.close()

        def stream_worker(should_anonymize):
            run_face_detector(
                rtsp_urls=[
                    os.getenv("CAMERA_1_URL", "rtsp://user:password@192.168.0.x:554/onvif1"),
                    os.getenv("CAMERA_2_URL", "rtsp://user:password@192.168.0.x:554/onvif1")
                ],
                video_name="FCBD_2026",
                anonymize=should_anonymize
            )
            st.session_state.streaming = False

        t = threading.Thread(target=stream_worker, args=(anonymize_live,), daemon=True)
        t.start()
        st.rerun()

with col4:
    if st.button("⏹ Stop", key="stop_btn", disabled=not st.session_state.streaming):
        stop_all_streams()
        for t in active_threads:
            t.join(timeout=10)
        st.session_state.streaming = False
        st.rerun()

# Show live dashboard while streaming
if st.session_state.streaming or st.session_state.stream_start:

    if st.session_state.stream_start:
        elapsed = int(time.time() - st.session_state.stream_start)
        mins, secs = divmod(elapsed, 60)
        st.metric("Session Duration", f"{mins:02d}:{secs:02d}")

    for emotion, category, duration, revisit in get_all_detections():
        process_detection(
            emotion,
            duration,
            category,
            st.session_state.signal_counts,
            st.session_state.category_scores,
            revisit=bool(revisit)
        )
    st.session_state.category_dwell = get_category_dwell()

    total_faces = sum(st.session_state.signal_counts.values())
    st.metric("Total Faces Detected", total_faces)

    st.subheader("Category Engagement")
    for cat, score in sorted(st.session_state.category_scores.items(),
                              key=lambda x: x[1], reverse=True):
        st.progress(min(score / 100, 1.0), text=f"{cat}: {score}")

    st.subheader("Recommendation")
    st.success(generate_recommendation(
        st.session_state.signal_counts,
        st.session_state.category_scores,
        st.session_state.category_dwell
    ))

    # Live feed display
    st.subheader("Live Feed")
    col_cam1, col_cam2 = st.columns(2)

    for col, cam_name in zip([col_cam1, col_cam2], ["Camera_1", "Camera_2"]):
        fpath = f"C:\\temp\\frame_{cam_name}.jpg"
        if os.path.exists(fpath):
            try:
                img = cv2.imread(fpath)
                if img is not None:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    with col:
                        st.image(img_rgb, caption=cam_name, width='stretch')
            except:
                pass
        else:
            with col:
                st.info(f"{cam_name}: no feed yet")

if st.session_state.streaming:
    time.sleep(2)
    st.rerun()

# ------------------------
# STREAM INPUTS
# ------------------------

st.subheader("Camera Streams")

cam1_url = st.text_input("Camera 1 RTSP URL", value=os.getenv("CAMERA_1_URL", "rtsp://user:password@192.168.0.x:554/onvif1"))
cam2_url = st.text_input("Camera 2 RTSP URL", value=os.getenv("CAMERA_2_URL", "rtsp://user:password@192.168.0.x:554/onvif1"))

if st.button("Start Live Analysis"):
    with st.spinner("Streaming and analyzing..."):
        run_face_detector(
            rtsp_urls=[cam1_url, cam2_url],
            video_name="FCBD_2026"
        )
        verifier_output = run_verifier()
        print(verifier_output)

        for emotion, category, duration, revisit in get_all
