# Retail Emotion Analytics

A real-time retail customer engagement analytics system using computer vision, live IP camera streaming, and privacy-preserving face anonymization.

Developed as a master's research project at Sejong University under the supervision of [Prof. Nhu-Ngoc Dao](https://nndao.github.io/).

---

## Overview

The system processes live RTSP camera feeds to detect customer faces, classify emotions, track dwell time by store zone, and generate product engagement recommendations — all without storing any identifiable biometric data.

Detected faces are replaced in real time with a synthetic proxy image before any display or storage, implementing a privacy-by-design architecture aligned with emerging AI governance frameworks.

---

## Architecture

```
IP Cameras (RTSP/H.265)
        │
        ▼
   ffmpeg pipe
        │
        ▼
 YOLOv8 Face Detection
        │
        ▼
 DeepFace Emotion Classification
        │
        ▼
 ByteTrack Dwell Time Tracking
        │
        ▼
 Zone-Based Category Mapping
        │
        ▼
 Signal Weighting & Recommendation Engine
        │
        ▼
 SQLite Persistence ──► Rust Verifier
        │
        ▼
 Streamlit Dashboard (Live Feed + Metrics)
```

---

## Features

- **Dual-camera RTSP streaming** via ffmpeg pipe with H.265/HEVC decoding
- **Face detection** using YOLOv8 (lindevs WIDERFace model)
- **Emotion classification** using DeepFace with CCTV bias calibration
- **Dwell time tracking** using ByteTrack across store zones
- **Privacy anonymization** — synthetic face overlay via seamlessClone, volatile biometric handling, metadata-only persistence
- **Zone-based engagement scoring** with configurable category boundaries
- **Dwell-aware recommendation engine** with signal weighting
- **SQLite session persistence** with multi-session accumulation
- **Rust-based data verifier** for session auditing and reporting
- **Streamlit UI** with live feed display, engagement metrics, and compliance toggle

---

## Stack

| Component | Technology |
|---|---|
| Face Detection | YOLOv8 (ultralytics) |
| Emotion Classification | DeepFace |
| Tracking | ByteTrack |
| Streaming | ffmpeg (H.265/HEVC) |
| UI | Streamlit |
| Database | SQLite |
| Verifier | Rust |
| CV | OpenCV |

---

## Project Structure

```
emotion_app/
├── app.py                  # Streamlit UI and dashboard
├── face_detector.py        # RTSP streaming, detection, anonymization
├── classifier.py           # Emotion-to-signal mapping, recommendations
├── db.py                   # SQLite schema and queries
├── models/
│   └── yolov8n-face-lindevs.pt
├── assets/
│   └── privacy/
│       └── neutral_proxy.jpg
├── rust/
│   └── dataset_verify/     # Rust verifier binary
└── data/
    └── emotions.db
```

---

## Setup

### Requirements

- Python 3.13
- ffmpeg (Windows: `C:\ffmpeg\ffmpeg-8.1-essentials_build\bin\ffmpeg.exe`)
- Rust (for verifier)

### Install dependencies

```bash
pip install streamlit ultralytics deepface opencv-python numpy
```

### Configuration

Copy `.env.example` to `.env` and set your camera credentials:

```
CAMERA_1_URL=rtsp://user:password@192.168.0.x:554/onvif1
CAMERA_2_URL=rtsp://user:password@192.168.0.x:554/onvif1
```

### Run

```bash
streamlit run app.py
```

### Verify data

```bash
rust\dataset_verify\target\release\dataset_verify.exe data\emotions.db
```

---

## Privacy Architecture

- Faces are detected and classified **in memory only**
- The detected face region is **replaced with a synthetic proxy** before any frame is written to disk or displayed
- No raw face images are stored — only emotion labels, dwell times, and zone metadata
- Face embeddings used for revisit detection are stored as numerical vectors, not images
- All biometric processing is **volatile** — embeddings are cleared between sessions

---

## Status

Active development. Current deployment stage: single-retailer pilot testing.

- [x] Dual-camera live stream pipeline
- [x] Face detection and emotion classification
- [x] Dwell time tracking and zone mapping
- [x] Privacy anonymization framework
- [x] SQLite persistence and Rust verifier
- [x] Streamlit live dashboard
- [ ] Zone boundary calibration (pending store deployment)
- [ ] Multi-session revisit tracking
- [ ] Formal accuracy benchmarking

---

## Academic Context

Master of Science in Computer Science — Sejong University, Seoul  
Supervisor: [Prof. Nhu-Ngoc Dao](https://nndao.github.io/)
