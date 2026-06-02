from ultralytics import YOLO
from deepface import DeepFace
import cv2
import numpy as np
import subprocess
import threading
import time
import os
import traceback
from db import initialize_db, create_session, insert_detection, insert_embedding, get_all_embeddings

PROXY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "privacy", "neutral_proxy.jpeg")

class FaceAnonymizer:
    def __init__(self, proxy_path):
        self.proxy_img = cv2.imread(proxy_path)
        if self.proxy_img is None:
            self.proxy_img = np.full((224, 224, 3), 128, dtype=np.uint8)
            print("WARNING: Proxy face image not found. Using gray placeholder.")

    def apply_mask(self, frame, x1, y1, x2, y2):
        """Replaces the face at the given coordinates with the synthetic proxy."""
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0: return frame
        resized_proxy = cv2.resize(self.proxy_img, (w, h))
        mask = 255 * np.ones(resized_proxy.shape, resized_proxy.dtype)
        center = (x1 + w // 2, y1 + h // 2)
        try:
            frame = cv2.seamlessClone(resized_proxy, frame, mask, center, cv2.NORMAL_CLONE)
        except:
            frame[y1:y2, x1:x2] = resized_proxy
        return frame

anonymizer = FaceAnonymizer(PROXY_PATH)

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "models",
    "yolov8n-face-lindevs.pt"
)

FFMPEG_PATH = os.getenv("FFMPEG_PATH", r"C:\ffmpeg\ffmpeg-8.1-essentials_build\bin\ffmpeg.exe")

print("FFMPEG exists:", os.path.exists(FFMPEG_PATH))
print("MODEL_PATH:", MODEL_PATH)
print("Exists:", os.path.exists(MODEL_PATH))

model = YOLO(MODEL_PATH)
model_lock = threading.Lock()

ZONES = {
    "DC":     (0,    320),
    "Marvel": (320,  640),
    "Indie":  (640,  960),
    "Manga":  (960,  1280),
}

def get_category(x_center):
    for category, (x_min, x_max) in ZONES.items():
        if x_min <= x_center < x_max:
            return category
    return "Unknown"

def preprocess_face(face_crop):
    face = cv2.resize(face_crop, (224, 224))
    return face

def get_emotion(face_crop):
    try:
        face = preprocess_face(face_crop)
        result = DeepFace.analyze(face, actions=["emotion"], enforce_detection=False)
        emotions = result[0]["emotion"]
        top_emotion = max(emotions, key=emotions.get)
        return top_emotion
    except:
        return "neutral"

def get_face_embedding(face_crop):
    try:
        result = DeepFace.represent(face_crop, enforce_detection=False)
        return result[0]["embedding"]
    except:
        return None

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def is_revisit(embedding, threshold=0.80):
    if embedding is None:
        return False
    stored = get_all_embeddings()
    for row in stored:
        stored_embedding = row[2]
        sim = cosine_similarity(embedding, stored_embedding)
        if sim > threshold:
            return True
    return False

def open_stream(rtsp_url, width=640, height=360):
    command = [
        FFMPEG_PATH,
        "-rtsp_transport", "udp",
        "-timeout", "10000000",
        "-fflags", "nobuffer",
        "-flags", "low_delay",
        "-i", rtsp_url,
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-vf", f"scale={width}:{height}",
        "pipe:1"
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        bufsize=10**8,
        creationflags=subprocess.CREATE_NO_WINDOW
    )
    time.sleep(15)
    print(f"Stream opened for {rtsp_url}")
    return process

def stop_all_streams():
    global stop_event
    stop_event.set()

def read_frame(process, width=640, height=360):
    expected = width * height * 3
    chunks = []
    bytes_read = 0
    while bytes_read < expected:
        chunk = process.stdout.read(expected - bytes_read)
        if not chunk:
            return None
        chunks.append(chunk)
        bytes_read += len(chunk)
    raw = b"".join(chunks)
    return np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))

def process_stream(rtsp_url, camera_name, session_id, thread_model, width=640, height=360, frame_interval=2, anonymize=True):
    print(f"Starting stream: {camera_name}")
    process = open_stream(rtsp_url, width, height)

    if process is None:
        print(f"{camera_name}: failed to open stream")
        return

    print(f"{camera_name}: stream opened, starting detection")
    os.makedirs("C:\\temp", exist_ok=True)

    face_tracker = {}
    frame_idx = 0
    FPS = 15.0
    empty_frame_count = 0
    MAX_EMPTY_FRAMES = 300

    try:
        while not stop_event.is_set():
            frame = read_frame(process, width, height)

            if frame is None:
                empty_frame_count += 1
                if empty_frame_count > MAX_EMPTY_FRAMES:
                    print(f"{camera_name}: stream ended or lost")
                    break
                continue

            empty_frame_count = 0
            frame_idx += 1

            if frame_idx % frame_interval != 0:
                continue

            try:
                results = []
                with model_lock:
                    results = thread_model.track(frame, persist=True, verbose=False)

                if not results or results[0].boxes is None:
                    continue

                boxes = results[0].boxes
                camera_num = 1 if "1" in camera_name else 2
                ids = boxes.id.tolist() if boxes.id is not None else [camera_num * 10000 for _ in range(len(boxes))]

                for b, track_id in zip(boxes, ids):
                    track_id = int(track_id)
                    conf = float(b.conf[0])
                    x1, y1, x2, y2 = b.xyxy[0].tolist()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    if conf < 0.4:
                        continue

                    face_crop = frame[y1:y2, x1:x2]
                    if face_crop.size == 0:
                        continue

                    emotion = get_emotion(face_crop)
                    x_center = (x1 + x2) // 2
                    category = get_category(x_center)

                    if anonymize:
                        frame = anonymizer.apply_mask(frame, x1, y1, x2, y2)

                    try:
                        cv2.imwrite(f"C:\\temp\\frame_{camera_name}.jpg", frame)
                    except:
                        pass

                    if track_id not in face_tracker:
                        face_tracker[track_id] = {
                            "start": frame_idx,
                            "last": frame_idx,
                            "emotions": [],
                            "categories": [],
                            "crops": []
                        }
                    else:
                        face_tracker[track_id]["last"] = frame_idx

                    face_tracker[track_id]["emotions"].append(emotion)
                    face_tracker[track_id]["categories"].append(category)
                    face_tracker[track_id]["crops"].append(face_crop)

            except Exception as e:
                print(f"{camera_name}: frame processing error: {e}")
                continue

    except Exception as e:
        print(f"ERROR in {camera_name}: {e}")
        traceback.print_exc()
    finally:
        print(f"{camera_name}: finally block reached, face_tracker size={len(face_tracker)}")
        process.terminate()
        from collections import Counter
        for track_id, data in face_tracker.items():
            dwell_seconds = (data["last"] - data["start"]) / FPS
            emotion = Counter(data["emotions"]).most_common(1)[0][0]
            category = Counter(data["categories"]).most_common(1)[0][0]
            embedding = get_face_embedding(data["crops"][0]) if data["crops"] else None
            revisit = is_revisit(embedding)
            insert_detection(session_id, track_id, emotion, category, dwell_seconds, revisit)
            if embedding:
                insert_embedding(session_id, track_id, embedding)
        print(f"{camera_name}: saved {len(face_tracker)} unique faces")

stop_event = threading.Event()
active_threads = []

def run_face_detector(rtsp_urls=None, video_path=None, video_name="live_stream", anonymize=True):
    global stop_event, active_threads
    stop_event.clear()
    active_threads.clear()

    try:
        initialize_db()
        session_id = create_session(video_name, 15.0)
        print(f"Session {session_id} started")

        if rtsp_urls:
            threads = []
            for i, url in enumerate(rtsp_urls):
                t = threading.Thread(
                    target=process_stream,
                    args=(url, f"Camera_{i+1}", session_id, model, 640, 360, 1, anonymize),
                    daemon=False
                )
                active_threads.append(t)
                threads.append(t)
                t.start()

            for t in threads:
                t.join()
            print(f"Session {session_id} complete")

        elif video_path:
            results = model.track(video_path, save=False, tracker="bytetrack.yaml")
            print(f"Total frames processed: {len(results)}")

            face_tracker = {}
            cap = cv2.VideoCapture(video_path)

            for frame_idx, r in enumerate(results):
                if frame_idx % 5 != 0:
                    continue

                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()

                if not ret:
                    continue

                if r.boxes.id is None:
                    continue

                for b, track_id in zip(r.boxes, r.boxes.id.tolist()):
                    track_id = int(track_id)
                    conf = float(b.conf[0])
                    x1, y1, x2, y2 = b.xyxy[0].tolist()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    if None in (x1, y1, x2, y2):
                        continue

                    if conf > 0.4:
                        face_crop = frame[y1:y2, x1:x2]
                        if face_crop.size == 0:
                            continue

                        emotion = get_emotion(face_crop)
                        x_center = (x1 + x2) // 2
                        category = get_category(x_center)

                        if track_id not in face_tracker:
                            face_tracker[track_id] = {
                                "start": frame_idx,
                                "last": frame_idx,
                                "emotions": [],
                                "categories": [],
                                "crops": []
                            }
                        else:
                            face_tracker[track_id]["last"] = frame_idx

                        face_tracker[track_id]["emotions"].append(emotion)
                        face_tracker[track_id]["categories"].append(category)
                        face_tracker[track_id]["crops"].append(face_crop)

            cap.release()

            from collections import Counter
            for track_id, data in face_tracker.items():
                cap_fps = cv2.VideoCapture(video_path)
                FPS = cap_fps.get(cv2.CAP_PROP_FPS)
                cap_fps.release()

                dwell_seconds = (data["last"] - data["start"]) / FPS
                emotion = Counter(data["emotions"]).most_common(1)[0][0]
                category = Counter(data["categories"]).most_common(1)[0][0]

                embedding = get_face_embedding(data["crops"][0]) if data["crops"] else None
                revisit = is_revisit(embedding)

                insert_detection(session_id, track_id, emotion, category, dwell_seconds, revisit)
                if embedding:
                    insert_embedding(session_id, track_id, embedding)

            print(f"Session {session_id} saved with {len(face_tracker)} unique faces")

        print(f"Session {session_id} complete")

    except Exception as e:
        print(f"ERROR in run_face_detector: {e}")
        traceback.print_exc()
