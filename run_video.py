"""Step 2 - run the detection pipeline over a video and drive the TriggerEngine.

Chain:
    video frames
      -> YOLOv8-face detection
      -> simple IoU tracker (stable person_id across frames)
      -> emotion classifier (DAN or POSTER++, already validated)
      -> TriggerEngine (the N-of-M persistence logic)
      -> alert / resolved events

Two outputs:
  1. a per-frame CSV of every classification, so trigger parameters can be
     re-tuned offline in seconds instead of re-running the video
  2. an events CSV of what the TriggerEngine actually fired

Frames are sampled at TARGET_FPS (default 3) rather than every frame: the
trigger was designed at 3 fps, and CPU inference cannot keep up with 30.

Usage:
    python3 run_video.py --video /path/to/clip.mp4
    python3 run_video.py --video clip.mp4 --classifier POSTER++ --annotate
"""
import os
import sys
import csv
import time
import argparse

import numpy as np
import cv2

EMOTION_SR = os.path.expanduser("~/datasets/emotion_sr")
sys.path.insert(0, EMOTION_SR)

from trigger_engine import TriggerEngine, TriggerConfig, LABELS  # noqa: E402

FACE_MODEL = "/mnt/c/Users/cwoss/OneDrive/Documents/emotion_app/models/yolov8n-face-lindevs.pt"
RESULTS_DIR = os.path.join(EMOTION_SR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

TARGET_FPS = 3.0
FACE_CONF_MIN = 0.60      # same threshold that cleaned up the EMOTIC analysis
FACE_PAD = 0.15
CLASSIFIER_INPUT = 224
IOU_MATCH_MIN = 0.30      # tracker: boxes overlapping this much are the same person
TRACK_MAX_MISSES = 5      # drop a track after this many sampled frames unseen


# ----------------------------------------------------------------------
#  A deliberately simple tracker.
#  Customers are continuously visible in a shop camera, so frame-to-frame
#  IoU matching is enough to keep a stable person_id. Full re-identification
#  (the OSNet work) would only be needed to recognise someone who left and
#  came back, which the trigger does not require.
# ----------------------------------------------------------------------
class IoUTracker:
    def __init__(self):
        self.tracks = {}       # tid -> dict(box, misses)
        self._next = 1

    @staticmethod
    def _iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        if inter == 0:
            return 0.0
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        return inter / (area_a + area_b - inter)

    def update(self, boxes):
        """boxes: list of (x1,y1,x2,y2). Returns list of (tid, box)."""
        assigned, used = [], set()
        for box in boxes:
            best_tid, best_iou = None, IOU_MATCH_MIN
            for tid, tr in self.tracks.items():
                if tid in used:
                    continue
                v = self._iou(box, tr['box'])
                if v > best_iou:
                    best_tid, best_iou = tid, v
            if best_tid is None:
                best_tid = f"p{self._next}"
                self._next += 1
                self.tracks[best_tid] = dict(box=box, misses=0)
            else:
                self.tracks[best_tid]['box'] = box
                self.tracks[best_tid]['misses'] = 0
            used.add(best_tid)
            assigned.append((best_tid, box))

        for tid in list(self.tracks):
            if tid not in used:
                self.tracks[tid]['misses'] += 1
                if self.tracks[tid]['misses'] > TRACK_MAX_MISSES:
                    del self.tracks[tid]
        return assigned


def detect_faces(yolo, bgr):
    res = yolo.predict(bgr, verbose=False)
    boxes = res[0].boxes
    out = []
    if boxes is None:
        return out
    for b in boxes:
        if float(b.conf[0]) < FACE_CONF_MIN:
            continue
        x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
        out.append((x1, y1, x2, y2))
    return out


def crop_face(bgr, box):
    H, W = bgr.shape[:2]
    x1, y1, x2, y2 = box
    pw, ph = int((x2 - x1) * FACE_PAD), int((y2 - y1) * FACE_PAD)
    x1, y1 = max(0, x1 - pw), max(0, y1 - ph)
    x2, y2 = min(W, x2 + pw), min(H, y2 + ph)
    crop = bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return crop


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--classifier", default="DAN", choices=["DAN", "POSTER++"])
    p.add_argument("--fps", type=float, default=TARGET_FPS,
                   help="frames per second to sample (trigger was designed at 3)")
    p.add_argument("--annotate", action="store_true",
                   help="also write an annotated output video")
    p.add_argument("--tag", default="run", help="label for the output files")
    args = p.parse_args()

    if not os.path.isfile(args.video):
        raise SystemExit(f"video not found: {args.video}")

    from ultralytics import YOLO
    from emotion_sr_sweep import load_dan, try_load_poster

    if args.classifier == "DAN":
        print("Loading DAN...")
        classify = load_dan()
    else:
        print("Loading POSTER++...")
        classify, err = try_load_poster()
        if classify is None:
            raise SystemExit(f"POSTER++ failed to load: {err}")

    yolo = YOLO(FACE_MODEL)
    tracker = IoUTracker()
    engine = TriggerEngine(TriggerConfig())

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"could not open video: {args.video}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, int(round(src_fps / args.fps)))
    print(f"video: {os.path.basename(args.video)}  {src_fps:.1f} fps, "
          f"{total} frames -> sampling every {step} ({args.fps:.1f} fps)")

    writer = None
    if args.annotate:
        out_path = os.path.join(RESULTS_DIR, f"annotated_{args.tag}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(out_path, fourcc, args.fps, (w, h))

    frame_rows, event_rows = [], []
    fi = 0
    sampled = 0
    t_start = time.time()
    infer_times = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if fi % step != 0:
            fi += 1
            continue

        ts = fi / src_fps          # video timestamp in seconds
        boxes = detect_faces(yolo, frame)
        tracked = tracker.update(boxes)

        for tid, box in tracked:
            crop = crop_face(frame, box)
            if crop is None:
                continue
            rgb = cv2.cvtColor(
                cv2.resize(crop, (CLASSIFIER_INPUT, CLASSIFIER_INPUT),
                           interpolation=cv2.INTER_CUBIC),
                cv2.COLOR_BGR2RGB)
            t0 = time.time()
            pred, probs = classify(rgb)
            infer_times.append(time.time() - t0)
            emotion = LABELS[int(pred)]
            conf = float(np.max(probs))
            face_px = min(box[2] - box[0], box[3] - box[1])

            frame_rows.append(dict(frame=fi, ts=round(ts, 3), person_id=tid,
                                   emotion=emotion, confidence=round(conf, 4),
                                   face_px=face_px,
                                   x1=box[0], y1=box[1], x2=box[2], y2=box[3]))

            for ev in engine.update(tid, emotion, conf, ts):
                event_rows.append(dict(ts=round(ev.ts, 3), kind=ev.kind,
                                       person_id=ev.person_id,
                                       emotion=ev.emotion or "",
                                       reason=ev.reason or "",
                                       held_for_s=ev.held_for_s or ""))
                print(f"  [{ts:6.2f}s] {ev}")

            if writer is not None:
                colour = (36, 165, 245) if any(
                    a['person_id'] == tid for a in engine.active_alerts(ts)
                ) else (139, 189, 91)
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), colour, 2)
                cv2.putText(frame, f"{tid} {emotion} {conf:.2f}",
                            (box[0], max(18, box[1] - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2, cv2.LINE_AA)

        for ev in engine.tick(ts):
            event_rows.append(dict(ts=round(ev.ts, 3), kind=ev.kind,
                                   person_id=ev.person_id, emotion="",
                                   reason=ev.reason or "", held_for_s=ev.held_for_s or ""))
            print(f"  [{ts:6.2f}s] {ev}")

        if writer is not None:
            writer.write(frame)

        sampled += 1
        if sampled % 25 == 0:
            print(f"  ...{sampled} sampled frames ({ts:.1f}s of video)")
        fi += 1

    cap.release()
    if writer is not None:
        writer.release()

    # ---- write outputs ----
    fcsv = os.path.join(RESULTS_DIR, f"video_frames_{args.tag}.csv")
    with open(fcsv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["frame", "ts", "person_id", "emotion",
                                          "confidence", "face_px",
                                          "x1", "y1", "x2", "y2"])
        w.writeheader(); w.writerows(frame_rows)

    ecsv = os.path.join(RESULTS_DIR, f"video_events_{args.tag}.csv")
    with open(ecsv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["ts", "kind", "person_id", "emotion",
                                          "reason", "held_for_s"])
        w.writeheader(); w.writerows(event_rows)

    # ---- summary ----
    wall = time.time() - t_start
    print("\n" + "=" * 60)
    print(f"sampled frames      : {sampled}")
    print(f"face classifications: {len(frame_rows)}")
    print(f"distinct people     : {len({r['person_id'] for r in frame_rows})}")
    if infer_times:
        mean_ms = 1000 * sum(infer_times) / len(infer_times)
        print(f"classifier speed    : {mean_ms:.1f} ms/face "
              f"({1000/mean_ms:.1f} faces/sec) - {args.classifier}")
    print(f"wall clock          : {wall:.1f}s")

    if frame_rows:
        counts = {}
        for r in frame_rows:
            counts[r['emotion']] = counts.get(r['emotion'], 0) + 1
        print("\nemotion distribution:")
        for emo, c in sorted(counts.items(), key=lambda kv: -kv[1]):
            print(f"  {emo:>10} | {c:>5} | {100*c/len(frame_rows):5.1f}%")
        sizes = sorted(r['face_px'] for r in frame_rows)
        n = len(sizes)
        print(f"\nface size (px): min {sizes[0]}  median {sizes[n//2]}  max {sizes[-1]}")

    alerts = [e for e in event_rows if e['kind'] == 'alert']
    print(f"\nALERTS FIRED: {len(alerts)}")
    for a in alerts:
        print(f"  {a['ts']}s  {a['person_id']}  {a['emotion']}")
    if not alerts:
        print("  (none - correct if nobody in the clip shows sustained "
              "anger/disgust/sadness)")

    print(f"\nWrote {fcsv}")
    print(f"Wrote {ecsv}")
    if writer is not None:
        print(f"Wrote {os.path.join(RESULTS_DIR, f'annotated_{args.tag}.mp4')}")


if __name__ == "__main__":
    main()
