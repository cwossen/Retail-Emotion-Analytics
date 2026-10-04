"""Confirm (or refute) the talking/open-mouth explanation for the video result.

The pipeline is exonerated: on RAF-DB stills the exact same crop path and model
score 0.897, identical to feeding aligned crops directly, with Happiness ->
Happiness at 94.9% and no leakage into Fear. Yet the same code on a video of a
person saying hello returned 62% Fear and 25% Surprise.

Hypothesis: the subject is TALKING. RAF-DB is posed and largely closed-mouth,
so a mid-speech open mouth carries the visual signature of surprise/fear.

This script tests that directly. It samples frames from the clip, groups them
by DAN's prediction, and writes contact-sheet PNGs - one per predicted emotion -
so you can look at what the model called "Fear" and see whether those frames
are open-mouthed speech.

It also reports, per predicted emotion, the mean mouth-openness of the frames
in that group, using the mouth landmarks YOLOv8-face provides when available.
That turns "they look open-mouthed to me" into a number.

If the Fear/Surprise groups are visibly (and numerically) the open-mouth frames
while Happiness/Neutral are the closed-mouth ones, the mechanism is proven.

Read-only. Writes results/speech_check/*.png and a per-frame CSV.
"""
import os
import sys
import csv
import argparse

import numpy as np
import cv2

EMOTION_SR = os.path.expanduser("~/datasets/emotion_sr")
sys.path.insert(0, EMOTION_SR)

FACE_MODEL = "/mnt/c/Users/cwoss/OneDrive/Documents/emotion_app/models/yolov8n-face-lindevs.pt"
RESULTS_DIR = os.path.join(EMOTION_SR, "results")
OUT_DIR = os.path.join(RESULTS_DIR, "speech_check")
os.makedirs(OUT_DIR, exist_ok=True)

LABELS = ('Surprise', 'Fear', 'Disgust', 'Happiness', 'Sadness', 'Anger', 'Neutral')
FACE_CONF_MIN = 0.60
FACE_PAD = 0.15
CLASSIFIER_INPUT = 224
THUMB = 150          # contact-sheet cell size
COLS = 6


def detect_best_face(yolo, bgr):
    """Return (box, keypoints or None). Keypoints, when present, are
    5-point: right eye, left eye, nose, right mouth corner, left mouth corner."""
    res = yolo.predict(bgr, verbose=False)
    r = res[0]
    boxes = r.boxes
    if boxes is None or len(boxes) == 0:
        return None, None
    best_i, best_dim = None, -1
    for i, b in enumerate(boxes):
        if float(b.conf[0]) < FACE_CONF_MIN:
            continue
        x1, y1, x2, y2 = map(float, b.xyxy[0].tolist())
        dim = min(x2 - x1, y2 - y1)
        if dim > best_dim:
            best_dim, best_i = dim, i
    if best_i is None:
        return None, None
    box = tuple(map(int, boxes[best_i].xyxy[0].tolist()))

    kps = None
    if getattr(r, "keypoints", None) is not None:
        try:
            k = r.keypoints.xy[best_i].cpu().numpy()
            if k.shape[0] >= 5:
                kps = k
        except Exception:
            kps = None
    return box, kps


def mouth_openness(kps, box):
    """Crude but serviceable: vertical gap between the mouth-corner line and
    the nose, normalised by face height. Higher = mouth lower/more open.
    Returns None if landmarks are unavailable."""
    if kps is None:
        return None
    try:
        nose = kps[2]
        mouth_r, mouth_l = kps[3], kps[4]
        mouth_mid_y = (mouth_r[1] + mouth_l[1]) / 2.0
        face_h = max(1.0, box[3] - box[1])
        return float((mouth_mid_y - nose[1]) / face_h)
    except Exception:
        return None


def crop_face(bgr, box):
    H, W = bgr.shape[:2]
    x1, y1, x2, y2 = box
    pw, ph = int((x2 - x1) * FACE_PAD), int((y2 - y1) * FACE_PAD)
    x1, y1 = max(0, x1 - pw), max(0, y1 - ph)
    x2, y2 = min(W, x2 + pw), min(H, y2 + ph)
    c = bgr[y1:y2, x1:x2]
    return c if c.size else None


def contact_sheet(crops, caption):
    if not crops:
        return None
    rows = (len(crops) + COLS - 1) // COLS
    sheet = np.full((rows * THUMB + 34, COLS * THUMB, 3), 28, dtype=np.uint8)
    cv2.putText(sheet, caption, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (240, 240, 240), 1, cv2.LINE_AA)
    for i, c in enumerate(crops):
        r, col = divmod(i, COLS)
        t = cv2.resize(c, (THUMB, THUMB), interpolation=cv2.INTER_AREA)
        y0 = 34 + r * THUMB
        x0 = col * THUMB
        sheet[y0:y0 + THUMB, x0:x0 + THUMB] = t
    return sheet


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--fps", type=float, default=3.0)
    p.add_argument("--max-per-class", type=int, default=24)
    p.add_argument("--classifier", default="DAN", choices=["DAN", "POSTER++"])
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
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit("could not open video")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(src_fps / args.fps)))

    by_class = {l: [] for l in LABELS}
    open_by_class = {l: [] for l in LABELS}
    rows = []
    fi = 0
    have_landmarks = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if fi % step != 0:
            fi += 1
            continue

        box, kps = detect_best_face(yolo, frame)
        if box is None:
            fi += 1
            continue
        crop = crop_face(frame, box)
        if crop is None:
            fi += 1
            continue

        rgb = cv2.cvtColor(
            cv2.resize(crop, (CLASSIFIER_INPUT, CLASSIFIER_INPUT),
                       interpolation=cv2.INTER_CUBIC),
            cv2.COLOR_BGR2RGB)
        pred, probs = classify(rgb)
        emo = LABELS[int(pred)]
        conf = float(np.max(probs))

        openness = mouth_openness(kps, box)
        if openness is not None:
            have_landmarks += 1
            open_by_class[emo].append(openness)

        if len(by_class[emo]) < args.max_per_class:
            by_class[emo].append(crop)

        rows.append(dict(frame=fi, ts=round(fi / src_fps, 3), emotion=emo,
                         confidence=round(conf, 4),
                         mouth_openness=(round(openness, 4)
                                         if openness is not None else "")))
        fi += 1

    cap.release()

    n = len(rows)
    print(f"\n{n} classified frames from {os.path.basename(args.video)}")
    print("\npredicted emotion distribution:")
    for l in LABELS:
        c = sum(1 for r in rows if r['emotion'] == l)
        if c:
            print(f"  {l:>10} | {c:>4} | {100*c/n:5.1f}%")

    # write contact sheets
    made = []
    for l in LABELS:
        if not by_class[l]:
            continue
        cnt = sum(1 for r in rows if r['emotion'] == l)
        sheet = contact_sheet(by_class[l],
                              f'DAN predicted "{l}"  ({cnt} frames)')
        if sheet is not None:
            path = os.path.join(OUT_DIR, f"predicted_{l}.png")
            cv2.imwrite(path, sheet)
            made.append(path)

    # the numeric half of the test
    print("\nmouth openness (mouth-to-nose gap / face height), by prediction:")
    if have_landmarks == 0:
        print("  landmarks unavailable from this detector build - rely on the")
        print("  contact sheets instead.")
    else:
        for l in LABELS:
            vals = open_by_class[l]
            if len(vals) >= 3:
                print(f"  {l:>10} | n={len(vals):>4} | mean {np.mean(vals):.4f} "
                      f"| median {np.median(vals):.4f}")
        print("\n  If Fear/Surprise show a HIGHER value than Happiness/Neutral,")
        print("  the open-mouth mechanism is confirmed numerically.")

    out_csv = os.path.join(RESULTS_DIR, "speech_check_frames.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["frame", "ts", "emotion",
                                          "confidence", "mouth_openness"])
        w.writeheader(); w.writerows(rows)

    print(f"\nWrote {len(made)} contact sheets to {OUT_DIR}")
    print(f"Wrote {out_csv}")
    print("\nLook at predicted_Fear.png and predicted_Surprise.png:")
    print("  are those frames open-mouthed / mid-speech?")
    print("Then compare with predicted_Happiness.png and predicted_Neutral.png.")


if __name__ == "__main__":
    main()
