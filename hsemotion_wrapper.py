"""HSEmotion integration - an in-the-wild-trained classifier for the comparison.

WHY THIS MODEL
--------------
DAN and POSTER++ are both trained on RAF-DB, which is posed and largely
closed-mouth. Measured consequence: on a video of an ordinary person talking,
DAN classified 62% of frames as Fear and 25% as Surprise, despite scoring 0.897
on RAF-DB stills through the identical crop path. The failure is a domain gap,
not a pipeline bug (proven: crop-path diagnostic showed 0.897 both ways).

HSEmotion (Savchenko et al.) is trained on AffectNet, which is collected
in-the-wild rather than posed. It is also lightweight (MobileNet / EfficientNet
backbones), so it sits at the fast end of the speed/accuracy tradeoff that
matters for a real-time app.

The open question this integration is meant to answer: does an in-the-wild
training set actually survive a talking face, or does the whole category of
static-image FER models fail the same way? Either answer is informative.

INSTALL
-------
    source ~/cv-env/bin/activate
    pip install hsemotion-onnx      # ONNX build: no torch version conflicts
  or
    pip install hsemotion           # PyTorch build

This module tries both, preferring whichever is present.

INTERFACE
---------
load_hsemotion() returns classify(img_224_rgb) -> (pred_idx, probs), matching
load_dan() / load_poster() exactly, so it drops into the existing harness,
run_video.py, and speech_check.py without changes elsewhere.

LABEL ORDER - the important part
--------------------------------
HSEmotion emits its own label order (AffectNet-style, 8 classes including
Contempt). The project's canonical RAF-DB order is:
    0 Surprise, 1 Fear, 2 Disgust, 3 Happiness, 4 Sadness, 5 Anger, 6 Neutral
This module remaps HSEmotion's output into that order and DROPS Contempt
(renormalising the remaining probabilities), so every model in the comparison
speaks the same 7-class language. Scoring against scrambled labels is exactly
the silent error the harness has always guarded against.
"""
import os
import sys
import numpy as np

CANONICAL = ['Surprise', 'Fear', 'Disgust', 'Happiness', 'Sadness', 'Anger', 'Neutral']

# HSEmotion's own label strings, lowercased for matching. Its 8-class models
# use AffectNet's set; the 7-class variants omit Contempt.
_HSE_TO_CANONICAL = {
    'surprise': 'Surprise',
    'surprised': 'Surprise',
    'fear': 'Fear',
    'fearful': 'Fear',
    'disgust': 'Disgust',
    'disgusted': 'Disgust',
    'happiness': 'Happiness',
    'happy': 'Happiness',
    'sadness': 'Sadness',
    'sad': 'Sadness',
    'anger': 'Anger',
    'angry': 'Anger',
    'neutral': 'Neutral',
    'contempt': None,        # deliberately dropped - no RAF-DB equivalent
}


def _build_remap(hse_labels):
    """Map HSEmotion's output vector onto the canonical 7-class order.

    Returns (indices, dropped) where indices[i] is the position in the
    HSEmotion output corresponding to CANONICAL[i], and dropped is the list of
    HSEmotion labels discarded (expected: Contempt)."""
    lower = [str(l).strip().lower() for l in hse_labels]
    indices = []
    for want in CANONICAL:
        pos = None
        for i, l in enumerate(lower):
            if _HSE_TO_CANONICAL.get(l) == want:
                pos = i
                break
        if pos is None:
            raise SystemExit(
                f"ABORT: HSEmotion has no output for canonical class {want!r}.\n"
                f"  its labels: {list(hse_labels)}\n"
                f"  refusing to score against a mismatched label set.")
        indices.append(pos)
    dropped = [l for i, l in enumerate(lower) if i not in indices]
    return indices, dropped


def load_hsemotion(model_name='enet_b0_8_best_afew', verbose=True):
    """Return classify(img_224_rgb) -> (pred_idx, probs) in canonical order.

    model_name options (varies by package version); good defaults:
      'enet_b0_8_best_afew'   - EfficientNet-B0, 8 classes, small and fast
      'enet_b0_8_best_vgaf'   - same size, different training mix
      'enet_b2_8'             - larger, more accurate, slower
    """
    recogniser = None
    backend = None

    try:
        from hsemotion_onnx.facial_emotions import HSEmotionRecognizer
        recogniser = HSEmotionRecognizer(model_name=model_name)
        backend = 'onnx'
    except Exception:
        try:
            from hsemotion.facial_emotions import HSEmotionRecognizer
            recogniser = HSEmotionRecognizer(model_name=model_name)
            backend = 'torch'
        except Exception as e:
            raise SystemExit(
                "Could not import HSEmotion. Install one of:\n"
                "    pip install hsemotion-onnx\n"
                "    pip install hsemotion\n"
                f"  (last error: {e})")

    # Discover the model's label order rather than assuming it.
    hse_labels = getattr(recogniser, 'idx_to_class', None)
    if isinstance(hse_labels, dict):
        hse_labels = [hse_labels[i] for i in sorted(hse_labels)]
    if hse_labels is None:
        raise SystemExit("ABORT: cannot read HSEmotion's label order; refusing "
                         "to guess it.")

    indices, dropped = _build_remap(hse_labels)

    if verbose:
        print(f"[HSEmotion] backend={backend} model={model_name}")
        print(f"[HSEmotion] native labels: {list(hse_labels)}")
        print(f"[HSEmotion] remapped to canonical: {CANONICAL}")
        if dropped:
            print(f"[HSEmotion] dropped (no RAF-DB equivalent): {dropped}")

    def classify(img_224_rgb):
        arr = np.asarray(img_224_rgb)
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
        # HSEmotion takes an RGB face image and does its own resizing.
        _label, scores = recogniser.predict_emotions(arr, logits=False)
        scores = np.asarray(scores, dtype=np.float64).ravel()
        picked = scores[indices]
        total = picked.sum()
        probs = picked / total if total > 0 else np.full(len(CANONICAL),
                                                         1.0 / len(CANONICAL))
        return int(np.argmax(probs)), probs

    return classify


# ---------------------------------------------------------------------
#  Smoke test - run before trusting this in the harness.
#  Verifies the model loads, the remap is sane, and that it produces
#  sensible predictions on RAF-DB faces with known labels.
# ---------------------------------------------------------------------
def _smoke(n=100, seed=42):
    import csv
    import random
    import cv2

    RAFDB_LABELS = os.path.expanduser(
        "~/datasets/RAF-DB/basic/EmoLabel/list_patition_label.txt")
    RAFDB_IMAGES = os.path.expanduser("~/datasets/RAF-DB/basic/Image/aligned")

    classify = load_hsemotion()

    items = []
    with open(RAFDB_LABELS) as f:
        for line in f:
            parts = line.split()
            if len(parts) != 2:
                continue
            stem = os.path.splitext(parts[0])[0]
            if not stem.startswith('test'):
                continue
            items.append((os.path.join(RAFDB_IMAGES, f"{stem}_aligned.jpg"),
                          int(parts[1]) - 1))

    rng = random.Random(seed)
    sample = rng.sample(items, min(n, len(items)))

    correct = 0
    counts = {}
    for path, gt in sample:
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(cv2.resize(bgr, (224, 224),
                                      interpolation=cv2.INTER_CUBIC),
                           cv2.COLOR_BGR2RGB)
        pred, probs = classify(rgb)
        counts[CANONICAL[pred]] = counts.get(CANONICAL[pred], 0) + 1
        correct += int(pred == gt)

    n_run = sum(counts.values())
    acc = correct / n_run if n_run else 0
    print(f"\n=== HSEmotion smoke test on {n_run} RAF-DB test images ===")
    print(f"accuracy: {acc:.3f}")
    print("prediction distribution:")
    for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {k:>10} | {v:>4} | {100*v/n_run:5.1f}%")

    print("\nHow to read this:")
    print("  HSEmotion is AffectNet-trained and tested here on RAF-DB, so this")
    print("  is CROSS-DATABASE accuracy. Expect it to be LOWER than DAN's 0.897")
    print("  in-domain figure - somewhere in the 0.6-0.8 range would be normal")
    print("  and healthy. What would signal a problem is a collapsed")
    print("  distribution (nearly everything one class), which would mean the")
    print("  label remap or preprocessing is wrong.")


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=100)
    p.add_argument('--model', default='enet_b0_8_best_afew')
    args = p.parse_args()
    _smoke(n=args.n)
