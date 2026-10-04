## Model selection: why HSEmotion

The two most common open FER models — DAN and POSTER++ — are trained on
RAF-DB, which is posed and largely closed-mouth. On real, talking faces they
don't degrade gracefully; they collapse, and not even in the same direction.

Same 44-second clip, 132 frames, one person saying hello, three classifiers
through the identical crop path:

| Classifier | Top reading | Speed | Real-time |
|---|---|---|---|
| DAN | Fear 37% + Surprise 27% | 39 ms/face | yes |
| POSTER++ | **Anger 77%** | 1194 ms/face | no (56× slower) |
| HSEmotion | Happiness 77% + Neutral 23% | 21 ms/face | yes (fastest) |

Three models, identical frames, three incompatible answers — and POSTER++
would have fired a false "angry customer" alert on someone greeting the
camera. DAN and POSTER++ share a training set yet fail into *opposite*
classes, which is the tell that the problem is out-of-distribution talking
faces, not one bad model.

HSEmotion (Savchenko et al., AffectNet-trained, in-the-wild) gives the
behaviorally plausible read, and does it fastest. It's the model this system
uses.

Guardrails around the swap: the harness verifies every model's label order at
load (silent label-scrambling is the classic FER integration bug), a
cross-database smoke test gates the model before it's trusted (0.765 on
RAF-DB, a healthy spread rather than a collapse), and low-confidence frames
abstain instead of forcing a label.

*Single-clip illustration; broader multi-subject evaluation in progress.*
