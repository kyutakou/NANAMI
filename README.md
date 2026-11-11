# NANAMI 🌸
### classroom computer‑vision that looks, but doesn’t *know* you

NANAMI is a **local‑only classroom CV + human‑verified incident review** project.
It detects people and tracks movement to help a human reviewer notice possible incidents —
**without identifying anyone**. No face recognition, no identity inference, no cloud upload.

> “A quiet little witness, not a judge.”

---

## ✨ What it is

- 🎥 Runs on **consented classroom footage only**
- 🧠 Detects people with **YOLOX‑Tiny ONNX** on CPU
- 👣 Tracks them with a **ByteTrack‑style tracker** (own implementation)
- 🧍 Will add **RTMPose‑t ONNX** pose estimation (own wrapper, no `rtmlib`)
- 📊 Will surface short, reviewable clips for a human to verify
- 🔒 **Local first** — no footage leaves the machine by default

## 🚫 What it is *not*

- ❌ Not a face‑recognition system
- ❌ Not an identity database
- ❌ Not an automatic punishment machine
- ❌ Not cloud‑dependent
- ❌ Not trained on your classroom’s private data

Every alert is a **suggestion for a human to review**, never an automatic conclusion.

---

## 🧰 Current stack

| Layer | Choice |
|---|---|
| Language | Python 3.11 |
| Env | `uv`‑built `.venv` |
| Detection | YOLOX‑Tiny ONNX (`onnxruntime`, CPU only) |
| Tracking | ByteTrack‑style, own code |
| Pose (next) | RTMPose‑t ONNX, own wrapper |
| CV utilities | `opencv-python-headless`, NumPy, SciPy |
| Tests | `pytest` |
| Frontend | Vite + React + TypeScript (starter) |
| OS | Windows 10 |

---

## 🗺️ Project status

| Stage | Status |
|---|---|
| Project skeleton + docs | ✅ done |
| YOLOX‑Tiny ONNX + model card | ✅ done |
| `Detector` interface + `YoloxDetector` | ✅ 5 tests passing |
| `try_detector.py` smoke test | ✅ ~76 ms/frame, 1 person @ 0.89 |
| ByteTrack‑style tracker | 🧪 written, tests not yet run |
| Pose (RTMPose‑t) | ⏳ next |
| Analysis layer | ⏳ planned |
| Backend | ⏳ planned |
| Dashboard | ⏳ planned |

---

## 🚀 Quick start (for future me)

```bash
# 1. clone
git clone https://github.com/kyutakou/NANAMI.git
cd NANAMI

# 2. create the uv environment
uv venv .venv
source .venv/Scripts/activate   # Windows Git Bash
# or: .venv\Scripts\activate   # cmd / PowerShell

# 3. install
uv pip install -e ".[dev]"

# 4. run the detector smoke test
python scripts/try_detector.py --image path/to/consented_photo.jpg
