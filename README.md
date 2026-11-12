
# NANAMI

**NANAMI** is a local-first computer vision research prototype for human detection, multi-object tracking, and upper-body pose estimation in video.

The project is being developed as a classroom research prototype. Its current focus is building and evaluating a lightweight CPU-based pipeline using independently implemented components and pretrained model checkpoints.

> **Status:** Active research and development. Not a production surveillance system.

## Features

- **Human detection:** YOLOX-Tiny-based person detection.
- **Multi-object tracking:** An independently implemented ByteTrack-style tracker.
- **Pose estimation:** RTMPose-t wrapper for 17-body-keypoint estimation.
- **Video processing:** Detection, tracking, and pose estimation components connected through a processing pipeline.
- **Track summaries:** Utilities for summarizing tracked-object information.
- **CPU inference:** ONNX Runtime CPU execution; no CUDA requirement.

### Current pose-estimation limitation

Initial testing showed unreliable lower-body keypoints in partially occluded scenes, particularly when people were seated behind desks.

For current experiments, NANAMI uses only the upper-body keypoint range (indices 0–10). Lower-body keypoints are excluded from the current analysis rather than treated as reliable measurements.

## Architecture

```text
Input Video
    |
    v
YOLOX-Tiny Detector
    |
    v
Person Detections
    |
    v
ByteTrack-style Tracker
    |
    v
Track IDs
    |
    v
RTMPose-t Pose Estimator
    |
    v
Upper-body Keypoints
    |
    v
Processing Pipeline
    |
    v
Track Summaries / Output
```

The detector, tracker, and pose estimator are separate components. This allows individual components to be tested and improved without requiring changes to the entire pipeline.

## Technology Stack

| Component | Technology |
|---|---|
| Language | Python 3.11 |
| Numerical computing | NumPy, SciPy |
| Image processing | OpenCV |
| Model inference | ONNX Runtime |
| Detection | YOLOX-Tiny |
| Tracking | Custom ByteTrack-style implementation |
| Pose estimation | RTMPose-t |
| Testing | pytest |
| Environment management | uv |
| Frontend prototype | React, TypeScript, Vite |

The frontend currently contains a starter application and is not yet a complete user interface for the computer vision pipeline.

## Repository Structure

```text
NANAMI/
├── backend/
│   ├── ...
│   └── tests/
├── frontend/
│   └── ...
├── models/
│   └── rtmpose_t/
│       └── MODEL_CARD.md
├── scripts/
│   └── run_video.py
├── docs/
│   └── LICENSING.md
├── data/
├── pyproject.toml
├── .gitignore
└── README.md
```

Some directories and files may change as development progresses.

## Requirements

- Windows 10 or newer
- Python 3.11
- `uv`
- A CPU-supported environment
- Git

The current backend dependencies are pinned in `backend/requirements.txt`.

## Setup

Clone the repository:

```powershell
git clone https://github.com/kyutakou/NANAMI.git
cd NANAMI
```

Create a virtual environment:

```powershell
uv venv --python 3.11
```

Activate it:

```powershell
.venv\Scripts\Activate.ps1
```

Install backend dependencies:

```powershell
uv pip install -r backend/requirements.txt
```

Install the development test dependency:

```powershell
uv pip install pytest
```

Model checkpoints are not included in the repository. Follow the corresponding model documentation and licensing records before obtaining or using them.

## Testing

Run the test suite from the repository root:

```powershell
pytest
```

Previously reported test results include:

- The tracker-specific test suite passed 15 tests.
- The pose-estimation wrapper tests were also reported passing.
- A complete end-to-end video-processing test has not yet been confirmed.

Test counts and results should be updated after running the current repository's full test suite.

## Running the Video Pipeline

The video-processing entry point is:

```text
scripts/run_video.py
```

Its command-line interface and required model paths should be checked in the current implementation before running it. The pipeline is still under development, so successful component tests should not be interpreted as proof of complete end-to-end functionality.

## Privacy and Responsible Use

NANAMI is intended for controlled, consent-based research.

The project follows these principles:

- Obtain informed permission from people who appear in recorded footage.
- Explain the purpose of recording and how the footage will be processed.
- Keep recordings and derived data on local storage.
- Do not upload identifiable footage to external services or AI tools.
- Avoid collecting or inferring personal identity.
- Do not use tracking IDs as real-world identities.
- Establish a reasonable retention period and delete recordings when they are no longer needed.
- Restrict access to recordings and generated outputs.

Tracking and pose estimation can produce inaccurate results, especially under occlusion, poor lighting, unusual camera angles, and crowded conditions. Outputs should not be treated as definitive descriptions of a person's actions or identity.

## Limitations

NANAMI is an experimental prototype.

Known limitations include:

- Lower-body pose estimation is unreliable in some partially occluded scenes.
- Tracking performance can degrade when people cross paths or become temporarily obscured.
- CPU inference may limit processing speed.
- End-to-end video pipeline behavior still requires validation.
- The frontend is not yet a completed application.
- Model checkpoint licensing and provenance must be verified independently of the source-code licenses.

## Licensing

NANAMI's own source-code license has not yet been finalized.

Third-party packages, pretrained model checkpoints, and datasets may have separate licensing conditions. Their terms must not be assumed to be identical to the project's eventual license.

See [`docs/LICENSING.md`](docs/LICENSING.md) for the current third-party component and licensing register.

**Do not assume that permission to use the source code automatically grants permission to redistribute or commercially use its pretrained weights.**

## Project Status

| Component | Status |
|---|---|
| YOLOX-Tiny detector | Implemented |
| ByteTrack-style tracker | Implemented; tracker tests passed |
| RTMPose-t wrapper | Implemented; component tests reported passing |
| Upper-body keypoint selection | Implemented |
| Video-processing pipeline | Implemented; end-to-end validation pending |
| Track summaries | Implemented; validation pending |
| Video runner | Implemented; validation pending |
| Frontend | Starter application |
| Full-system validation | Pending |

## Author

Developed by **Coral** as a computer vision research project.

Repository: https://github.com/kyutakou/NANAMI
