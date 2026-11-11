 
<div align="center">

<img src="https://capsule-render.vercel.app/api?type=venom&color=0:FCE4EC,50:F48FB1,100:EC407A&height=220&section=header&text=NANAMI&fontSize=75&fontColor=FFFFFF&animation=fadeIn&fontAlignY=45&desc=classroom%20computer%20vision%20%7C%20privacy-first%20research&descAlignY=68&descSize=14" width="100%" />

<img src="https://readme-typing-svg.demolab.com?font=JetBrains+Mono&weight=500&size=14&duration=3000&pause=1000&color=EC6C9A&center=true&vCenter=true&width=550&lines=A+quiet+little+witness%2C+not+a+judge.;Computer+vision+with+privacy+in+mind.;Built+with+curiosity+%26+a+little+bit+of+pink." />

<br/>

[![Python](https://img.shields.io/badge/Python-3.11-EC8FB3?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![ONNX](https://img.shields.io/badge/ONNX-Runtime-F4A6C1?style=for-the-badge)](https://onnxruntime.ai/)
[![Status](https://img.shields.io/badge/Status-In%20Development-EF80AA?style=for-the-badge)](https://github.com/kyutakou/NANAMI)
[![License](https://img.shields.io/badge/License-TBD-F8BBD0?style=for-the-badge)](https://github.com/kyutakou/NANAMI)

<br/>

**classroom computer-vision that looks, but doesn't know you.**

*A local-first, privacy-conscious classroom computer vision research project.*

[Overview](#-overview) · [Architecture](#-architecture) · [Progress](#-development-progress) · [Setup](#-getting-started)

</div>

---

## ୨୧ Overview

NANAMI is a classroom computer-vision research project designed to detect people, track movement, and eventually assist human reviewers in examining potentially relevant classroom events.

It uses local inference and anonymous tracking rather than facial recognition or personal identity identification.

### ♡ Design philosophy

- **Privacy first.** No facial recognition or identity inference.
- **Local-first.** Processing is designed to happen on the local machine.
- **Human-centered.** Automated observations are never treated as proof of misconduct.
- **Transparent.** Limitations and false detections must be acknowledged.
- **Research-driven.** Every component should be tested and evaluated before integration.

## ୨୧ What NANAMI can do

| Feature | Status |
|---|---|
| YOLOX-Tiny person detection | Working |
| ONNX CPU inference | Working |
| ByteTrack-style tracking | Working |
| Green bounding-box visualization | Working |
| RTMPose-t pose estimation | Planned |
| Behavioral analysis | Planned |
| Backend | Planned |
| Dashboard | Planned |

### Current detection and tracking

NANAMI currently combines:

- **YOLOX-Tiny ONNX** for person detection.
- **ByteTrack-style tracking** using a custom implementation.
- **OpenCV** for image processing and visualization.

Current observed performance:

| Metric | Result |
|---|---|
| Processing speed | ~13 FPS |
| Detection visualization | Green bounding boxes |
| Person identification | Not implemented |
| Inference | CPU-based |

*Performance is preliminary and may vary depending on hardware, input resolution, and the number of people in a frame.*

## ୨୧ Architecture

```mermaid
flowchart TD
    A[Video Input] --> B[Frame Processing]
    B --> C[YOLOX-Tiny ONNX]
    C --> D[Person Detection]
    D --> E[ByteTrack-style Tracker]
    E --> F[Anonymous Track Continuity]
    F --> G[Visualization]

    G --> H[Future Pose Estimation]
    H --> I[Future Analysis Layer]
    I --> J[Human Review]
    
    style A fill:#FCE4EC,stroke:#EC407A,color:#880E4F
    style C fill:#F8BBD0,stroke:#EC407A,color:#880E4F
    style E fill:#F8BBD0,stroke:#EC407A,color:#880E4F
    style J fill:#FCE4EC,stroke:#EC407A,color:#880E4F
```

## ୨୧ Tech stack

| Layer | Technology |
|---|---|
| Primary language | Python 3.11 |
| Environment | uv |
| Detection | YOLOX-Tiny ONNX |
| Tracking | Custom ByteTrack-style implementation |
| Inference | ONNX Runtime (CPU) |
| Image processing | OpenCV |
| Numerical computing | NumPy, SciPy |
| Testing | pytest |
| Future pose estimation | RTMPose-t ONNX |
| Frontend | Vite, React, TypeScript |
| Development OS | Windows 10 |

## ୨୧ Development progress

<details open>
<summary><b>Phase 1 — Person Detection</b> ♡ COMPLETE</summary>

- YOLOX-Tiny ONNX integration.
- Detector interface.
- Person detection pipeline.
- Initial automated detector tests.
- Bounding-box visualization.

</details>

<details open>
<summary><b>Phase 2 — Person Tracking</b> ♡ WORKING</summary>

- Custom ByteTrack-style tracker.
- Detection-to-track association.
- Multiple-frame tracking pipeline.
- Green bounding-box visualization.
- Initial tracking tests and debugging.

</details>

<details open>
<summary><b>Phase 3 — Integration and Testing</b> ♡ WORKING</summary>

- Detection and tracking pipeline tested together.
- Initial performance testing.
- Approximately 13 FPS observed.
- Further reliability and edge-case testing remains necessary.

</details>

<details>
<summary><b>Phase 4 — Pose Estimation</b> ♡ PLANNED</summary>

- RTMPose-t ONNX integration.
- Keypoint extraction.
- Pose estimation validation.

</details>

<details>
<summary><b>Phase 5 — Analysis Layer</b> ♡ PLANNED</summary>

- Temporal observation.
- Research-based analysis.
- False-positive evaluation.
- Human-verifiable outputs.

</details>

<details>
<summary><b>Phase 6 — Backend and Dashboard</b> ♡ PLANNED</summary>

- Backend development.
- Dashboard interface.
- Review workflow.
- Results visualization.

</details>

## ୨୧ Getting started

### Requirements

- Python 3.11
- uv
- Git
- Windows development environment

### Installation

```bash
# Clone the repository
git clone https://github.com/kyutakou/NANAMI.git

# Enter the project
cd NANAMI

# Create virtual environment
uv venv .venv

# Activate environment (Windows Git Bash)
source .venv/Scripts/activate

# Or Windows PowerShell
.venv\Scripts\Activate.ps1

# Install dependencies
uv pip install -e ".[dev]"
```

### Run detector test

```bash
python scripts/try_detector.py --image path/to/consented_photo.jpg
```

### Run tests

```bash
pytest
```

## ୨୧ Privacy and ethical considerations

NANAMI is designed around the principle that observing an event is not the same as knowing what happened.

The project deliberately avoids:

- Facial recognition.
- Real-world identity identification.
- Personal identity databases.
- Automated disciplinary decisions.
- Default cloud-based footage uploads.

Additional safeguards required for research deployment include:

- Appropriate consent and institutional approval.
- Restricted access to footage and results.
- Data minimization and defined retention periods.
- Testing for false detections and tracking errors.
- Human verification before interpreting any observation.

NANAMI must not be used as an autonomous cheating detector or as the sole basis for disciplinary action.

## ୨୧ Limitations

NANAMI is an experimental research prototype.

Its performance and reliability may be affected by:

- Occlusion and overlapping people.
- Lighting conditions.
- Camera placement and viewing angle.
- Detection confidence thresholds.
- Tracking interruptions and ID switches.
- Hardware limitations and processing speed.

A detection or tracking result does not establish intent or misconduct.

## ୨୧ Repository structure

```text
NANAMI/
├── scripts/
│   └── try_detector.py
├── README.md
├── pyproject.toml
└── ...
```

*The repository structure will expand as additional modules are implemented.*

## ୨୧ Contributing and research

NANAMI is being developed as a student research project.

Suggestions, technical feedback, and discussions about computer vision, model performance, and responsible classroom technology are welcome.

Please avoid submitting private classroom footage, identifiable student information, or sensitive personal data.

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:FCE4EC,100:F48FB1&height=3&section=footer" width="100%"/>

**NANAMI** ♡

*Seeing patterns. Preserving privacy.*

Made with curiosity, Python, and a little bit of pink.

<sub>© NANAMI Research Project · kyutakou</sub>

</div>
