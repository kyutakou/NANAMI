# NANAMI Licensing Register

Status: DRAFT. Last updated 2026-11-09.
This document is an engineering record, not legal advice. Anything marked
"UNVERIFIED" or "LEGAL REVIEW" must be resolved before commercial distribution.

## Rules

1. Code license, weights license, and training-data terms are recorded separately.
2. A repository license does not automatically cover pretrained weights.
3. AGPL, GPL, non-commercial, and platform-restricted licenses are not allowed
   unless explicitly approved after legal review.
4. Secondary sources (blogs, mirrors, comparison sites) are NOT proof. Confirm
   against the LICENSE file or model card at a pinned version and record it here.

## Selected components

| Component | Role | Code license | Weights / data terms | Status |
|---|---|---|---|---|
| YOLOX-Tiny (Megvii-BaseDetection/YOLOX) | Person detection | Apache 2.0 (reported by search results) | COCO-trained official weights; COCO image terms vary | UNVERIFIED against repo LICENSE file; LEGAL REVIEW for COCO |
| RTMPose-t/s (OpenMMLab MMPose) | Pose estimation | Apache 2.0 (reported) | Trained on multiple public datasets ("Body7"); some carry their own terms | UNVERIFIED per checkpoint; LEGAL REVIEW for datasets |
| ByteTrack (own implementation from the paper) | Tracking | n/a (own code) | n/a | Paper credited; original repo license UNVERIFIED |
| onnxruntime | Inference | verify (believed MIT) | n/a | UNVERIFIED |
| numpy, scipy, opencv-python-headless | Core libraries | verify | n/a | UNVERIFIED; run pip-licenses |
| flatbuffers, protobuf, packaging | Dependencies | verify | n/a | UNVERIFIED; run pip-licenses |
| React, Vite, ESLint | Frontend | verify (believed MIT) | n/a | UNVERIFIED; run license-checker |

## Candidates (not selected)

| Candidate | Notes | Status |
|---|---|---|
| RF-DETR Nano | Reported Apache 2.0 for Nano to Large. XL/2XL (and reportedly Atto/Femto/Pico) use Roboflow PML 1.0. Never install rfdetr[plus]. | Swap-in test only; UNVERIFIED |
| MoveNet | Reported Apache 2.0. Trained on COCO plus an internal Google dataset ("Active"). | UNVERIFIED against TF Hub / Kaggle model card |
| RTMO | One-stage multi-person pose (MMPose). | Not evaluated |
| OC-SORT | Motion-based tracker, believed MIT. | Not evaluated |
| D-FINE, DEIM, PP-PicoDet, NanoDet | Possible detector alternatives. | Not evaluated; licenses from memory only |

## Excluded

| Item | Reason |
|---|---|
| ultralytics (YOLOv5/8/11, RT-DETR wrapper), YOLOv10 | AGPL-3.0 |
| YOLOv6/7/9 | GPL (reported) |
| YOLO-NAS | Pretrained weights under a special license |
| boxmot | AGPL (believed; verify before ever reconsidering) |
| OpenPose, AlphaPose | Non-commercial academic licenses (believed) |
| Any tracker using ReID/appearance embeddings | Conflicts with the no-identity-inference rule |
| rtmlib | Apache 2.0, but requires non-headless opencv-python; we write our own ONNX wrapper instead |
| Prototype code and weights from earlier Ultralytics-based work | Clean-start decision; do not reuse |

## Open items

- [ ] Read the LICENSE file in Megvii-BaseDetection/YOLOX at a pinned commit; record the commit hash.
- [ ] Identify the exact YOLOX-Tiny ONNX/weights source, its checksum, and its license.
- [ ] Identify the exact RTMPose checkpoint, its source URL, checksum, and listed training datasets.
- [ ] Review terms of each training dataset (COCO and the RTMPose training sets).
- [ ] Run pip-licenses on the final venv; paste results here.
- [ ] Run license-checker on frontend dependencies; paste results here.
- [ ] Review the Philippine Data Privacy Act (RA 10173) implications with the school's data protection officer.
- [ ] Obtain legal review before any commercial distribution.

## Record of checks

| Date | Item | Source consulted | Result |
|---|---|---|---|
| 2026-11-09 | YOLOX | Search results (third-party mirror model card, Roboflow page) | Apache 2.0 reported; repo LICENSE not yet read |
| 2026-11-09 | RF-DETR | Roboflow repo README, Roboflow licensing pages | Apache 2.0 Nano-Large; PML 1.0 for larger/plus sizes |
| 2026-11-09 | RTMPose / rtmlib | Hugging Face model cards, rtmlib repo metadata | Apache 2.0 code; dataset terms vary |
| 2026-11-09 | MoveNet | Google blog, third-party project | Apache 2.0 reported; model card not yet read |
'