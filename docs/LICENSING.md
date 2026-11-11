# NANAMI Licensing Register

**Status:** DRAFT — NOT CLEARED FOR COMMERCIAL DISTRIBUTION  
**Last updated:** 2026-11-11  
**Purpose:** Track the licenses, attribution requirements, model-weight terms, dataset rights, and distribution restrictions relevant to NANAMI.

This is an engineering compliance record, not legal advice. A component marked **UNVERIFIED**, **PENDING**, or **LEGAL REVIEW REQUIRED** must not be treated as approved for redistribution or commercial use.

---

## 1. Licensing principles

1. Track source-code licenses, pretrained-weight licenses, and training-data terms separately.
2. A repository's license does not automatically apply to separately distributed model weights, datasets, or third-party assets.
3. Record the exact repository, release or commit, checkpoint, download source, and checksum for every third-party component.
4. Do not treat search snippets, blogs, mirrors, or package metadata as sufficient proof when an authoritative license file or model card is available.
5. Preserve required copyright notices, license texts, attribution, and modification notices.
6. Do not redistribute a model or dataset unless its applicable terms have been checked.
7. Any AGPL, GPL, non-commercial, platform-restricted, or otherwise incompatible component requires explicit review before adoption.
8. NANAMI's own license must be selected separately. Do not assume third-party licenses determine the license of the entire project.

## 2. Selected components

| Component | Purpose | Code license | Weights / data terms | Current status |
|---|---|---|---|---|
| YOLOX-Tiny — Megvii-BaseDetection/YOLOX | Person detection | Apache-2.0, as identified by the official repository | Exact ONNX checkpoint and its source still need identification. COCO image rights require separate review. | **PARTIALLY VERIFIED** |
| RTMPose-t/s — OpenMMLab MMPose | Pose estimation | Apache-2.0 for the MMPose repository | Exact checkpoint, model-card terms, and training dataset provenance remain unverified. | **CODE VERIFIED; WEIGHTS PENDING** |
| ByteTrack algorithm | Multi-object tracking | NANAMI's independently written implementation will use NANAMI's chosen project license. | No pretrained weights currently identified. | **OWN CODE; LICENSE DECISION PENDING** |
| ONNX Runtime | Model inference | MIT, according to the official repository | No separate model-weight rights implied. | **CODE LICENSE VERIFIED** |
| NumPy | Numerical operations | Pending verification against installed distribution metadata | Not applicable | **PENDING INVENTORY** |
| SciPy | Numerical operations | Pending verification against installed distribution metadata | Not applicable | **PENDING INVENTORY** |
| OpenCV | Image processing | Pending verification against installed distribution metadata and build variant | Not applicable | **PENDING INVENTORY** |
| FlatBuffers | Serialization support | Pending verification against installed distribution metadata | Not applicable | **PENDING INVENTORY** |
| Protobuf | Serialization support | Pending verification against installed distribution metadata | Not applicable | **PENDING INVENTORY** |
| Packaging | Python package utilities | Pending verification against installed distribution metadata | Not applicable | **PENDING INVENTORY** |
| React, Vite, ESLint | Frontend development | Pending verification against the actual installed dependency tree | Not applicable | **PENDING INVENTORY** |

### Important notes

- The official YOLOX repository identifies its repository license as Apache-2.0. This does not establish the redistribution rights of an independently downloaded ONNX checkpoint.
- MMPose's repository license does not independently clear every RTMPose checkpoint.
- The original ByteTrack repository identifies MIT licensing. That does not automatically license independently written NANAMI code under MIT. If upstream source code is copied or adapted, preserve the applicable upstream notices and review the resulting obligations.
- ONNX Runtime's MIT license applies to the ONNX Runtime project, not to the model being executed.

## 3. Model and dataset register

### YOLOX-Tiny

**Upstream project:** https://github.com/Megvii-BaseDetection/YOLOX  
**Repository code license:** Apache-2.0  
**Checkpoint:** Not yet recorded  
**Checkpoint source:** Not yet recorded  
**Checksum:** Not yet recorded  
**Training data:** Reported as COCO; exact provenance for the selected checkpoint must be confirmed.

COCO has separate terms for annotations and images. Its annotations and website are licensed under CC BY 4.0, while the images remain subject to their respective copyright and Flickr terms. Do not describe all COCO images as CC BY 4.0.

**Required before distribution:**
- Identify the exact checkpoint source and version.
- Record its checksum.
- Check the checkpoint's own terms and accompanying notices.
- Confirm whether its training-data documentation matches the selected checkpoint.
- Review whether the intended use and redistribution are permitted.

### RTMPose

**Upstream project:** https://github.com/open-mmlab/mmpose  
**Repository code license:** Apache-2.0  
**Checkpoint:** Not yet recorded  
**Checkpoint source:** Not yet recorded  
**Checksum:** Not yet recorded  
**Training datasets:** Not yet fully identified.

The MMPose repository license is not sufficient evidence that every RTMPose checkpoint or dataset has identical terms.

**Required before distribution:**
- Identify the exact RTMPose variant and checkpoint.
- Record the model-card or release URL.
- Record the checkpoint checksum.
- Record every training dataset explicitly disclosed by the model documentation.
- Review each dataset's applicable terms.

### ByteTrack

**Reference paper:** ByteTrack: Multi-Object Tracking by Associating Every Detection Box.  
**Reference implementation:** https://github.com/FoundationVision/ByteTrack  
**Reference repository license:** MIT.

NANAMI intends to implement the tracking algorithm independently. The paper should be cited in research documentation. If code is copied or adapted from the reference implementation, record the source and comply with its license.

**NANAMI implementation license:** Pending the project's own license decision.

## 4. Candidates — not selected

| Candidate | Notes | Status |
|---|---|---|
| RF-DETR | The official repository distinguishes Apache-2.0 model variants from certain Platform Model License variants. Check the exact model size and package extras before selection. | NOT SELECTED |
| MoveNet | Exact model artifact, model card, and applicable terms have not been checked. | NOT SELECTED |
| RTMO | Potential multi-person pose-estimation alternative. Exact checkpoint and terms need review. | NOT EVALUATED |
| OC-SORT | Potential motion-based tracking alternative. Verify the exact repository and license before adoption. | NOT EVALUATED |
| D-FINE, DEIM, PP-PicoDet, NanoDet | Potential detector alternatives. Do not rely on remembered license information. | NOT EVALUATED |

Candidate status does not constitute approval for use.

## 5. Excluded or restricted components

| Component or category | Reason | Status |
|---|---|---|
| Ultralytics-based components | Excluded from the current clean-start architecture because their licensing and distribution implications require separate review. | EXCLUDED |
| YOLO-NAS | Not selected. Exact code and pretrained-weight terms would need independent review. | EXCLUDED |
| BoxMOT | Not selected. Its exact license and the licenses of any included trackers must be checked before reconsideration. | EXCLUDED |
| OpenPose and AlphaPose | Not selected. Review exact code, model, and commercial-use terms before reconsideration. | EXCLUDED |
| Trackers using ReID or appearance embeddings | Conflicts with NANAMI's no-identity-inference design requirement. | DESIGN EXCLUSION |
| rtmlib | Not selected for the current implementation because of the project's preferred inference and dependency architecture. | DESIGN EXCLUSION |
| Earlier Ultralytics-based prototype code and weights | Clean-start decision. Do not reuse without a separate provenance and license review. | EXCLUDED |

**Note:** These exclusions are project decisions, not universal legal conclusions about the named projects.

## 6. NANAMI's own project license

**Current status:** Not selected.

Before choosing a root `LICENSE`, decide:

- Whether NANAMI source code may be reused in other projects.
- Whether commercial use is permitted.
- Whether modified versions may be redistributed.
- Whether attribution is required.
- Whether the repository will remain open source.
- Whether any third-party components impose additional distribution conditions.

Do not add a license badge or advertise NANAMI as commercially reusable until the project license and third-party component obligations have been reviewed.

A root project license cannot override the terms of third-party dependencies, pretrained weights, or datasets.

## 7. Privacy and research compliance

Licensing is separate from privacy and research ethics.

Before collecting or processing classroom footage:

- Obtain the required school authorization and research approvals.
- Consult the school's Data Protection Officer regarding the Philippine Data Privacy Act of 2012 (RA 10173).
- Establish a clear purpose and appropriate legal basis for processing.
- Limit access to footage and derived data.
- Define retention and deletion procedures.
- Avoid collecting unnecessary personal information.
- Document limitations, false positives, and the need for human review.
- Do not use automated outputs as the sole basis for disciplinary decisions.

This section records engineering requirements, not a legal determination that the system is compliant.

## 8. Open items

### Source code and dependencies

- [ ] Record the exact YOLOX upstream commit or release used.
- [ ] Preserve the YOLOX Apache-2.0 license text and required notices.
- [ ] Run a dependency license inventory on the final Python environment.
- [ ] Run a production dependency license inventory for the frontend.
- [ ] Review transitive dependencies and bundled third-party assets.
- [ ] Add a root project license only after deciding NANAMI's own licensing policy.

### Models and datasets

- [ ] Identify the exact YOLOX-Tiny ONNX file and its download source.
- [ ] Record its SHA-256 checksum.
- [ ] Verify the ONNX checkpoint's redistribution and commercial-use terms.
- [ ] Identify the exact RTMPose checkpoint and model card.
- [ ] Record the RTMPose checkpoint's SHA-256 checksum.
- [ ] Identify and review the disclosed RTMPose training datasets.
- [ ] Review COCO annotation terms separately from image terms.
- [ ] Record model and dataset attribution requirements.

### Privacy and distribution

- [ ] Consult the school's Data Protection Officer.
- [ ] Document school approval and research safeguards.
- [ ] Review privacy and data-retention procedures.
- [ ] Obtain qualified legal review before commercial distribution.

## 9. Suggested inventory commands

Run these inside the actual project environments. Save the complete output with the date and environment details.

**Python environment:**

```bash
python -m pip install pip-licenses
pip-licenses --format=markdown --with-urls --with-description
```

**Frontend production dependencies:**

```bash
npx license-checker --production --summary
```

These tools provide an inventory, not a legal clearance. Manually inspect licenses, bundled assets, model files, and dataset terms as well.

## 10. Record of checks

| Date | Item | Source consulted | Result |
|---|---|---|---|
| 2026-11-11 | YOLOX repository | Official Megvii-BaseDetection/YOLOX repository | Apache-2.0 identified at repository level; exact commit and checkpoint still pending |
| 2026-11-11 | MMPose repository | Official OpenMMLab/MMPose LICENSE | Apache-2.0 verified at repository level; checkpoint and dataset terms pending |
| 2026-11-11 | ByteTrack | Official FoundationVision/ByteTrack repository | MIT identified for reference repository; NANAMI's independent implementation requires its own license decision |
| 2026-11-11 | ONNX Runtime | Official Microsoft/ONNX Runtime repository | MIT identified |
| 2026-11-11 | COCO | Official COCO Terms of Use | Annotations CC BY 4.0; image rights remain with their respective copyright holders |
| 2026-11-11 | RF-DETR | Official Roboflow RF-DETR repository | Different model variants have different licensing terms; candidate only |

---

**Distribution status: NOT CLEARED.**

NANAMI may continue as a research and development project while these checks remain open. Do not represent the current register as proof of commercial-use permission.

*Document owner: NANAMI project maintainers.*
