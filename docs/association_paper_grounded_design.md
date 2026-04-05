# Paper-Grounded Design Note for Cross-Camera Association

## Scope

This note extracts the usable ideas from three local papers and maps them into the association core of the graduation project:

- [Kim_SapiensID_Foundation_for_Human_Recognition_CVPR_2025_paper.pdf](d:/ĐỒ%20ÁN%20TỐT%20NGHIỆP/PAPERS/Kim_SapiensID_Foundation_for_Human_Recognition_CVPR_2025_paper.pdf)
- [MCTR Multi Camera Tracking Transformer.pdf](d:/ĐỒ%20ÁN%20TỐT%20NGHIỆP/PAPERS/MCTR%20Multi%20Camera%20Tracking%20Transformer.pdf)
- [Multi-Camera Industrial Open-Set Person.pdf](d:/ĐỒ%20ÁN%20TỐT%20NGHIỆP/PAPERS/Multi-Camera%20Industrial%20Open-Set%20Person.pdf)

The goal is not to copy any single paper directly. The goal is to build a paper-grounded association framework for this project:

- candidate filtering by topology + travel time
- gallery and reference-frame update
- association score / decision policy
- open-set acceptance / rejection logic

This note intentionally avoids a single ad-hoc weighted sum across unrelated signals. The recommended near-term implementation is a gated, paper-inspired rule framework. A heavier end-to-end model can be added later without changing the data schema.

## What each paper contributes

### 1. SapiensID: appearance representation and modality handling

Key extraction:

- Human recognition is framed as metric learning: same-person features should be closer than different-person features.
- The paper explicitly argues that face-only or body-only models are insufficient in real scenarios.
- It highlights score-level or feature-level fusion between face and body cues.
- It also shows that scenario-specific body-part emphasis matters. Some datasets benefit more from upper-body cues than full-body cues.

Usable project implication:

- The association core should treat appearance as a structured evidence block, not as one raw cosine score.
- Face and body should be separate signals.
- The system should allow modality-aware policy:
  - strong face available
  - weak face but usable body
  - body-only fallback
- The appearance module should keep quality metadata together with embeddings.

### 2. MCTR: probabilistic association across overlapping cameras

Key extraction:

- MCTR handles overlapping cameras by maintaining global track embeddings and view-specific detection embeddings.
- It uses probabilistic assignment of detections to tracks instead of one hard assignment too early.
- For detections from two cameras or two frames, it computes the probability that they belong to the same track by integrating over track assignment.
- The paper is explicitly designed for overlapping multi-camera setups where simultaneous visibility is possible.

Direct paper formula:

`P_same_track(d1, d2) = sum_t A_v1[d1, t] * A_v2[d2, t]`

where `A_v[d, t]` is the probabilistic assignment of detection `d` in view `v` to track `t`.

Usable project implication:

- The association module should not assume every cross-camera event is sequential.
- The same person may be visible at nearly the same time in two overlapping cameras.
- Candidate generation must support:
  - overlapping same-time visibility
  - near-simultaneous visibility
  - delayed sequential re-appearance
- Even in a light rule-based implementation, ranking candidates should be "probability-like" and ambiguity-aware, not just threshold-on-cosine.

### 3. MICRO-TRACK: open-set gallery, quality gate, orchestrator

Key extraction:

- The system is modular: tracker per camera, Re-ID module, and a global orchestrator.
- Re-ID is not invoked on every frame blindly. A decision module checks whether the detection is good enough.
- A detection score threshold `thscore` controls when Re-ID is applied.
- A gallery threshold `themb` controls acceptance vs rejection in the open-set gallery.
- Tracking maintains the assigned identity between re-identification triggers.
- Stored embeddings expire after a time horizon to limit drift and memory growth.

Usable project implication:

- The project should use a quality gate before expensive or high-risk identity comparison.
- The gallery should be open-set and dynamic.
- Unknown profiles should have expiration / refresh rules.
- The tracker is responsible for temporal continuity inside each camera.
- The orchestrator is responsible for cross-camera identity reuse.

## Matching formula extracted or adapted from the papers

### Paper-grounded appearance principle

From SapiensID:

`d(f_A_i, f_A_j) < d(f_A_i, f_B_k)`

Interpretation for this project:

- use face embedding similarity when face quality is good
- use body embedding similarity when face is absent or weak
- keep modality quality and availability attached to the score

### Paper-grounded cross-camera association principle

From MCTR:

`P_same_track(d1, d2) = sum_t A_v1[d1, t] * A_v2[d2, t]`

Interpretation for this project:

- association should be read as "probability-like compatibility with an existing global identity"
- multiple candidates may remain plausible until the final decision stage
- overlap cameras must allow `delta_t ~= 0`

### Paper-grounded open-set decision principle

From MICRO-TRACK:

- if detection quality is below `thscore`, do not trust Re-ID yet
- if gallery distance is worse than `themb`, reject match and create new open-set identity
- tracker maintains identity continuity between explicit re-identification steps

Interpretation for this project:

- separate "should compare now?" from "who should it match?"
- keep acceptance and rejection as explicit branches

## Recommended replacement for the current weighted rule

### Do not use

`final_score = a * appearance + b * topology + c * time + d * zone`

Problem:

- weights are hard to justify across datasets
- a large appearance score can hide topology failure
- overlap and sequential cases behave very differently
- one scalar score makes ambiguity analysis weak

### Replace with a gated paper-inspired framework

Use four stages:

1. quality gate
2. candidate filter
3. evidence evaluation
4. accept / reject / defer decision

This framework is still light and rule-based, but it is closer to the papers:

- MICRO-TRACK contributes the quality gate and open-set accept/reject logic
- MCTR contributes probability-like candidate reasoning and overlap support
- SapiensID contributes modality-aware appearance evidence

## Candidate filtering by topology + travel time

Define the candidate set for a new event `e` as:

`C(e) = { u in gallery | topology_ok(e, u) and time_ok(e, u) and zone_ok(e, u) }`

### Topology rules

Each camera pair must have one `relation_type`:

- `overlap`
- `sequential`
- `weak_link`

Required per edge:

- `min_travel_time_sec`
- `avg_travel_time_sec`
- `max_travel_time_sec`
- `same_area_overlap`
- optional `exit_zone -> entry_zone` compatibility

### Time compatibility rules

For `overlap`:

- allow simultaneous or near-simultaneous observations
- accept `abs(delta_t) <= max_travel_time_sec`
- center the compatibility near `0`

For `sequential`:

- require `min_travel_time_sec <= delta_t <= max_travel_time_sec`
- compatibility peaks near `avg_travel_time_sec`

For `weak_link`:

- allow only if appearance evidence is strong
- time window may be wider, but threshold must be stricter later

### Zone compatibility rules

Optional but strongly recommended:

- exit zone from source camera
- entry zone from destination camera
- if zone transition is impossible, reject candidate early

This is the minimum "map" needed for the system even when a full floorplan image is not available.

## Gallery and reference-frame update logic

Inspired primarily by MICRO-TRACK, extended for this project.

For each `unknown_global_id`, keep:

- `first_seen_camera`
- `first_seen_time`
- `latest_seen_camera`
- `latest_seen_time`
- `cameras_seen`
- `zones_seen`
- `top_k_face_refs`
- `top_k_body_refs`
- `representative_face_embedding`
- `representative_body_embedding`
- `quality_stats`
- `expiry_at`

### Update policy

Update a profile only when at least one of these is true:

- the new frame is higher quality than existing references
- the new frame comes from a new camera
- the new frame adds a new zone transition
- the new frame improves pose or scale diversity

### Expiry policy

Expire or down-rank old references when:

- the person has not been seen for `ttl_sec`
- appearance is likely stale
- gallery growth becomes too large

This keeps the gallery dynamic and avoids reference drift.

## Acceptance-rejection logic

### Stage 1. Quality gate

Do not compare an event yet if:

- detector confidence is too low
- bbox is too small
- face crop is unusable and body crop is also poor
- occlusion is severe

Decision:

- `DEFER` if the event is too poor for safe association
- let per-camera tracking keep the local identity until better evidence arrives

### Stage 2. Appearance evidence

Do not collapse face and body into one arbitrary number too early.

Keep:

- `face_sim`
- `body_sim`
- `modality_state`
  - `face_and_body`
  - `body_only`
  - `face_only`
  - `weak_both`

Recommended rule:

- if strong face exists, use face as primary evidence and body as consistency check
- if face is weak or missing, use body as primary evidence
- if both are weak, do not accept reuse unless topology is extremely strong and ambiguity is low

### Stage 3. Candidate ranking

For each candidate `u` in `C(e)`, compute:

- `appearance_primary`
- `appearance_secondary`
- `time_compatibility`
- `zone_compatibility`
- `relation_type`
- `quality_reliability`

The ranking should be lexicographic or probability-like, not a blind weighted sum.

Recommended comparison key:

`K(e, u) = (gate_pass, appearance_primary, appearance_secondary, time_compatibility, quality_reliability)`

### Stage 4. Accept / reject / create / defer

Accept existing unknown `u*` only if:

- `u*` passed topology, time, and zone gates
- `appearance_primary >= tau_primary[relation_type, modality_state]`
- `appearance_secondary >= tau_secondary[relation_type, modality_state]` when secondary evidence exists
- `margin(top1, top2) >= tau_margin`

Create new unknown if:

- no candidate survives the gates, or
- best candidate is below acceptance threshold, or
- best-vs-second-best margin is too small

Defer if:

- event quality is too poor, or
- ambiguity is high and waiting for a better frame is safer

Known-ID acceptance should be stricter than unknown reuse:

- `best_known >= tau_known_accept`
- `best_known - second_known >= tau_known_margin`
- otherwise keep the event in open-set flow

## Paper idea -> system module mapping

| Paper | Extracted idea | Module in this project | Adaptation for this project |
|---|---|---|---|
| SapiensID | Metric-learning appearance space | `appearance_encoder` | Face/body kept as separate evidence streams |
| SapiensID | Score-level or feature-level fusion across modalities | `appearance_evidence_builder` | Use modality-aware policy instead of one raw cosine |
| SapiensID | Scenario-specific useful body regions vary | `camera_quality_profile` | Allow camera-specific preference for face or upper-body |
| MCTR | Probabilistic association instead of early hard assignment | `association_core` | Use probability-like ranking and ambiguity margin |
| MCTR | Overlapping cameras may see the same identity simultaneously | `candidate_filter` | `relation_type=overlap` allows `delta_t ~= 0` |
| MCTR | Global identity consistency across views and time | `global_identity_state` | Maintain persistent unknown profiles, not one-shot events |
| MICRO-TRACK | Quality gate before Re-ID | `quality_gate` | Skip unsafe comparisons and defer when needed |
| MICRO-TRACK | Open-set gallery with accept/reject threshold | `open_set_gallery` | Known/unknown branches share orchestrator logic |
| MICRO-TRACK | Tracker keeps temporal continuity locally | `per_camera_tracker_handoff` | Local track stabilizes evidence until cross-camera compare |
| MICRO-TRACK | Embedding expiry and bounded gallery | `gallery_lifecycle_manager` | TTL and reference refresh for unknown identities |

## Short spec for association API and data schema

### 1. ObservationEvent

```json
{
  "event_id": "string",
  "camera_id": "C5",
  "local_track_id": "string",
  "timestamp_sec": 123.45,
  "event_type": "ENTRY_IN | OVERLAP_OBSERVATION | FOLLOWUP_OBSERVATION",
  "bbox": {"xmin": 0, "ymin": 0, "xmax": 0, "ymax": 0},
  "zone_id": "optional-string",
  "quality": {
    "det_score": 0.98,
    "face_quality": 0.74,
    "body_quality": 0.88
  },
  "embeddings": {
    "face": "vector-or-null",
    "body": "vector-or-null"
  },
  "refs": {
    "head_crop_path": "string",
    "body_crop_path": "string"
  }
}
```

### 2. TopologyEdge

```json
{
  "src_camera_id": "C5",
  "dst_camera_id": "C6",
  "relation_type": "overlap | sequential | weak_link",
  "same_area_overlap": true,
  "min_travel_time_sec": 0.0,
  "avg_travel_time_sec": 1.2,
  "max_travel_time_sec": 2.5,
  "allowed_exit_zones": ["stairs_left"],
  "allowed_entry_zones": ["stairs_right"]
}
```

### 3. GalleryProfile

```json
{
  "global_id": "UNK_0123",
  "identity_status": "unknown | known",
  "first_seen_camera": "C5",
  "first_seen_time": 123.45,
  "latest_seen_camera": "C6",
  "latest_seen_time": 125.10,
  "cameras_seen": ["C5", "C6"],
  "zones_seen": ["door_in", "hall_mid"],
  "face_refs": [
    {"event_id": "e1", "quality": 0.82, "embedding": "vector"}
  ],
  "body_refs": [
    {"event_id": "e1", "quality": 0.91, "embedding": "vector"}
  ],
  "expiry_at_sec": 600.0
}
```

### 4. AssociationDecision

```json
{
  "event_id": "string",
  "decision_type": "attach_known | reuse_unknown | create_unknown | defer",
  "chosen_global_id": "string-or-empty",
  "candidate_ids_considered": ["id1", "id2"],
  "top1_score": 0.0,
  "top2_score": 0.0,
  "margin": 0.0,
  "reason_code": "known_accept | unknown_reuse | below_threshold | ambiguous | no_candidate | poor_quality"
}
```

## Proposed paper-inspired rule-based framework

This is the recommended replacement for the current ad-hoc weighted rule before moving to a heavier learned model.

### Rule block A. Quality-aware trigger

- run association only on events that pass quality gate
- otherwise keep local tracking and wait for better evidence

This block is from MICRO-TRACK thinking.

### Rule block B. Topology-first candidate generation

- do not compare with all gallery identities
- compare only with identities reachable by topology + travel time + zone logic

This block adapts MCTR's view-consistency idea to a lighter system with explicit map priors.

### Rule block C. Modality-aware appearance evaluation

- evaluate face and body separately
- choose primary modality by quality and availability
- use the other modality as consistency evidence, not as arbitrary bonus weight

This block is grounded in SapiensID.

### Rule block D. Open-set accept/reject with margin

- accept a match only when:
  - gate passed
  - candidate is above threshold
  - best candidate is sufficiently better than the second best
- otherwise create a new unknown or defer

This block combines MICRO-TRACK open-set logic with MCTR-style ambiguity awareness.

## Recommended decision policy

For each event `e`:

1. Apply quality gate.
2. Build candidate set `C(e)` from topology, time, and zone rules.
3. Compute appearance evidence against each candidate profile.
4. Rank candidates.
5. Accept best candidate only if:
   - hard gates passed
   - appearance threshold passed
   - best-vs-second-best margin passed
6. Otherwise:
   - `create_unknown` if quality is usable but no safe match exists
   - `defer` if quality is too poor or ambiguity is too high
7. Update gallery only with good references.

## Why this framework is better than the current weighted rule

- It uses one common engine across datasets.
- Only the topology file and thresholds change per dataset.
- It naturally supports both overlap datasets and wide-area sequential datasets.
- It separates "candidate eligibility" from "candidate ranking".
- It makes rejection and defer explicit, which is essential in open-set surveillance.
- It is still lightweight and explainable for thesis/demo use.

## Minimal deployment inputs required per dataset

The system does not require a beautiful floorplan image, but it does require at least a topology config:

- camera list
- camera pair relation type
- min / avg / max travel time
- overlap flag
- optional zone transition map
- optional camera-specific face/body quality priors

If a floorplan exists, it should be treated as an additional source for building this topology config, not as a mandatory input to the runtime.

## Recommended next implementation step

Refactor the current association code into these modules:

- `quality_gate.py`
- `topology_filter.py`
- `appearance_evidence.py`
- `open_set_gallery.py`
- `decision_policy.py`

Do not train a heavy new model yet.

First replace the ad-hoc weighted sum with:

- topology/time/zone hard gating
- modality-aware appearance evidence
- margin-based accept/reject/defer
- dynamic gallery update with TTL and top-k references

That will already move the system much closer to the three papers while staying practical for the current project stage.
