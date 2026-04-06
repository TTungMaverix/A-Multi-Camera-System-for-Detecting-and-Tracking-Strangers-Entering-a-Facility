# Association Runtime Config

Association runtime policy is externalized at:

- [association_policy.example.yaml](../insightface_demo_assets/runtime/config/association_policy.example.yaml)
- [camera_transition_map.example.yaml](../insightface_demo_assets/runtime/config/camera_transition_map.example.yaml)

The loader order is:

1. `association_policy_config` from [face_demo_config.json](../insightface_demo_assets/runtime/face_demo_config.json), if provided
2. `insightface_demo_assets/runtime/config/association_policy.yaml`
3. `insightface_demo_assets/runtime/config/association_policy.example.yaml`
4. built-in defaults from `association_core/config_loader.py`

Camera transition metadata is loaded in this order:

1. `camera_transition_map_config` from [face_demo_config.json](../insightface_demo_assets/runtime/face_demo_config.json), if provided
2. `insightface_demo_assets/runtime/config/camera_transition_map.yaml`
3. `insightface_demo_assets/runtime/config/camera_transition_map.example.yaml`
4. a fallback map derived from `wildtrack_demo/wildtrack_demo_config.json`

If a config file is missing or only partially filled:

- the loader falls back to defaults
- the applied default keys are recorded in `association_logs/association_policy_runtime.json`
- the demo command still runs without manual code edits

## Config Sections

`quality_gate`
- bbox and face-quality thresholds before association

`topology_filter`
- relation priors
- overlap/sequential/weak-link travel-time tolerance
- zone fallback behavior
- subzone fallback behavior

`appearance_evidence`
- face-vs-body modality preference
- body fallback behavior
- secondary evidence availability floor

`gallery_lifecycle`
- `ttl_sec`
- `top_k_face_refs`
- `top_k_body_refs`
- representative update behavior
- expiry policy metadata

`decision_policy`
- known accept threshold and margin
- unknown reuse threshold
- per-relation thresholds for `overlap`, `sequential`, `weak_link`
- per-relation margin requirements
- minimum evidence requirements
- defer and create rules

## Dataset Adaptation

To adapt the same association engine to a new dataset:

1. keep the code unchanged
2. update camera topology in the dataset config
3. copy the example policy file to a dataset-specific policy file
4. tune thresholds and margins there

This keeps the system paper-grounded and avoids re-editing core logic for every dataset.

## Zone / Transition Metadata

`camera_transition_map.example.yaml` externalizes:

- per-camera default zones
- per-camera default subzones
- entry zones and exit zones
- entry/exit/transit/overlap subzones
- directed camera-pair transitions
- `relation_type` per edge
- `min / avg / max` travel-time priors
- overlap flags and weak-link support
- optional allowed entry/exit subzones per transition

At runtime, observations can carry `zone_id`, `zone_type`, `subzone_id`, and `subzone_type`. If a dataset does not provide fine-grained subzones yet:

- the runtime falls back to the camera default zone
- then falls back to the camera default subzone
- the association logs mark that fallback explicitly
- the demo command still runs without code edits
