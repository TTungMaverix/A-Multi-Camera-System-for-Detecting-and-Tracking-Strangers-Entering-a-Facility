# Association Runtime Config

Association runtime policy is externalized at:

- [association_policy.example.yaml](../insightface_demo_assets/runtime/config/association_policy.example.yaml)

The loader order is:

1. `association_policy_config` from [face_demo_config.json](../insightface_demo_assets/runtime/face_demo_config.json), if provided
2. `insightface_demo_assets/runtime/config/association_policy.yaml`
3. `insightface_demo_assets/runtime/config/association_policy.example.yaml`
4. built-in defaults from `association_core/config_loader.py`

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
