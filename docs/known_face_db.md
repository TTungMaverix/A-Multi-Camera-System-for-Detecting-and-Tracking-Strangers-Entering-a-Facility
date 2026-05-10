# Facility Known Face DB

The active Known DB for the New Dataset is:

`D:\ĐỒ ÁN TỐT NGHIỆP\New Dataset\Known ID`

Each identity is a folder such as `known_001`, `known_002`, `known_003`, and future `known_004` entries. The runtime normalizes these into stable facility IDs like `FACILITY_001_known_001`.

## Build Command

```powershell
& "D:\ĐỒ ÁN TỐT NGHIỆP\.venv_insightface_demo\Scripts\python.exe" `
  insightface_demo_assets/runtime/run_build_known_facility_db.py `
  --project-root . `
  --known-root "D:\ĐỒ ÁN TỐT NGHIỆP\New Dataset\Known ID" `
  --manifest-csv insightface_demo_assets/known_face_facility_manifest.csv `
  --embeddings-csv insightface_demo_assets/runtime/known_face_facility_embeddings.csv `
  --summary-json outputs/evaluations/known_facility_db/known_db_build_summary.json
```

## Generated Files

- Manifest: `insightface_demo_assets/known_face_facility_manifest.csv`
- Embedding CSV: `insightface_demo_assets/runtime/known_face_facility_embeddings.csv`
- Grayscale aligned crops: `insightface_demo_assets/runtime/known_facility_grayscale_faces`
- Build summary: `outputs/evaluations/known_facility_db/known_db_build_summary.json`

## Matching Policy

- Only `Direction = IN` tracks are sent to Known DB matching.
- Face embedding cosine similarity is the primary signal.
- Aligned grayscale face similarity is an auxiliary signal for audit and lighting-domain-gap analysis.
- If a known match passes threshold, the track is labeled `KNOWN` and is not converted into an Unknown stranger profile.
- If no known match passes threshold, the track continues into the Unknown ID and cross-camera association flow.

## P0 Build Result

The P0 build loaded 3 identities and 12 images:

- `known_001`
- `known_002`
- `known_003`

All 12 images produced embeddings and aligned grayscale crops in the validation run.
