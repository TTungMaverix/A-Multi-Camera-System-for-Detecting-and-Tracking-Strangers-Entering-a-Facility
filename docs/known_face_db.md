# Facility Known Face DB

The active Known DB for the New Dataset demo is:

`D:\ĐỒ ÁN TỐT NGHIỆP\New Dataset\Known ID`

Each identity is stored as a folder such as `known_001`, `known_002`, `known_003`, and future `known_004` entries. Runtime code normalizes these into stable facility IDs such as `FACILITY_001_known_001`.

## Runtime Defaults

If `KNOWN_DB_ROOT` is not provided through the environment, runtime face matching now defaults to:

- Known DB root: `D:\ĐỒ ÁN TỐT NGHIỆP\New Dataset\Known ID`
- Manifest CSV: `insightface_demo_assets/known_face_facility_manifest.csv`
- Embedding CSV: `insightface_demo_assets/runtime/known_face_facility_embeddings.csv`

If the manifest is missing or empty but the Known DB root contains images, runtime now auto-discovers manifest rows from that root instead of silently falling back to an old demo gallery.

## Build Command

```cmd
cd /d "C:\Users\Admin\AppData\Local\Temp\doantn-p0-phase"
set "KNOWN_DB_ROOT=D:\ĐỒ ÁN TỐT NGHIỆP\New Dataset\Known ID"
"D:\ĐỒ ÁN TỐT NGHIỆP\.venv_insightface_demo\Scripts\python.exe" ".\insightface_demo_assets\runtime\run_build_known_facility_db.py" --project-root "." --known-root "%KNOWN_DB_ROOT%" --manifest-csv ".\insightface_demo_assets\known_face_facility_manifest.csv" --embeddings-csv ".\insightface_demo_assets\runtime\known_face_facility_embeddings.csv" --summary-json ".\outputs\evaluations\known_facility_db\known_db_build_summary.json"
```

## Generated Files

- Manifest: `insightface_demo_assets/known_face_facility_manifest.csv`
- Embedding CSV: `insightface_demo_assets/runtime/known_face_facility_embeddings.csv`
- Grayscale aligned face crops: `insightface_demo_assets/runtime/known_facility_grayscale_faces`
- Build summary: `outputs/evaluations/known_facility_db/known_db_build_summary.json`

Generated embeddings and crops are runtime artifacts. They should not be committed unless there is an explicit repo convention for doing so.

## Matching Policy

- Only `Direction = IN` tracks are sent to Known DB matching.
- Face embedding cosine similarity is the primary signal.
- Aligned grayscale face similarity is an auxiliary audit signal only.
- Body Re-ID keeps color. Do not convert body crops or the full dataset to grayscale.
- If a known match passes threshold, the track is labeled `KNOWN` and does not create an Unknown stranger profile.
- If no known match passes threshold, the track continues into the Unknown ID and cross-camera association flow.

## Current Build Result

The current Known DB build for the New Dataset demo loaded:

- `known_001`
- `known_002`
- `known_003`

The validation run produced embeddings and aligned grayscale face crops for all 12 currently indexed images.
