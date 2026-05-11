# Worktree Status Midterm

Date: 2026-05-11

## Main Repository

- Path: `D:\DO AN TOT NGHIEP`
- Branch observed: `fix/dataset-quality-pooling-osnet-face-capture`
- Status: dirty.
- Dirty content includes runtime server/UI files, scene calibration configs, template CSV/JSON outputs, `New Dataset/`, `models/`, and `outputs/`.
- Those dirty changes were not reset, stashed, staged, or committed during the calibration-tool hygiene pass.
- Recommendation: do not continue new feature work directly in the main repository until those changes are reviewed and either committed to their correct branch, stashed, or intentionally discarded by the owner.

## Task 2 Calibration Tool Worktree

- Path: `C:\Users\Admin\AppData\Local\Temp\doantn-p0-phase`
- Branch observed: `feat/roi-zone-calibration-tool`
- Remote branch: `origin/feat/roi-zone-calibration-tool`
- Baseline pushed commit before this hygiene pass: `653e0c5 Add ROI and entry-line calibration tool`.
- Worktree also contains untracked local data/artifacts such as `New Dataset/`, `models/`, `outputs/`, and generated Known DB files. These must remain excluded from source commits unless explicitly versioned later.

## Task 2 Current Readiness

- OpenCV calibration tool supports ROI polygon, entry line, clicked IN side, runtime-compatible YAML merge, and per-camera JSON export.
- Manual GUI smoke output was saved under `outputs/evaluations/manual_calibration_smoke/` in the main repository and was not staged.
- Subzone readiness is implemented as schema-compatible placeholder generation through `--with-default-subzones`.
- Full interactive subzone drawing remains covered by `/calibration.html`; the OpenCV tool intentionally stays lightweight.

## Recommended Next Base

- Continue Task 2 polish from `feat/roi-zone-calibration-tool`.
- Start unrelated future phases from the latest pushed P0 branch or main only after the dirty main repository state is cleaned or reviewed.
