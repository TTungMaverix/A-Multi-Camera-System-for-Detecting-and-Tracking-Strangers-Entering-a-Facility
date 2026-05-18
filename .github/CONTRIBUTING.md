## Branch policy

- `main`: production-ready code and validated thesis demo configuration only.
- Do not create a new branch when the change is only a small config or threshold adjustment.
- Small config or threshold changes may be committed directly to `main` after validation.
- Larger code changes must use exactly one clearly named branch such as `fix/<issue>` or `feat/<feature>`.
- Do not commit `New Dataset/`, `models/`, `outputs/`, videos, generated crops, generated embeddings, or cache files.
- Do not fake metrics. Every reported number must include the command, config, output path, and observed result. If it was not measured, report `NOT_MEASURED` or `NOT_PROVEN`.
