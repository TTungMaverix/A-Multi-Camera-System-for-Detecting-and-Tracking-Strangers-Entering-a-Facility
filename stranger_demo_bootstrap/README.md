# Stranger Demo Bootstrap Pack

Bootstrap pack nay da duoc chinh lai de khop voi moi truong demo da chay duoc tren may hien tai:
- uu tien Python that tai `C:\Users\Admin\.platformio\python3\python.exe`
- dung `onnxruntime` CPU mac dinh
- khong cai `insightface` truc tiep tu PyPI vi flow do da tung ket build C++
- sao chep source `InsightFace` tu repo local `C:\Users\Admin\insightface-master\python-package\insightface`
- patch runtime toi thieu roi cai vao `venv`
- tai su dung model cache `C:\Users\Admin\.insightface\models\buffalo_l` neu da co

Muc tieu cua template nay van bam sat de tai:
- 4 stream camera
- chi xu ly nguoi di vao facility
- check mat voi known DB
- neu khong match thi tao hoac tai su dung `unknown_global_id`
- dong nhat thong tin giua cac stream dua tren topology, travel time, va overlap
- output chinh la `snapshot + timestamp + camera_id + global_id`

## Trang thai pack

Pack nay gio gom:
- `scripts/bootstrap_windows.ps1`: bootstrap Windows phu hop voi may hien tai
- `scripts/bootstrap_unix.sh`: bootstrap Unix voi cung logic template
- `scripts/build_known_db.py`: script tao known face DB embeddings
- `scripts/run_demo.py`: launcher/stub de kiem tra config va khoi tao runtime demo
- `src/`: skeleton module cho config, direction filter, topology, association, API, dashboard
- `tests/`: unit test mau cho topology va association
- `docs/CODEX_IMPLEMENTATION_BRIEF.md`: brief cap nhat theo huong Wildtrack demo + thesis flow
- `docs/CODEX_PROMPT.txt`: prompt ngan de tiep tuc code bang Codex

## Kha nang hien thi demo

Co the lam duoc hai kieu:
- hien thi tren man hinh theo dang `quad-view` 4 camera, overlay `track_id`, `global_id`, `known/unknown`, `direction`
- hien thi tren web qua `FastAPI + Streamlit`, tap trung vao event table, stranger timeline, snapshot, va sau nay co the them khung video preview

Trong pack nay, lop `visualization` da duoc dua vao config va skeleton API/dashboard da co san. Day la lop trinh dien phu tro. Output chinh van la:
- anh capture cua tung stranger
- thoi diem xuat hien
- camera xuat hien
- lich su xuat hien giua cac stream

## Quick start tren Windows

Neu chay ngay trong folder pack:

```powershell
cd D:\ĐỒ ÁN TỐT NGHIỆP\stranger_demo_bootstrap
powershell -ExecutionPolicy Bypass -File .\scripts\bootstrap_windows.ps1 -ProjectRoot .
```

Neu bootstrap sang mot folder moi:

```powershell
powershell -ExecutionPolicy Bypass -File D:\ĐỒ ÁN TỐT NGHIỆP\stranger_demo_bootstrap\scripts\bootstrap_windows.ps1 -ProjectRoot D:\path\to\new-project
```

## Quick start tren Unix

```bash
bash scripts/bootstrap_unix.sh .
```

## Sau bootstrap

1. Bo sung 4 stream trong `config/cameras.yaml`.
2. Dat known face images vao `data/known_db/<person_id>/`.
3. Chinh `config/topology.yaml` theo map/thoi gian di chuyen cua ban.
4. Chay build known DB:
   `.\.venv_insightface_demo\Scripts\python.exe scripts\build_known_db.py`
5. Chay khoi tao demo:
   `.\.venv_insightface_demo\Scripts\python.exe scripts\run_demo.py`
6. Neu muon web/API:
   - `.\.venv_insightface_demo\Scripts\python.exe -m uvicorn src.api.app:app --reload --host 127.0.0.1 --port 8000`
   - `.\.venv_insightface_demo\Scripts\python.exe -m streamlit run src\ui\dashboard.py --server.port 8501`

## Ghi chu quan trong

- Demo mau duoc canh chinh theo 4 camera Wildtrack `cam03`, `cam05`, `cam06`, `cam07` vi day la bo du lieu demo hien tai hop ly nhat.
- `cam05` va `cam06` duoc xem la camera "entry" uu tien cho rule `IN`.
- `cam03` va `cam07` duoc xem la camera follow-up de giu global ID xuyen stream.
- Neu may co NVIDIA/CUDA, ban co the chuyen sang `onnxruntime-gpu`, nhung tren may hien tai chi co Intel Iris Xe nen CPU la lua chon dung.
