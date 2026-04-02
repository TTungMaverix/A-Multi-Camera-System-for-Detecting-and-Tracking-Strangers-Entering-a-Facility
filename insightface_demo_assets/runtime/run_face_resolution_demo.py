import csv
import json
import math
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault('ALBUMENTATIONS_DISABLE_VERSION_CHECK', '1')
os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')
os.environ.setdefault('PYTHONUTF8', '1')

import cv2
import numpy as np
from insightface.app import FaceAnalysis

CONFIG_DEFAULT = Path(__file__).with_name('face_demo_config.json')


def load_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def read_csv(path: Path):
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def cosine_similarity(v1, v2):
    denom = float(np.linalg.norm(v1) * np.linalg.norm(v2))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(v1, v2) / denom)


def normalize(vec):
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-12:
        return vec
    return vec / norm


def choose_best_face(faces):
    def score(face):
        bbox = np.asarray(face.bbox).astype(float)
        area = max(1.0, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
        det = float(face.det_score or 0.0)
        return area * max(det, 0.01)
    return max(faces, key=score)


def load_image_unicode(image_path: Path):
    try:
        data = np.fromfile(str(image_path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def extract_embedding_from_image(app, image_path: Path):
    result = {
        'status': 'missing',
        'embedding': None,
        'face_count': 0,
        'det_score': 0.0,
        'bbox': '',
        'message': '',
    }
    if not image_path or not image_path.exists():
        result['message'] = f'missing_image:{image_path}'
        return result
    img = load_image_unicode(image_path)
    if img is None:
        result['message'] = f'failed_to_read:{image_path}'
        return result
    faces = app.get(img)
    result['face_count'] = len(faces)
    if not faces:
        result['status'] = 'no_face'
        result['message'] = 'no_face_detected'
        return result
    face = choose_best_face(faces)
    emb = np.asarray(face.normed_embedding, dtype=np.float32)
    bbox = [int(round(x)) for x in np.asarray(face.bbox).tolist()]
    result.update({
        'status': 'ok',
        'embedding': emb,
        'det_score': float(face.det_score or 0.0),
        'bbox': json.dumps(bbox),
        'message': 'ok',
    })
    return result


def enroll_demo_authorized_identities(app, queue_rows, known_root: Path, count: int = 2):
    counts = Counter(row['global_gt_id'] for row in queue_rows)
    ordered_rows = sorted(
        queue_rows,
        key=lambda row: (
            -counts[row['global_gt_id']],
            float(row.get('relative_sec') or 0.0),
            row.get('camera_id') or '',
        ),
    )
    selected = []
    used_gt = set()
    for row in ordered_rows:
        gt_id = row['global_gt_id']
        if gt_id in used_gt:
            continue
        for crop_key in ('best_head_crop', 'best_body_crop'):
            crop_path = Path(row[crop_key])
            emb = extract_embedding_from_image(app, crop_path)
            if emb['status'] != 'ok':
                continue
            identity_id = f'known_demo_gt_{int(gt_id):04d}'
            display_name = f'Authorized Demo GT {gt_id}'
            identity_dir = known_root / identity_id
            identity_dir.mkdir(parents=True, exist_ok=True)
            ext = crop_path.suffix or '.png'
            dest_path = identity_dir / f'seed_{crop_key}{ext}'
            shutil.copy2(crop_path, dest_path)
            selected.append({
                'identity_id': identity_id,
                'display_name': display_name,
                'source_repo_path': str(crop_path),
                'gallery_rel_path': str(dest_path.relative_to(known_root.parent)),
                'seed_type': 'wildtrack_entry_event',
                'status': 'demo_authorized_seed',
                'notes': f"Auto-enrolled from {row['event_id']} using {crop_key}",
            })
            used_gt.add(gt_id)
            break
        if len(selected) >= count:
            break
    return selected


def build_gallery_embeddings(app, manifest_rows, base_dir: Path, output_csv: Path):
    per_image_rows = []
    per_identity_vectors = defaultdict(list)
    for row in manifest_rows:
        gallery_rel_path = row['gallery_rel_path']
        image_path = base_dir / gallery_rel_path
        emb = extract_embedding_from_image(app, image_path)
        out_row = {
            'identity_id': row['identity_id'],
            'display_name': row['display_name'],
            'image_rel_path': gallery_rel_path,
            'embedding_status': emb['status'],
            'embedding_dim': len(emb['embedding']) if emb['embedding'] is not None else '',
            'model_name': 'buffalo_l',
            'embedding_json': json.dumps(emb['embedding'].tolist()) if emb['embedding'] is not None else '',
            'notes': emb['message'],
        }
        per_image_rows.append(out_row)
        if emb['embedding'] is not None:
            per_identity_vectors[row['identity_id']].append(emb['embedding'])
    write_csv(output_csv, per_image_rows, [
        'identity_id', 'display_name', 'image_rel_path', 'embedding_status', 'embedding_dim', 'model_name', 'embedding_json', 'notes'
    ])

    identity_means = {}
    for identity_id, vectors in per_identity_vectors.items():
        mean_vec = normalize(np.mean(np.stack(vectors, axis=0), axis=0).astype(np.float32))
        identity_means[identity_id] = mean_vec
    return identity_means, per_image_rows


def analyze_events(app, queue_rows):
    analyzed = []
    for row in sorted(queue_rows, key=lambda r: (float(r.get('relative_sec') or 0.0), r.get('camera_id') or '')):
        best = None
        used_crop = ''
        for crop_key in ('best_head_crop', 'best_body_crop'):
            crop_path = Path(row[crop_key])
            emb = extract_embedding_from_image(app, crop_path)
            if best is None:
                best = emb.copy()
                best['crop_key'] = crop_key
                best['crop_path'] = str(crop_path)
            if emb['status'] == 'ok':
                best = emb.copy()
                best['crop_key'] = crop_key
                best['crop_path'] = str(crop_path)
                used_crop = crop_key
                break
        analyzed.append({
            'row': row,
            'embedding': best['embedding'] if best else None,
            'embedding_status': best['status'] if best else 'missing',
            'face_count': best['face_count'] if best else 0,
            'det_score': best['det_score'] if best else 0.0,
            'bbox': best['bbox'] if best else '',
            'used_crop': used_crop or (best['crop_key'] if best else ''),
            'used_crop_path': best['crop_path'] if best else '',
            'message': best['message'] if best else 'missing',
        })
    return analyzed


def assign_identities(analyzed_events, identity_means, threshold: float, unknown_prefix: str, unknown_start: int):
    by_gt = defaultdict(list)
    for item in analyzed_events:
        by_gt[item['row']['global_gt_id']].append(item)

    assignments = {}
    unknown_index = unknown_start
    for gt_id, items in by_gt.items():
        best_known = None
        for item in items:
            emb = item['embedding']
            if emb is None or not identity_means:
                continue
            for identity_id, ref_vec in identity_means.items():
                score = cosine_similarity(emb, ref_vec)
                if best_known is None or score > best_known['score']:
                    best_known = {
                        'identity_id': identity_id,
                        'score': score,
                        'event_id': item['row']['event_id'],
                        'used_crop': item['used_crop'],
                    }
        if best_known and best_known['score'] >= threshold:
            assignments[gt_id] = {
                'identity_status': 'known',
                'matched_known_id': best_known['identity_id'],
                'matched_known_score': f"{best_known['score']:.4f}",
                'unknown_global_id': '',
                'resolution_source': f"face_match:{best_known['used_crop']}",
                'decision_reason': 'best_known_match_over_threshold',
                'representative_event_id': best_known['event_id'],
            }
        else:
            assignments[gt_id] = {
                'identity_status': 'unknown',
                'matched_known_id': '',
                'matched_known_score': f"{best_known['score']:.4f}" if best_known else '',
                'unknown_global_id': f"{unknown_prefix}_{unknown_index:04d}",
                'resolution_source': 'unknown_assignment',
                'decision_reason': 'no_known_match_over_threshold',
                'representative_event_id': items[0]['row']['event_id'],
            }
            unknown_index += 1
    return assignments


def build_resolved_events(analyzed_events, assignments):
    rows = []
    for item in analyzed_events:
        row = item['row']
        assignment = assignments[row['global_gt_id']]
        rows.append({
            'event_id': row['event_id'],
            'camera_id': row['camera_id'],
            'frame_id': row['frame_id'],
            'relative_sec': row['relative_sec'],
            'global_gt_id': row['global_gt_id'],
            'best_head_crop': row['best_head_crop'],
            'best_body_crop': row['best_body_crop'],
            'identity_status': assignment['identity_status'],
            'matched_known_id': assignment['matched_known_id'],
            'matched_known_score': assignment['matched_known_score'],
            'unknown_global_id': assignment['unknown_global_id'],
            'resolved_global_id': assignment['matched_known_id'] or assignment['unknown_global_id'],
            'resolution_source': assignment['resolution_source'],
            'decision_reason': assignment['decision_reason'],
            'embedding_status': item['embedding_status'],
            'face_count': item['face_count'],
            'det_score': f"{item['det_score']:.4f}" if item['det_score'] else '',
            'used_crop': item['used_crop'],
            'used_crop_path': item['used_crop_path'],
            'face_bbox': item['bbox'],
        })
    return rows


def build_unknown_profiles(resolved_rows):
    first_unknown = {}
    for row in resolved_rows:
        unk = row['unknown_global_id']
        if not unk or unk in first_unknown:
            continue
        first_unknown[unk] = {
            'unknown_global_id': unk,
            'source_event_id': row['event_id'],
            'camera_id': row['camera_id'],
            'created_at_frame': row['frame_id'],
            'created_at_sec': row['relative_sec'],
            'seed_body_crop': row['best_body_crop'],
            'seed_head_crop': row['best_head_crop'],
            'body_feature_status': 'pending',
            'face_feature_status': 'embedded' if row['embedding_status'] == 'ok' else 'no_face',
            'notes': 'Demo profile seeded from Wildtrack entry event',
        }
    return list(first_unknown.values())


def build_stream_timeline(tracks_csv: Path, assignments, output_csv: Path):
    track_rows = read_csv(tracks_csv)
    out_rows = []
    for row in track_rows:
        gt_id = row['global_gt_id']
        if gt_id not in assignments:
            continue
        assignment = assignments[gt_id]
        out_rows.append({
            'resolved_global_id': assignment['matched_known_id'] or assignment['unknown_global_id'],
            'identity_status': assignment['identity_status'],
            'matched_known_id': assignment['matched_known_id'],
            'unknown_global_id': assignment['unknown_global_id'],
            'global_gt_id': gt_id,
            'camera_id': row['camera_id'],
            'frame_id': row['frame_id'],
            'relative_sec': row['relative_sec'],
            'bbox_xmin': row['xmin'],
            'bbox_ymin': row['ymin'],
            'bbox_xmax': row['xmax'],
            'bbox_ymax': row['ymax'],
            'stream_unification_basis': 'wildtrack_global_gt_id_demo_bridge',
        })
    write_csv(output_csv, out_rows, [
        'resolved_global_id', 'identity_status', 'matched_known_id', 'unknown_global_id', 'global_gt_id',
        'camera_id', 'frame_id', 'relative_sec', 'bbox_xmin', 'bbox_ymin', 'bbox_xmax', 'bbox_ymax',
        'stream_unification_basis'
    ])
    return out_rows


def main(config_path: Path):
    config = load_json(config_path)
    base_dir = config_path.parents[1]
    known_root = Path(config['known_face_gallery_root'])
    queue_csv = Path(config['wildtrack_identity_queue_csv'])
    known_manifest_csv = Path(config['known_face_manifest_csv'])
    known_embeddings_csv = Path(config['known_face_embeddings_csv'])
    resolved_events_csv = Path(config['resolved_events_csv'])
    unknown_profiles_csv = Path(config['unknown_profiles_csv'])
    stream_timeline_csv = resolved_events_csv.with_name('stream_identity_timeline.csv')
    summary_json = resolved_events_csv.with_name('face_resolution_summary.json')
    runtime_manifest_csv = known_manifest_csv.with_name('known_face_manifest_runtime.csv')

    queue_rows = read_csv(queue_csv)
    base_manifest_rows = read_csv(known_manifest_csv)

    app = FaceAnalysis(
        name=config['insightface_runtime'].get('recommended_model_name', 'buffalo_l'),
        root=str(Path(config['insightface_runtime'].get('recommended_model_root', 'C:/Users/Admin/.insightface'))),
        providers=['CPUExecutionProvider'],
    )
    app.prepare(ctx_id=-1, det_size=(640, 640))

    auto_enrolled = enroll_demo_authorized_identities(app, queue_rows, known_root, count=2)
    manifest_rows = base_manifest_rows + auto_enrolled
    write_csv(runtime_manifest_csv, manifest_rows, [
        'identity_id', 'display_name', 'source_repo_path', 'gallery_rel_path', 'seed_type', 'status', 'notes'
    ])

    identity_means, gallery_rows = build_gallery_embeddings(app, manifest_rows, base_dir, known_embeddings_csv)
    analyzed_events = analyze_events(app, queue_rows)
    assignments = assign_identities(
        analyzed_events,
        identity_means,
        threshold=float(config['matching'].get('known_match_threshold', 0.65)),
        unknown_prefix=config['unknown_handling'].get('seed_prefix', 'UNK'),
        unknown_start=int(config['unknown_handling'].get('start_index', 1)),
    )
    resolved_rows = build_resolved_events(analyzed_events, assignments)
    write_csv(resolved_events_csv, resolved_rows, list(resolved_rows[0].keys()) if resolved_rows else [
        'event_id', 'camera_id', 'frame_id', 'relative_sec', 'global_gt_id', 'best_head_crop', 'best_body_crop',
        'identity_status', 'matched_known_id', 'matched_known_score', 'unknown_global_id', 'resolved_global_id',
        'resolution_source', 'decision_reason', 'embedding_status', 'face_count', 'det_score', 'used_crop',
        'used_crop_path', 'face_bbox'
    ])

    unknown_profiles = build_unknown_profiles(resolved_rows)
    write_csv(unknown_profiles_csv, unknown_profiles, [
        'unknown_global_id', 'source_event_id', 'camera_id', 'created_at_frame', 'created_at_sec', 'seed_body_crop',
        'seed_head_crop', 'body_feature_status', 'face_feature_status', 'notes'
    ])

    tracks_csv = queue_csv.parents[1] / 'tracks' / 'all_tracks_filtered.csv'
    timeline_rows = build_stream_timeline(tracks_csv, assignments, stream_timeline_csv)

    known_count = sum(1 for item in assignments.values() if item['identity_status'] == 'known')
    unknown_count = sum(1 for item in assignments.values() if item['identity_status'] == 'unknown')
    embedded_event_count = sum(1 for row in resolved_rows if row['embedding_status'] == 'ok')
    summary = {
        'queue_events': len(queue_rows),
        'resolved_events': len(resolved_rows),
        'gallery_identities_runtime': len({row['identity_id'] for row in manifest_rows}),
        'gallery_images_runtime': len(manifest_rows),
        'auto_enrolled_demo_identities': [row['identity_id'] for row in auto_enrolled],
        'known_assignments': known_count,
        'unknown_assignments': unknown_count,
        'events_with_face_embedding': embedded_event_count,
        'stream_timeline_rows': len(timeline_rows),
        'stream_unification_basis': 'wildtrack_global_gt_id_demo_bridge',
        'notes': [
            'This demo uses InsightFace for face detection and embedding on entry-event crops.',
            'To demonstrate the Known branch, a small authorized gallery is auto-enrolled from a few Wildtrack entry events.',
            'Cross-stream identity propagation in this demo is bridged by Wildtrack global_gt_id; replace this with body re-id association for a full deployment pipeline.'
        ]
    }
    save_json(summary_json, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    config_path = Path(os.environ.get('FACE_DEMO_CONFIG', str(CONFIG_DEFAULT)))
    main(config_path)
