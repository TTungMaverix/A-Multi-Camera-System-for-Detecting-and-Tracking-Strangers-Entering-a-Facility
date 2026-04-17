from calibration_editor import build_shape_catalog, commit_draft_shape, delete_shape, undo_last_draft_point


def _base_camera_cfg():
    return {
        "camera_id": "C5",
        "processing_roi": {"polygon": []},
        "entry_line": {"points": [], "in_side_point": []},
        "zones": [],
        "subzones": [],
        "default_zone_id": "",
        "default_subzone_id": "",
    }


def test_commit_processing_roi_and_undo_last_point():
    draft = [[0.1, 0.2], [0.7, 0.2], [0.7, 0.9]]
    assert undo_last_draft_point(draft) == [[0.1, 0.2], [0.7, 0.2]]
    cfg = commit_draft_shape(_base_camera_cfg(), shape_type="processing_roi", draft_points=draft)
    assert cfg["processing_roi"]["polygon"] == draft


def test_commit_zone_and_delete_shape_cascades_subzones():
    cfg = commit_draft_shape(
        _base_camera_cfg(),
        shape_type="zone",
        draft_points=[[0.1, 0.1], [0.8, 0.1], [0.8, 0.8]],
        shape_id="c5_entry_main",
        kind_value="entry",
    )
    cfg = commit_draft_shape(
        cfg,
        shape_type="subzone",
        draft_points=[[0.2, 0.2], [0.7, 0.2], [0.7, 0.6]],
        shape_id="c5_inner_overlap",
        kind_value="overlap",
        parent_zone_id="c5_entry_main",
    )
    updated = delete_shape(cfg, shape_type="zone", shape_id="c5_entry_main")
    assert updated["zones"] == []
    assert updated["subzones"] == []
    assert updated["default_zone_id"] == ""
    assert updated["default_subzone_id"] == ""


def test_build_shape_catalog_lists_existing_shapes():
    cfg = {
        "processing_roi": {"polygon": [[0.1, 0.2], [0.7, 0.2], [0.7, 0.9]]},
        "entry_line": {"points": [[0.3, 0.4], [0.8, 0.5]], "in_side_point": [0.6, 0.8]},
        "zones": [{"zone_id": "c6_entry_main", "zone_type": "entry", "polygon": [[0.1, 0.1], [0.8, 0.1], [0.8, 0.8]]}],
        "subzones": [{"subzone_id": "c6_inner_overlap", "subzone_type": "overlap", "parent_zone_id": "c6_entry_main", "polygon": [[0.2, 0.2], [0.7, 0.2], [0.7, 0.6]]}],
    }
    catalog = build_shape_catalog(cfg)
    assert [item["shape_type"] for item in catalog] == ["processing_roi", "entry_line", "zone", "subzone"]
    assert catalog[2]["shape_id"] == "c6_entry_main"
    assert catalog[3]["parent_zone_id"] == "c6_entry_main"
