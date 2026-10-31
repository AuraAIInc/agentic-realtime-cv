"""Memory config defaults + MEMORY_* env overrides.

Run from repo root:
    .venv/bin/python -m pytest agent/memory/tests -q
"""

from __future__ import annotations

import pytest

from agent.memory import MemoryConfig


def test_config_defaults_match_design_doc():
    cfg = MemoryConfig.from_yaml()
    assert (cfg.version, cfg.view_id) == (1, 3)
    assert (cfg.frame_width, cfg.frame_height) == (800, 600)
    assert [(lv.name, lv.cols, lv.rows) for lv in cfg.spatial_levels] == [
        ("cells", 8, 6),
        ("zones", 4, 3),
        ("full_view", 1, 1),
    ]
    assert [(lv.name, lv.minutes) for lv in cfg.temporal_levels] == [
        ("slot_5min", 5),
        ("hour", 60),
        ("day", 1440),
        ("week", 10080),
    ]
    assert cfg.retention_days == {"slot_5min": 7, "hour": 30, "day": 180, "week": 0}
    assert (cfg.spatial_chebyshev_radius, cfg.temporal_history_days, cfg.semantic_top_k) == (1, 7, 5)
    assert cfg.idle_seconds == 300
    assert (cfg.min_pattern_support, cfg.stable_weight_cap, cfg.flip_threshold, cfg.recent_kept) == (3, 50, 3, 5)
    assert (cfg.queue_warn_depth, cfg.promote_coalesce) == (100, True)
    assert cfg.store_dir == "agent/memory/data"
    assert (cfg.llm_model_role, cfg.max_units_per_call) == ("memory", 8)


def test_config_env_overrides(monkeypatch):
    monkeypatch.setenv("MEMORY_VIEW_ID", "5")
    monkeypatch.setenv("MEMORY_IDLE_SECONDS", "60")
    monkeypatch.setenv("MEMORY_PROMOTE_COALESCE", "false")
    monkeypatch.setenv("MEMORY_STORE_DIR", "/tmp/mem")
    cfg = MemoryConfig.from_yaml()
    assert cfg.view_id == 5 and isinstance(cfg.view_id, int)
    assert cfg.idle_seconds == 60.0
    assert cfg.promote_coalesce is False
    assert cfg.store_dir == "/tmp/mem"
    assert cfg.frame_width == 800  # untouched knobs keep defaults


def test_config_store_path_and_missing_file(tmp_path):
    cfg = MemoryConfig.from_yaml()
    assert cfg.store_path().name == "view_3_long_term.yaml"
    assert cfg.store_path(view_id=7).name == "view_7_long_term.yaml"
    with pytest.raises(FileNotFoundError):
        MemoryConfig.from_yaml(tmp_path / "nope.yaml")
