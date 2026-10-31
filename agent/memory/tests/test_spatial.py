"""Grid mapping: corners, edges, parent coverage, neighbors."""

from __future__ import annotations

import pytest

from agent.memory import MemoryConfig, SpatialIndex


@pytest.fixture()
def index():
    return SpatialIndex(MemoryConfig.from_yaml())


def test_corners(index):
    assert index.point_to_cell(0, 0) == (0, 0, "r0c0")
    assert index.point_to_cell(799, 0) == (7, 0, "r0c7")
    assert index.point_to_cell(0, 599) == (0, 5, "r5c0")
    assert index.point_to_cell(799.9, 599.9) == (7, 5, "r5c7")
    assert index.point_to_cell(800, 600) == (7, 5, "r5c7")  # far edge clamps in


def test_edges_and_cell_boundaries(index):
    assert index.point_to_cell(99.9, 50)[:2] == (0, 0)
    assert index.point_to_cell(100, 50)[:2] == (1, 0)
    assert index.point_to_cell(550, 350) == (5, 3, "r3c5")  # design-doc example cell
    assert index.point_to_cell(-5, -5)[:2] == (0, 0)


def test_box_to_cell_uses_center(index):
    assert index.box_to_cell([500, 300, 600, 400]) == (5, 3, "r3c5")
    assert index.box_to_cell(None) is None


def test_parent_chain_reaches_full_view(index):
    assert index.parent(0, 5, 3) == (1, 2, 1, "r1c2")
    assert index.parent(1, 2, 1) == (2, 0, 0, "r0c0")
    assert index.parent(2, 0, 0) is None
    for row in range(6):
        for col in range(8):
            lvl, pc, pr, _ = index.parent(0, col, row)
            assert index.parent(lvl, pc, pr) == (2, 0, 0, "r0c0")


def test_parent_coverage(index):
    seen_l0 = set()
    for prow in range(3):
        for pcol in range(4):
            kids = index.children(1, pcol, prow)
            assert len(kids) == 4  # each L1 covers 2x2 L0 cells
            seen_l0.update(cid for _, _, cid in kids)
    assert len(seen_l0) == 48
    assert len(index.children(2, 0, 0)) == 12  # L2 covers all L1 zones
    assert index.children(0, 0, 0) == []


def test_neighbor_counts(index):
    assert len(index.neighbors(0, 3, 2, radius=1)) == 9  # interior
    assert len(index.neighbors(0, 0, 0, radius=1)) == 4  # corner
    assert len(index.neighbors(0, 0, 2, radius=1)) == 6  # edge
    assert len(index.neighbors(0, 3, 2, radius=0)) == 1
    assert len(index.neighbors(0, 3, 2, radius=1, include_self=False)) == 8
    assert len(index.neighbors(0, 3, 2, radius=2)) == 25


def test_cell_bounds(index):
    assert index.cell_bounds(0, 5, 3) == [500, 300, 600, 400]
    assert index.cell_bounds(2, 0, 0) == [0, 0, 800, 600]


def test_level_index_accepts_label_or_name(index):
    assert index.level_index("L0") == 0
    assert index.level_index("cells") == 0
    assert index.level_index("full_view") == 2
    with pytest.raises(KeyError):
        index.level_index("nope")
    with pytest.raises(ValueError):
        index.parse_cell_id("bogus")
