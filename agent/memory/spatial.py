"""2D hierarchical grid math: point/box -> cell, parent/children, neighbors."""

from __future__ import annotations

import re

from agent.memory.memory_config import MemoryConfig

_CELL_RE = re.compile(r"^r(\d+)c(\d+)$")


class SpatialIndex:
    """Pure grid math over the configured spatial levels (L0 finest .. Ln coarsest)."""

    def __init__(self, config: MemoryConfig):
        self.frame_width = config.frame_width
        self.frame_height = config.frame_height
        self._levels = [(lv.cols, lv.rows) for lv in config.spatial_levels]
        self._names = [lv.name for lv in config.spatial_levels]

    @property
    def n_levels(self) -> int:
        return len(self._levels)

    def grid(self, level: int) -> tuple[int, int]:
        return self._levels[level]

    def name(self, level: int) -> str:
        return self._names[level]

    @staticmethod
    def level_label(level: int) -> str:
        return f"L{level}"

    def level_index(self, label_or_name: str) -> int:
        for i in range(self.n_levels):
            if label_or_name == f"L{i}" or label_or_name == self._names[i]:
                return i
        raise KeyError(f"unknown spatial level: {label_or_name!r}")

    @staticmethod
    def cell_id(level: int, col: int, row: int) -> str:
        return f"r{row}c{col}"

    @staticmethod
    def parse_cell_id(cell_id: str) -> tuple[int, int]:
        """Return ``(col, row)`` for an ``r<row>c<col>`` id."""
        m = _CELL_RE.match(cell_id)
        if not m:
            raise ValueError(f"bad cell id: {cell_id!r}")
        return int(m.group(2)), int(m.group(1))

    def point_to_cell(
        self, x: float, y: float, level: int = 0
    ) -> tuple[int, int, str]:
        """Map view pixels to ``(col, row, id)``; out-of-frame points clamp in."""
        cols, rows = self._levels[level]
        col = min(int(max(x, 0.0) / self.frame_width * cols), cols - 1)
        row = min(int(max(y, 0.0) / self.frame_height * rows), rows - 1)
        return col, row, self.cell_id(level, col, row)

    def box_to_cell(
        self, box: tuple[float, float, float, float] | list | None, level: int = 0
    ) -> tuple[int, int, str] | None:
        """Map a box to a cell via its center; ``None`` stays ``None``."""
        if box is None:
            return None
        cx = (box[0] + box[2]) / 2.0
        cy = (box[1] + box[3]) / 2.0
        return self.point_to_cell(cx, cy, level)

    def cell_bounds(self, level: int, col: int, row: int) -> list[float]:
        cols, rows = self._levels[level]
        return [
            col * self.frame_width / cols,
            row * self.frame_height / rows,
            (col + 1) * self.frame_width / cols,
            (row + 1) * self.frame_height / rows,
        ]

    def parent(
        self, level: int, col: int, row: int
    ) -> tuple[int, int, int, str] | None:
        """Return ``(level, col, row, id)`` of the parent cell, or ``None`` at top."""
        if level + 1 >= self.n_levels:
            return None
        ccols, crows = self._levels[level]
        pcols, prows = self._levels[level + 1]
        pcol = col * pcols // ccols
        prow = row * prows // crows
        return level + 1, pcol, prow, self.cell_id(level + 1, pcol, prow)

    def children(self, level: int, col: int, row: int) -> list[tuple[int, int, str]]:
        """Return ``(col, row, id)`` of every cell one level down; ``[]`` at L0."""
        if level <= 0:
            return []
        ccols, crows = self._levels[level - 1]
        pcols, prows = self._levels[level]
        out = []
        for cr in range(crows):
            if cr * prows // crows != row:
                continue
            for cc in range(ccols):
                if cc * pcols // ccols != col:
                    continue
                out.append((cc, cr, self.cell_id(level - 1, cc, cr)))
        return out

    def neighbors(
        self,
        level: int,
        col: int,
        row: int,
        radius: int = 1,
        include_self: bool = True,
    ) -> list[tuple[int, int, str]]:
        """Chebyshev ring(s) around a cell, clipped to the grid, row-major."""
        cols, rows = self._levels[level]
        out = []
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if dc == 0 and dr == 0 and not include_self:
                    continue
                cc, cr = col + dc, row + dr
                if 0 <= cc < cols and 0 <= cr < rows:
                    out.append((cc, cr, self.cell_id(level, cc, cr)))
        return out
