from dataclasses import dataclass, field
from typing import List, Dict, Literal

GeometryType = Literal["mesh", "pcd"]


@dataclass
class SelectedEntry:
    display: str
    path: str
    type: GeometryType
    options: Dict[str, object] = field(default_factory=lambda: {
        "wireframe": False,
        "scale": 1.0,
        "visible": True,
        "texture": None,
        "texture_enabled": True,
        "color_mode": "default",  # default | rgb | continuous | position (meshes)
    })


class SelectionList:
    def __init__(self) -> None:
        self._items: List[SelectedEntry] = []
        self._sel_index: int = -1

    @property
    def items(self) -> List[SelectedEntry]:
        return self._items

    @property
    def selected_index(self) -> int:
        return self._sel_index

    def set_selected_index(self, idx: int) -> None:
        self._sel_index = idx if 0 <= idx < len(self._items) else -1

    def add(self, entry: SelectedEntry) -> None:
        self._items.append(entry)
        if self._sel_index == -1:
            self._sel_index = 0

    def remove_at(self, idx: int) -> None:
        if 0 <= idx < len(self._items):
            self._items.pop(idx)
            if not self._items:
                self._sel_index = -1
            elif idx >= len(self._items):
                self._sel_index = len(self._items) - 1

    def move(self, idx: int, delta: int) -> None:
        new_idx = idx + delta
        if 0 <= idx < len(self._items) and 0 <= new_idx < len(self._items):
            self._items[idx], self._items[new_idx] = self._items[new_idx], self._items[idx]
            self._sel_index = new_idx

    def update_options(self, idx: int, **opts) -> None:
        if 0 <= idx < len(self._items):
            self._items[idx].options.update(opts)
