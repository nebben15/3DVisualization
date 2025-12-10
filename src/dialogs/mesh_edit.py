import os
from pathlib import Path
from typing import Callable, Dict, Any

import open3d.visualization.gui as gui


class MeshEditWindow:
    def __init__(self, app: gui.Application, title: str, entry: Dict[str, Any], apply_change: Callable[[Dict[str, Any]], None]):
        self._app = app
        self._entry = entry
        self._apply_change = apply_change
        self._win = app.create_window(title, 380, 280)
        self._container = gui.Vert(6, gui.Margins(8, 8, 8, 8))
        self._build_ui()
        self._win.add_child(self._container)

    def _build_ui(self):
        # Wireframe checkbox for meshes
        chk = gui.Checkbox("Wireframe")
        try:
            chk.checked = bool(self._entry.get("options", {}).get("wireframe", False))
        except Exception:
            pass

        def _on_chk(_checked):
            try:
                self._entry.setdefault("options", {})["wireframe"] = bool(chk.checked)
            except Exception:
                pass
            # Apply change immediately
            self._apply_change(self._entry)

        chk.set_on_checked(_on_chk)
        self._container.add_child(chk)

    # No extra Close button; window has a native close control.


def open_mesh_edit_window(app: gui.Application, parent_window: gui.Window, entry: Dict[str, Any], apply_change: Callable[[Dict[str, Any]], None]):
    """
    Opens a movable mesh edit window.

    Parameters:
        app: gui.Application.instance
        parent_window: parent window (unused except for API symmetry)
        entry: geometry entry dict {display, path, type, options}
        apply_change: callback invoked after any change to entry options
    """
    title = f"Edit: {Path(entry['path']).name}"
    # Returns an instance to keep a reference alive, but caller may ignore
    return MeshEditWindow(app, title, entry, apply_change)
