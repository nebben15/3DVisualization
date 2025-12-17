from typing import Callable, Dict, Any

import open3d.visualization.gui as gui


class GeometryEditWindowBase:
    """
    Base class for simple geometry edit dialogs.
    Subclasses should implement _build_ui() and may use helpers like add_scale_control().
    """

    def __init__(self, app: gui.Application, title: str, entry: Dict[str, Any], apply_change: Callable[[Dict[str, Any]], None]):
        self._app = app
        self._entry = entry
        self._apply_change = apply_change
        self._win = app.create_window(title, 380, 260)
        self._container = gui.Vert(6, gui.Margins(8, 8, 8, 8))
        self._build_ui()
        self._win.add_child(self._container)

    def _build_ui(self):
        raise NotImplementedError

    # Helper: add a labeled double slider for scale and wire the change to apply_change
    def add_scale_control(self, min_v: float = 0.1, max_v: float = 3.0):
        self._container.add_child(gui.Label("Scale"))
        sld = gui.Slider(gui.Slider.DOUBLE)
        sld.set_limits(min_v, max_v)
        try:
            sld.double_value = float(self._entry.get("options", {}).get("scale", 1.0))
        except Exception:
            sld.double_value = 1.0

        def _on_scale(_val):
            try:
                self._entry.setdefault("options", {})["scale"] = float(sld.double_value)
            except Exception:
                pass
            self._apply_change(self._entry)

        sld.set_on_value_changed(_on_scale)
        self._container.add_child(sld)
