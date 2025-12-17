import os
from pathlib import Path
from typing import Callable, Dict, Any

import open3d.visualization.gui as gui

# Prefer relative import; fallback to absolute when running as script
try:
    from .base_edit import GeometryEditWindowBase
except Exception:
    from dialogs.base_edit import GeometryEditWindowBase


class MeshEditWindow(GeometryEditWindowBase):
    def __init__(self, app: gui.Application, title: str, entry: Dict[str, Any], apply_change: Callable[[Dict[str, Any]], None]):
        super().__init__(app, title, entry, apply_change)

    def _build_ui(self):
        # Info section: number of vertices and whether a texture is linked in the mesh file
        try:
            try:
                from services.geometry_loader import load_mesh, find_obj_mtl_texture
            except Exception:
                from ..services.geometry_loader import load_mesh, find_obj_mtl_texture
            m = load_mesh(self._entry.get("path", ""))
            vcount = int(len(m.vertices)) if m is not None else 0
            tex_link = None
            try:
                tex_link = find_obj_mtl_texture(self._entry.get("path", ""))
            except Exception:
                tex_link = None
            info_lines = []
            info_lines.append(f"Vertices: {vcount}")
            if tex_link:
                from pathlib import Path as _P
                info_lines.append(f"Texture linked: Yes ({_P(tex_link).name})")
            else:
                info_lines.append("Texture linked: No")
            self._container.add_child(gui.Label("Info"))
            for ln in info_lines:
                self._container.add_child(gui.Label(ln))
            # Spacer
            self._container.add_child(gui.Label(""))
        except Exception:
            pass

        # Coloring mode selector
        self._container.add_child(gui.Label("Coloring"))
        cmb_mode = gui.Combobox()
        modes = ["Default", "Position (XYZ)"]
        for m in modes:
            cmb_mode.add_item(m)
        current = str(self._entry.get("options", {}).get("color_mode", "default")).lower()
        idx_map = {"default":0, "position":1}
        try:
            cmb_mode.selected_index = idx_map.get(current, 0)
        except Exception:
            cmb_mode.selected_index = 0

        def _on_mode_changed(*_):
            sel = int(getattr(cmb_mode, 'selected_index', 0))
            val = "default" if sel == 0 else "position"
            try:
                self._entry.setdefault("options", {})["color_mode"] = val
            except Exception:
                pass
            self._apply_change(self._entry)

        cmb_mode.set_on_selection_changed(_on_mode_changed)
        self._container.add_child(cmb_mode)

        # Wireframe checkbox (meshes only)
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

        # Scale slider
        self.add_scale_control()

        # Histogram controls: bins and button
        self._container.add_child(gui.Label("Histogram bins"))
        sld_bins = gui.Slider(gui.Slider.INT)
        sld_bins.set_limits(5, 128)
        try:
            sld_bins.int_value = int(self._entry.get("options", {}).get("hist_bins", 32))
        except Exception:
            sld_bins.int_value = 32

        def _on_bins_changed(_val):
            try:
                self._entry.setdefault("options", {})["hist_bins"] = int(sld_bins.int_value)
            except Exception:
                pass
            # No need to re-render scene; bins affect histogram window only

        sld_bins.set_on_value_changed(_on_bins_changed)
        self._container.add_child(sld_bins)

        btn_hist = gui.Button("Show Histogram")
        def _on_hist():
            try:
                # Lazy import to avoid cyclic dependencies
                try:
                    from .histogram_window import open_histogram_window
                except Exception:
                    from dialogs.histogram_window import open_histogram_window
                bins = int(self._entry.get("options", {}).get("hist_bins", 32))
                try:
                    print(f"[MeshEdit] Opening histogram: path={self._entry.get('path','')}, bins={bins}")
                except Exception:
                    pass
                # Keep a reference so the window is not garbage-collected
                self._hist_window = open_histogram_window(self._app, self._win, self._entry, bins)
            except Exception as e:
                try:
                    print(f"[MeshEdit] Failed to open histogram window: {e}")
                except Exception:
                    pass
        btn_hist.set_on_clicked(_on_hist)
        self._container.add_child(btn_hist)

        # Texture selection
        self._container.add_child(gui.Label("Texture"))
        cmb = gui.Combobox()
        try:
            from services.geometry_loader import find_textures
        except Exception:
            from ..services.geometry_loader import find_textures  # if running as module
        tex_list = []
        try:
            tex_list = find_textures(self._entry.get("path", ""))
        except Exception:
            tex_list = []
        # Populate: None + discovered textures
        cmb.add_item("None")
        for tp in tex_list:
            cmb.add_item(Path(tp).name)
        # Select current texture if present
        cur_tex = self._entry.get("options", {}).get("texture", None)
        try:
            if not cur_tex:
                cmb.selected_index = 0
            else:
                # match by basename
                names = ["None"] + [Path(t).name for t in tex_list]
                idx = names.index(Path(cur_tex).name) if Path(cur_tex).name in names else 0
                cmb.selected_index = idx
        except Exception:
            cmb.selected_index = 0

        def _on_tex_changed(*_args):
            sel = cmb.selected_index
            try:
                if sel <= 0:
                    self._entry.setdefault("options", {})["texture"] = None
                else:
                    self._entry.setdefault("options", {})["texture"] = tex_list[sel - 1]
            except Exception:
                pass
            self._apply_change(self._entry)

        cmb.set_on_selection_changed(_on_tex_changed)
        self._container.add_child(cmb)

        # Texture enable/disable checkbox
        chk_tex = gui.Checkbox("Enable Texture")
        try:
            chk_tex.checked = bool(self._entry.get("options", {}).get("texture_enabled", True))
        except Exception:
            chk_tex.checked = True

        def _on_tex_enable(_checked):
            try:
                self._entry.setdefault("options", {})["texture_enabled"] = bool(chk_tex.checked)
            except Exception:
                pass
            self._apply_change(self._entry)

        chk_tex.set_on_checked(_on_tex_enable)
        self._container.add_child(chk_tex)

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
