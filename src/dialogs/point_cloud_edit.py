from pathlib import Path
from typing import Callable, Dict, Any

import open3d.visualization.gui as gui

# Prefer relative import; fallback to absolute when running as script
try:
    from .base_edit import GeometryEditWindowBase
except Exception:
    from dialogs.base_edit import GeometryEditWindowBase


class PointCloudEditWindow(GeometryEditWindowBase):
    def __init__(self, app: gui.Application, title: str, entry: Dict[str, Any], apply_change: Callable[[Dict[str, Any]], None]):
        super().__init__(app, title, entry, apply_change)

    def _build_ui(self):
        # Inspect header to determine available coloring options and info
        has_rgb = False
        has_feats3 = False
        feat_count = 0
        point_count = 0
        p = str(self._entry.get("path", ""))
        if p.lower().endswith(".ply"):
            try:
                try:
                    from services.geometry_loader import inspect_ply_header_modes
                except Exception:
                    from ..services.geometry_loader import inspect_ply_header_modes
                has_rgb, has_feats3 = inspect_ply_header_modes(p)
            except Exception:
                pass
        # Attempt to read positions to determine point count (works for any format)
        try:
            try:
                from services.geometry_loader import read_point_positions_fast
            except Exception:
                from ..services.geometry_loader import read_point_positions_fast
            pos = read_point_positions_fast(p)
            if pos is not None:
                import numpy as np
                PA = np.asarray(pos)
                point_count = int(PA.shape[0])
        except Exception:
            pass
        # Attempt to read features to determine dimension count (works for any format)
        try:
            try:
                from services.geometry_loader import read_point_features_fast
            except Exception:
                from ..services.geometry_loader import read_point_features_fast
            feats = read_point_features_fast(p)
            if feats is not None:
                import numpy as np
                arr = np.asarray(feats)
                if arr.ndim == 2:
                    feat_count = int(arr.shape[1])
                elif arr.ndim == 1:
                    feat_count = 1
        except Exception:
            pass
        # If we detected >=3 feature dims, enable 3D feature coloring
        if feat_count >= 3:
            has_feats3 = True

        # Info section
        info = []
        info.append(f"Points: {point_count}")
        info.append(f"RGB: {'yes' if has_rgb else 'no'}")
        info.append(f"Feature dims: {feat_count}")
        self._container.add_child(gui.Label("Info"))
        for ln in info:
            self._container.add_child(gui.Label(ln))
        # Spacer
        self._container.add_child(gui.Label(""))

        # Coloring mode selector
        self._container.add_child(gui.Label("Coloring"))
        cmb = gui.Combobox()
        # Build modes based on header detection
        modes = ["Default"]
        if has_rgb:
            modes.append("RGB (from header)")
        if has_feats3:
            modes.append("Feature RGB (3D)")
            modes.append("Feature (normalized 3D)")
        for m in modes:
            cmb.add_item(m)
        current = str(self._entry.get("options", {}).get("color_mode", "default")).lower()
        # Map current option to index in dynamically built list
        desired_label = "Default"
        if current == "rgb" and has_rgb:
            desired_label = "RGB (from header)"
        elif current == "rgb" and not has_rgb:
            desired_label = "Default"
        elif current == "continuous" and has_feats3:
            desired_label = "Feature (normalized 3D)"
        try:
            cmb.selected_index = modes.index(desired_label)
        except Exception:
            cmb.selected_index = 0

        def _on_mode_changed(*_):
            sel = int(getattr(cmb, 'selected_index', 0))
            label = modes[sel] if 0 <= sel < len(modes) else "Default"
            if label == "RGB (from header)" and not has_rgb:
                label = "Default"
            if label in ("Feature RGB (3D)", "Feature (normalized 3D)") and not has_feats3:
                label = "Default"
            val = "default"
            if label == "RGB (from header)":
                val = "rgb"
            elif label == "Feature RGB (3D)":
                val = "rgb"
            elif label == "Feature (normalized 3D)":
                val = "continuous"
            try:
                self._entry.setdefault("options", {})["color_mode"] = val
            except Exception:
                pass
            self._apply_change(self._entry)

        cmb.set_on_selection_changed(_on_mode_changed)
        self._container.add_child(cmb)

        # For point clouds: Scale only (texture and wireframe not applicable)
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
        sld_bins.set_on_value_changed(_on_bins_changed)
        self._container.add_child(sld_bins)

        btn_hist = gui.Button("Show Histogram")
        def _on_hist():
            try:
                try:
                    from .histogram_window import open_histogram_window
                except Exception:
                    from dialogs.histogram_window import open_histogram_window
                bins = int(self._entry.get("options", {}).get("hist_bins", 32))
                try:
                    print(f"[PointCloudEdit] Opening histogram: path={self._entry.get('path','')}, bins={bins}")
                except Exception:
                    pass
                # Keep a reference so the window is not garbage-collected
                self._hist_window = open_histogram_window(self._app, self._win, self._entry, bins)
            except Exception as e:
                try:
                    print(f"[PointCloudEdit] Failed to open histogram window: {e}")
                except Exception:
                    pass
        btn_hist.set_on_clicked(_on_hist)
        self._container.add_child(btn_hist)
        # Optional future: per-entry point size
        # lbl = gui.Label("Point Size")
        # self._container.add_child(lbl)
        # sld = gui.Slider(gui.Slider.INT)
        # sld.set_limits(1, 10)
        # try:
        #     sld.int_value = int(self._entry.get("options", {}).get("point_size", 3))
        # except Exception:
        #     sld.int_value = 3
        # def _on_ps(_val):
        #     try:
        #         self._entry.setdefault("options", {})["point_size"] = int(sld.int_value)
        #     except Exception:
        #         pass
        #     self._apply_change(self._entry)
        # sld.set_on_value_changed(_on_ps)
        # self._container.add_child(sld)


def open_pointcloud_edit_window(app: gui.Application, parent_window: gui.Window, entry: Dict[str, Any], apply_change: Callable[[Dict[str, Any]], None]):
    title = f"Edit: {Path(entry['path']).name} (PCD)"
    return PointCloudEditWindow(app, title, entry, apply_change)
