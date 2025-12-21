from typing import Dict, Any, List, Tuple

import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path

# Prefer relative imports for services
try:
    from ..services.geometry_loader import read_point_positions_fast, read_point_features_fast, load_mesh
except Exception:
    from services.geometry_loader import read_point_positions_fast, read_point_features_fast, load_mesh


def _fig_to_o3d_image(fig: matplotlib.figure.Figure) -> o3d.geometry.Image:
    """Convert a Matplotlib figure into an Open3D Image robustly across backends."""
    fig.canvas.draw()
    try:
        # Preferred: RGBA buffer
        w, h = fig.canvas.get_width_height()
        buf = fig.canvas.buffer_rgba()
        # Some backends return a memoryview; convert to bytes buffer first
        if isinstance(buf, (memoryview, bytearray)):
            buf = bytes(buf)
        arr = np.frombuffer(buf, dtype=np.uint8)
        # Ensure expected size
        expected = h * w * 4
        if arr.size != expected:
            # Resize defensively
            arr = arr[:expected]
        arr = arr.reshape(h, w, 4)
        rgb = arr[:, :, :3]
        rgb = np.ascontiguousarray(rgb)
        return o3d.geometry.Image(rgb)
    except Exception as e:
        # Fallback: try tostring_rgb if available
        try:
            w, h = fig.canvas.get_width_height()
            buf = fig.canvas.tostring_rgb()
            if isinstance(buf, (memoryview, bytearray)):
                buf = bytes(buf)
            img_arr = np.frombuffer(buf, dtype=np.uint8)
            img_arr = img_arr.reshape(h, w, 3)
            img_arr = np.ascontiguousarray(img_arr)
            return o3d.geometry.Image(img_arr)
        except Exception as e2:
            try:
                print(f"[Histogram] Figure to image conversion failed: {e} / fallback {e2}")
            except Exception:
                pass
            raise


def _render_hist_image(data: np.ndarray, bins: int, label: str, size: Tuple[int, int] = (800, 300)) -> o3d.geometry.Image:
    """Render a histogram using Matplotlib and convert to Open3D Image."""
    # Use a non-interactive backend for offscreen rendering
    try:
        matplotlib.use("Agg")
    except Exception:
        pass
    fig = plt.figure(figsize=(size[0] / 100.0, size[1] / 100.0), dpi=100)
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(data, bins=bins, color='steelblue', alpha=0.85, edgecolor='black')
    ax.set_title(label)
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    img = _fig_to_o3d_image(fig)
    plt.close(fig)
    return img


def _collect_data_for_entry(entry: Dict[str, Any]) -> Tuple[List[Tuple[str, np.ndarray]], int]:
    """Collect per-dimension arrays and suggest default bins.
    Returns list of (label, 1D ndarray) and default bins.
    For meshes: XYZ from vertices.
    For point clouds: XYZ + features (feat_dim_*).
    """
    dims: List[Tuple[str, np.ndarray]] = []
    default_bins = 32
    path = str(entry.get("path", ""))
    typ = str(entry.get("type", "")).lower()
    def _to_2d_float(arr, cols=3):
        A = np.asarray(arr)
        if A.ndim == 2 and A.shape[1] >= cols and np.issubdtype(A.dtype, np.number):
            return A.astype(np.float32, copy=False)
        # Try to coerce a nested/object array into numeric 2D
        try:
            A2 = np.vstack([np.array(x).ravel() for x in A]).astype(np.float32, copy=False)
            return A2
        except Exception:
            return np.array([], dtype=np.float32).reshape(0, cols)

    if typ == "pcd":
        # Positions
        P = read_point_positions_fast(path)
        P = _to_2d_float(P, cols=3) if P is not None else np.array([], dtype=np.float32).reshape(0, 3)
        if P.ndim == 2 and P.shape[1] >= 3 and P.size > 0:
            dims.append(("X (pos)", P[:, 0]))
            dims.append(("Y (pos)", P[:, 1]))
            dims.append(("Z (pos)", P[:, 2]))
        # Features
        F = read_point_features_fast(path)
        if F is not None:
            F = np.asarray(F)
            # Inspect header to decide labeling (RGB vs feat dims)
            has_rgb = False
            has_feats3 = False
            try:
                from ..services.geometry_loader import inspect_ply_header_modes as _insp
            except Exception:
                from services.geometry_loader import inspect_ply_header_modes as _insp
            try:
                has_rgb, has_feats3 = _insp(path)
            except Exception:
                has_rgb, has_feats3 = (False, False)
            if F.ndim == 2 and F.shape[1] > 0:
                if has_rgb and not has_feats3 and F.shape[1] == 3:
                    labels = ["red", "green", "blue"]
                    for j in range(3):
                        dims.append((labels[j], F[:, j]))
                else:
                    for j in range(F.shape[1]):
                        dims.append((f"feat_dim_{j}", F[:, j]))
            elif F.ndim == 1 and F.size > 0:
                # Single channel
                label = "red" if has_rgb and not has_feats3 else "feat_dim_0"
                dims.append((label, F.ravel()))
    else:
        # Mesh: use vertex positions only
        mesh = load_mesh(path)
        if mesh is not None:
            V = _to_2d_float(np.asarray(mesh.vertices), cols=3)
            if V.ndim == 2 and V.shape[1] >= 3 and V.size > 0:
                dims.append(("X (pos)", V[:, 0]))
                dims.append(("Y (pos)", V[:, 1]))
                dims.append(("Z (pos)", V[:, 2]))
    return dims, default_bins


def open_histogram_window(app: gui.Application, parent_window: gui.Window, entry: Dict[str, Any], bins: int = 32):
    # Debug: start opening
    try:
        print(f"[Histogram] Request open: path={entry.get('path','')}, bins={bins}, type={entry.get('type','')}")
    except Exception:
        pass
    try:
        title = f"Histogram: {Path(entry.get('path', '')).name}"
    except Exception:
        title = "Histogram"
    win = app.create_window(title, 920, 720)
    # Scrollable container for multiple histograms
    vert = gui.Vert(8, gui.Margins(8, 8, 8, 8))
    dims, def_bins = _collect_data_for_entry(entry)
    try:
        print(f"[Histogram] Collected dims: {len(dims)} (default_bins={def_bins})")
    except Exception:
        pass
    bins = int(bins) if isinstance(bins, (int, float)) else def_bins
    bins = max(2, min(256, bins))
    # Build images
    if not dims:
        vert.add_child(gui.Label("No data available to plot."))
    else:
        for label, arr in dims:
            try:
                data = np.asarray(arr).astype(np.float32, copy=False).ravel()
                img = _render_hist_image(data, bins, label)
                # Label + image
                vert.add_child(gui.Label(label))
                vert.add_child(gui.ImageWidget(img))
            except Exception as e:
                try:
                    print(f"[Histogram] Failed to render histogram for '{label}': {e}")
                except Exception:
                    pass
                vert.add_child(gui.Label(f"Failed to render histogram: {label}"))
    # Some Open3D builds may lack ScrollView; add content directly
    try:
        scroll = gui.ScrollView()
        scroll.set_content(vert)
        win.add_child(scroll)
    except Exception as e:
        try:
            print(f"[Histogram] ScrollView unavailable, adding content directly: {e}")
        except Exception:
            pass
        win.add_child(vert)
    # Keep a global reference to prevent GC
    # Register window by path for later cleanup
    try:
        register_hist_window(entry.get('path', ''), win)
    except Exception:
        pass
    return win


# --- Registry helpers for cleanup ---
# Initialize registry for histogram windows keyed by geometry path
_HIST_REGISTRY = {}

def register_hist_window(path: str, win: gui.Window):
    key = str(path or "")
    lst = _HIST_REGISTRY.get(key, [])
    lst.append(win)
    _HIST_REGISTRY[key] = lst
    # Ensure cleanup when user closes the window manually
    try:
        def _on_close():
            try:
                # Remove this window from the registry list
                cur = _HIST_REGISTRY.get(key, [])
                _HIST_REGISTRY[key] = [w for w in cur if w is not win]
            except Exception:
                pass
            return True
        win.set_on_close(_on_close)
    except Exception:
        pass

def close_hist_windows_for(path: str):
    key = str(path or "")
    wins = _HIST_REGISTRY.pop(key, [])
    for w in wins:
        try:
            # Close via window API on the UI thread
            gui.Application.instance.post_to_main_thread(w, lambda win=w: win.close())
        except Exception:
            pass

def close_all_hist_windows():
    keys = list(_HIST_REGISTRY.keys())
    for k in keys:
        close_hist_windows_for(k)
