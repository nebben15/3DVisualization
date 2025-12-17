from functools import lru_cache
from pathlib import Path
import open3d as o3d
from typing import Optional, Generator, Tuple
from typing import Optional
import numpy as np
import open3d as o3d


@lru_cache(maxsize=128)
def load_mesh(path: str) -> Optional[o3d.geometry.TriangleMesh]:
    mesh = o3d.io.read_triangle_mesh(path)
    if mesh is None or mesh.is_empty():
        return None
    try:
        mesh.compute_vertex_normals()
    except Exception:
        pass
    return mesh


@lru_cache(maxsize=128)
def load_pointcloud(path: str) -> Optional[o3d.geometry.PointCloud]:
    # Prefer tensor point cloud if available
    try:
        pcd_t = o3d.t.io.read_point_cloud(path)
        if pcd_t is not None and pcd_t.point.positions is not None:
            positions = np.asarray(pcd_t.point.positions)
            pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(positions))
            if "colors" in pcd_t.point:
                colors = np.asarray(pcd_t.point.colors)
                pcd.colors = o3d.utility.Vector3dVector(colors[:, :3] / 255.0 if colors.max() > 1.0 else colors[:, :3])
            return pcd
    except Exception:
        pass
    pcd = o3d.io.read_point_cloud(path)
    if pcd is None or pcd.is_empty():
        return None
    return pcd

def read_point_positions_fast(path: str) -> Optional[np.ndarray]:
    """Return point positions as a NumPy array as quickly as possible (tensor API preferred)."""
    try:
        pcd_t = o3d.t.io.read_point_cloud(path)
        if pcd_t is not None and pcd_t.point.positions is not None:
            try:
                return pcd_t.point.positions.numpy()
            except Exception:
                return np.asarray(pcd_t.point.positions)
    except Exception:
        pass
    try:
        pcd = o3d.io.read_point_cloud(path)
        if pcd is None or pcd.is_empty():
            return None
        P = np.asarray(pcd.points)
        # Ensure numeric 2D
        if P.ndim == 1:
            try:
                P = np.vstack([np.array(x).ravel() for x in P]).astype(np.float32, copy=False)
            except Exception:
                P = P.reshape(-1, 3)
        return P
    except Exception:
        return None

def read_point_features_fast(path: str) -> Optional[np.ndarray]:
    """Return point features as a NumPy array if present.
    Supports Open3D tensor point clouds with keys 'feat_dim_*' (stacked in order)
    or a single NxD 'features' attribute. Returns None if no features found.
    """
    # Try tensor API first
    try:
        pcd_t = o3d.t.io.read_point_cloud(path)
        if pcd_t is not None and getattr(pcd_t, 'point', None) is not None:
            keys = list(pcd_t.point.keys())
            # Collect custom feature dims
            dim_keys = [k for k in keys if k.startswith("feat_dim_")]
            if dim_keys:
                dim_keys = sorted(dim_keys, key=lambda k: int(k.split('_')[-1]))
                cols = [np.asarray(pcd_t.point[k])[:, 0] for k in dim_keys]
                feats = np.stack(cols, axis=1)
                return feats
            if "features" in pcd_t.point:
                arr = np.asarray(pcd_t.point["features"])
                if arr.ndim == 1:
                    arr = arr.reshape(-1, 1)
                return arr
            # Fallback: use colors if present
            if "colors" in pcd_t.point:
                arr = np.asarray(pcd_t.point.colors)
                # Ensure Nx3
                if arr.ndim == 2 and arr.shape[1] >= 3:
                    return arr[:, :3]
    except Exception:
        pass
    # Legacy API fallback
    try:
        pcd = o3d.io.read_point_cloud(path)
        if pcd is None or pcd.is_empty():
            return None
        # Use legacy colors if available
        try:
            cols = np.asarray(pcd.colors)
            if cols.ndim == 2 and cols.shape[1] >= 3 and cols.size > 0:
                return cols[:, :3]
        except Exception:
            pass
        # No features found
        return None
    except Exception:
        return None

def stream_point_positions(path: str, points_per_chunk: int = 10000) -> Generator[np.ndarray, None, None]:
    """
    Stream point positions from simple text formats without loading the entire file.
    Supports:
      - .xyz (x y z per line, space-separated)
      - ASCII .ply with vertex properties including x,y,z
    For other formats, this generator will yield nothing (caller can fall back to full load).
    """
    ext = Path(path).suffix.lower()
    if ext == ".xyz" or ext == ".txt":
        buf = []
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                import numpy as np
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 3:
                        continue
                    try:
                        x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                        buf.append((x, y, z))
                    except Exception:
                        continue
                    if len(buf) >= points_per_chunk:
                        yield np.asarray(buf, dtype=np.float32)
                        buf = []
                if buf:
                    yield np.asarray(buf, dtype=np.float32)
        except Exception:
            return
        return
    if ext == ".ply":
        # Minimal ASCII PLY parser
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                header = []
                for line in f:
                    header.append(line.rstrip("\n"))
                    if line.strip() == "end_header":
                        break
                # Parse header
                fmt_ascii = any(h.startswith("format ascii") for h in header)
                if not fmt_ascii:
                    return
                # Find vertex count and property order
                vert_count = 0
                prop_names = []
                in_vertex = False
                for h in header:
                    hs = h.strip().split()
                    if len(hs) >= 3 and hs[0] == "element" and hs[1] == "vertex":
                        vert_count = int(hs[2])
                        in_vertex = True
                        continue
                    if in_vertex and len(hs) >= 3 and hs[0] == "property":
                        prop_names.append(hs[-1])
                    if len(hs) >= 2 and hs[0] == "element" and hs[1] != "vertex":
                        in_vertex = False
                # Determine x,y,z indices
                try:
                    ix = prop_names.index("x")
                    iy = prop_names.index("y")
                    iz = prop_names.index("z")
                except Exception:
                    return
                # Read vertex lines
                import numpy as np
                buf = []
                read_n = 0
                for line in f:
                    if read_n >= vert_count:
                        break
                    parts = line.strip().split()
                    if len(parts) < max(ix, iy, iz) + 1:
                        continue
                    try:
                        x, y, z = float(parts[ix]), float(parts[iy]), float(parts[iz])
                        buf.append((x, y, z))
                        read_n += 1
                    except Exception:
                        continue
                    if len(buf) >= points_per_chunk:
                        yield np.asarray(buf, dtype=np.float32)
                        buf = []
                if buf:
                    yield np.asarray(buf, dtype=np.float32)
        except Exception:
            return
        return
    # Unsupported for streaming; caller should fall back
    return


def to_lineset(mesh: o3d.geometry.TriangleMesh) -> Optional[o3d.geometry.LineSet]:
    try:
        return o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
    except Exception:
        return None

def find_textures(mesh_path: str):
    """Return a list of candidate texture image file paths in the same folder as the mesh."""
    try:
        p = Path(mesh_path)
        folder = p.parent
        tex_exts = {".png", ".jpg", ".jpeg", ".bmp"}
        candidates = []
        for img in sorted(folder.glob("*")):
            if img.is_file() and img.suffix.lower() in tex_exts:
                candidates.append(str(img))
        return candidates
    except Exception:
        return []

def load_texture_image(path: str):
    """Load an image for use as a base color map; returns open3d.geometry.Image or None."""
    try:
        return o3d.io.read_image(path)
    except Exception:
        return None

def find_obj_mtl_texture(mesh_path: str) -> Optional[str]:
    """
    If the mesh is an OBJ, try to locate an accompanying MTL file and parse a diffuse map (map_Kd).
    Returns the texture image path if found, else None.
    """
    try:
        p = Path(mesh_path)
        if p.suffix.lower() != ".obj":
            return None
        folder = p.parent
        # Common convention: same basename .mtl
        mtl_candidates = [folder / (p.stem + ".mtl")]
        # Also scan for any .mtl if the named one doesn't exist
        if not mtl_candidates[0].exists():
            mtl_candidates = list(folder.glob("*.mtl"))
        for mtl in mtl_candidates:
            try:
                with open(mtl, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("map_Kd"):
                            parts = line.split()
                            if len(parts) >= 2:
                                tex_rel = parts[1]
                                tex_path = (mtl.parent / tex_rel).resolve()
                                if tex_path.exists():
                                    return str(tex_path)
            except Exception:
                continue
    except Exception:
        return None
    return None

def inspect_ply_header_modes(path: str) -> Tuple[bool, bool]:
    """
    Inspect PLY header to determine available coloring modes for point clouds.
    Returns (has_rgb_colors, has_3d_features).
    - has_rgb_colors: True if header defines vertex properties 'red','green','blue'.
    - has_3d_features: True if header includes at least three 'feat_dim_*' properties.
    """
    has_rgb = False
    has_feats3 = False
    try:
        with open(path, "rb") as f:
            header_bytes = b""
            while True:
                line = f.readline()
                if not line:
                    break
                header_bytes += line
                if line.strip() == b"end_header":
                    break
        header = header_bytes.decode("utf-8", errors="ignore").strip().splitlines()
        prop_names = []
        in_vertex = False
        for h in header:
            hs = h.strip().split()
            if len(hs) >= 3 and hs[0] == "element" and hs[1] == "vertex":
                in_vertex = True
                continue
            if in_vertex and len(hs) >= 3 and hs[0] == "property":
                # last token is property name
                prop_names.append(hs[-1])
            if len(hs) >= 2 and hs[0] == "element" and hs[1] != "vertex":
                in_vertex = False
        names = set(p.lower() for p in prop_names)
        if {"red", "green", "blue"}.issubset(names):
            has_rgb = True
        # Count feat_dim_* occurrences
        feat_props = [p for p in prop_names if p.startswith("feat_dim_")]
        if len(feat_props) >= 3:
            has_feats3 = True
    except Exception:
        pass
    return has_rgb, has_feats3
