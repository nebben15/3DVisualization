from functools import lru_cache
from pathlib import Path
import open3d as o3d
from typing import Optional
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
