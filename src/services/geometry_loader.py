from functools import lru_cache
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
