from typing import List
import open3d.visualization.rendering as rendering
import open3d as o3d

# Support both module and script execution import styles
try:
    from ..models.selection import SelectedEntry
    from ..services.geometry_loader import load_mesh, load_pointcloud, to_lineset
except Exception:
    from models.selection import SelectedEntry
    from services.geometry_loader import load_mesh, load_pointcloud, to_lineset


def render(scene: rendering.Open3DScene, selection: List[SelectedEntry], point_size: float, preserve_camera: bool = False) -> None:
    scene.clear_geometry()
    offset_x = 0.0
    gap = 1.5
    for i, entry in enumerate(selection):
        name = f"geom_{i}"
        try:
            if entry.type == "pcd":
                pcd = load_pointcloud(entry.path)
                if not pcd:
                    continue
                # Apply scale around center if provided
                scale = float(entry.options.get("scale", 1.0))
                try:
                    center = pcd.get_center()
                    pcd = o3d.geometry.PointCloud(pcd)  # clone
                    pcd.scale(scale, center)
                except Exception:
                    pass
                pcd.translate([offset_x, 0.0, 0.0])
                mat = rendering.MaterialRecord()
                mat.point_size = float(point_size)
                scene.add_geometry(name, pcd, mat)
                bb = pcd.get_axis_aligned_bounding_box()
            else:
                mesh = load_mesh(entry.path)
                if not mesh:
                    continue
                scale = float(entry.options.get("scale", 1.0))
                mesh = o3d.geometry.TriangleMesh(mesh)  # clone
                try:
                    center = mesh.get_center()
                    mesh.scale(scale, center)
                except Exception:
                    pass
                mesh.translate([offset_x, 0.0, 0.0])
                if bool(entry.options.get("wireframe", False)):
                    lines = to_lineset(mesh)
                    if lines:
                        mat = rendering.MaterialRecord()
                        mat.shader = "unlitLine"
                        mat.line_width = 1.0
                        scene.add_geometry(name, lines, mat)
                    else:
                        mat = rendering.MaterialRecord()
                        mat.shader = "unlitLine"
                        scene.add_geometry(name, mesh, mat)
                else:
                    mat = rendering.MaterialRecord()
                    mat.shader = "defaultLit"
                    mat.base_color = [0.8, 0.8, 0.85, 1.0]
                    scene.add_geometry(name, mesh, mat)
                bb = mesh.get_axis_aligned_bounding_box()
            w = (bb.get_max_bound() - bb.get_min_bound())[0]
            offset_x += w + gap
        except Exception as e:
            print(f"[Renderer] Error rendering {entry.path}: {e}")
    if not preserve_camera:
        try:
            bounds = scene.bounding_box
            # 60 deg FOV; center camera on bounding box center
            # The actual scene widget should handle setup_camera
        except Exception:
            pass
