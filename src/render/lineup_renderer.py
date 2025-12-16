from typing import List
import open3d.visualization.rendering as rendering
import open3d as o3d

# Support both module and script execution import styles
try:
    from ..models.selection import SelectedEntry
    from ..services.geometry_loader import (
        load_mesh,
        load_pointcloud,
        to_lineset,
        load_texture_image,
        find_textures,
        find_obj_mtl_texture,
        read_point_features_fast,
    )
except Exception:
    from models.selection import SelectedEntry
    from services.geometry_loader import (
        load_mesh,
        load_pointcloud,
        to_lineset,
        load_texture_image,
        find_textures,
        find_obj_mtl_texture,
        read_point_features_fast,
    )


def render(scene: rendering.Open3DScene, selection: List[SelectedEntry], point_size: float, preserve_camera: bool = False, overlay: bool = False) -> None:
    scene.clear_geometry()
    offset_x = 0.0
    gap = 1.5
    for i, entry in enumerate(selection):
        name = f"geom_{i}"
        try:
            # Skip entries still loading to avoid blocking UI
            if bool(entry.options.get("loading", False)):
                continue
            # Skip rendering if entry is hidden
            if not bool(entry.options.get("visible", True)):
                continue
            if entry.type == "pcd":
                # Support progressive rendering via 'progress_positions' option
                prog = entry.options.get("progress_positions", None)
                if prog is not None:
                    try:
                        import numpy as np
                        pts = np.asarray(prog)
                        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
                    except Exception:
                        pcd = load_pointcloud(entry.path)
                else:
                    pcd = load_pointcloud(entry.path)
                if not pcd:
                    continue
                # Coloring for point clouds based on options
                color_mode = str(entry.options.get("color_mode", "default")).lower()
                if color_mode == "rgb":
                    # If pcd has colors, keep; otherwise try to pull from tensor 'colors'
                    pass  # load_pointcloud already tries to populate colors
                elif color_mode == "continuous" and prog is None:
                    # Only apply when not in progressive mode (requires full features)
                    try:
                        import numpy as np
                        feats = read_point_features_fast(entry.path)
                        if feats is not None and len(feats) == len(np.asarray(pcd.points)):
                            # Map features to RGB
                            F = feats
                            if F.ndim == 1:
                                F = F.reshape(-1, 1)
                            if F.shape[1] >= 3:
                                C = F[:, :3].astype(np.float32)
                                # Min-max per channel
                                for c in range(3):
                                    mn, mx = float(C[:, c].min()), float(C[:, c].max())
                                    if mx > mn:
                                        C[:, c] = (C[:, c] - mn) / (mx - mn)
                                    else:
                                        C[:, c] = 0.5
                            else:
                                # 1D or 2D: expand via simple normalization and padding
                                C = np.zeros((F.shape[0], 3), dtype=np.float32)
                                for c in range(min(3, F.shape[1])):
                                    mn, mx = float(F[:, c].min()), float(F[:, c].max())
                                    if mx > mn:
                                        C[:, c] = (F[:, c] - mn) / (mx - mn)
                                    else:
                                        C[:, c] = 0.5
                            pcd.colors = o3d.utility.Vector3dVector(C)
                    except Exception:
                        pass
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
                # Ensure normals for lit shading
                try:
                    if not mesh.has_vertex_normals():
                        mesh.compute_vertex_normals()
                except Exception:
                    pass
                mesh.translate([offset_x, 0.0, 0.0])
                # Coloring for meshes
                mesh_color_mode = str(entry.options.get("color_mode", "default")).lower()
                if bool(entry.options.get("wireframe", False)):
                    # Render lit mesh AND a line overlay for wireframe clarity
                    try:
                        # base shaded mesh for context
                        mat_mesh = rendering.MaterialRecord()
                        mat_mesh.shader = "defaultLit"
                        mat_mesh.base_color = [0.75, 0.75, 0.8, 1.0]
                        scene.add_geometry(f"{name}_mesh", mesh, mat_mesh)
                    except Exception:
                        pass
                    lines = to_lineset(mesh)
                    if lines:
                        mat = rendering.MaterialRecord()
                        mat.shader = "unlitLine"
                        mat.line_width = 1.0
                        scene.add_geometry(f"{name}_wire", lines, mat)
                    else:
                        # fallback: render mesh in line mode if LineSet unavailable
                        mat = rendering.MaterialRecord()
                        mat.shader = "unlitLine"
                        scene.add_geometry(f"{name}_wire", mesh, mat)
                else:
                    mat = rendering.MaterialRecord()
                    mat.shader = "defaultLit"
                    # Apply texture if available/enabled; otherwise base color
                    tex_path = entry.options.get("texture", None)
                    tex_enabled = bool(entry.options.get("texture_enabled", True))
                    applied = False
                    # Check for UVs
                    try:
                        has_uv = bool(getattr(mesh, "has_triangle_uvs", lambda: False)())
                    except Exception:
                        has_uv = False
                    # Priority: explicit selection -> embedded mesh texture -> fallback color
                    img_to_use = None
                    if tex_enabled:
                        if isinstance(tex_path, str) and tex_path:
                            img_to_use = load_texture_image(tex_path)
                        elif hasattr(mesh, "textures") and mesh.textures and len(mesh.textures) > 0:
                            # Use first embedded texture if present
                            try:
                                img_to_use = mesh.textures[0]
                            except Exception:
                                img_to_use = None
                        else:
                            # Fast-path: try OBJ+MTL diffuse map (map_Kd)
                            try:
                                mtl_tex = find_obj_mtl_texture(entry.path)
                                if mtl_tex:
                                    img_to_use = load_texture_image(mtl_tex)
                            except Exception:
                                pass
                    if img_to_use is not None and has_uv:
                        try:
                            # Prefer explicit material texture assignment for GUI renderer
                            # Newer Open3D builds support base_color_texture; fallback to base_color_map
                            try:
                                mat.base_color_texture = img_to_use  # preferred
                            except Exception:
                                mat.base_color_map = img_to_use      # fallback name
                            # Neutral base color to avoid tinting
                            mat.base_color = [1.0, 1.0, 1.0, 1.0]
                            applied = True
                        except Exception:
                            applied = False
                        # Also set legacy mesh texture list for broader compatibility
                        try:
                            mesh.textures = [img_to_use]
                        except Exception:
                            pass
                    if not applied:
                        # If requested, color by normalized vertex positions (XYZ)
                        if mesh_color_mode == "position":
                            try:
                                import numpy as np
                                V = np.asarray(mesh.vertices)
                                if V.size > 0:
                                    mn = V.min(axis=0)
                                    mx = V.max(axis=0)
                                    denom = np.where((mx - mn) > 0, (mx - mn), 1.0)
                                    C = (V - mn) / denom
                                    mesh.vertex_colors = o3d.utility.Vector3dVector(C[:, :3])
                                mat.base_color = [1.0, 1.0, 1.0, 1.0]
                            except Exception:
                                mat.base_color = [0.8, 0.8, 0.85, 1.0]
                        else:
                            mat.base_color = [0.8, 0.8, 0.85, 1.0]
                    scene.add_geometry(name, mesh, mat)
                bb = mesh.get_axis_aligned_bounding_box()
            w = (bb.get_max_bound() - bb.get_min_bound())[0]
            # In overlay mode, do not advance offset; stack at origin for comparison
            if not overlay:
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
