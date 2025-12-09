import argparse
import os
from pathlib import Path

import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering


try:
	from . import config as viz_config  # when running as a module
except Exception:
	import config as viz_config  # when running as a script


class VisualizationApp:
	def __init__(self):
		gui.Application.instance.initialize()
		self.window = gui.Application.instance.create_window("3D Visualization", 1280, 800)

		# Scene widget
		self.scene = gui.SceneWidget()
		self.scene.scene = rendering.Open3DScene(self.window.renderer)
		# Softer background for better contrast
		self.scene.scene.set_background([0.15, 0.17, 0.20, 1.0])
		# Try enabling a sun/directional light for depth perception (compatibility guarded)
		try:
			# Direction slightly from camera
			self.scene.scene.set_sun_light([0.577, 0.577, 0.577], [1.0, 1.0, 1.0], 75000)
			self.scene.scene.enable_sun_light(True)
		except Exception:
			pass
		self.window.add_child(self.scene)

		# Right panel for controls
		em = self.window.theme.font_size
		margin = int(round(0.5 * em))
		self.panel = gui.Vert(0, gui.Margins(margin, margin, margin, margin))

		# Geometry sources: directories to scrape and discovered geometry list
		self.source_dirs = self._init_default_dirs()
		self.geom_index = []  # list of (display_name, full_path)
		self.cmb_geometry = gui.Combobox()
		# Acts as "Add Geometry" dropdown: selecting immediately adds to list
		self.cmb_geometry.set_on_selection_changed(self._on_add_selected_from_dropdown)
		# Selected geometries management
		self.selected_geoms = []  # list of dicts: {path, type: 'mesh'|'pcd', options}
		# List to show current selected geometries (no ScrollView on this build)
		self.geom_list_container = gui.Vert(0, gui.Margins(0, 0, 0, 0))
		self.btn_clear_selected = gui.Button("Clear All")
		self.btn_clear_selected.set_on_clicked(self._on_clear_selected)
		# Simple default option: wireframe for meshes (per-entry toggled via list)
		self.default_wireframe = False
		self.btn_add_folder = gui.Button("Add Folder…")
		self.btn_add_folder.set_on_clicked(self._on_add_folder)
		self.btn_refresh = gui.Button("Refresh List")
		self.btn_refresh.set_on_clicked(self._refresh_geometry_list)
		# Initial scrape
		self._refresh_geometry_list()

		# Point size slider (kept minimal for visibility of point clouds)
		self.sld_point_size = gui.Slider(gui.Slider.INT)
		self.sld_point_size.set_limits(1, 10)
		self.sld_point_size.int_value = 3
		self.sld_point_size.set_on_value_changed(self._on_point_size_change)

		# Hover info
		self.scene.set_on_mouse(self._on_mouse)

		# Light controls: azimuth and elevation
		self.sld_light_az = gui.Slider(gui.Slider.INT)
		self.sld_light_az.set_limits(0, 360)
		self.sld_light_az.int_value = 45
		self.sld_light_el = gui.Slider(gui.Slider.INT)
		self.sld_light_el.set_limits(-80, 80)
		self.sld_light_el.int_value = 30
		self.sld_light_az.set_on_value_changed(self._on_light_changed)
		self.sld_light_el.set_on_value_changed(self._on_light_changed)

		# Assemble panel
		self.panel.add_child(gui.Label("Sources"))
		row_title = gui.Horiz()
		row_title.add_child(gui.Label("Add Geometry"))
		row_title.add_stretch()
		self.panel.add_child(row_title)
		self.panel.add_child(self.cmb_geometry)
		row = gui.Horiz()
		row.add_stretch()
		row.add_child(self.btn_add_folder)
		row.add_child(self.btn_refresh)
		# Rename and place Clear All on same row
		self.btn_clear_selected.text = "Clear All Geometries"
		row.add_child(self.btn_clear_selected)
		self.panel.add_child(row)
		# Spacer
		self.panel.add_child(gui.Label(""))
		self.panel.add_child(gui.Label("Selected Geometries"))
		self.panel.add_child(self.geom_list_container)
		self.panel.add_child(gui.Label(""))
		self.panel.add_child(gui.Label("Point Size"))
		self.panel.add_child(self.sld_point_size)
		# Light controls UI
		self.panel.add_child(gui.Label(""))
		self.panel.add_child(gui.Label("Light Azimuth (deg)"))
		self.panel.add_child(self.sld_light_az)
		self.panel.add_child(gui.Label("Light Elevation (deg)"))
		self.panel.add_child(self.sld_light_el)

		# Layout
		self.window.set_on_layout(self._on_layout)
		self.window.add_child(self.panel)

		# State
		self.geometry = None
		self.is_pointcloud = False
		self.features = None  # np.ndarray (N, D)
		self.category_mapping = None  # placeholder (categorical mapping disabled)

		# Initialize sun light from sliders if available
		try:
			self._apply_light()
			self.scene.scene.enable_sun_light(True)
		except Exception:
			pass

		# Build initial empty list UI
		self._rebuild_geom_list_ui()

	def _on_light_changed(self, _val):
		self._apply_light()

	def _apply_light(self):
		# Convert azimuth/elevation to a unit direction vector and apply.
		# Fallback: rotate indirect light (IBL) by azimuth if sun light API is unavailable.
		az_deg = float(self.sld_light_az.int_value)
		try:
			az = az_deg * np.pi / 180.0
			sel = float(self.sld_light_el.int_value) * np.pi / 180.0
			cx = np.cos(sel)
			dir_vec = [cx * np.cos(az), cx * np.sin(az), np.sin(sel)]
			self.scene.scene.set_sun_light(dir_vec, [1.0, 1.0, 1.0], 75000)
			return
		except Exception:
			pass
		# Indirect light rotation (around up-axis) as a compatible alternative
		try:
			self.scene.scene.set_indirect_light_rotation(az_deg)
		except Exception:
			# No lighting control available; silently ignore
			return

	def _init_default_dirs(self):
		candidates = []
		# From config if provided
		if hasattr(viz_config, "DEFAULT_GEOMETRY_DIRS"):
			for p in getattr(viz_config, "DEFAULT_GEOMETRY_DIRS"):
				try:
					pp = Path(p).expanduser().resolve()
					candidates.append(pp)
				except Exception:
					continue
		# De-duplicate and keep existing
		seen = set()
		result = []
		for pp in candidates:
			if pp in seen:
				continue
			seen.add(pp)
			if pp.exists() and pp.is_dir():
				result.append(str(pp))
		return result

	def _refresh_geometry_list(self, *_):
		# Scrape source_dirs for known geometry files and populate combobox
		exts = {".ply", ".pcd", ".xyz", ".obj", ".stl", ".off", ".gltf", ".glb"}
		items = []
		for d in self.source_dirs:
			dp = Path(d)
			if not dp.exists():
				continue
			for p in sorted(dp.glob("**/*")):
				if p.is_file() and p.suffix.lower() in exts:
					display = f"{dp.name}/{p.name}"
					items.append((display, str(p)))
		# Update UI
		self.geom_index = items
		self.cmb_geometry.clear_items()
		for disp, _ in items:
			self.cmb_geometry.add_item(disp)
		if items:
			self.cmb_geometry.selected_index = 0
			# Do not auto-load to avoid popping error dialogs on startup.
			# User can pick from the dropdown to load.

	def _on_add_folder(self):
		# Use file dialog to pick a file; add its directory
		dlg = gui.FileDialog(gui.FileDialog.OPEN, "Select any file in folder to add", self.window.theme)
		dlg.set_on_cancel(lambda: self.window.close_dialog())
		def _on_ok(path):
			self.window.close_dialog()
			dirpath = os.path.dirname(path)
			if dirpath and dirpath not in self.source_dirs:
				self.source_dirs.append(dirpath)
				self._refresh_geometry_list()
		dlg.set_on_done(_on_ok)
		self.window.show_dialog(dlg)

	def _on_add_selected_from_dropdown(self, *_args):
		# Selecting from the dropdown adds to the list (acts as "Add Geometry")
		idx = self.cmb_geometry.selected_index
		if 0 <= idx < len(self.geom_index):
			display, path = self.geom_index[idx]
			ext = Path(path).suffix.lower()
			gtype = 'pcd' if ext in {'.ply', '.pcd', '.xyz'} else 'mesh'
			entry = {
				"display": display,
				"path": path,
				"type": gtype,
				"options": {
					"wireframe": bool(self.default_wireframe),
				}
			}
			self.selected_geoms.append(entry)
			# Rebuild UI and render on main thread to avoid GUI reentrancy issues
			gui.Application.instance.post_to_main_thread(self.window, lambda: (self._rebuild_geom_list_ui(), self._render_selected_lineup()))

	# Deprecated button-based add kept out per new UX

	def _on_clear_selected(self):
		self.selected_geoms.clear()
		self._clear_scene()
		self._rebuild_geom_list_ui()

	# Per-entry options handled via list UI

	def _rebuild_geom_list_ui(self):
		# Clear container and rebuild rows representing selected_geoms
		# Some Open3D builds don't expose clear_children; remove manually.
		# Remove existing children safely, keep container anchored under its label
		try:
			children = list(self.geom_list_container.get_children())
			for ch in children:
				self.geom_list_container.remove_child(ch)
		except Exception:
			# If removal fails for some child, ignore and continue building fresh rows
			pass
		for i, entry in enumerate(self.selected_geoms):
			row = gui.Horiz()
			label = gui.Label(entry["display"])
			row.add_child(label)
			row.add_stretch()
			# Wireframe toggle for meshes
			if entry["type"] == "mesh":
				chk = gui.Checkbox("Wireframe")
				chk.checked = bool(entry["options"].get("wireframe", False))
				def _on_chk(_checked, idx=i):
					self.selected_geoms[idx]["options"]["wireframe"] = bool(chk.checked)
					self._render_selected_lineup()
				chk.set_on_checked(_on_chk)
				row.add_child(chk)
			# Reorder buttons
			btn_up = gui.Button("↑")
			btn_down = gui.Button("↓")
			def _move(delta, idx=i):
				new_idx = idx + delta
				if 0 <= new_idx < len(self.selected_geoms):
					self.selected_geoms[idx], self.selected_geoms[new_idx] = self.selected_geoms[new_idx], self.selected_geoms[idx]
					self._rebuild_geom_list_ui()
					self._render_selected_lineup()
			btn_up.set_on_clicked(lambda idx=i: _move(-1, idx))
			btn_down.set_on_clicked(lambda idx=i: _move(1, idx))
			row.add_child(btn_up)
			row.add_child(btn_down)
			# Remove button
			btn_rm = gui.Button("Remove")
			def _rm(idx=i):
				self.selected_geoms.pop(idx)
				self._rebuild_geom_list_ui()
				self._render_selected_lineup()
			btn_rm.set_on_clicked(lambda idx=i: _rm(idx))
			row.add_child(btn_rm)
			self.geom_list_container.add_child(row)

	def _load_geometry_from_path(self, path: str):
		# Route based on file extension to avoid spurious warnings (e.g., OBJ as point cloud)
		ext = Path(path).suffix.lower()
		pcd_exts = {".ply", ".pcd", ".xyz"}
		mesh_exts = {".obj", ".stl", ".off", ".gltf", ".glb"}
		if ext in pcd_exts:
			self._load_pointcloud_from_path(path)
			return
		if ext in mesh_exts:
			self._load_mesh_from_path(path)
			return
		# Unknown or no extension: try tensor point cloud then mesh as fallback
		try:
			pcd_t = o3d.t.io.read_point_cloud(path)
			if pcd_t is not None and pcd_t.point.positions is not None and len(pcd_t.point.positions) > 0:
				self._load_pointcloud_from_path(path)
				return
		except Exception:
			pass
		self._load_mesh_from_path(path)

	def _on_layout(self, layout_context):
		r = self.window.content_rect
		# Scene fills left 3/4, panel on right 1/4
		panel_width = int(0.28 * r.width)
		self.scene.frame = gui.Rect(r.x, r.y, r.width - panel_width, r.height)
		self.panel.frame = gui.Rect(r.get_right() - panel_width, r.y, panel_width, r.height)

	def _clear_scene(self):
		self.scene.scene.clear_geometry()
		self.geometry = None
		self.features = None
		self.is_pointcloud = False

	def _on_load_mesh(self):
		dlg = gui.FileDialog(gui.FileDialog.OPEN, "Select mesh", self.window.theme)
		dlg.add_filter("Mesh", ".ply .obj .stl .off .gltf .glb")
		dlg.set_on_cancel(lambda: self.window.close_dialog())
		def _on_ok(path):
			self.window.close_dialog()
			self._load_mesh_from_path(path)
		dlg.set_on_done(_on_ok)
		self.window.show_dialog(dlg)

	def _on_load_pointcloud(self):
		dlg = gui.FileDialog(gui.FileDialog.OPEN, "Select point cloud", self.window.theme)
		dlg.add_filter("PointCloud", ".ply .pcd .xyz")
		dlg.set_on_cancel(lambda: self.window.close_dialog())
		def _on_ok(path):
			self.window.close_dialog()
			self._load_pointcloud_from_path(path)
		dlg.set_on_done(_on_ok)
		self.window.show_dialog(dlg)

	# Category mapping UI removed for now

	def _load_mesh_from_path(self, path):
		try:
			mesh = o3d.io.read_triangle_mesh(path)
			if mesh is None or mesh.is_empty():
				self._show_message("Failed to load mesh")
				return
			mesh.compute_vertex_normals()
			self._clear_scene()
			# Use a lit material so normals contribute to shading
			mat = rendering.MaterialRecord()
			mat.shader = "defaultLit"
			mat.base_color = [0.8, 0.8, 0.85, 1.0]
			self.scene.scene.add_geometry("mesh", mesh, mat)
			bounds = mesh.get_axis_aligned_bounding_box()
			self.scene.setup_camera(60.0, bounds, bounds.get_center())
			self.geometry = mesh
			self.is_pointcloud = False
		except Exception as e:
			self._show_message(f"Error loading mesh: {e}")

	def _load_pointcloud_from_path(self, path):
		try:
			# Prefer tensor point cloud if available
			pcd_t = o3d.t.io.read_point_cloud(path)
			if pcd_t is not None and pcd_t.point.positions is not None:
				positions = np.asarray(pcd_t.point.positions)
				pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(positions))
				# Gather features: feat_dim_* or 'features'
				dims = []
				for key in pcd_t.point.keys():
					if key.startswith("feat_dim_"):
						dims.append(key)
				dims_sorted = sorted(dims, key=lambda k: int(k.split('_')[-1]))
				if dims_sorted:
					cols = [np.asarray(pcd_t.point[k])[:, 0] for k in dims_sorted]
					self.features = np.stack(cols, axis=1)
				elif "features" in pcd_t.point:
					arr = np.asarray(pcd_t.point["features"])
					# features might be (N,D)
					self.features = arr if arr.ndim == 2 else arr.reshape(-1, 1)
				# Colors if present
				if "colors" in pcd_t.point:
					colors = np.asarray(pcd_t.point.colors)
					pcd.colors = o3d.utility.Vector3dVector(colors[:, :3] / 255.0 if colors.max() > 1.0 else colors[:, :3])
			else:
				pcd = o3d.io.read_point_cloud(path)
				if pcd is None or pcd.is_empty():
					self._show_message("Failed to load point cloud")
					return
				self.features = None

			self._clear_scene()
			mat = rendering.MaterialRecord()
			mat.point_size = float(self.sld_point_size.int_value)
			self.scene.scene.add_geometry("pcd", pcd, mat)
			bounds = pcd.get_axis_aligned_bounding_box()
			self.scene.setup_camera(60.0, bounds, bounds.get_center())
			self.geometry = pcd
			self.is_pointcloud = True
			# Update feature dim combobox
			# No feature-based coloring in this simplified UI
		except Exception as e:
			self._show_message(f"Error loading point cloud: {e}")

	def _on_point_size_change(self, _val):
		# Re-render lineup to apply point size to all point clouds
		self._render_selected_lineup()
	def _render_selected_lineup(self):
		# Clear scene and render each selected geometry side-by-side along +X
		self.scene.scene.clear_geometry()
		offset_x = 0.0
		gap = 1.5  # spacing between items
		for i, entry in enumerate(self.selected_geoms):
			path = entry["path"]
			gtype = entry["type"]
			wire = entry["options"].get("wireframe", False)
			name = f"geom_{i}"
			try:
				if gtype == 'pcd':
					pcd = o3d.io.read_point_cloud(path)
					if pcd is None or pcd.is_empty():
						continue
					# translate by offset
					pcd.translate([offset_x, 0.0, 0.0])
					mat = rendering.MaterialRecord()
					mat.point_size = float(self.sld_point_size.int_value)
					self.scene.scene.add_geometry(name, pcd, mat)
				else:
					mesh = o3d.io.read_triangle_mesh(path)
					if mesh is None or mesh.is_empty():
						continue
					mesh.compute_vertex_normals()
					mesh.translate([offset_x, 0.0, 0.0])
					mat = rendering.MaterialRecord()
					if wire:
						mat.shader = "unlitLine"
					else:
						mat.shader = "defaultLit"
						mat.base_color = [0.8, 0.8, 0.85, 1.0]
					self.scene.scene.add_geometry(name, mesh, mat)
				# advance offset by bounding box width + gap
				bb = None
				if gtype == 'pcd':
					bb = pcd.get_axis_aligned_bounding_box()
				else:
					bb = mesh.get_axis_aligned_bounding_box()
				w = (bb.get_max_bound() - bb.get_min_bound())[0]
				offset_x += w + gap
			except Exception as e:
				self._show_message(f"Render lineup error for {path}: {e}")
		# Fit camera to all
		try:
			bounds = self.scene.scene.bounding_box
			self.scene.setup_camera(60.0, bounds, bounds.get_center())
		except Exception:
			pass


	def _on_mouse(self, event):
		if event.type == gui.MouseEvent.Type.MOVE and self.is_pointcloud and self.geometry is not None:
			# Perform a pick near mouse to show tooltip with feature value
			# Note: GUI picking helpers are limited; do approximate nearest search in screen space
			# Here we skip heavy picking and just show a generic hint
			return gui.Widget.EventCallbackResult.HANDLED
		return gui.Widget.EventCallbackResult.IGNORED

	def _show_message(self, msg: str):
		# Avoid modal dialogs that can obscure the UI; log to console for now.
		print(f"[Viewer] {msg}")


def main():
	app = VisualizationApp()
	gui.Application.instance.run()


if __name__ == "__main__":
	main()
