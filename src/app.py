import argparse
import os
from pathlib import Path
import threading

import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

from models.selection import SelectionList, SelectedEntry
from services.geometry_registry import scan_directories
from render.lineup_renderer import render as render_lineup

try:
	# local dialog modules
	from .dialogs import mesh_edit
except Exception:
	# fallback when running as script
	from dialogs import mesh_edit


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

		# Geometry sources and dropdown
		self.source_dirs = self._init_default_dirs()
		self.geom_index = []  # list of GeometryInfo
		self.cmb_geometry = gui.Combobox()
			# Acts as "Add Geometry" dropdown: selecting immediately adds to list
		self.cmb_geometry.set_on_selection_changed(self._on_add_selected_from_dropdown)
		# Selected geometries model
		self.selection = SelectionList()
			# ListView to show current selected geometries
		self.list_view = gui.ListView()
		self.list_view.set_items([])
		self.list_view.set_on_selection_changed(self._on_list_selection_changed)
		# Initialize selection index
		self.sel_index = -1
		# Controls for selected entry
		# Use ASCII labels to avoid font issues with arrows on some builds
		self.btn_up = gui.Button("Up")
		self.btn_down = gui.Button("Down")
		self.btn_remove = gui.Button("Remove")
		self.chk_wireframe_sel = gui.Checkbox("Wireframe (mesh)")
		self.btn_up.set_on_clicked(lambda: self._move_selected(-1))
		self.btn_down.set_on_clicked(lambda: self._move_selected(1))
		self.btn_remove.set_on_clicked(self._remove_selected)
		self.chk_wireframe_sel.set_on_checked(self._on_selected_wireframe)
		self.btn_clear_selected = gui.Button("Clear All")
		self.btn_clear_selected.set_on_clicked(self._on_clear_selected)
		# Simple default option: wireframe for meshes (per-entry toggled via list)
		self.default_wireframe = False
		self.btn_add_folder = gui.Button("Add Folder")
		self.btn_add_folder.set_on_clicked(self._on_add_folder)
		self.btn_refresh = gui.Button("Refresh List")
		self.btn_refresh.set_on_clicked(self._refresh_geometry_list)
		# Initial scrape
		self._refresh_geometry_list()

		# Default point size for point clouds
		self.default_point_size = 3.0

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

		# Backward compatibility: keep selected_geoms list while migrating to SelectionList
		self.selected_geoms = []
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
		self.panel.add_child(self.list_view)
		# Controls row (always visible) for selected geometry actions
		self.controls_row = gui.Horiz()
		# Add buttons: Up, Down, Remove, Edit
		self.btn_edit = gui.Button("Edit")
		self.btn_edit.set_on_clicked(self._open_edit_dialog)
		self.controls_row.add_child(self.btn_up)
		self.controls_row.add_child(self.btn_down)
		self.controls_row.add_child(self.btn_remove)
		self.controls_row.add_child(self.btn_edit)
		# Hide/Show toggle for selected entry
		self.btn_toggle_visible = gui.Button("Hide")
		self.btn_toggle_visible.set_on_clicked(self._on_toggle_visible)
		self.controls_row.add_child(self.btn_toggle_visible)
		# Overlay toggle: render all geometries at the same origin for comparison
		self.overlay_mode = False
		self.btn_overlay = gui.Button("Overlay: Off")
		self.btn_overlay.set_on_clicked(self._on_toggle_overlay)
		self.controls_row.add_child(self.btn_overlay)
		self.controls_row.add_stretch()
		self.panel.add_child(self.controls_row)
		# Controls row will sit below the list
		# Light controls UI
		self.panel.add_child(gui.Label(""))
		self.panel.add_child(gui.Label("Light Azimuth (deg)"))
		self.panel.add_child(self.sld_light_az)
		self.panel.add_child(gui.Label("Light Elevation (deg)"))
		self.panel.add_child(self.sld_light_el)

		# Scale control moved to Edit dialog

		# Layout
		self.window.set_on_layout(self._on_layout)
		self.window.add_child(self.panel)
		# Initialize controls state
		self._sync_controls()

		# State
		self.geometry = None
		self.is_pointcloud = False
		self.features = None
		# Use registry service to scan directories for supported geometry files
		infos = scan_directories(self.source_dirs)
		# Store as (display, path, type) for accurate type handling (mesh vs pcd)
		self.geom_index = [(gi.display, gi.path, gi.type) for gi in infos]
		self.cmb_geometry.clear_items()
		# Use plain ASCII to avoid font fallback showing question marks
		self.cmb_geometry.add_item("Select to add")
		for disp, _path, _type in self.geom_index:
			self.cmb_geometry.add_item(disp)
		self.cmb_geometry.selected_index = 0
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
		# Use registry service to collect supported files recursively with types
		try:
			infos = scan_directories(self.source_dirs)
		except Exception:
			infos = []
		self.geom_index = [(gi.display, gi.path, gi.type) for gi in infos]
		# Update UI
		self.cmb_geometry.clear_items()
		self.cmb_geometry.add_item("Select to add")
		for disp, _, _ in self.geom_index:
			self.cmb_geometry.add_item(disp)
		self.cmb_geometry.selected_index = 0
			# Do not auto-load to avoid popping error dialogs on startup.
			# User can pick from the dropdown to load.

	def _on_light_changed(self, _val):
		# React to slider changes by applying light settings
		self._apply_light()

	def _apply_light(self):
		# Convert azimuth/elevation to a unit direction vector and apply sun light if available.
		az_deg = float(getattr(self.sld_light_az, 'int_value', 45))
		el_deg = float(getattr(self.sld_light_el, 'int_value', 30))
		try:
			az = az_deg * np.pi / 180.0
			sel = el_deg * np.pi / 180.0
			cx = np.cos(sel)
			dir_vec = [cx * np.cos(az), cx * np.sin(az), np.sin(sel)]
			self.scene.scene.set_sun_light(dir_vec, [1.0, 1.0, 1.0], 75000)
			return
		except Exception:
			pass
		# Fallback: rotate indirect light (IBL) by azimuth
		try:
			self.scene.scene.set_indirect_light_rotation(az_deg)
		except Exception:
			return

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
		if idx <= 0:
			return
		adj_idx = idx - 1
		if 0 <= adj_idx < len(self.geom_index):
			display, path, gtype = self.geom_index[adj_idx]
			# Determine if we should mark as loading (point clouds can be heavy)
			is_pcd = (gtype == 'pcd')
			entry = SelectedEntry(
				display=display,
				path=path,
				type=gtype,
				options={
					"wireframe": bool(self.default_wireframe),
					"scale": 1.0,
					"visible": True,
					"texture": None,
					"texture_enabled": True,
					"loading": True if is_pcd else False,
				},
			)
			self.selection.add(entry)
			# Update current selection index to the newly added item
			self.sel_index = len(self.selection.items) - 1 if self.selection.items else -1
			# Update UI immediately; for meshes, render right away; for pcd, show loading and warm cache in background
			def _after_add():
				self._rebuild_geom_list_ui()
				if not is_pcd:
					self._render_selected_lineup(camera_preserve=False)
				self._sync_controls()
			gui.Application.instance.post_to_main_thread(self.window, _after_add)
			gui.Application.instance.post_to_main_thread(self.window, lambda: setattr(self.cmb_geometry, 'selected_index', 0))
			if is_pcd:
				# Progressive load in background: try streaming for text formats; fallback to batched updates
				def _bg_load_progress(path_=path):
					try:
						from services.geometry_loader import stream_point_positions, read_point_positions_fast
					except Exception:
						from .services.geometry_loader import stream_point_positions, read_point_positions_fast
					# First, attempt streaming chunks (for .xyz and ASCII .ply)
					streamed_any = False
					max_display = 50000
					accum = None
					for chunk in stream_point_positions(path_):
						streamed_any = True
						# Append and cap to avoid overwhelming the GUI
						try:
							accum = chunk if accum is None else np.vstack((accum, chunk))
							if len(accum) > max_display:
								accum = accum[-max_display:]
						except Exception:
							accum = chunk
						def _on_chunk(sp=accum):
							for idx_, e in enumerate(self.selection.items):
								if e.path == path_:
									opts = dict(e.options)
									opts["progress_positions"] = sp
									self.selection.update_options(idx_, **opts)
									break
							self._render_selected_lineup(camera_preserve=True)
						gui.Application.instance.post_to_main_thread(self.window, _on_chunk)
					if streamed_any:
						def _on_done_stream():
							for idx_, e in enumerate(self.selection.items):
								if e.path == path_:
									opts = dict(e.options)
									opts["loading"] = False
									self.selection.update_options(idx_, **opts)
									break
							self._rebuild_geom_list_ui()
							self._render_selected_lineup(camera_preserve=True)
							self._sync_controls()
						gui.Application.instance.post_to_main_thread(self.window, _on_done_stream)
						return
					# Fallback: read full array, then post downsampled batched updates
					pts = read_point_positions_fast(path_)
					if pts is None or len(pts) == 0:
						def _on_fail():
							for idx_, e in enumerate(self.selection.items):
								if e.path == path_:
									opts = dict(e.options)
									opts["loading"] = False
									opts.pop("progress_positions", None)
									self.selection.update_options(idx_, **opts)
									break
							self._rebuild_geom_list_ui()
							self._render_selected_lineup(camera_preserve=False)
							self._sync_controls()
						gui.Application.instance.post_to_main_thread(self.window, _on_fail)
						return
					batch = 100000
					for end in range(batch, len(pts) + batch, batch):
						cur = min(end, len(pts))
						stride = max(1, cur // 10000)
						slice_pts = pts[:cur:stride]
						def _on_chunk(sp=slice_pts):
							for idx_, e in enumerate(self.selection.items):
								if e.path == path_:
									opts = dict(e.options)
									opts["progress_positions"] = sp
									self.selection.update_options(idx_, **opts)
									break
							self._render_selected_lineup(camera_preserve=True)
						gui.Application.instance.post_to_main_thread(self.window, _on_chunk)
						try:
							import time; time.sleep(0.02)
						except Exception:
							pass
					def _on_done_fallback():
						for idx_, e in enumerate(self.selection.items):
							if e.path == path_:
								opts = dict(e.options)
								opts["loading"] = False
								# Keep last progressive sample for display; avoid forced full reload
								self.selection.update_options(idx_, **opts)
								break
						self._rebuild_geom_list_ui()
						self._render_selected_lineup(camera_preserve=True)
						self._sync_controls()
					gui.Application.instance.post_to_main_thread(self.window, _on_done_fallback)
				threading.Thread(target=_bg_load_progress, daemon=True).start()

	# Deprecated button-based add kept out per new UX

	def _on_clear_selected(self):
		self.selection.items.clear()
		self.sel_index = -1
		self._clear_scene()
		self._rebuild_geom_list_ui()

	def _move_selected(self, delta: int):
		# Move currently selected entry up/down by delta (selection model)
		idx = self.sel_index
		new_idx = idx + delta
		if 0 <= idx < len(self.selection.items) and 0 <= new_idx < len(self.selection.items):
			self.selection.move(idx, delta)
			self.sel_index = new_idx
			self._rebuild_geom_list_ui()
			self._render_selected_lineup(camera_preserve=True)
			self._sync_controls()

	def _remove_selected(self):
		# Remove currently selected entry (selection model)
		idx = self.sel_index
		if 0 <= idx < len(self.selection.items):
			self.selection.remove_at(idx)
			# Adjust selection
			if not self.selection.items:
				self.sel_index = -1
			elif idx >= len(self.selection.items):
				self.sel_index = len(self.selection.items) - 1
			# Close edit window if open
			try:
				if getattr(self, "_last_edit_window", None):
					win = getattr(self._last_edit_window, "_win", None) or self._last_edit_window
					gui.Application.instance.close_window(win)
					self._last_edit_window = None
			except Exception:
				self._last_edit_window = None
			self._rebuild_geom_list_ui()
			self._render_selected_lineup(camera_preserve=False)
			self._sync_controls()

	# Per-entry options handled via list UI

	def _rebuild_geom_list_ui(self):
		# Update ListView items and sync controls visibility
		items = []
		for e in self.selection.items:
			label = e.display
			if bool(e.options.get("loading", False)):
				label = f"{label} (loading...)"
			items.append(label)
		try:
			self.list_view.set_items(items)
		except Exception:
			# Some builds require clearing via setting again
			self.list_view.set_items([])
			self.list_view.set_items(items)
		# Keep selection in range
		if not (0 <= self.sel_index < len(self.selection.items)):
			self.sel_index = -1 if not self.selection.items else 0
		try:
			self.list_view.selected_index = self.sel_index if self.sel_index >= 0 else -1
			self.window.set_needs_layout()
		except Exception:
			pass
		# Ensure controls reflect current selection state
		self._sync_controls()

	def _on_toggle_visible(self):
		idx = self.sel_index
		if 0 <= idx < len(self.selection.items):
			vis = bool(self.selection.items[idx].options.get("visible", True))
			self.selection.update_options(idx, visible=not vis)
			self._render_selected_lineup(camera_preserve=True)
			self._sync_controls()

	def _on_list_selection_changed(self, *args):
		# Sync selected index from ListView callback or widget state
		idx = -1
		try:
			if args:
				idx = int(args[0])
			else:
				idx = int(getattr(self.list_view, 'selected_index', -1))
		except Exception:
			idx = int(getattr(self.list_view, 'selected_index', -1)) if hasattr(self.list_view, 'selected_index') else -1
		self.sel_index = idx
		self._sync_controls()

	def _render_selected_lineup(self, camera_preserve: bool = False):
		# Use selection model entries
		entries = self.selection.items
		# Resolve point size
		try:
			point_size = float(getattr(self, 'default_point_size', 3.0))
		except Exception:
			point_size = 3.0
		# Delegate rendering
		render_lineup(self.scene.scene, entries, point_size, preserve_camera=camera_preserve, overlay=self.overlay_mode)
		# Fit camera if not preserved
		if not camera_preserve:
			try:
				bounds = self.scene.scene.bounding_box
				self.scene.setup_camera(60.0, bounds, bounds.get_center())
			except Exception:
				pass

	def _sync_controls(self):
		# Enable/disable the fixed controls row based on selection and position
		N = len(self.selection.items)
		idx = self.sel_index
		en_has_sel = 0 <= idx < N
		try:
			self.btn_remove.enabled = bool(en_has_sel)
			self.btn_edit.enabled = bool(en_has_sel)
			self.btn_toggle_visible.enabled = bool(en_has_sel)
			if en_has_sel:
				vis = bool(self.selection.items[idx].options.get("visible", True))
				self.btn_toggle_visible.text = "Hide" if vis else "Show"
			self.btn_up.enabled = bool(en_has_sel and idx > 0)
			self.btn_down.enabled = bool(en_has_sel and idx < N - 1)
			# Overlay button reflects global overlay mode
			self.btn_overlay.text = "Overlay: On" if self.overlay_mode else "Overlay: Off"
			# No inline wireframe checkbox; wireframe is controlled via Edit dialog
		except Exception:
			pass
		try:
			self.window.set_needs_layout()
		except Exception:
			pass

	def _on_layout(self, layout_context):
		# Layout: left scene, right panel; in panel, list view top, controls row bottom
		r = self.window.content_rect
		panel_width = int(0.28 * r.width)
		self.scene.frame = gui.Rect(r.x, r.y, r.width - panel_width, r.height)
		self.panel.frame = gui.Rect(r.get_right() - panel_width, r.y, panel_width, r.height)
		# Pin list view and controls row regions
		panel_rect = self.panel.frame
		margin = 8
		px = panel_rect.x + margin
		py = panel_rect.y + margin
		pw = panel_rect.width - 2 * margin
		ph = panel_rect.height - 2 * margin
		controls_h = 48
		list_h = max(0, ph - controls_h - margin)
		try:
			self.list_view.frame = gui.Rect(px, py, pw, list_h)
		except Exception:
			pass
		try:
			self.controls_row.frame = gui.Rect(px, py + list_h + margin, pw, controls_h)
		except Exception:
			pass

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
			mat.point_size = float(self.default_point_size)
			self.scene.scene.add_geometry("pcd", pcd, mat)
			bounds = pcd.get_axis_aligned_bounding_box()
			self.scene.setup_camera(60.0, bounds, bounds.get_center())
			self.geometry = pcd
			self.is_pointcloud = True
			# Update feature dim combobox
			# No feature-based coloring in this simplified UI
		except Exception as e:
			self._show_message(f"Error loading point cloud: {e}")



	def _on_mouse(self, event):
		if event.type == gui.MouseEvent.Type.MOVE and self.is_pointcloud and self.geometry is not None:
			# Perform a pick near mouse to show tooltip with feature value
			# Note: GUI picking helpers are limited; do approximate nearest search in screen space
			# Here we skip heavy picking and just show a generic hint
			return gui.Widget.EventCallbackResult.HANDLED
		return gui.Widget.EventCallbackResult.IGNORED

	def _open_edit_dialog(self):
		# Open a movable editor window via dialogs.mesh_edit
		idx = self.sel_index
		if not (0 <= idx < len(self.selection.items)):
			return
		entry = self.selection.items[idx]
		def _apply_change(updated_entry_dict):
			# Update model options from dialog and re-render
			options = dict(updated_entry_dict.get("options", {}))
			self.selection.update_options(idx, **options)
			self._render_selected_lineup(camera_preserve=True)
		# Prepare legacy-like dict for dialog
		legacy_like = {
			"display": entry.display,
			"path": entry.path,
			"type": entry.type,
			"options": dict(entry.options),
		}
		# Keep reference to prevent GC
		self._last_edit_window = mesh_edit.open_mesh_edit_window(gui.Application.instance, self.window, legacy_like, _apply_change)

	def _on_selected_wireframe(self, _checked):
		# Deprecated: wireframe is controlled via Edit dialog only
		return

	def _on_toggle_overlay(self):
		# Toggle overlay mode and re-render
		self.overlay_mode = not bool(self.overlay_mode)
		self._sync_controls()
		self._render_selected_lineup(camera_preserve=True)

	def _show_message(self, msg: str):
		# Avoid modal dialogs that can obscure the UI; log to console for now.
		print(f"[Viewer] {msg}")


def main():
	app = VisualizationApp()
	gui.Application.instance.run()


if __name__ == "__main__":
	main()
