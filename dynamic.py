import polyscope as ps
import numpy as np
import h5py
import polyscope.imgui as psim


# =========================
# DATA LOADING
# =========================

def load_dfaust_sequence(h5_path, sequence_key):
    with h5py.File(h5_path, "r") as f:
        print("Available sequences:", list(f.keys()))

        verts = f[sequence_key][:]              # (V, 3, T)
        verts = np.transpose(verts, (2, 0, 1)) # (T, V, 3)

        faces = f["faces"][:]

    return verts, faces


def load_ply_with_time(path):
    points, colors, times = [], [], []

    with open(path, "r") as f:
        header = True
        for line in f:
            if header:
                if line.strip() == "end_header":
                    header = False
                continue

            x, y, z, r, g, b, t = line.split()
            points.append([float(x), float(y), float(z)])
            colors.append([int(r), int(g), int(b)])
            times.append(float(t))

    points = np.array(points)
    colors = np.array(colors) / 255.0
    times = np.clip(np.array(times), 0.0, 1.0)

    return points, colors, times


def compute_time_bins(times, num_frames):
    return (times * (num_frames - 1)).astype(int)


# =========================
# MAIN
# =========================

def main():
    h5_path = "/home/ben/LRZSyncShare/Thesis/Data/data/DFAUST/registrations_m.hdf5"
    sequence_key = "50002_chicken_wings"
    cloud_path = "/home/ben/LRZSyncShare/Thesis/Data/samples/dfaust_test/shape-B_e0005_n3000000.ply"

    ps.init()

    # --- Mesh ---
    verts, faces = load_dfaust_sequence(h5_path, sequence_key)
    print("Sequence shape:", verts.shape)

    ps_mesh = ps.register_surface_mesh(
        "dfaust_mesh",
        verts[0],
        faces,
        smooth_shade=True
    )

    # Wireframe-like look
    ps_mesh.set_edge_width(1.0)
    ps_mesh.set_edge_color((0.0, 1.0, 0.0))
    ps_mesh.set_transparency(0.4)

    # --- Point cloud ---
    points, colors, times = load_ply_with_time(cloud_path)
    time_bins = compute_time_bins(times, len(verts))

    # =========================
    # PRECOMPUTE BINS
    # =========================

    pcs = {}

    print("Precomputing point cloud bins...")

    for i in range(len(verts)):
        mask = time_bins == i

        if np.any(mask):
            pc_i = ps.register_point_cloud(f"samples_{i}", points[mask])
            pc_i.add_color_quantity("color", colors[mask], enabled=True)
            pc_i.set_enabled(False)
            pcs[i] = pc_i

    print(f"Created {len(pcs)} non-empty bins")

    # =========================
    # CALLBACK
    # =========================

    current_frame = 0
    autoplay = False
    show_points = True
    show_all = False
    point_radius = 1e-3  # initial

    def callback():
        nonlocal current_frame, autoplay, show_points, show_all, point_radius

        psim.Text("Controls")
        psim.Separator()

        # --- toggles ---
        _, autoplay = psim.Checkbox("Autoplay", autoplay)
        _, show_points = psim.Checkbox("Show Points", show_points)
        _, show_all = psim.Checkbox("Show All Bins", show_all)

        # --- frame slider ---
        changed, current_frame = psim.SliderInt(
            "Frame", current_frame, 0, len(verts) - 1
        )

        # --- log radius slider ---
        log_radius = np.log10(point_radius)
        _, log_radius = psim.SliderFloat(
            "log10(Point Size)", log_radius, -5, -2
        )
        point_radius = 10 ** log_radius

        # --- autoplay ---
        if autoplay and not changed:
            current_frame = (current_frame + 1) % len(verts)

        # --- update mesh ---
        ps_mesh.update_vertex_positions(verts[current_frame])

        # --- update point clouds ---
        for i, pc_i in pcs.items():
            visible = show_points and (show_all or i == current_frame)
            pc_i.set_enabled(visible)

            if visible:
                pc_i.set_radius(point_radius)

    ps.set_user_callback(callback)
    ps.show()


if __name__ == "__main__":
    main()