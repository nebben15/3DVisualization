import open3d as o3d
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle

###############################################################################################
### Histograms
###############################################################################################

def load_features_with_metadata(file_path, expand_dims=True):
    """
    Load features using the new unified header format only.

    Header lines supported:
        - FEATURES_COUNT <int>
        - FEATURES_DIM <int>
        - FEATURES_MAX_ABS <float ... float>

    Returns:
        features: np.ndarray of shape (rows, dim)
        max_meta: np.ndarray of shape (dim,), per-dimension max magnitude (from header if present, else derived)
        dim_out: int, number of feature dimensions (columns)
    """
    with open(file_path, 'r') as f:
        # Read header lines
        first = f.readline().strip()
        header = {"FEATURES_COUNT": None, "FEATURES_DIM": None, "FEATURES_MAX_ABS": None}

        def _try_parse_header(line: str):
            if not line:
                return False
            if line.startswith('FEATURES_COUNT'):
                parts = line.split()
                header["FEATURES_COUNT"] = int(parts[1])
                return True
            if line.startswith('FEATURES_DIM'):
                parts = line.split()
                header["FEATURES_DIM"] = int(parts[1])
                return True
            if line.startswith('FEATURES_MAX_ABS'):
                parts = line.split()
                # parts[0] is label; rest are floats
                header["FEATURES_MAX_ABS"] = np.array([float(p) for p in parts[1:]], dtype=float)
                return True
            return False

        if not _try_parse_header(first):
            raise ValueError("Missing unified header. Expected FEATURES_COUNT/DIM/MAX_ABS lines.")

        # Read additional header lines if present (up to 2 more expected)
        for _ in range(2):
            pos = f.tell()
            ln = f.readline()
            if not ln:
                break
            if not _try_parse_header(ln.strip()):
                # Not a header line; rewind to start of this data line
                f.seek(pos)
                break

        # Load remaining lines as numeric features
        features = np.loadtxt(f, dtype=float)
        if features.ndim == 0:
            features = features.reshape(-1, 1)
        elif features.ndim == 1:
            # If header provides DIM, reshape accordingly; else assume single column
            dim = header["FEATURES_DIM"]
            if dim is not None and features.size % dim == 0:
                features = features.reshape(-1, dim)
            else:
                features = features.reshape(-1, 1)

        # Validate vs header
        if header["FEATURES_COUNT"] is not None and features.shape[0] != header["FEATURES_COUNT"]:
            raise ValueError(f"Feature row count mismatch: header {header['FEATURES_COUNT']} vs data {features.shape[0]}")
        if header["FEATURES_DIM"] is not None and features.shape[1] != header["FEATURES_DIM"]:
            raise ValueError(f"Feature dimension mismatch: header {header['FEATURES_DIM']} vs data {features.shape[1]}")

        max_meta = header["FEATURES_MAX_ABS"]
        if max_meta is None:
            # Derive maxima if not provided
            max_meta = np.max(np.abs(features), axis=0) if features.size else np.zeros((features.shape[1],), dtype=float)

        # Optionally expand dims for 1D inputs (rare with explicit DIM)
        if expand_dims and features.ndim == 1:
            features = np.expand_dims(features, axis=1)

    dim_out = features.shape[1]
    return features, max_meta, dim_out

###############################################################################################
### Histograms
###############################################################################################

def plot_histogram(features, bins=20, title="Feature Histogram", xlabel="Feature Value", ylabel="Frequency"):
    """
    Plots a histogram for 1D features.

    Args:
        features (np.ndarray): 1D array of feature values.
        bins (int): Number of bins for the histogram.
        title (str): Title of the histogram.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
    """
    #matplotlib.use('TkAgg')
    plt.figure(figsize=(8, 6))
    plt.hist(features, bins=bins, color='blue', alpha=0.7, edgecolor='black')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

def plot_feature_histogram(pcd_path, bins=50, title="Feature Histogram", xlable="Feature Value", ylabel="Frequency"):
    """
    Plot one histogram per feature dimension for a point cloud.

    Supports features embedded as separate point properties (feat_dim_*) or a single
    'features' tensor. If multiple dims are present, creates subplots.
    """
    pcd_tensor_api = o3d.t.io.read_point_cloud(pcd_path)
    # Prefer new embedded per-dimension properties feat_dim_* if present
    try:
        point_keys = [str(k) for k in pcd_tensor_api.point]
    except Exception:
        try:
            point_keys = list(pcd_tensor_api.point.keys())
        except Exception:
            point_keys = []
    feat_keys = [k for k in point_keys if k.startswith('feat_dim_')]

    def _to_np(t):
        try:
            return t.numpy()
        except Exception:
            return np.asarray(t)

    if feat_keys:
        # Sort by numeric suffix to ensure correct column order
        def _suffix_idx(k):
            try:
                return int(k.split('feat_dim_')[-1])
            except Exception:
                return 0
        feat_keys_sorted = sorted(feat_keys, key=_suffix_idx)
        feats_list = [_to_np(pcd_tensor_api.point[k]).reshape(-1, 1) for k in feat_keys_sorted]
        pcd_features = np.concatenate(feats_list, axis=1)
        dim = pcd_features.shape[1]
    elif 'features' in point_keys:
        pcd_features = _to_np(pcd_tensor_api.point['features'])
        if pcd_features.ndim == 1:
            pcd_features = pcd_features.reshape(-1, 1)
        dim = pcd_features.shape[1]
    else:
        raise ValueError("No Features found in PCD (expected 'feat_dim_*' or 'features').")

    # Plot one histogram per dimension
    cols = min(dim, 3)
    rows = int(np.ceil(dim / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
    axes = np.array(axes).reshape(-1)
    for d in range(dim):
        ax = axes[d]
        ax.hist(pcd_features[:, d], bins=bins, color='blue', alpha=0.7, edgecolor='black')
        ax.set_title(f"{title} (dim {d})")
        ax.set_xlabel(xlable)
        ax.set_ylabel(ylabel)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
    # Hide any unused subplots
    for i in range(dim, len(axes)):
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()


def plot_mesh_vertices_histogram(mesh_path, bins=50, title="Vertex Coordinate Histogram", xlable="Value", ylabel="Frequency"):
    """
    Plot one histogram per geometric coordinate (X, Y, Z) of mesh vertices.

    This function intentionally ignores any per-vertex feature attributes and
    uses only the vertex positions as the data source.
    """
    # Try tensor mesh first to read positions reliably
    positions = None
    try:
        mesh_t = o3d.t.io.read_triangle_mesh(mesh_path)
        try:
            positions = mesh_t.vertex['positions']
            # Convert to numpy
            try:
                positions = positions.numpy()
            except Exception:
                positions = np.asarray(positions)
        except Exception:
            positions = None
    except Exception:
        mesh_t = None

    # Legacy mesh fallback
    if positions is None:
        try:
            mesh = o3d.io.read_triangle_mesh(mesh_path)
            positions = np.asarray(mesh.vertices)
        except Exception:
            positions = None

    if positions is None or positions.size == 0:
        raise ValueError("No mesh vertex positions found to plot.")

    # Ensure 2D array and select first 3 columns for X,Y,Z
    if positions.ndim == 1:
        positions = positions.reshape(-1, 1)
    if positions.shape[1] < 3:
        raise ValueError(f"Expected at least 3 coordinate dimensions, got {positions.shape[1]}.")
    coords = positions[:, :3]

    labels = ['X', 'Y', 'Z']
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    for i in range(3):
        ax = axes[i]
        ax.hist(coords[:, i], bins=bins, color='blue', alpha=0.7, edgecolor='black')
        ax.set_title(f"{title} ({labels[i]})")
        ax.set_xlabel(xlable)
        ax.set_ylabel(ylabel)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()



###############################################################################################
### 3D Visualizations
###############################################################################################

def get_categorical_colormap(num_features):
    colormap = plt.get_cmap("Dark2")
    colors = colormap(np.linspace(0, 1, len(num_features)))[:, :3]
    return {f: colors[idx] for idx, f in enumerate(num_features)}

def normalize_feats(features, min_feat=None, max_feat=None):
    """
    Normalize features to [0, 1].

    Supports arbitrary feature dimensions:
    - If features is 1D or shape (N, 1), uses scalar min/max.
    - If features is shape (N, D), performs per-dimension normalization using
      provided min_feat/max_feat vectors (shape (D,)) or derives them per dim.
    """
    feats = np.asarray(features)
    if feats.ndim == 1:
        # Treat as single column
        if min_feat is None:
            min_feat = float(np.min(feats))
        if max_feat is None:
            max_feat = float(np.max(feats))
        denom = (max_feat - min_feat) if (max_feat - min_feat) != 0 else 1.0
        return (feats - min_feat) / denom
    else:
        # Per-dimension normalization
        if min_feat is None:
            min_feat = np.min(feats, axis=0)
        if max_feat is None:
            max_feat = np.max(feats, axis=0)
        min_feat = np.asarray(min_feat)
        max_feat = np.asarray(max_feat)
        denom = (max_feat - min_feat)
        # Avoid division by zero per dimension
        denom = np.where(denom == 0, 1.0, denom)
        return (feats - min_feat) / denom



def color_mesh(mesh, features, mapping=None, color_mode='categorical', color_dims=None):
    """
    Color a mesh using features with arbitrary dimensions.

    Args:
        mesh: Open3D triangle mesh
        features: np.ndarray shape (N,) or (N, D)
        mapping: for categorical mode, dict mapping feature id -> name/color index
        color_mode: 'categorical' or 'continuous'
        color_dims: None, int, or sequence of 3 ints
            - None: auto behavior. If D==3, use RGB per-dim normalization; if D>3, use first 3 dims; if D==1, use colormap.
            - int: use that dimension as scalar for colormap.
            - sequence of 3 ints: use these dims as RGB after per-dim normalization.
    """
    feats = np.asarray(features)
    if color_mode == 'categorical':
        if mapping is None:
            raise ValueError("No mapping provided!")
        unique_features = mapping.keys()
        feature_to_color = get_categorical_colormap(unique_features)
        # If multi-dim, default to first dim as label
        labels = feats if feats.ndim == 1 else feats[:, 0]
        vertex_colors = np.array([feature_to_color[int(round(f))] if int(round(f)) in feature_to_color else [0, 0, 0]
                                  for f in labels])
    elif color_mode == 'continuous':
        colormap = plt.get_cmap("turbo")
        if feats.ndim == 1 or (feats.ndim == 2 and feats.shape[1] == 1) or isinstance(color_dims, int):
            # Scalar -> colormap
            if isinstance(color_dims, int):
                scalar = feats[:, color_dims] if feats.ndim == 2 else feats
            else:
                scalar = feats[:, 0] if feats.ndim == 2 else feats
            scalar_norm = normalize_feats(scalar)
            vertex_colors = colormap(scalar_norm)[:, :3]
        else:
            # Use 3 dims as RGB, normalize per dim
            if color_dims is None:
                # Auto-pick: use first 3 dims if available
                if feats.ndim == 2 and feats.shape[1] >= 3:
                    dims = [0, 1, 2]
                else:
                    # Fallback: treat 1D as scalar (handled above), or pad
                    dims = [0]
            elif isinstance(color_dims, (list, tuple)) and len(color_dims) == 3:
                dims = list(color_dims)
            else:
                # If invalid color_dims provided, default to first dim scalar colormap
                scalar = feats[:, 0] if feats.ndim == 2 else feats
                scalar_norm = normalize_feats(scalar)
                vertex_colors = colormap(scalar_norm)[:, :3]
                mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
                return

            if len(dims) == 3:
                rgb = feats[:, dims]
                rgb_norm = normalize_feats(rgb)
                # Ensure values in [0,1]
                rgb_norm = np.clip(rgb_norm, 0.0, 1.0)
                vertex_colors = rgb_norm
            else:
                # Single dim fallback
                scalar = feats[:, dims[0]] if feats.ndim == 2 else feats
                scalar_norm = normalize_feats(scalar)
                vertex_colors = colormap(scalar_norm)[:, :3]
    else:
        raise ValueError(f"Unsupported color_mode: {color_mode}")
    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

def color_pcd(pcd, pcd_features, mesh_features, mapping=None, color_mode='categorical', color_dims=None, use_mesh_normalization=True):
    """
    Color a point cloud using features with arbitrary dimensions.

    For continuous mode, normalization uses mesh feature min/max when provided
    to keep colors consistent across mesh and point cloud. Otherwise, derives
    per-dimension min/max from the point cloud features.
    """
    feats_p = np.asarray(pcd_features)
    if color_mode == 'categorical':
        if mapping is None:
            raise ValueError("No mapping provided!")
        unique_features = mapping.keys()
        feature_to_color = get_categorical_colormap(unique_features)
        labels = feats_p[:, 0] if feats_p.ndim == 2 else feats_p
        pcd_colors = np.array([
            feature_to_color[int(round(f))] if int(round(f)) in feature_to_color else [0, 0, 0]
            for f in labels
        ])
    elif color_mode == 'continuous':
        colormap = plt.get_cmap("turbo")
        # Determine normalization anchors from mesh if available and desired
        if mesh_features is not None and use_mesh_normalization:
            mesh_feats = np.asarray(mesh_features)
            if feats_p.ndim == 1 or (feats_p.ndim == 2 and feats_p.shape[1] == 1):
                min_mesh_feat = float(np.min(mesh_feats))
                max_mesh_feat = float(np.max(mesh_feats))
            else:
                min_mesh_feat = np.min(mesh_feats, axis=0)
                max_mesh_feat = np.max(mesh_feats, axis=0)
        else:
            min_mesh_feat = None
            max_mesh_feat = None

        if feats_p.ndim == 1 or (feats_p.ndim == 2 and feats_p.shape[1] == 1) or isinstance(color_dims, int):
            # Scalar -> colormap
            if isinstance(color_dims, int):
                scalar = feats_p[:, color_dims] if feats_p.ndim == 2 else feats_p
            else:
                scalar = feats_p[:, 0] if feats_p.ndim == 2 else feats_p
            scalar_norm = normalize_feats(scalar, min_feat=min_mesh_feat, max_feat=max_mesh_feat)
            pcd_colors = colormap(scalar_norm)[:, :3]
        else:
            # Use 3 dims as RGB
            if color_dims is None:
                if feats_p.ndim == 2 and feats_p.shape[1] >= 3:
                    dims = [0, 1, 2]
                else:
                    dims = [0]
            elif isinstance(color_dims, (list, tuple)) and len(color_dims) == 3:
                dims = list(color_dims)
            else:
                scalar = feats_p[:, 0] if feats_p.ndim == 2 else feats_p
                scalar_norm = normalize_feats(scalar, min_feat=min_mesh_feat, max_feat=max_mesh_feat)
                pcd_colors = colormap(scalar_norm)[:, :3]
                pcd.colors = o3d.utility.Vector3dVector(pcd_colors)
                return

            if len(dims) == 3:
                rgb = feats_p[:, dims]
                # Per-dim normalization using mesh anchors if available
                rgb_norm = normalize_feats(rgb, min_feat=(min_mesh_feat if isinstance(min_mesh_feat, np.ndarray) else None),
                                           max_feat=(max_mesh_feat if isinstance(max_mesh_feat, np.ndarray) else None))
                pcd_colors = np.clip(rgb_norm, 0.0, 1.0)
            else:
                # Single dim fallback
                scalar = feats_p[:, dims[0]] if feats_p.ndim == 2 else feats_p
                scalar_norm = normalize_feats(scalar, min_feat=min_mesh_feat, max_feat=max_mesh_feat)
                pcd_colors = colormap(scalar_norm)[:, :3]
    else:
        raise ValueError(f"Unsupported color_mode: {color_mode}")
    pcd.colors = o3d.utility.Vector3dVector(pcd_colors)

def visualize_mesh_cloud(mesh_path, pcd_path, pcd2_path=None, feature_path=None,
                         mapping_path=None, side_by_side=True, color_mode='categorical',
                         color_dims=None, use_mesh_normalization=True):
    """
    Visualize a mesh and point cloud, with optional feature-based coloring.

    Args:
        mesh_path (str): Path to the mesh file.
        pcd_path (str): Path to the point cloud file.
        feature_path (str, optional): Path to the .txt file containing feature vectors for the mesh.
        mapping_path (str, optional): Path to the .pkl file mapping feature IDs to class names.
        side_by_side (bool): Whether to display the point cloud next to the mesh.
    """
    # Initialize the GUI app
    o3d.visualization.gui.Application.instance.initialize()
    
    # Load mesh and point cloud
    mesh = None
    mesh_tensor_api = None
    if mesh_path:
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        mesh.compute_vertex_normals()
        # Also read tensor mesh to access per-vertex attributes if present
        try:
            mesh_tensor_api = o3d.t.io.read_triangle_mesh(mesh_path)
        except Exception:
            mesh_tensor_api = None
    pcd = None
    pcd_tensor_api = None
    if pcd_path:
        pcd = o3d.io.read_point_cloud(pcd_path)
        pcd_tensor_api = o3d.t.io.read_point_cloud(pcd_path)
    pcd2 = None
    pcd2_tensor_api = None
    if pcd2_path:
        pcd2 = o3d.io.read_point_cloud(pcd2_path)
        pcd2_tensor_api = o3d.t.io.read_point_cloud(pcd2_path)

    # Optionally move the point cloud to the side
    if side_by_side:
        bbox = mesh.get_axis_aligned_bounding_box()
        offset = bbox.get_extent()[0] * 1.5
        pcd.translate([offset, 0, 0])
        if pcd2:
            pcd2.translate([2*offset, 0, 0])
    # Define a callback to toggle side_by_side
    def toggle_side_by_side(action):
        nonlocal side_by_side
        side_by_side = not side_by_side
        bbox = mesh.get_axis_aligned_bounding_box()
        offset = bbox.get_extent()[0] * 1.5
        if side_by_side:
            pcd.translate([offset, 0, 0])
        else:
            pcd.translate([-offset, 0, 0])  # Move back to original position
        vis.remove_geometry("pointcloud")
        vis.add_geometry("pointcloud", pcd)
    
    # Helper to extract features from o3d.t point cloud with embedded columns
    def _extract_pcd_features(pcd_t):
        # Prefer new embedded per-dimension properties feat_dim_* if present
        try:
            point_keys = [str(k) for k in pcd_t.point]
        except Exception:
            try:
                point_keys = list(pcd_t.point.keys())
            except Exception:
                point_keys = []
        feat_keys = [k for k in point_keys if k.startswith('feat_dim_')]
        if feat_keys:
            def _suffix_idx(k):
                try:
                    return int(k.split('feat_dim_')[-1])
                except Exception:
                    return 0
            feat_keys_sorted = sorted(feat_keys, key=_suffix_idx)
            def _to_np(t):
                try:
                    return t.numpy()
                except Exception:
                    return np.asarray(t)
            feats_list = [_to_np(pcd_t.point[k]).reshape(-1, 1) for k in feat_keys_sorted]
            return np.concatenate(feats_list, axis=1)
        elif 'features' in point_keys:
            return _to_np(pcd_t.point['features'])
        else:
            return None

    # pcd features
    pcd_features = None
    if pcd_tensor_api is not None:
        pcd_features = _extract_pcd_features(pcd_tensor_api)
    pcd2_feaures = None
    if pcd2_tensor_api is not None:
        pcd2_feaures = _extract_pcd_features(pcd2_tensor_api)

    # mesh features
    mesh_features = None
    if feature_path is not None:
        mesh_features, _, _ = load_features_with_metadata(file_path=feature_path, expand_dims=False)
    # If no external feature file, try to extract from mesh vertex attributes (feat_dim_*)
    if mesh_features is None and mesh_tensor_api is not None:
        def _extract_vertex_features(mesh_t):
            try:
                vkeys = [str(k) for k in mesh_t.vertex]
            except Exception:
                try:
                    vkeys = list(mesh_t.vertex.keys())
                except Exception:
                    vkeys = []
            feat_keys = [k for k in vkeys if k.startswith('feat_dim_')]
            if not feat_keys:
                # Fallback: use vertex positions (XYZ) as features if available
                try:
                    if 'positions' in vkeys:
                        t = mesh_t.vertex['positions']
                        # positions may be (N,3)
                        try:
                            arr = t.numpy()
                        except Exception:
                            arr = np.asarray(t)
                        return np.asarray(arr)
                except Exception:
                    pass
                return None
            def _suffix_idx(k):
                try:
                    return int(k.split('feat_dim_')[-1])
                except Exception:
                    return 0
            feat_keys_sorted = sorted(feat_keys, key=_suffix_idx)
            def _to_np(t):
                try:
                    return t.numpy()
                except Exception:
                    return np.asarray(t)
            feats_list = [_to_np(mesh_t.vertex[k]).reshape(-1, 1) for k in feat_keys_sorted]
            return np.concatenate(feats_list, axis=1)
        mesh_features = _extract_vertex_features(mesh_tensor_api)
    # If tensor mesh isn't available or yielded None, try legacy mesh vertices (XYZ)
    if mesh_features is None and mesh is not None:
        try:
            mesh_features = np.asarray(mesh.vertices)
        except Exception:
            mesh_features = None

    # mapping
    mapping = None
    if mapping_path is not None:
        with open(mapping_path, "rb") as f:
            mapping = pickle.load(f)

    # Sanity Checks
    # if features provided -> mapping provided or continuous
    assert pcd_features is None or (mapping is not None or color_mode == 'continuous'), "pcd-features provided but no mapping provided"
    assert mesh_features is None or (mapping is not None or color_mode == 'continuous'), "mesh-features provided but no mapping provided"
    # num feature dims 
    feat_dim = 0
    if pcd_features is not None and mesh_features is not None:
        # feat_dim = pcd_features.shape[1]
        # assert feat_dim == mesh_features.shape[1], "#vertex-featurs != #pcd-features"
        # TODO check mapping features and unique features
        pass
    elif pcd_features is not None:
        # feat_dim = pcd_features.shape[1]
        # TODO check mapping features and unique features\
        pass
    elif mesh_features is not None:
        # feat_dim = mesh_features.shape[1]
        # TODO check mapping features and unique features
        pass

    
    # Mesh coloring
    if mesh_features is not None:
        color_mesh(mesh=mesh, features=mesh_features, mapping=mapping, color_mode=color_mode, color_dims=color_dims)
    
    # PCD coloring
    if pcd_features is not None:
        print(np.max(pcd_features), np.min(pcd_features))
        color_pcd(pcd=pcd, pcd_features=pcd_features, mesh_features=mesh_features, mapping=mapping,
                  color_mode=color_mode, color_dims=color_dims, use_mesh_normalization=use_mesh_normalization)
    else:
        # Set all points to gray if no features are present
        gray_color = np.array([[0.5, 0.5, 0.5]] * np.asarray(pcd.points).shape[0])
        pcd.colors = o3d.utility.Vector3dVector(gray_color)

    # Second PCD coloring
    if pcd2 is not None:
        if pcd2_feaures is not None:
            color_pcd(pcd=pcd2, pcd_features=pcd2_feaures, mesh_features=mesh_features, mapping=mapping,
                      color_mode=color_mode, color_dims=color_dims, use_mesh_normalization=use_mesh_normalization)
        else:
            # Set all points to gray for the second cloud if no features are present
            gray_color2 = np.array([[0.5, 0.5, 0.5]] * np.asarray(pcd2.points).shape[0])
            pcd2.colors = o3d.utility.Vector3dVector(gray_color2)

    # Legend
    print(mapping)

    # Create the visualizer window
    vis = o3d.visualization.O3DVisualizer("Mesh + PointCloud", 1024, 768)
    vis.add_geometry("mesh", mesh)
    vis.add_geometry("pointcloud", pcd)
    if pcd2:
        vis.add_geometry("pointcloud2", pcd2)
    vis.show_settings = True  # makes right-hand menu visible
    vis.add_action("Toggle Side-by-Side", toggle_side_by_side)

    # Run the app
    app = o3d.visualization.gui.Application.instance
    app.add_window(vis)
    app.run()
    

if __name__ == "__main__":
    mapping_path = None
    feature_path = None
    pcd2_path = None
    color_mode = 'categorical'
    ### loong
    # mesh_path = "../data/shapes/wukong.obj"
    # pcd_path = "../samples/shapes/wukong/sample-15.ply"
    # pcd2_path = "../samples/shapes/wukong/sample-10.ply"
    ### spot
    #mesh_path = "../data/shapes/spot/spot_uv_normalized.obj"
    # pcd_path = "../samples/shapes/spot_color/self_trained_45_epochs.ply"
    # pcd2_path = "../samples/shapes/spot_color/self_trained_200_epochs.ply"
    #pcd_path = "../samples/shapes/spot_color/sample_45.ply"
    #pcd_path = "../samples/shapes/spot/sample_5.ply"
    ### FAUST
    # mesh_path = "../MPI-FAUST/training/registrations/tr_reg_000.ply"
    # pcd_path = "../samples/FAUST/sample.ply"
    ### FAUST with semantic features
    # mesh_path = "../MPI-FAUST/training/registrations/tr_reg_000.ply"
    # feature_path = "../SMPL_python_v.1.1.0/smpl_vert_segmentation.txt"
    # mapping_path = "../SMPL_python_v.1.1.0/smpl_vert_segmentation_mapping.pkl"
    # pcd_path = "../samples/FAUST_scaling/sample-5.ply"
    # pcd2_path = "../samples/FAUST_scaling_depth8/sample-20.ply"
    #pcd_path = "../samples/FAUST_wrong_scaling/sample-5.ply"
    ### FAUST with vertex-ids
    # color_mode = 'continuous'
    # mesh_path = "../MPI-FAUST/training/registrations/tr_reg_000.ply"
    # feature_path = "../SMPL_python_v.1.1.0/smpl_template_indices.txt"
    # pcd_path = "../samples/FAUST_vertexid/sample-30.ply"
    ### FAUST with template vertices
    color_mode = 'continuous'
    # Optionally select dims for coloring: int for scalar colormap or list of 3 for RGB
    # Examples: color_dims = 0  or color_dims = [0,1,2]
    color_dims = [0,1,2]
    # Control whether point cloud normalization uses mesh min/max anchors (for direct comparison)
    use_mesh_normalization = True
    mesh_path = '/home/ben/Desktop/LRZ Sync+Share/Thesis/Data/data/MPI-FAUST/training/registrations/tr_reg_000.ply'
    pcd_path = '/home/ben/Desktop/LRZ Sync+Share/Thesis/Data/samples/FAUST_smpl_template_000/sample_e40_n300000.ply'
    visualize_mesh_cloud(mesh_path=mesh_path, pcd_path=pcd_path, pcd2_path=pcd2_path,
                         feature_path=feature_path, mapping_path=mapping_path,
                         side_by_side=True, color_mode=color_mode, color_dims=color_dims,
                         use_mesh_normalization=use_mesh_normalization)