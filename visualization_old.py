import open3d as o3d
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle
from util import features as fe

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

def plot_feature_histogram(pcd_path, bins=20, title="Feature Histogram", xlable="Feature Value", ylabel="Frequency"):
    pcd_tensor_api= o3d.t.io.read_point_cloud(pcd_path)
    if 'features' in pcd_tensor_api.point:
        pcd_features = pcd_tensor_api.point['features'].numpy()
    else:
        raise ValueError("No Features in PCD.")
    plot_histogram(features=pcd_features, bins=bins, title=title, xlabel=xlable, ylabel=ylabel)



###############################################################################################
### 3D Visualizations
###############################################################################################

def get_categorical_colormap(num_features):
    colormap = plt.get_cmap("Dark2")
    colors = colormap(np.linspace(0, 1, len(num_features)))[:, :3]
    return {f: colors[idx] for idx, f in enumerate(num_features)}

def normalize_feats(features, min_feat=None, max_feat=None):
    if min_feat is None or max_feat is None:
        min_feat = np.min(features)
        max_feat = np.max(features)
    return (features - min_feat) / (max_feat - min_feat)

def color_mesh(mesh, features, mapping=None, color_mode='categorical'):
    if color_mode == 'categorical':
        if mapping is not None:
            unique_features = mapping.keys()
        else:
            raise ValueError("No mapping provided!")
        feature_to_color = get_categorical_colormap(unique_features)
        vertex_colors = np.array([feature_to_color[f] for f in features])
    elif color_mode == 'continuous':
        colormap = plt.get_cmap("turbo")
        normalized_feats = normalize_feats(features)
        vertex_colors = colormap(normalized_feats)[:, :3]
    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

def color_pcd(pcd, pcd_features, mesh_features, mapping=None, color_mode='categorical'):
    if color_mode == 'categorical':
        if mapping is not None:
            unique_features = mapping.keys()
        else:
            raise ValueError("No mapping provided!")
        feature_to_color = get_categorical_colormap(unique_features)
        pcd_colors = np.array([
            feature_to_color[round(f)] if round(f) in feature_to_color else [0, 0, 0]
                for f in pcd_features[:, 0]
        ])
    elif color_mode == 'continuous':
        colormap = plt.get_cmap("turbo")
        min_mesh_feat = np.min(mesh_features)
        max_mesh_feat = np.max(mesh_features)
        normalized_features = normalize_feats(pcd_features, min_feat=min_mesh_feat, max_feat=max_mesh_feat)
        print(np.squeeze(colormap(normalized_features[0])[:, :3]))
        pcd_colors = np.array([
            np.squeeze(colormap(f)[:, :3]) if f >= 0 and f <= 1 else [0, 0, 0]
                for f in normalized_features
        ])
        print(np.min(pcd_colors), np.max(pcd_colors))
    pcd.colors = o3d.utility.Vector3dVector(pcd_colors)

def visualize_mesh_cloud(mesh_path, pcd_path, pcd2_path=None, feature_path=None,
                         mapping_path=None, side_by_side=True, color_mode='categorical'):
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
    if mesh_path:
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        mesh.compute_vertex_normals()
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
    
    # pcd features
    pcd_features = None
    if 'features' in pcd_tensor_api.point:
        pcd_features = pcd_tensor_api.point['features'].numpy()
    pcd2_feaures = None
    if pcd2_tensor_api is not None and 'features' in pcd2_tensor_api.point:
        pcd2_feaures = pcd2_tensor_api.point['features'].numpy()

    # mesh features
    mesh_features = None
    if feature_path is not None:
        mesh_features, _, _ = fe.load_features_with_metadata(file_path=feature_path, expand_dims=False)

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
        color_mesh(mesh=mesh, features=mesh_features, mapping=mapping, color_mode=color_mode)
    
    # PCD coloring
    if pcd_features is not None:
        print(np.max(pcd_features), np.min(pcd_features))
        color_pcd(pcd=pcd, pcd_features=pcd_features, mesh_features=mesh_features, mapping=mapping, color_mode=color_mode)
    else:
        # Set all points to gray if no features are present
        gray_color = np.array([[0.5, 0.5, 0.5]] * np.asarray(pcd.points).shape[0])
        pcd.colors = o3d.utility.Vector3dVector(gray_color)
    if pcd2_feaures is not None:
        color_pcd(pcd=pcd2, pcd_features=pcd2_feaures, mesh_features=mesh_features, mapping=mapping, color_mode=color_mode)
    else:
        # Set all points to gray if no features are present
        gray_color = np.array([[0.5, 0.5, 0.5]] * np.asarray(pcd.points).shape[0])
        pcd.colors = o3d.utility.Vector3dVector(gray_color)

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
    mesh_path = "../data/shapes/wukong.obj"
    pcd_path = "../samples/shapes/wukong/sample-15.ply"
    pcd2_path = "../samples/shapes/wukong/sample-10.ply"
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
    visualize_mesh_cloud(mesh_path=mesh_path, pcd_path=pcd_path, pcd2_path=pcd2_path, 
                         feature_path=feature_path, mapping_path=mapping_path, 
                         side_by_side=True, color_mode=color_mode)