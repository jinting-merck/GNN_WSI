#!/usr/bin/env python3
"""
Fixed coordinate system for patch visualization on binary mask.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

def identify_and_clean_main_subject(path_annotation, fseg_data):
    """Identify main subject and clean both path_annotation and fseg_data."""
    print("\n=== Identifying and Cleaning Main Subject ===")
    
    from skimage.filters import threshold_otsu
    from skimage.measure import label, regionprops
    
    # Apply Otsu thresholding on path_annotation
    thresh = threshold_otsu(path_annotation)
    binary = path_annotation > thresh
    
    # Find connected components
    labeled = label(binary)
    regions = regionprops(labeled)
    
    print(f"Found {len(regions)} distinct subjects in the region")
    
    if len(regions) == 1:
        print("✓ Single subject found - using entire region")
        return path_annotation, fseg_data  # No changes needed
    
    elif len(regions) > 1:
        print("⚠ Multiple subjects found - selecting main one")
        
        # Calculate center of LOCAL bounding box (NPY region)
        height, width = path_annotation.shape
        local_center_y, local_center_x = height // 2, width // 2
        print(f"Local center of bounding box: ({local_center_x}, {local_center_y})")
        
        # Find region closest to local center
        main_region = None
        min_distance = float('inf')
        
        for i, region in enumerate(regions):
            centroid = region.centroid
            distance = np.sqrt((centroid[0] - local_center_y)**2 + (centroid[1] - local_center_x)**2)
            print(f"  Region {i+1}: centroid {centroid}, distance to center: {distance:.2f}")
            
            if distance < min_distance:
                min_distance = distance
                main_region = region
        
        # Create mask with only the main region
        main_mask = np.zeros_like(path_annotation)
        main_mask[labeled == main_region.label] = 1
        
        # Clean both masks
        cleaned_path_annotation = path_annotation * main_mask
        cleaned_fseg_data = fseg_data * main_mask
        
        print(f"✓ Selected main subject (centroid: {main_region.centroid})")
        print(f"✓ Cleaned irrelevant regions from both masks")
        
        return cleaned_path_annotation, cleaned_fseg_data
    
    else:
        print("✗ No subjects found!")
        return None, None

def load_region_data(npy_file_path):
    """Load region data from NPY file."""
    try:
        region_data = np.load(npy_file_path, allow_pickle=True).item()
        print(f"✓ Loaded: {region_data['wsi_id']} - Region {region_data['region_id']}")
        return region_data
    except Exception as e:
        print(f"✗ Error loading {npy_file_path}: {e}")
        raise

def validate_patches(fseg_data, path_annotation, patches):
    """Validate patches and extract FSEG percentage features using LOCAL coordinates."""
    print("\n=== Patch Validation and Feature Extraction (Local Coordinates) ===")
    
    valid_patches = []
    valid_features = []
    valid_centers = []
    
    # Calculate region offset for coordinate conversion
    all_minr = [patch[0] for patch in patches]
    all_minc = [patch[1] for patch in patches]
    region_offset_r = min(all_minr)  # Offset from WSI to region
    region_offset_c = min(all_minc)  # Offset from WSI to region
    
    print(f"Region offset from WSI: ({region_offset_r}, {region_offset_c})")
    
    for i, patch in enumerate(patches):
        # Extract patch coordinates (GLOBAL WSI coordinates)
        minr, minc, maxr, maxc = patch
        
        # Convert to LOCAL region coordinates
        rel_minr = minr - region_offset_r
        rel_minc = minc - region_offset_c
        rel_maxr = maxr - region_offset_r
        rel_maxc = maxc - region_offset_c
        
        # Extract submatrices using LOCAL coordinates
        patch_fseg = fseg_data[rel_minr:rel_maxr, rel_minc:rel_maxc]
        patch_path = path_annotation[rel_minr:rel_maxr, rel_minc:rel_maxc]
        
        # Check if path_annotation contains any foreground (1s)
        has_foreground = np.any(patch_path > 0)
        
        if has_foreground:
            # Valid patch - calculate FSEG percentage features
            
            # Calculate percentage of each FSEG value (0-10)
            feature_vector = []
            for fseg_value in range(11):  # 0 to 10
                percentage = np.sum(patch_fseg == fseg_value) / patch_fseg.size
                feature_vector.append(percentage)
            
            # Verify sum equals 1.0
            feature_sum = np.sum(feature_vector)
            
            # Calculate patch center for spatial positioning (use LOCAL coordinates)
            center_x = rel_minc + (rel_maxc - rel_minc) / 2
            center_y = rel_minr + (rel_maxr - rel_minr) / 2
            
            valid_patches.append({
                'patch_id': i+1,
                'coordinates': patch,  # Keep original global coordinates for reference
                'local_coordinates': (rel_minr, rel_minc, rel_maxr, rel_maxc),  # Add local coordinates
                'center': (center_x, center_y),  # Local center
                'size': patch_fseg.shape
            })
            valid_features.append(feature_vector)
            valid_centers.append([center_x, center_y])
    
    print(f"\nSummary: {len(valid_patches)} valid patches out of {len(patches)} total")
    
    return valid_patches, np.array(valid_features), np.array(valid_centers)

def check_connectivity(edges, num_nodes):
    """Check if graph is connected using NetworkX."""
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(edges)
    
    # Check if graph is connected
    is_connected = nx.is_connected(G)
    
    # Get number of connected components
    num_components = nx.number_connected_components(G)
    
    # Get size of largest component
    largest_component_size = len(max(nx.connected_components(G), key=len))
    
    # Get component sizes
    component_sizes = [len(c) for c in nx.connected_components(G)]
    
    return {
        'is_connected': is_connected,
        'num_components': num_components,
        'largest_component_size': largest_component_size,
        'connectivity_percentage': largest_component_size / num_nodes * 100,
        'component_sizes': component_sizes
    }

def filter_edges_by_similarity(edges, node_features, top_percentile=0.8):
    """Filter edges to keep top X% most similar neighbors based on cosine similarity."""
    print(f"\n=== Filtering Edges by Feature Similarity (Top {top_percentile*100}%) ===")
    
    edge_similarities = []
    
    for edge in edges:
        i, j = edge
        # Calculate cosine similarity between patch features
        similarity = cosine_similarity([node_features[i]], [node_features[j]])[0][0]
        edge_similarities.append((edge, similarity))
    
    # Sort by similarity (highest first)
    edge_similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Keep top X%
    num_to_keep = int(len(edge_similarities) * top_percentile)
    filtered_edges = [edge for edge, similarity in edge_similarities[:num_to_keep]]
    
    print(f"Original edges: {len(edges)}")
    print(f"Filtered edges (top {top_percentile*100}%): {len(filtered_edges)}")
    print(f"Removed {len(edges) - len(filtered_edges)} edges due to low similarity")
    
    # Check connectivity after filtering
    connectivity_info = check_connectivity(filtered_edges, len(node_features))
    
    if not connectivity_info['is_connected']:
        print(f"⚠ WARNING: Graph is disconnected after filtering!")
        print(f"  Components: {connectivity_info['num_components']}")
        print(f"  Largest component: {connectivity_info['largest_component_size']}/{len(node_features)} ({connectivity_info['connectivity_percentage']:.1f}%)")
        
        # Use fallback strategy: keep more edges until connected
        print("  Using fallback: keeping more edges for connectivity...")
        fallback_percentile = min(0.95, top_percentile + 0.1)  # Increase by 10% up to 95%
        
        while fallback_percentile <= 0.95 and not connectivity_info['is_connected']:
            fallback_edges = [edge for edge, similarity in edge_similarities[:int(len(edge_similarities) * fallback_percentile)]]
            connectivity_info = check_connectivity(fallback_edges, len(node_features))
            
            if connectivity_info['is_connected']:
                print(f"  ✓ Connected graph achieved with {fallback_percentile*100:.0f}% cutoff")
                filtered_edges = fallback_edges
                break
            else:
                fallback_percentile += 0.05
                print(f"  Trying {fallback_percentile*100:.0f}% cutoff...")
        
        if not connectivity_info['is_connected']:
            print(f"  ⚠ Could not achieve connectivity even at 95% - using all edges")
            filtered_edges = edges
            connectivity_info = check_connectivity(filtered_edges, len(node_features))
    
    print(f"Final edges: {len(filtered_edges)}")
    print(f"Connectivity: {'✓ Connected' if connectivity_info['is_connected'] else '❌ Disconnected'}")
    
    return filtered_edges

def generate_spatial_knn_edges(patch_centers, k=5):
    """Generate spatial KNN edges (k=5, undirected, no self-loops)."""
    print(f"\n=== Generating Spatial KNN Edges (k={k}) ===")
    
    n_patches = len(patch_centers)
    
    if n_patches <= k:
        print(f"Warning: Only {n_patches} patches, using all-to-all connections")
        k = n_patches - 1
    
    # Use sklearn's NearestNeighbors for KNN
    nn = NearestNeighbors(n_neighbors=k+1, algorithm='auto')  # +1 to include self
    nn.fit(patch_centers)
    
    # Find k+1 nearest neighbors (including self)
    distances, indices = nn.kneighbors(patch_centers)
    
    # Create edges (exclude self-loops)
    edges = []
    for i, neighbors in enumerate(indices):
        for j in neighbors[1:]:  # Skip first neighbor (self)
            # Create undirected edge (avoid duplicates)
            edge = tuple(sorted([i, j]))
            if edge not in edges:
                edges.append(edge)
    
    print(f"Generated {len(edges)} undirected edges")
    print(f"Average edges per node: {len(edges) * 2 / n_patches:.1f}")
    
    return edges

def create_three_subplot_visualization(path_annotation, all_patches, valid_patches, 
                                        edges_k5_filtered, edges_k10_filtered, viz_path):
    """
    Create a three-subplot visualization:
    1. path_annotation + all patches (background)
    2. Graph visualization (k=5 + 80% similarity)
    3. Graph visualization (k=10 + 30% similarity)
    """
    # Get image dimensions
    img_height, img_width = path_annotation.shape
    
    # Calculate region offset for coordinate conversion
    # Find the minimum coordinates to get the offset from WSI to region
    all_minr = [patch[0] for patch in all_patches]
    all_minc = [patch[1] for patch in all_patches]
    region_offset_r = min(all_minr)  # Offset from WSI to region
    region_offset_c = min(all_minc)  # Offset from WSI to region
    
    # Helper function to convert WSI coordinates to region coordinates
    def convert_to_region_coords(patch):
        minr, minc, maxr, maxc = patch
        rel_minr = minr - region_offset_r
        rel_minc = minc - region_offset_c
        rel_maxr = maxr - region_offset_r
        rel_maxc = maxc - region_offset_c
        return rel_minr, rel_minc, rel_maxr, rel_maxc

    # Convert ALL patches to local coordinates first (before filtering)
    all_patches_local = []
    for i, patch in enumerate(all_patches):
        rel_minr, rel_minc, rel_maxr, rel_maxc = convert_to_region_coords(patch)
        all_patches_local.append({
            'original_patch': patch,
            'local_coords': (rel_minr, rel_minc, rel_maxr, rel_maxc),
            'local_center': (rel_minc + (rel_maxc - rel_minc)/2, rel_minr + (rel_maxr - rel_minr)/2),
            'index': i # Store the original index
        })
    
    # Select patches for visualization (keep their original local coordinates)
    # CRITICAL: Must maintain the same order as local_patch_centers for correct edge indexing
    valid_patches_local = []
    
    # Create a mapping from original patch coordinates to their index in all_patches_local
    patch_to_index = {local_patch['original_patch']: i for i, local_patch in enumerate(all_patches_local)}
    
    # Map valid_patches to local coordinates in the SAME ORDER as local_patch_centers
    for patch_info in valid_patches:
        original_coords = patch_info['coordinates']
        if original_coords in patch_to_index:
            local_index = patch_to_index[original_coords]
            valid_patches_local.append(all_patches_local[local_index])
        else:
            print(f"Warning: Could not find local coordinates for patch {patch_info['patch_id']}")
    
    # Verify the order is maintained (silent check)
    if len(valid_patches_local) != len(valid_patches):
        print(f"Warning: Patch order mismatch detected")
    
    # Create three subplots
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    
    # Subplot 1: path_annotation + all patches (background)
    ax1 = axes[0]
    ax1.imshow(path_annotation, cmap='gray', alpha=0.7)
    ax1.set_title("All Patches on Background", fontsize=12, pad=10)
    
    # Draw all patch boundaries with their local coordinates
    for i, local_patch in enumerate(all_patches_local):
        rel_minr, rel_minc, rel_maxr, rel_maxc = local_patch['local_coords']
        width = rel_maxc - rel_minc
        height = rel_maxr - rel_minr
        
        rect = plt.Rectangle((rel_minc, rel_minr), width, height, 
                           fill=False, edgecolor='blue', linewidth=1, alpha=0.6)
        ax1.add_patch(rect)
    
    ax1.axis('off')
    
    # Subplot 2: Clean graph visualization (k=5 + 80% similarity) without background
    ax2 = axes[1]
    ax2.set_facecolor('white')
    
    # Draw patch boundaries as thin lines
    for local_patch in valid_patches_local:
        rel_minr, rel_minc, rel_maxr, rel_maxc = local_patch['local_coords']
        width = rel_maxc - rel_minc
        height = rel_maxr - rel_minr
        
        rect = plt.Rectangle((rel_minc, rel_minr), width, height, 
                           fill=False, edgecolor='lightblue', linewidth=0.1, alpha=0.7)
        ax2.add_patch(rect)
    
    # Draw nodes at patch centers
    centers_x = [local_patch['local_center'][0] for local_patch in valid_patches_local]
    centers_y = [local_patch['local_center'][1] for local_patch in valid_patches_local]
    
    ax2.scatter(centers_x, centers_y, c='red', s=5, alpha=0.9, 
               edgecolors='red', linewidth=0.1, zorder=5)
    
    # Draw edges between connected patches
    for edge in edges_k5_filtered:
        i, j = edge
        if i < len(valid_patches_local) and j < len(valid_patches_local):
            center_i = valid_patches_local[i]['local_center']
            center_j = valid_patches_local[j]['local_center']
            
            ax2.plot([center_i[0], center_j[0]], [center_i[1], center_j[1]], 
                   'black', linewidth=1, alpha=0.8, zorder=4)
    
    # Set title and labels
    ax2.set_title(f"Graph (k=5 + 80%) - {len(valid_patches_local)} Nodes, {len(edges_k5_filtered)} Edges", 
                fontsize=12, pad=10)
    ax2.set_xlabel("X Coordinate", fontsize=10)
    ax2.set_ylabel("Y Coordinate", fontsize=10)
    
    # Set axis limits to show all patches with some padding
    if valid_patches_local:
        all_x = [local_patch['local_center'][0] for local_patch in valid_patches_local]
        all_y = [local_patch['local_center'][1] for local_patch in valid_patches_local]
        
        x_margin = max(50, (max(all_x) - min(all_x)) * 0.1)
        y_margin = max(50, (max(all_y) - min(all_y)) * 0.1)
        
        ax2.set_xlim(min(all_x) - x_margin, max(all_x) + x_margin)
        ax2.set_ylim(max(all_y) + y_margin, min(all_y) - y_margin)  # Invert Y-axis for image coordinates
    
    # Add grid for better readability
    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Equal aspect ratio to maintain patch shapes
    ax2.set_aspect('equal')
    
    # Subplot 3: Clean graph visualization (k=10 + 30% similarity) without background
    ax3 = axes[2]
    ax3.set_facecolor('white')
    
    # Draw patch boundaries as thin lines
    for local_patch in valid_patches_local:
        rel_minr, rel_minc, rel_maxr, rel_maxc = local_patch['local_coords']
        width = rel_maxc - rel_minc
        height = rel_maxr - rel_minr
        
        rect = plt.Rectangle((rel_minc, rel_minr), width, height, 
                           fill=False, edgecolor='lightblue', linewidth=0.1, alpha=0.7)
        ax3.add_patch(rect)
    
    # Draw nodes at patch centers
    centers_x = [local_patch['local_center'][0] for local_patch in valid_patches_local]
    centers_y = [local_patch['local_center'][1] for local_patch in valid_patches_local]
    
    ax3.scatter(centers_x, centers_y, c='red', s=5, alpha=0.9, 
               edgecolors='red', linewidth=0.1, zorder=5)
    
    # Draw edges between connected patches
    for edge in edges_k10_filtered:
        i, j = edge
        if i < len(valid_patches_local) and j < len(valid_patches_local):
            center_i = valid_patches_local[i]['local_center']
            center_j = valid_patches_local[j]['local_center']
            
            ax3.plot([center_i[0], center_j[0]], [center_i[1], center_j[1]], 
                   'black', linewidth=1, alpha=0.8, zorder=4)
    
    # Set title and labels
    ax3.set_title(f"Graph (k=10 + 30%) - {len(valid_patches_local)} Nodes, {len(edges_k10_filtered)} Edges", 
                fontsize=12, pad=10)
    ax3.set_xlabel("X Coordinate", fontsize=10)
    ax3.set_ylabel("Y Coordinate", fontsize=10)
    
    # Set axis limits to show all patches with some padding
    if valid_patches_local:
        all_x = [local_patch['local_center'][0] for local_patch in valid_patches_local]
        all_y = [local_patch['local_center'][1] for local_patch in valid_patches_local]
        
        x_margin = max(50, (max(all_x) - min(all_x)) * 0.1)
        y_margin = max(50, (max(all_y) - min(all_y)) * 0.1)
        
        ax3.set_xlim(min(all_x) - x_margin, max(all_x) + x_margin)
        ax3.set_ylim(max(all_y) + y_margin, min(all_y) - y_margin)  # Invert Y-axis for image coordinates
    
    # Add grid for better readability
    ax3.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Equal aspect ratio to maintain patch shapes
    ax3.set_aspect('equal')
    
    plt.tight_layout()
    
    if viz_path:
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {viz_path}")
    
    plt.show()
    
    return fig

def save_extended_region_data(region_data, valid_patches, node_features, patch_centers, 
                             region_offset_r, region_offset_c, save_path):
    """Save extended region data with local coordinates and filtering info."""
    print(f"\n=== Saving Extended Region Data ===")
    
    # Create extended data structure
    extended_data = {
        # Original data
        'fseg_data': region_data['fseg_data'],
        'path_annotation': region_data['path_annotation'],
        'patches': region_data['patches'],
        'wsi_id': region_data['wsi_id'],
        'region_id': region_data['region_id'],
        
        # Additional traceability data
        'patches_local_coords': [],
        'valid_patch_indices': [],
        'removed_patch_indices': [],
        'coordinate_mapping': {
            'region_offset_r': region_offset_r,
            'region_offset_c': region_offset_c
        },
        'node_features': node_features,
        'patch_centers': patch_centers,
        'valid_patches': valid_patches
    }
    
    # Convert all patches to local coordinates
    for i, patch in enumerate(region_data['patches']):
        local_coords = (
            patch[0] - region_offset_r,  # minr
            patch[1] - region_offset_c,  # minc
            patch[2] - region_offset_r,  # maxr
            patch[3] - region_offset_c   # maxc
        )
        extended_data['patches_local_coords'].append(local_coords)
        
        # Check if this patch is valid
        if any(p['coordinates'] == patch for p in valid_patches):
            extended_data['valid_patch_indices'].append(i)
        else:
            extended_data['removed_patch_indices'].append(i)
    
    # Save extended data
    np.save(save_path, extended_data)
    print(f"✓ Extended region data saved to: {save_path}")
    print(f"  Valid patches: {len(extended_data['valid_patch_indices'])}")
    print(f"  Removed patches: {len(extended_data['removed_patch_indices'])}")
    
    return extended_data

def save_graph_data(node_features, edges, patch_centers, method_name, save_path):
    """Save node features and edges for a specific graph generation method."""
    print(f"\n=== Saving Graph Data for {method_name} ===")
    
    graph_data = {
        'node_features': node_features,
        'edges': edges,
        'patch_centers': patch_centers
    }
    
    np.save(save_path, graph_data)
    print(f"✓ Graph data saved to: {save_path}")
    print(f"  Nodes: {len(node_features)}")
    print(f"  Edges: {len(edges)}")
    print(f"  Patch centers: {len(patch_centers)}")
    
    return graph_data

def main(npy_filename=None):
    """Main function to process one region NPY file."""
    # Check if region files exist
    region_dir = "/media/data2/jinting/GNN/Cervical/region_npy_files_new"
    if not os.path.exists(region_dir):
        print(f"Region directory not found: {region_dir}")
        return
    
    # Use provided filename or get list of region NPY files
    if npy_filename:
        npy_file_path = os.path.join(region_dir, npy_filename)
        if not os.path.exists(npy_file_path):
            print(f"Specified NPY file not found: {npy_file_path}")
            return
    else:
        # Get list of region NPY files
        npy_files = [f for f in os.listdir(region_dir) if f.endswith('.npy')]
        if not npy_files:
            print(f"No NPY files found in {region_dir}")
            return
        # Use the first available file
        npy_file_path = os.path.join(region_dir, npy_files[0])
    
    print(f"\n=== Processing: {os.path.basename(npy_file_path)} ===")
    
    try:
        # Step 1: Load region data
        region_data = load_region_data(npy_file_path)
        
        # Extract data
        fseg_data = region_data['fseg_data']
        path_annotation = region_data['path_annotation']
        patches = region_data['patches']
        
        print(f"Data shapes: FSEG {fseg_data.shape}, Path {path_annotation.shape}")
        print(f"Number of patches: {len(patches)}")
        
        # Step 2: Identify and clean main subject
        cleaned_path_annotation, cleaned_fseg_data = identify_and_clean_main_subject(path_annotation, fseg_data)
        if cleaned_path_annotation is None or cleaned_fseg_data is None:
            print("No valid subject found or cleaning failed. Exiting.")
            return
        
        # Step 2.5: Validate patches and extract features
        valid_patches, node_features, patch_centers = validate_patches(
            cleaned_fseg_data, cleaned_path_annotation, patches
        )
        
        if len(valid_patches) == 0:
            print("No valid patches found!")
            return
        
        # Step 2.5.5: Convert patch centers to local coordinates for consistent graph generation
        # Calculate region offset for coordinate conversion
        all_minr = [patch[0] for patch in patches]
        all_minc = [patch[1] for patch in patches]
        region_offset_r = min(all_minr)  # Offset from WSI to region
        region_offset_c = min(all_minc)  # Offset from WSI to region
        
        # Convert global centers to local centers
        local_patch_centers = []
        for center in patch_centers:
            local_x = center[0] - region_offset_c
            local_y = center[1] - region_offset_r
            local_patch_centers.append([local_x, local_y])
        
        # Step 3: Generate spatial KNN edges using LOCAL coordinates
        # Generate k=5 edges and filter by similarity
        edges_k5_raw = generate_spatial_knn_edges(local_patch_centers, k=5)
        edges_k5_filtered = filter_edges_by_similarity(edges_k5_raw, node_features, top_percentile=0.8)

        # Generate k=10 edges and filter by similarity
        edges_k10_raw = generate_spatial_knn_edges(local_patch_centers, k=10)
        edges_k10_filtered = filter_edges_by_similarity(edges_k10_raw, node_features, top_percentile=0.3)
        
        # Step 4: Create three-subplot visualization
        os.makedirs("results/visualizations", exist_ok=True)
        # Use the complete filename (before .npy) as the image_id
        image_id = os.path.splitext(os.path.basename(npy_file_path))[0]
        viz_path = f"results/visualizations/{image_id}_three_subplot_comparison.png"
        create_three_subplot_visualization(cleaned_path_annotation, patches, valid_patches, 
                                        edges_k5_filtered, edges_k10_filtered, viz_path)
        
        # Step 5: Save extended region data and graph results
        # Use the complete filename (before .npy) as the image_id
        image_id = os.path.splitext(os.path.basename(npy_file_path))[0]
        print(f"Using image_id: {image_id}")
        
        # Create results directory structure
        os.makedirs("results", exist_ok=True)
        os.makedirs("results/Extended_NPY", exist_ok=True)
        os.makedirs("results/k5_80", exist_ok=True)
        os.makedirs("results/k10_30", exist_ok=True)
        os.makedirs("results/visualizations", exist_ok=True)
        
        # Save extended region data
        extended_npy_path = f"results/Extended_NPY/{image_id}_extended_region_data.npy"
        save_extended_region_data(region_data, valid_patches, node_features, 
                                local_patch_centers, region_offset_r, region_offset_c, 
                                extended_npy_path)
        
        # Save graph data for each method
        save_graph_data(node_features, edges_k5_filtered, local_patch_centers, 
                       "k5_80percent", f"results/k5_80/{image_id}_graph_k5_80percent.npy")
        save_graph_data(node_features, edges_k10_filtered, local_patch_centers, 
                       "k10_30percent", f"results/k10_30/{image_id}_graph_k10_30percent.npy")
        
        # Final summary
        print(f"\n=== Final Graph Summary ===")
        print(f"Nodes: {len(valid_patches)} (valid patches)")
        print(f"Edges (k=5 + 80% similarity): {len(edges_k5_filtered)} (filtered)")
        print(f"Edges (k=10 + 30% similarity): {len(edges_k10_filtered)} (filtered)")
        print(f"Node features: {node_features.shape[1]}D (FSEG percentages 0-10)")
        print(f"✓ Graph generation complete! Results saved to results/ folder")
        print(f"✓ Visualization saved to: {viz_path}")
        
    except Exception as e:
        print(f"✗ Error during processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import sys
    
    # Check if filename was provided as command line argument
    if len(sys.argv) > 1:
        npy_filename = sys.argv[1]
        print(f"Processing specified file: {npy_filename}")
        main(npy_filename)
    else:
        print("No filename specified. Processing first available NPY file.")
        main()
