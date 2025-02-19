import slicer
import numpy as np
from skimage import measure, morphology
import vtk

def generate_3D_model(volume_node, threshold_value=1, min_size=50):
    """
    Generates a 3D surface model from a volume using thresholding & filtering.

    Parameters:
    - volume_node: The input 3D volume node
    - threshold_value: Intensity threshold for surface extraction
    - min_size: Minimum size for objects to keep (remove noise)

    Returns:
    - modelNode: The generated 3D model node
    """
    print("🔍 Generating 3D model...")

    # Create segmentation node
    segmentationNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
    slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(volume_node, segmentationNode)

    # Apply threshold to filter out noise
    slicer.modules.segmentations.logic().ThresholdSegmentationNode(segmentationNode, threshold_value, 1e9)

    # Remove small components
    segmentationArray = slicer.util.arrayFromVolume(volume_node)
    labeled, num_features = measure.label(segmentationArray, return_num=True)
    sizes = np.bincount(labeled.ravel())
    
    # Keep only objects larger than min_size
    filtered_array = np.isin(labeled, np.where(sizes >= min_size)[0])
    
    # Convert back to slicer volume
    slicer.util.updateVolumeFromArray(volume_node, filtered_array.astype(np.uint8))

    # Extract 3D model
    slicer.modules.segmentations.logic().ExportAllSegmentsToModels(segmentationNode)

    return segmentationNode

def detect_spheres(volume_node, min_radius=3, max_radius=10):
    """
    Detects sphere-like structures in a 3D volume using connected components.

    Parameters:
    - volume_node: The input 3D volume node
    - min_radius, max_radius: Size constraints for spheres

    Returns:
    - centers: List of detected sphere coordinates
    """
    print("🔍 Detecting spheres...")

    # Get numpy array from Slicer volume
    volume_array = slicer.util.arrayFromVolume(volume_node)

    # Connected component labeling
    labeled, num_features = measure.label(volume_array, return_num=True)

    # Measure region properties
    properties = measure.regionprops(labeled)

    # Find spheres based on size & shape
    centers = []
    for prop in properties:
        if min_radius**3 < prop.area < max_radius**3:  # Approximate volume filter
            centers.append(prop.centroid)

    print(f"✅ Found {len(centers)} potential spheres")
    return centers

def create_markups(centers):
    """
    Converts a list of 3D coordinates into a Slicer Markups Node.

    Parameters:
    - centers: List of (x, y, z) coordinates

    Returns:
    - markupsNode: The created Markups node
    """
    print("🎯 Creating markups...")

    markupsNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode")
    markupsNode.SetName("SphereCenters")

    for c in centers:
        markupsNode.AddFiducial(c[2], c[1], c[0])  # Convert from (z, y, x) to (x, y, z)

    return markupsNode


def merge_2D_3D(centers_2D, centers_3D, distance_threshold=5):
    """
    Merges 2D and 3D detected points by checking proximity.

    Parameters:
    - centers_2D: List of 2D-detected coordinates
    - centers_3D: List of 3D-detected coordinates
    - distance_threshold: Maximum distance to consider a match

    Returns:
    - merged_centers: Final list of merged 3D points
    """
    print("🔗 Merging 2D and 3D detections...")

    merged_centers = centers_3D.copy()  # Start with 3D points

    # Add 2D points that are NOT near existing 3D points
    for p2d in centers_2D:
        if all(np.linalg.norm(np.array(p2d) - np.array(p3d)) > distance_threshold for p3d in centers_3D):
            merged_centers.append(p2d)

    print(f"✅ Merged {len(merged_centers)} total points")
    return merged_centers
