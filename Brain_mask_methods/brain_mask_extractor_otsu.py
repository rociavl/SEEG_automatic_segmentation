import logging
import time
import numpy as np
import vtk
import vtk.util.numpy_support
from scipy import ndimage
from skimage import filters
import slicer
from slicer import vtkMRMLScalarVolumeNode

class BrainMaskExtractor:
    """
    Class responsible for extracting brain masks from input volumes.
    This class implements the masking algorithm based on thresholding and morphological operations.
    """
    
    def __init__(self):
        """Initialize the brain mask extractor."""
        pass
        
    def remove_annotations(self):
        """Removes invalid annotation and markup nodes from the scene."""
        nodesToDelete = []
        
        for i in range(slicer.mrmlScene.GetNumberOfNodes()):
            node = slicer.mrmlScene.GetNthNode(i)
            
            # Check if the node is a Markups node (generic)
            if node.IsA("vtkMRMLMarkupsNode"):
                if node.IsA("vtkMRMLMarkupsFiducialNode"):
                    # Remove empty Markups Fiducial Nodes
                    if node.GetNumberOfControlPoints() == 0:
                        nodesToDelete.append(node)

                # Orphaned Annotation Hierarchy Nodes
                elif node.IsA("vtkMRMLAnnotationHierarchyNode") and not node.GetAssociatedNode():
                    nodesToDelete.append(node)

        for node in nodesToDelete:
            slicer.mrmlScene.RemoveNode(node)
            logging.info(f"Deleted invalid node: {node.GetName()}")
    
    def extract_mask(self, inputVolume: vtkMRMLScalarVolumeNode, threshold_value: int = 20, 
                    show_result: bool = True) -> vtkMRMLScalarVolumeNode:
        """
        Extract a binary mask from the input volume.
        
        Parameters:
        -----------
        inputVolume : vtkMRMLScalarVolumeNode
            The input volume from which to extract the mask
        threshold_value : int, optional
            The threshold value for binarization (default is 20)
        show_result : bool, optional
            Whether to show the result in the Slicer viewer (default is True)
            
        Returns:
        --------
        vtkMRMLScalarVolumeNode
            The output volume node containing the mask
        """
        if not inputVolume:
            raise ValueError("Input volume is invalid")

        startTime = time.time()
        logging.info("Mask extraction started")

        # Create a new vtkMRMLScalarVolumeNode for the mask
        maskVolumeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
        maskVolumeNode.SetName(f"Generated Mask_{inputVolume.GetName()}")
        maskVolumeNode.CopyOrientation(inputVolume)  

        if inputVolume.GetImageData():
            maskVolumeNode.SetAndObserveImageData(inputVolume.GetImageData())
            
        # Remove any annotations or markups from the scene   
        self.remove_annotations()

        inputImage = maskVolumeNode.GetImageData()
        # Convert the VTK array to a NumPy array
        inputArrayVtk = inputImage.GetPointData().GetScalars()
        inputArray = vtk.util.numpy_support.vtk_to_numpy(inputArrayVtk)

        dims = inputImage.GetDimensions()

        # Apply Gaussian smoothing
        smooth_input = filters.gaussian(inputArray, sigma=2)
        logging.info(f"Image stats - Min: {inputArray.min()}, Max: {inputArray.max()}, Mean: {inputArray.mean()}")

        # Apply thresholding - can use Otsu's method or the provided threshold value
        if threshold_value <= 0:
            # Use Otsu's thresholding method
            thresh = filters.threshold_otsu(smooth_input)
            logging.info(f"Calculated Otsu threshold: {thresh}")
            maskArray = (inputArray > thresh).astype(np.uint8)
        else:
            # Use the provided threshold value
            maskArray = (inputArray > threshold_value).astype(np.uint8)
            
        # Reshape to match the original dimensions
        maskArray = maskArray.reshape(dims)
        logging.info(f"Mask array shape: {maskArray.shape}, dtype: {maskArray.dtype}")
        logging.info(f"Unique values in mask array: {np.unique(maskArray)}")
        
        # Apply morphological operations
        # Create a 3D structuring element for closing operation
        selem_close = ndimage.generate_binary_structure(3, 1)
        
        # Apply binary closing to connect nearby structures
        closed = ndimage.binary_closing(maskArray, structure=selem_close, iterations=6).astype(np.uint8)
        
        # Fill holes in the binary mask
        filled = ndimage.binary_fill_holes(closed).astype(np.uint8)
        
        # Flatten the array for VTK conversion
        final_flat = filled.ravel()
  
        # Convert the mask (NumPy array) back to VTK format
        outputArrayVtk = vtk.util.numpy_support.numpy_to_vtk(final_flat, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)

        # Create output image data
        outputImage = vtk.vtkImageData()
        outputImage.CopyStructure(inputImage)
        outputImage.GetPointData().SetScalars(outputArrayVtk)
        
        # Set the mask as the image data for the new volume node
        maskVolumeNode.SetAndObserveImageData(outputImage)

        # Ensure the output volume is properly displayed
        if show_result:
            slicer.app.processEvents()  # Ensure the viewer updates with new data

        stopTime = time.time()
        logging.info(f"Mask extraction completed in {stopTime - startTime:.2f} seconds")

        return maskVolumeNode