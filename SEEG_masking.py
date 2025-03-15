import logging
import os
from typing import Annotated, Optional

import slicer.util
import vtk
import numpy as np
import vtk.util.numpy_support
import time


import slicer
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import (
    parameterNodeWrapper,
    WithinRange)

from slicer import vtkMRMLScalarVolumeNode
from skimage import (exposure, filters, feature, morphology)
from skimage import img_as_ubyte, measure
import cv2
from skimage.measure import label, regionprops
import qt
from enhance_ctp import enhance_ctp
from enhance_ctp import add_more_filter
from scipy import ndimage



#
# SEEG_masking
#

class SEEG_masking(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("SEEG Masking")
        # TODO: set categories (folders where the module shows up in the module selector)
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "Examples")]
        self.parent.dependencies = []  # TODO: add here list of module names that this module requires
        self.parent.contributors = ["Rocio Avalos (AnyWare Corp.)"]  # TODO: replace with "Firstname Lastname (Organization)"
        # TODO: update with short description of the module and a link to online module documentation
        # _() function marks text as translatable to other languages
        self.parent.helpText = _("""
This is an example of scripted loadable module bundled in an extension.
See more information in <a href="https://github.com/organization/projectname#SEEG_masking">module documentation</a>.
""")
        # TODO: replace with organization, grant and thanks
        self.parent.acknowledgementText = _("""
This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc., Andras Lasso, PerkLab,
and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
""")

        # Additional initialization step after application startup is complete
        slicer.app.connect("startupCompleted()", registerSampleData)


#
# Register sample data sets in Sample Data module?
#


def registerSampleData():
    """Add data sets to Sample Data module."""
    # It is always recommended to provide sample data for users to make it easy to try the module,
    # but if no sample data is available then this method (and associated startupCompeted signal connection) can be removed.

    import SampleData

    iconsPath = os.path.join(os.path.dirname(__file__), "Resources/Icons")

    # To ensure that the source code repository remains small (can be downloaded and installed quickly)
    # it is recommended to store data sets that are larger than a few MB in a Github release.

    # SEEG_masking1
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category="SEEG_masking",
        sampleName="SEEG_masking1",
        # Thumbnail should have size of approximately 260x280 pixels and stored in Resources/Icons folder.
        # It can be created by Screen Capture module, "Capture all views" option enabled, "Number of images" set to "Single".
        thumbnailFileName=os.path.join(iconsPath, "SEEG_masking1.png"),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        fileNames="SEEG_masking1.nrrd",
        # Checksum to ensure file integrity. Can be computed by this command:
        #  import hashlib; print(hashlib.sha256(open(filename, "rb").read()).hexdigest())
        checksums="SHA256:998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        # This node name will be used when the data set is loaded
        nodeNames="SEEG_masking1",
    )

    # SEEG_masking2
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category="SEEG_masking",
        sampleName="SEEG_masking2",
        thumbnailFileName=os.path.join(iconsPath, "SEEG_masking2.png"),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        fileNames="SEEG_masking2.nrrd",
        checksums="SHA256:1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        # This node name will be used when the data set is loaded
        nodeNames="SEEG_masking2",
    )


#
# SEEG_maskingParameterNode
#
@parameterNodeWrapper
class SEEG_maskingParameterNode:
    """
    The parameters needed by module.

    inputVolume - The volume to threshold.
    imageThreshold - The value at which to threshold the input volume.
    invertThreshold - If true, will invert the threshold.
    thresholdedVolume - The output volume that will contain the thresholded volume.
    invertedVolume - The output volume that will contain the inverted thresholded volume.
    """

    inputVolume: vtkMRMLScalarVolumeNode
    #imageThreshold: Annotated[float, WithinRange(-100, 500)] = 100
    #invertThreshold: bool = False
    outputVolume: vtkMRMLScalarVolumeNode
    #invertedVolume: vtkMRMLScalarVolumeNode
    inputROI: vtkMRMLScalarVolumeNode



#
# SEEG_maskingWidget
#


class SEEG_maskingWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._parameterNodeGuiTag = None

    def setup(self) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/SEEG_masking.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = SEEG_maskingLogic()

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # Buttons
        self.ui.applyButton.connect("clicked(bool)", self.onApplyButton)
        #self.ui.newButton_save.connect("clicked(bool)", self.onNewButton_save) # New Button
        #self.ui.inputSelector_saveMaskAI.setMRMLScene(slicer.mrmlScene)
        
        self.ui.saveButton.connect("clicked(bool)", self.onSaveButton) # Save Button
        # Set up the Save button
        self.ui.saveButton.setText("Save Mask") # Save Button

        # Set up the CTP button
        self.ui.pushButton_ctp.connect('clicked(bool)', self.onCTPButton) # Push Button for CTP
        # Set up input selector for CTP
        self.ui.inputSelector_ctp.setMRMLScene(slicer.mrmlScene)
        self.ui.inputSelector_ROI_norm.setMRMLScene(slicer.mrmlScene)

        self.ui.pushButton_filters.connect('clicked(bool)', self.onApplyFiltersButton)
        self.ui.inputSelector_filter.setMRMLScene(slicer.mrmlScene)


        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()

    def cleanup(self) -> None:
        """Called when the application closes and the module widget is destroyed."""
        self.removeObservers()

    def enter(self) -> None:
        """Called each time the user opens this module."""
        # Make sure parameter node exists and observed
        self.initializeParameterNode()

    def exit(self) -> None:
        """Called each time the user opens a different module."""
        # Do not react to parameter node changes (GUI will be updated when the user enters into the module)
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self._parameterNodeGuiTag = None
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)

    def onSceneStartClose(self, caller, event) -> None:
        """Called just before the scene is closed."""
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event) -> None:
        """Called just after the scene is closed."""
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self) -> None:
        """Ensure parameter node exists and observed."""
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.

        self.setParameterNode(self.logic.getParameterNode())

        # Select default input nodes if nothing is selected yet to save a few clicks for the user
        if not self._parameterNode.inputVolume:
            firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
            if firstVolumeNode:
                self._parameterNode.inputVolume = firstVolumeNode

    def setParameterNode(self, inputParameterNode: Optional[SEEG_maskingParameterNode]) -> None:
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """

        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
        self._parameterNode = inputParameterNode
        if self._parameterNode:
            # Note: in the .ui file, a Qt dynamic property called "SlicerParameterName" is set on each
            # ui element that needs connection.
            self._parameterNodeGuiTag = self._parameterNode.connectGui(self.ui)
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
            self._checkCanApply()

    
    # check if I have to put more boxes!!
    def _checkCanApply(self, caller=None, event=None) -> None:
        """  Check if the input volume is selected. """
        if self._parameterNode and self._parameterNode.inputVolume:
            self.ui.applyButton.toolTip = _("Compute output volume")
            self.ui.applyButton.enabled = True
            self.ui.pushButton_ctp.toolTip = _("Compute output volume")
            self.ui.pushButton_ctp.enabled = True
        else:
            self.ui.applyButton.toolTip = _("Select input volume nodes")
            self.ui.applyButton.enabled = False
            self.ui.pushButton_ctp.toolTip = _("Select input volume nodes")
            self.ui.pushButton_ctp.enabled = False

    # Handle 'Apply' button click event (run processing)
    def onApplyButton(self) -> None:
        """Run processing when user clicks 'Apply' button."""
        with slicer.util.tryWithErrorDisplay(_("Failed to compute results."), waitCursor=True):
            # Process the input and create the mask (but don't save it yet)
            #self.logic.process(self.ui.inputSelector.currentNode(), self.ui.inputSelector.currentNode(), self.ui.invertOutputCheckBox.checked)
            self.outputVolumeNode = self.logic.process(self.ui.inputSelector.currentNode())
            # Optionally, update the UI to indicate the mask was created
            slicer.util.infoDisplay("Mask created! Now, select a location to save the mask.")

    # def onNewButton_save(self) -> None:
    #     """Create mask and save it when user clicks the button"""
    #     with slicer.util.tryWithErrorDisplay(_("Failed to compute results."), waitCursor = True):
    #         self.outputVolumeNode = self.logic.brain_mask_modified(self.ui.inputSelector_saveMaskAI.currentNode())
    #         print("Brain mask created and saved ")


    
    def onSaveButton(self) -> None:
        """Handle 'Save Mask' button click event."""
        print("✅ Save button clicked!")  # Debugging print

        if not self.outputVolumeNode:
            slicer.util.errorDisplay("❌ No mask to save! Apply first.")
            print("❌ No mask to save!")
            return

        print(f"🛠️ Saving mask: {self.outputVolumeNode.GetName()}")  # Debugging print

        # Create a file dialog
        fileDialog = qt.QFileDialog()
        fileDialog.setAcceptMode(qt.QFileDialog.AcceptSave)
        fileDialog.setNameFilter("NRRD files (*.nrrd)")
        fileDialog.setDefaultSuffix('nrrd')
        fileDialog.setOption(qt.QFileDialog.ShowDirsOnly, False)

        # Show file dialog and get selected folder
        if fileDialog.exec_():
            savePath = fileDialog.selectedFiles()[0]
            print(f"📂 Selected save path: {savePath}")  # Debugging print

            success = slicer.util.saveNode(self.outputVolumeNode, savePath)

            if success:
                slicer.util.infoDisplay(f"✅ Mask saved to:\n{savePath}")
                print(f"✅ Mask saved successfully at: {savePath}")  # Debugging print
            else:
                slicer.util.errorDisplay("❌ Failed to save mask")
                print("❌ Failed to save mask")  # Debugging print


    def saveMaskToFile(self, maskVolumeNode: slicer.vtkMRMLScalarVolumeNode, filePath: str) -> None:
        """Save the output volume (mask) to the given file path."""
        
        if not maskVolumeNode:
            slicer.util.errorDisplay("No output volume available.")
            return

        # Ensure file path is valid
        if not filePath or not filePath.strip():
            slicer.util.errorDisplay("Invalid file path.")
            return

        # Save the mask using Slicer's saveNode function
        success = slicer.util.saveNode(maskVolumeNode, filePath)
        
        if not success:
            slicer.util.errorDisplay(f"Failed to save output volume to {filePath}")
        else:
            slicer.util.infoDisplay(f"Mask saved successfully at: {filePath}")

    
    def onCTPButton(self) -> None:
        """Handle 'CTP' button click event."""
        
        # Get selected nodes from the UI
        inputVolume = self.ui.inputSelector_ctp.currentNode()
        inputROI = self.ui.inputSelector_ROI_norm.currentNode()

        # Optional: Get additional inputs for methods and outputDir
        methods = 'all'  # Placeholder, will be replaced by actual UI input later
        outputDir = r'C:\\Users\\rocia\\Downloads\\TFG\\Cohort\\Enhance_ctp_tests'  
        
        # Validate inputs
        if not inputVolume or not inputROI:
            slicer.util.errorDisplay("Please select both the input volume and the ROI.")
            return
        

        # Call the logic function from the logic class
        enhancedVolumeNodes = self.logic.enhance_ctp_logic(inputVolume, inputROI, methods, outputDir)

        # Handle the returned enhancedVolumeNodes dictionary (which contains all enhanced volume nodes)
        if enhancedVolumeNodes:
            # Loop over all the enhanced volume nodes (stored in the dictionary)
            for method_name, enhancedVolumeNode in enhancedVolumeNodes.items():
                # Check if enhancedVolumeNode is a valid volume node
                if enhancedVolumeNode:
                    # Set the reference active volume for each enhanced volume node
                    slicer.app.applicationLogic().GetSelectionNode().SetReferenceActiveVolumeID(enhancedVolumeNode.GetID())
                    slicer.app.applicationLogic().PropagateVolumeSelection()
                    slicer.app.layoutManager().resetSliceViews()
                    slicer.app.processEvents()

                    # Optionally, inform the user for each enhancement
                    slicer.util.infoDisplay(f"Enhanced volume for method '{method_name}' has been created and set as active.")
                
        else:
            slicer.util.errorDisplay("Error: No enhanced volume nodes were returned.")


    def onApplyFiltersButton(self) -> None:
        """Handle 'Apply Filters' button click event with debugging statements."""
        
        print("[DEBUG] onApplyFiltersButton triggered")

        # Get the current selected volume from the UI (assuming you have a selector for the volume)
        inputVolume = self.ui.inputSelector_filter.currentNode()  # Assuming you have a volume selector in your UI
        
        if not inputVolume:
            slicer.util.errorDisplay("Please select a valid volume to apply filters.")
            print("[ERROR] No input volume selected.")
            return

        print(f"[DEBUG] Selected input volume: {inputVolume.GetName()}")
        print(f"[DEBUG] Checkable_filters initialized: {self.ui.Checkable_filters}")

        # Collect the selected filters from the UI
        try:
             # Get the checked indexes from the ctkCheckableComboBox
            checked_indexes = self.ui.Checkable_filters.checkedIndexes()
            print(f"[DEBUG] Checked filter indexes: {checked_indexes}")
            selected_items = [index.data() for index in checked_indexes]  # Get selected filters
            print(f"[DEBUG] Selected filter UI items: {selected_items}")
        except AttributeError as e:
            print(f"[ERROR] Could not retrieve selected filters: {e}")
            slicer.util.errorDisplay("Error retrieving selected filters. Check if Checkable_filters is properly initialized.")
            return


        # Define a mapping of combo box display names to internal filter names
        filter_names = {
            "Morphological Operations": "morph_operations",
            "Canny Edge Detection": "canny_edge",
            "High Pass Sharpening": "high_pass_sharpening"
        }

        # Map selected items to internal names
        selected_filters = [filter_names[item] for item in selected_items if item in filter_names]
        
        if not selected_filters:
            print("[WARNING] No valid filters selected.")
            slicer.util.infoDisplay("No additional filters selected. The volume will remain unchanged.")
            return

        print(f"[DEBUG] Mapped selected filters: {selected_filters}")

        # Define the output directory
        outputDir = r'C:\\Users\\rocia\\Downloads\\TFG\\Cohort\\Enhance_ctp_tests'
        print(f"[DEBUG] Output directory: {outputDir}")

        # Apply filters by calling the add_more_filter function
        try:
            enhancedVolumeNode = add_more_filter(inputVolume, selected_filters, outputDir)
        except Exception as e:
            print(f"[ERROR] Failed to apply filters: {e}")
            slicer.util.errorDisplay(f"Error while applying filters: {e}")
            return

        if enhancedVolumeNode:
            try:
                slicer.app.applicationLogic().GetSelectionNode().SetReferenceActiveVolumeID(enhancedVolumeNode.GetID())
                slicer.app.applicationLogic().PropagateVolumeSelection()
                slicer.app.layoutManager().resetSliceViews()
                slicer.app.processEvents()
                print("[DEBUG] Enhanced volume set as active.")

                # Inform the user that the filters have been applied and saved
                slicer.util.infoDisplay("Filters applied successfully to the selected volume.")
                slicer.util.infoDisplay(f"Filtered volume saved as: {outputDir}\\Enhanced_more_filters_{inputVolume.GetName()}.nrrd")
            except Exception as e:
                print(f"[ERROR] Failed to update UI with enhanced volume: {e}")
                slicer.util.errorDisplay(f"Error updating UI with enhanced volume: {e}")
        else:
            print("[ERROR] add_more_filter did not return a valid volume node.")
            slicer.util.errorDisplay("Error: No enhanced volume was created.")
#
# SEEG_maskingLogic
#

class SEEG_maskingLogic:
    """This class implements all the actual computation done by your module."""

    def __init__(self, parent=None):
        """Called when the user opens the module the first time and the widget is initialized."""
        #ScriptedLoadableModuleLogic.__init__(self)
        super().__init__() # Call the super class initializer

    
    def createParameterNode(self):
        """Create and initialize new parameter node"""
        node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScriptedModuleNode")
        node.SetName("SEEG_maskingParameters")
        return node

    def getParameterNode(self):
        node = slicer.mrmlScene.GetFirstNodeByName("SEEG_maskingParameters")
        if not node:
            node = self.createParameterNode()
        return SEEG_maskingParameterNode(node)  
    
    #Dealing with markups :c
    def remove_Annotations(self):
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
            print(f"Deleted invalid node: {node.GetName()}")

    def process(self,
                inputVolume: vtkMRMLScalarVolumeNode,
                #outputVolume: vtkMRMLScalarVolumeNode,
                #imageThreshold: float,
                #invert: bool = False,
                showResult: bool = True) -> vtkMRMLScalarVolumeNode:
        """
        Run the processing algorithm.
        Creates a mask where scalar > 0 is set to 1, otherwise set to 0.
        :param inputVolume: volume to be thresholded
        :param outputVolume: output mask volume
        :param imageThreshold: values above/below this threshold will be set to 1
        :param invert: if True then values above the threshold will be set to 0, otherwise values below are set to 0
        :param showResult: show output volume in slice viewers
        """
        
        if not inputVolume:
            raise ValueError("Input volume is invalid")

        startTime = time.time()
        logging.info("Processing started")



        # Create a new vtkMRMLScalarVolumeNode for the mask and save it
        maskVolumeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
        #maskVolumeNode.DeepCopy(inputVolume)
        maskVolumeNode.SetName(f"Generated Mask_{inputVolume.GetName()}")
        maskVolumeNode.CopyOrientation(inputVolume)  


        
        if inputVolume.GetImageData():
            maskVolumeNode.SetAndObserveImageData(inputVolume.GetImageData())
        # Remove any annotations or markups from the scene   
        self.remove_Annotations()

        inputImage = maskVolumeNode.GetImageData()
        # Convert the VTK array to a NumPy array
        inputArrayVtk = inputImage.GetPointData().GetScalars()
        inputArray = vtk.util.numpy_support.vtk_to_numpy(inputArrayVtk)

        dims = inputImage.GetDimensions() # No funciona

        smooth_input = filters.gaussian(inputArray, sigma=2)
        print(f"Min: {inputArray.min()}, Max: {inputArray.max()}, Mean: {inputArray.mean()}")

        
        # Apply binarization logic: all non-zero values become 1, zero stays as 0
        thresh = filters.threshold_otsu(smooth_input)
        
        maskArray = (inputArray > 25).astype(np.uint8)  
        maskArray = maskArray.reshape(dims)
        print("Mask array dtype:", maskArray.dtype)
        print("Unique values in mask array:", np.unique(maskArray))
        selem_close = ndimage.generate_binary_structure(3, 1) # 3D structuring element
        print("Input array shape:", inputArray.shape)
        print("Mask array shape:", maskArray.shape)
        print("Selem shape:", selem_close.shape)
        closed = ndimage.binary_closing(maskArray, structure=selem_close, iterations=6).astype(np.uint8)
        
        filled = ndimage.binary_fill_holes(closed).astype(np.uint8)
        
        final_flat = filled.ravel()
  

        # Convert the mask (NumPy array) back to VTK format
        outputArrayVtk = vtk.util.numpy_support.numpy_to_vtk(final_flat, deep=True, array_type = vtk.VTK_UNSIGNED_CHAR)
        # Create an output image and set its scalar data
        #outputImage.GetPointData().SetScalars(outputArrayVtk)

        outputImage = vtk.vtkImageData()
        outputImage.CopyStructure(inputImage)
        outputImage.GetPointData().SetScalars(outputArrayVtk)
        
        # Set the mask as the image data for the new volume node
        maskVolumeNode.SetAndObserveImageData(outputImage)


        # Ensure the output volume is properly displayed
        slicer.app.processEvents()  # Ensures the UI of Slicer is updated


        if showResult:
            slicer.app.processEvents()  # Ensure the viewer updates with new data

        stopTime = time.time()
        logging.info(f"Processing completed in {stopTime - startTime:.2f} seconds")

        return maskVolumeNode
    
    # def brain_mask_modified(self, inputVolume: vtkMRMLScalarVolumeNode) -> vtkMRMLScalarVolumeNode:
    #     """
    #     Generate a binary brain mask using DeepBrain
    #     param inputVolume: volume to be masked
    #     param outputVolume: volume to be used as a mask
    #     """


    #     return brain_mask_modified(self, inputVolume)



    ### Finding points of contact (creating a mask and enhancing the image) (no funciona aun :c) en al archivo ctp_enhance.py sí


    def enhance_ctp_logic(self, inputVolume, inputROI, methods='all', outputDir=None):
        """
        Logic function to handle the volume enhancement.
        :param inputVolume: The input volume to enhance
        :param inputROI: The input ROI mask for enhancement
        :param methods: Methods used for enhancement (default 'all')
        :param outputDir: Directory to save the output (optional)
        :return: vtkMRMLScalarVolumeNode or None
        """
        # Call the enhance_ctp function from the external script
        enhancedVolumeNode = enhance_ctp(inputVolume, inputROI, methods, outputDir)

        # Handle the result (e.g., update the active volume, reset slice views)
        if enhancedVolumeNode:
            slicer.app.applicationLogic().GetSelectionNode().SetReferenceActiveVolumeID(enhancedVolumeNode.GetID())
            slicer.app.applicationLogic().PropagateVolumeSelection()
            slicer.app.layoutManager().resetSliceViews()
            slicer.app.processEvents()

            slicer.util.infoDisplay("Enhancement complete! Mask created!")
        else:
            slicer.util.errorDisplay("Enhancement failed. Please check your inputs and try again.")
        
        return enhancedVolumeNode
    
    def apply_filters_logic(self, enhancedVolume, selected_filters):
        """
        Logic function to handle applying additional filters to the enhanced volume.
        :param enhancedVolume: The enhanced volume after running `enhance_ctp`
        :param selected_filters: List of filters to apply (e.g., "morph_operations", "canny_edge")
        :return: vtkMRMLScalarVolumeNode or None
        """
        # Call the add_more_filter function from the external script
        enhancedVolumeNode = add_more_filter(enhancedVolume, selected_filters)

        # Handle the result (e.g., update the active volume, reset slice views)
        if enhancedVolumeNode:
            slicer.app.applicationLogic().GetSelectionNode().SetReferenceActiveVolumeID(enhancedVolumeNode.GetID())
            slicer.app.applicationLogic().PropagateVolumeSelection()
            slicer.app.layoutManager().resetSliceViews()
            slicer.app.processEvents()

            slicer.util.infoDisplay("Additional filters applied successfully!")
        else:
            slicer.util.errorDisplay("Applying filters failed. Please check your inputs and try again.")
        
        return enhancedVolumeNode

    

    

        
        


#
# SEEG_maskingTest
#


class SEEG_maskingTest(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """Do whatever is needed to reset the state - typically a scene clear will be enough."""
        slicer.mrmlScene.Clear()

    def runTest(self):
        """Run as few or as many tests as needed here."""
        self.setUp()
        self.test_SEEG_masking1()

    def test_SEEG_masking1(self):
        """Ideally you should have several levels of tests.  At the lowest level
        tests should exercise the functionality of the logic with different inputs
        (both valid and invalid).  At higher levels your tests should emulate the
        way the user would interact with your code and confirm that it still works
        the way you intended.
        One of the most important features of the tests is that it should alert other
        developers when their changes will have an impact on the behavior of your
        module.  For example, if a developer removes a feature that you depend on,
        your test should break so they know that the feature is needed.
        """

        self.delayDisplay("Starting the test")

        # Get/create input data

        import SampleData

        registerSampleData()
        inputVolume = SampleData.downloadSample("SEEG_masking1")
        self.delayDisplay("Loaded test data set")

        inputScalarRange = inputVolume.GetImageData().GetScalarRange()
        self.assertEqual(inputScalarRange[0], 0)
        self.assertEqual(inputScalarRange[1], 695)

        outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
        threshold = 100

        # Test the module logic

        logic = SEEG_maskingLogic()

        # Test algorithm with non-inverted threshold
        logic.process(inputVolume, outputVolume, threshold, True)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], threshold)

        # Test algorithm with inverted threshold
        logic.process(inputVolume, outputVolume, threshold, False)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], inputScalarRange[1])

        self.delayDisplay("Test passed")


    
