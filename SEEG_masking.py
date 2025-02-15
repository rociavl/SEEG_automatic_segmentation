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
from scipy import ndimage

import nibabel as nib
import synthstrip 
import tempfile 


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
        self.parent.contributors = ["John Doe (AnyWare Corp.)"]  # TODO: replace with "Firstname Lastname (Organization)"
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
# Register sample data sets in Sample Data module
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
        self.ui.newButton.connect("clicked(bool)", self.onNewButton) # New Button
        self.ui.saveButton.connect("clicked(bool)", self.onSaveButton) # Save Button
        # Set up the Save button
        self.ui.saveButton.setText("Save Mask") # Save Button

        # Set up the CTP button
        self.ui.pushButton_ctp.connect('clicked(bool)', self.onCTPButton) # Push Button for CTP
        # Set up input selector for CTP
        self.ui.inputSelector_ctp.setMRMLScene(slicer.mrmlScene)
        self.ui.inputSelector_ROI_norm.setMRMLScene(slicer.mrmlScene)

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

    def onNewButton(self) -> None:
        """Handle 'New' button click event"""
        print("New Button Clicked")

    
    def onSaveButton(self) -> None:
        """Handle 'Save Mask' button click event."""
        # Use current working directory or specify a default folder for the file dialog
        defaultDirectory = os.getcwd()  
        
        # Alternatively, use a predefined directory such as the user's home directory
        # defaultDirectory = os.path.expanduser("~")  # User's home directory
        
        # Create a file dialog
        fileDialog = qt.QFileDialog()
        fileDialog.setAcceptMode(qt.QFileDialog.AcceptSave)
        fileDialog.setNameFilter("NRRD files (*.nrrd)")
        fileDialog.setDefaultSuffix('nrrd')  # Set the default directory
        fileDialog.setOption(qt.QFileDialog.ShowDirsOnly, False)

        # Show file dialog and get selected folder
        if fileDialog.exec_():
            savePath = fileDialog.selectedFiles()[0]  # Get the selected folder path
            
            if self.outputVolumeNode:
                success = slicer.util.saveNode(self.outputVolumeNode, savePath)
                if success:
                    slicer.util.infoDisplay(f"Mask saved to:\n{savePath}")
                else:
                    slicer.util.errorDisplay("Failed to save mask")
            else:
                slicer.util.errorDisplay("No mask to save")   

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
        # Process the input and create the mask (but don't save it yet)
        
        #self.outputVolumeNode = self.logic.enhance_ctp(self.ui.inputSelector_ctp.currentNode(), self.ui.inputSelector_ROI_norm.currentNode())
        inputVolume = self.ui.inputSelector_ctp.currentNode()
        inputROI = self.ui.inputSelector_ROI_norm.currentNode()
        
        if not inputVolume or not inputROI:
            slicer.util.errorDisplay("Please select both the input volume and the ROI.")
            return
        
        self.outputVolumeNode = self.logic.enhance_ctp(inputVolume, inputROI, methods, outputDir =None)

        if self.outputVolumeNode:
            slicer.app.applicationLogic().GetSelectionNode().SetReferenceActiveVolumeID(self.outputVolumeNode.GetID())
            slicer.app.applicationLogic().PropagateVolumeSelection()
            slicer.app.layoutManager().resetSliceViews()
            slicer.app.processEvents()

        # Optionally, update the UI to indicate the mask was created
        slicer.util.infoDisplay("Mask created! Now, select a location to save the mask. :D")

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
        
        # Set essential properties from the input volume
        #maskVolumeNode.SetSpacing(inputVolume.GetSpacing())
        #maskVolumeNode.SetOrigin(inputVolume.GetOrigin())

        maskVolumeNode.CopyOrientation(inputVolume)  


        #matrix = vtk.vtkMatrix4x4()
        #inputVolume.GetIJKToRASMatrix(matrix)
        #maskVolumeNode.SetIJKToRASMatrix(matrix)
        
        # Copying the display node from the input volume
        #inputImage = inputVolume.GetImageData()
        #if inputImage:
         #   outputImage = vtk.vtkImageData()
          #  outputImage.DeepCopy(inputImage)
           # maskVolumeNode.SetAndObserveImageData(outputImage) # Set the image data to the new volume node
        
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
        
        # Apply binarization logic: all non-zero values become 1, zero stays as 0
        thresh = filters.threshold_otsu(smooth_input)
        
        maskArray = (smooth_input > thresh).astype(np.uint8)  # All non-zero become 1
        maskArray = maskArray.reshape(dims)
        print("Mask array dtype:", maskArray.dtype)
        print("Unique values in mask array:", np.unique(maskArray))
        selem_close = ndimage.generate_binary_structure(3, 1) # 3D structuring element
        print("Input array shape:", inputArray.shape)
        print("Mask array shape:", maskArray.shape)
        print("Selem shape:", selem_close.shape)
        closed = ndimage.binary_closing(maskArray, structure=selem_close, iterations=3).astype(np.uint8)
        
        filled = ndimage.binary_fill_holes(closed).astype(np.uint8)

        selem_dilate = ndimage.generate_binary_structure(3, 1) # 3D structuring element
        dilated = ndimage.binary_dilation(filled, structure=selem_dilate, iterations=4).astype(np.uint8)
        
        final_flat = dilated.ravel()
  

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
    
    def 
    

    


    ### Finding points of contact (creating a mask and enhancing the image) (no funciona aun :c) en al archivo ctp_enhance.py sí

    def enhance_ctp(
            inputVolume: vtkMRMLScalarVolumeNode,
            inputROI: vtkMRMLScalarVolumeNode, # ROI (norm mask), do I have to change the name? 
            methods='all', outputDir=None 
    ) -> vtkMRMLScalarVolumeNode:
        """
        Enhancing the image using different methods
        param inputVolume: volume to be enhanced
        param inputROI: volume to be used as a mask
        param methods: methods to be used for enhancing the image
        
        """

        return enhance_ctp(inputVolume, inputROI, methods = 'all', outputDir = r'C:\\Users\\rocia\\Downloads\\TFG\\Cohort\\Enhance_ctp_tests') 
        


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
