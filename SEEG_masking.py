import logging
import os
from typing import Annotated, Optional
import sys

import slicer.util
import vtk
import numpy as np
import time

# Add the Brain_mask_methods directory to the Python path
module_dir = os.path.dirname(__file__)
brain_mask_methods_dir = os.path.join(module_dir, "Brain_mask_methods")
if brain_mask_methods_dir not in sys.path:
    sys.path.append(brain_mask_methods_dir)

# Import the BrainMaskExtractor from the correct module
from brain_mask_extractor_otsu import BrainMaskExtractor # import from Brain_mask_methods

# Add the Threshold_mask directory to the Python path
threshold_mask_dir = os.path.join(module_dir, "Threshold_mask")
if threshold_mask_dir not in sys.path:
    sys.path.append(threshold_mask_dir)

# Import CTPEnhancer from enhance_ctp
from Threshold_mask.ctp_enhancer import CTPEnhancer


# Define path to the model
MODEL_PATH = os.path.join(module_dir, "models", "random_forest_modelP1.joblib")


import slicer
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import (
    parameterNodeWrapper,
    WithinRange)

from slicer import vtkMRMLScalarVolumeNode
import qt


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
    outputVolume - The output volume that will contain the thresholded volume.
    """

    inputVolume: vtkMRMLScalarVolumeNode
    inputVolumeCT: vtkMRMLScalarVolumeNode 
    outputVolume: vtkMRMLScalarVolumeNode



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
        self.outputVolumeNode = None  # Track the generated mask volume

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
        
        self.ui.saveButton.connect("clicked(bool)", self.onSaveButton) # Save Button
        # Set up the Save button
        self.ui.saveButton.setText("Save Mask") # Save Button

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

    
    def _checkCanApply(self, caller=None, event=None) -> None:
        """  Check if the input volume is selected. """
        if self._parameterNode and self._parameterNode.inputVolume:
            self.ui.applyButton.toolTip = _("Compute output volume")
            self.ui.applyButton.enabled = True
        else:
            self.ui.applyButton.toolTip = _("Select input volume nodes")
            self.ui.applyButton.enabled = False

    # Handle 'Apply' button click event (run processing)
    def onApplyButton(self) -> None:
        """Run processing when user clicks 'Apply' button."""
        with slicer.util.tryWithErrorDisplay(_("Failed to compute results."), waitCursor=True):
            input_volume_node = self.ui.inputSelector.currentNode()
            input_volume_node_CT = self.ui.inputSelectorCT.currentNode()
            
            # Generate brain mask
            self.outputVolumeNode = self.logic.process(input_volume_node)
            
            # Initialize CTPEnhancer with model
            ctp_enhancer = CTPEnhancer()
            try:
                # Use the predefined MODEL_PATH
                enhanced_volumes = ctp_enhancer.enhance_ctp(
                    inputVolume=input_volume_node_CT,
                    inputROI=self.outputVolumeNode,  # Brain mask as ROI
                    model_path=MODEL_PATH  # Pass model path
                )
                slicer.util.infoDisplay(f"Generated {len(enhanced_volumes)} enhanced volumes.")
            except FileNotFoundError as e:
                slicer.util.errorDisplay(f"Model file not found: {MODEL_PATH}")
                logging.error(str(e))

    
    def onSaveButton(self) -> None:
        """Handle 'Save Mask' button click event."""
        logging.info("Save button clicked!")

        if not self.outputVolumeNode:
            slicer.util.errorDisplay("No mask to save! Apply first.")
            logging.error("No mask to save!")
            return

        logging.info(f"Saving mask: {self.outputVolumeNode.GetName()}")

        # Create a file dialog
        fileDialog = qt.QFileDialog()
        fileDialog.setAcceptMode(qt.QFileDialog.AcceptSave)
        fileDialog.setNameFilter("NRRD files (*.nrrd)")
        fileDialog.setDefaultSuffix('nrrd')
        fileDialog.setOption(qt.QFileDialog.ShowDirsOnly, False)

        # Show file dialog and get selected folder
        if fileDialog.exec_():
            savePath = fileDialog.selectedFiles()[0]
            logging.info(f"Selected save path: {savePath}")

            success = slicer.util.saveNode(self.outputVolumeNode, savePath)

            if success:
                slicer.util.infoDisplay(f"Mask saved to:\n{savePath}")
                logging.info(f"Mask saved successfully at: {savePath}")
            else:
                slicer.util.errorDisplay("Failed to save mask")
                logging.error("Failed to save mask")


#
# SEEG_maskingLogic
#

class SEEG_maskingLogic:
    """This class implements all the actual computation done by your module."""

    def __init__(self, parent=None):
        """Called when the user opens the module the first time and the widget is initialized."""
        # Initialize the brain mask extractor
        self.maskExtractor = BrainMaskExtractor()

    
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

    def process(self, inputVolume: vtkMRMLScalarVolumeNode, showResult: bool = True) -> vtkMRMLScalarVolumeNode:
        """
        Run the processing algorithm.
        Creates a mask using the BrainMaskExtractor.
        
        Parameters:
        -----------
        inputVolume : vtkMRMLScalarVolumeNode
            The input volume to process
        showResult : bool, optional
            Whether to show the result in the Slicer viewer (default is True)
            
        Returns:
        --------
        vtkMRMLScalarVolumeNode
            The output volume node containing the mask
        """
        if not inputVolume:
            raise ValueError("Input volume is invalid")

        # Use the brain mask extractor to create the mask
        return self.maskExtractor.extract_mask(inputVolume, threshold_value=20, show_result=showResult)


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

        # Test the module logic
        logic = SEEG_maskingLogic()
        
        # Process the input volume
        outputVolume = logic.process(inputVolume, True)
        
        # Check that output volume exists
        self.assertIsNotNone(outputVolume)
        
        # Basic validation that the output is a binary mask (0s and 1s only)
        outputArray = slicer.util.arrayFromVolume(outputVolume)
        unique_values = np.unique(outputArray)
        self.assertTrue(len(unique_values) <= 2, "Mask should only contain binary values")
        self.assertTrue(np.all(np.isin(unique_values, [0, 1])), "Values should be 0 or 1")

        self.delayDisplay("Test passed")