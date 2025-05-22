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
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "Examples")]
        self.parent.dependencies = []
        self.parent.contributors = ["Rocio Avalos (AnyWare Corp.)"]
        self.parent.helpText = _("""
This is an example of scripted loadable module bundled in an extension.
See more information in <a href="https://github.com/organization/projectname#SEEG_masking">module documentation</a>.
""")
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
    import SampleData

    iconsPath = os.path.join(os.path.dirname(__file__), "Resources/Icons")

    # SEEG_masking1
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        category="SEEG_masking",
        sampleName="SEEG_masking1",
        thumbnailFileName=os.path.join(iconsPath, "SEEG_masking1.png"),
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        fileNames="SEEG_masking1.nrrd",
        checksums="SHA256:998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        nodeNames="SEEG_masking1",
    )

    # SEEG_masking2
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        category="SEEG_masking",
        sampleName="SEEG_masking2",
        thumbnailFileName=os.path.join(iconsPath, "SEEG_masking2.png"),
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        fileNames="SEEG_masking2.nrrd",
        checksums="SHA256:1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        nodeNames="SEEG_masking2",
    )

#
# SEEG_maskingWidget - Pure Direct UI Access
#

class SEEG_maskingWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class with pure direct UI access."""

    def __init__(self, parent=None) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)
        self.logic = None
        self.outputVolumeNode = None  # Track the generated mask volume

    def setup(self) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/SEEG_masking.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets - This is crucial!
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class
        self.logic = SEEG_maskingLogic()

        # Debug: Check what UI elements we have
        self.debugUIElements()

        # Configure volume selectors explicitly
        self.configureVolumeSelectors()

        # Connect UI elements directly
        self.ui.applyButton.connect("clicked(bool)", self.onApplyButton)
        self.ui.saveButton.connect("clicked(bool)", self.onSaveButton)
        self.ui.saveButton.setText("Save Mask")

        # Connect volume selectors to update button state
        if hasattr(self.ui, 'inputSelector'):
            self.ui.inputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onInputChanged)
        if hasattr(self.ui, 'inputSelectorCT'):
            self.ui.inputSelectorCT.connect("currentNodeChanged(vtkMRMLNode*)", self.onInputChanged)

        # Initial button state check
        self.updateApplyButtonState()

    def debugUIElements(self):
        """Debug what UI elements we have access to."""
        print("=== UI Elements Debug ===")
        print(f"Available UI attributes: {[attr for attr in dir(self.ui) if not attr.startswith('_')]}")
        print(f"Has inputSelector: {hasattr(self.ui, 'inputSelector')}")
        print(f"Has inputSelectorCT: {hasattr(self.ui, 'inputSelectorCT')}")
        
        if hasattr(self.ui, 'inputSelector'):
            print(f"inputSelector type: {type(self.ui.inputSelector)}")
        if hasattr(self.ui, 'inputSelectorCT'):
            print(f"inputSelectorCT type: {type(self.ui.inputSelectorCT)}")

    def configureVolumeSelectors(self):
        """Configure the volume selectors explicitly."""
        print("=== Configuring Volume Selectors ===")
        
        # Configure MRI input selector
        if hasattr(self.ui, 'inputSelector'):
            self.ui.inputSelector.setMRMLScene(slicer.mrmlScene)
            self.ui.inputSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
            self.ui.inputSelector.selectNodeUponCreation = False
            self.ui.inputSelector.addEnabled = False
            self.ui.inputSelector.removeEnabled = False
            self.ui.inputSelector.noneEnabled = True
            self.ui.inputSelector.showHidden = False
            self.ui.inputSelector.showChildNodeTypes = False
            self.ui.inputSelector.setToolTip("Select MRI volume for brain mask generation")
            print("✓ inputSelector configured")
        else:
            print("✗ inputSelector not found!")
        
        # Configure CT input selector
        if hasattr(self.ui, 'inputSelectorCT'):
            self.ui.inputSelectorCT.setMRMLScene(slicer.mrmlScene)
            self.ui.inputSelectorCT.nodeTypes = ["vtkMRMLScalarVolumeNode"]
            self.ui.inputSelectorCT.selectNodeUponCreation = False
            self.ui.inputSelectorCT.addEnabled = False
            self.ui.inputSelectorCT.removeEnabled = False
            self.ui.inputSelectorCT.noneEnabled = True
            self.ui.inputSelectorCT.showHidden = False
            self.ui.inputSelectorCT.showChildNodeTypes = False
            self.ui.inputSelectorCT.setToolTip("Select CT volume for enhancement processing")
            print("✓ inputSelectorCT configured")
        else:
            print("✗ inputSelectorCT not found!")

        # Check available volumes
        volumes = slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode")
        print(f"Available volumes in scene: {len(volumes)}")
        for i, vol in enumerate(volumes):
            print(f"  {i+1}: {vol.GetName()}")

    def cleanup(self) -> None:
        """Called when the application closes and the module widget is destroyed."""
        self.removeObservers()

    def enter(self) -> None:
        """Called each time the user opens this module."""
        # Refresh selectors when entering
        self.configureVolumeSelectors()
        self.updateApplyButtonState()

    def exit(self) -> None:
        """Called each time the user opens a different module."""
        pass

    def onInputChanged(self) -> None:
        """Called when input volume selection changes."""
        print("Input selection changed")
        self.updateApplyButtonState()

    def updateApplyButtonState(self) -> None:
        """Update the Apply button state based on current selections."""
        # Get current selections
        input_volume = None
        input_volume_CT = None
        
        if hasattr(self.ui, 'inputSelector'):
            input_volume = self.ui.inputSelector.currentNode()
        if hasattr(self.ui, 'inputSelectorCT'):
            input_volume_CT = self.ui.inputSelectorCT.currentNode()

        print(f"Current selections - MRI: {input_volume.GetName() if input_volume else 'None'}, CT: {input_volume_CT.GetName() if input_volume_CT else 'None'}")

        # Determine button state
        canApply = (input_volume is not None and input_volume_CT is not None)
        
        if not input_volume and not input_volume_CT:
            tooltip = "Select both MRI and CT input volumes"
        elif not input_volume:
            tooltip = "Select MRI input volume for brain mask generation"
        elif not input_volume_CT:
            tooltip = "Select CT input volume for enhancement processing"
        else:
            tooltip = "Generate brain mask and enhance CT volume"

        # Update button
        if hasattr(self.ui, 'applyButton'):
            self.ui.applyButton.enabled = canApply
            self.ui.applyButton.toolTip = _(tooltip)

    def onApplyButton(self) -> None:
        """Run processing when user clicks 'Apply' button."""
        print("Apply button clicked!")
        
        with slicer.util.tryWithErrorDisplay(_("Failed to compute results."), waitCursor=True):
            
            # Get input volumes directly from UI
            input_volume_node = None
            input_volume_node_CT = None
            
            if hasattr(self.ui, 'inputSelector'):
                input_volume_node = self.ui.inputSelector.currentNode()
            if hasattr(self.ui, 'inputSelectorCT'):
                input_volume_node_CT = self.ui.inputSelectorCT.currentNode()
            
            print(f"Selected volumes - MRI: {input_volume_node}, CT: {input_volume_node_CT}")
            
            # Validate inputs
            if not input_volume_node:
                slicer.util.errorDisplay("Please select an MRI input volume for brain mask generation.")
                return
                
            if not input_volume_node_CT:
                slicer.util.errorDisplay("Please select a CT input volume for enhancement processing.")
                return
            
            logging.info(f"Processing MRI volume: {input_volume_node.GetName()}")
            logging.info(f"Processing CT volume: {input_volume_node_CT.GetName()}")
            
            # Generate brain mask from MRI volume
            try:
                self.outputVolumeNode = self.logic.process(input_volume_node)
                if not self.outputVolumeNode:
                    slicer.util.errorDisplay("Failed to generate brain mask.")
                    return
                
                logging.info(f"Generated brain mask: {self.outputVolumeNode.GetName()}")
                    
            except Exception as e:
                slicer.util.errorDisplay(f"Error generating brain mask: {str(e)}")
                logging.error(f"Brain mask generation error: {str(e)}")
                return
            
            # Initialize CTPEnhancer and process CT volume
            try:
                # Import CTPEnhancer here to avoid import issues
                from Threshold_mask.ctp_enhancer import CTPEnhancer
                
                # Create CTPEnhancer instance
                ctp_enhancer = CTPEnhancer()
                
                # Verify the enhance_ctp method exists and is callable
                if not hasattr(ctp_enhancer, 'enhance_ctp'):
                    slicer.util.errorDisplay("CTPEnhancer does not have enhance_ctp method.")
                    return
                
                if not callable(getattr(ctp_enhancer, 'enhance_ctp')):
                    slicer.util.errorDisplay("enhance_ctp is not callable.")
                    return
                
                print(f"CTPEnhancer instance created successfully: {type(ctp_enhancer)}")
                print(f"enhance_ctp method: {type(ctp_enhancer.enhance_ctp)}")
                
                # Call the enhance_ctp method with simplified parameters
                enhanced_volumes = ctp_enhancer.enhance_ctp(
                    inputVolume=input_volume_node_CT,
                    inputROI=self.outputVolumeNode,  # Brain mask as ROI
                    outputDir=r"C:\Users\rocia\Downloads\TFG\Cohort\Extension",  
                    model_path=MODEL_PATH,
    
                )
                
                if enhanced_volumes and len(enhanced_volumes) > 0:
                    volume_names = list(enhanced_volumes.keys())
                    slicer.util.infoDisplay(f"Successfully generated {len(enhanced_volumes)} enhanced volumes:\n" + 
                                        "\n".join(volume_names))
                    logging.info(f"Enhanced volumes: {volume_names}")
                else:
                    slicer.util.warningDisplay("No enhanced volumes were generated.")
                    
            except ImportError as e:
                slicer.util.errorDisplay(f"Failed to import CTPEnhancer: {str(e)}\n\nPlease check that ctp_enhancer.py is in the Threshold_mask directory.")
                logging.error(f"Import error: {str(e)}")
            except FileNotFoundError as e:
                slicer.util.errorDisplay(f"Model file not found: {MODEL_PATH}\n\nPlease check that the model file exists.")
                logging.error(str(e))
            except TypeError as e:
                slicer.util.errorDisplay(f"Type error during CT enhancement: {str(e)}\n\nThis might be a parameter type mismatch.")
                logging.error(f"Type error: {str(e)}")
            except Exception as e:
                slicer.util.errorDisplay(f"Error during CT enhancement: {str(e)}")
                logging.error(f"CT enhancement error: {str(e)}")
                # Print more detailed error info for debugging
                import traceback
                traceback.print_exc()


    def onSaveButton(self) -> None:
        """Handle 'Save Mask' button click event."""
        logging.info("Save button clicked!")

        if not self.outputVolumeNode:
            slicer.util.errorDisplay("No mask to save! Please click 'Apply' first to generate a brain mask.")
            logging.error("No mask to save!")
            return

        logging.info(f"Saving mask: {self.outputVolumeNode.GetName()}")

        # Create a file dialog
        fileDialog = qt.QFileDialog()
        fileDialog.setAcceptMode(qt.QFileDialog.AcceptSave)
        fileDialog.setNameFilter("NRRD files (*.nrrd);;All files (*)")
        fileDialog.setDefaultSuffix('nrrd')
        fileDialog.setOption(qt.QFileDialog.ShowDirsOnly, False)
        
        # Set default filename
        default_name = f"BrainMask_{self.outputVolumeNode.GetName()}.nrrd"
        fileDialog.selectFile(default_name)

        # Show file dialog and get selected file path
        if fileDialog.exec_():
            savePath = fileDialog.selectedFiles()[0]
            logging.info(f"Selected save path: {savePath}")

            try:
                success = slicer.util.saveNode(self.outputVolumeNode, savePath)

                if success:
                    slicer.util.infoDisplay(f"Brain mask saved successfully to:\n{savePath}")
                    logging.info(f"Mask saved successfully at: {savePath}")
                else:
                    slicer.util.errorDisplay("Failed to save mask. Please check the file path and permissions.")
                    logging.error("Failed to save mask")
            except Exception as e:
                slicer.util.errorDisplay(f"Error saving mask: {str(e)}")
                logging.error(f"Error saving mask: {str(e)}")

#
# SEEG_maskingLogic - Simplified without parameter node
#

class SEEG_maskingLogic:
    """This class implements all the actual computation done by your module."""

    def __init__(self, parent=None):
        """Called when the user opens the module the first time and the widget is initialized."""
        # Initialize the brain mask extractor
        self.maskExtractor = BrainMaskExtractor()

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
    """

    def setUp(self):
        """Do whatever is needed to reset the state - typically a scene clear will be enough."""
        slicer.mrmlScene.Clear()

    def runTest(self):
        """Run as few or as many tests as needed here."""
        self.setUp()
        self.test_SEEG_masking1()

    def test_SEEG_masking1(self):
        """Test the module logic with sample data."""
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