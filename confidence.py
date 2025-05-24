import csv
from qt import QSlider, QLabel, QVBoxLayout, QWidget
import slicer

class ConfidenceThresholdViewer(QWidget):
    def __init__(self, csv_path):
        super().__init__()
        self.csv_path = csv_path
        self.points = []
        self.threshold = 0.1  
        self.markupNode = None
        self.setWindowTitle("Confidence Threshold Viewer")
        self.setMinimumWidth(300)
        self.loadPoints()
        self.initUI()
        self.refreshPoints()

    def initUI(self):
        layout = QVBoxLayout()
        self.label = QLabel(f"Threshold: {self.threshold:.2f}")
        layout.addWidget(self.label)

        self.slider = QSlider()
        self.slider.setOrientation(1)  # Horizontal
        self.slider.setMinimum(0)
        self.slider.setMaximum(100)
        self.slider.setValue(int(self.threshold * 100))
        self.slider.valueChanged.connect(self.updateThreshold)
        layout.addWidget(self.slider)

        self.setLayout(layout)
        self.show()

    def loadPoints(self):
        self.points = []
        with open(self.csv_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                try:
                    x = float(row['RAS_X'])
                    y = float(row['RAS_Y'])
                    z = float(row['RAS_Z'])
                    confidence = float(row['Ensemble_Confidence'])
                    self.points.append((x, y, z, confidence))
                except Exception as e:
                    print(f"Skipping row due to error: {e}")

    def updateThreshold(self, value):
        self.threshold = value / 100.0
        self.label.setText(f"Threshold: {self.threshold:.2f}")
        self.refreshPoints()

    def refreshPoints(self):
        if not self.markupNode:
            self.markupNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", "PE")
        else:
            self.markupNode.RemoveAllControlPoints()

        for x, y, z, confidence in self.points:
            if confidence >= self.threshold:
                self.markupNode.AddFiducial(x, y, z)

# === Path  CSV ===
csv_path = r"C:\Users\rocia\Downloads\TFG\Cohort\Extension\P8_predictions.csv"

# === Run the interactive viewer ===
confidenceViewer = ConfidenceThresholdViewer(csv_path)

#exec(open(r'C:\Users\rocia\AppData\Local\slicer.org\Slicer 5.6.2\SEEG_module\SEEG_masking\confidence.py').read())