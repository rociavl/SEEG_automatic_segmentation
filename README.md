# SEEG_automatic_segmentation

![image](https://github.com/user-attachments/assets/be33d580-feb4-4caa-9a48-30ebd59ee7e1)
![image](https://github.com/user-attachments/assets/4e0f3fa7-2de5-4efc-b5d4-10d8878caf77)
<p align="center">
  <a href="https://www.youtube.com/watch?v=mSGtHaBInJM">
    <img src="https://img.youtube.com/vi/mSGtHaBInJM/0.jpg" alt="Watch the demo video">
  </a>
</p>
The video demonstrates real-time electrode detection, trajectory reconstruction, and quality assessment with interactive 3D visualization.


A comprehensive 3D Slicer extension for automated localization of Stereoelectroencephalography (SEEG) electrode contacts in post-operative CT scans. This system transforms the manual electrode localization process from 4+ hours to 30 minutes while maintaining clinical accuracy standards.

## 🎯 Overview

This project addresses a critical need identified by neurosurgeons and neurophysiologists worldwide: the development of a user-friendly module to help automate the localization of contact points on stereo electroencephalography (SEEG) electrodes. The system has been successfully deployed at Hospital del Mar's Epilepsy Unit, demonstrating real-world clinical applicability.

### Key Features

- **🔄 Complete Automation**: End-to-end processing from CT/MRI to validated electrode coordinates
- **🧠 AI-Powered**: Multi-mask ensemble with confidence scoring for clinical decision support
- **⚡ Efficient**: Reduces processing time by >95% (4+ hours → 30 minutes)
- **🎯 Accurate**: Achieves 98.8% accuracy within 2mm clinical threshold
- **🔧 Clinical Ready**: Integrated into 3D Slicer with intuitive interface
- **🤝 Human-AI Collaboration**: Preserves clinical decision-making authority

## 🏥 Clinical Impact

- **Patient Safety**: Reduced electrode implantation duration and infection risk
- **Clinical Efficiency**: Dramatic reduction in specialist workload
- **Quality**: Maintains clinical accuracy standards with automated quality control
- **Accessibility**: User-friendly interface for medical professionals without technical background

## 🔬 Technical Innovation

### Pipeline Architecture

The system implements a novel 6-stage processing pipeline:

1. **Brain Extraction**: MONAI-based 3D U-Net for automated brain segmentation
2. **Multi-Modal Enhancement**: 7 complementary approaches for electrode visibility
3. **Adaptive Thresholding**: Machine learning-based threshold prediction
4. **Global Voting Ensemble**: 38 segmentation variants with consensus formation
5. **Contact Authentication**: Confidence-based electrode validation
6. **Trajectory Reconstruction**: Complete pathway mapping from entry to target

### Key Innovations

- **Multi-Mask Ensemble**: Generates 38 binary mask variants for robust detection
- **Conservative Confidence Scoring**: Enables graduated clinical decision-making
- **Regression-Based Authentication**: 98.8% accuracy with continuous confidence scores
- **Hybrid Clustering**: DBSCAN + Louvain community detection for trajectory reconstruction

## 📊 Performance Metrics

| Metric | Result |
|--------|--------|
| Accuracy (2mm threshold) | 98.8% |
| Mean localization distance | 0.33 mm |
| Processing time reduction | >95% |
| Clinical deployment | Hospital del Mar |
| Patient validation | 8 patients, external validation |

## 🚀 Quick Start

### Prerequisites

- [3D Slicer](https://www.slicer.org/) (version 5.0+)
- Python 3.9+
- Required dependencies (see `requirements.txt`)

### Installation

#### Option 1: 3D Slicer Extension Manager (Coming Soon!)
🚀 **The extension will soon be available through the official 3D Slicer Extension Manager for easy one-click installation.**

- Open 3D Slicer
- Go to Extension Manager → Install Extensions
- Search for "SEEG Masking" or "SEEG Automatic Segmentation"
- Click Install and restart 3D Slicer

#### Option 2: Manual Installation (Current)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/rociavl/SEEG_automatic_segmentation.git
   cd SEEG_automatic_segmentation
   ```

2. **Install in 3D Slicer:**
   - Open 3D Slicer
   - Go to Extension Manager → Developer Tools
   - Add module path: `path/to/SEEG_automatic_segmentation/SEEG_masking`
   - Restart 3D Slicer

3. **Load the extension:**
   - Navigate to Modules → Examples → SEEG Masking
   - The extension interface will appear in the module panel

### Basic Usage

1. **Input Selection:**
   - Load post-operative CT scan
   - Load or generate brain mask (MRI-based or automated)

2. **Processing:**
   - Configure parameters (defaults work for most cases)
   - Click "Apply" to start automated processing
   - Monitor progress through console output

3. **Review Results:**
   - Adjust confidence threshold for electrode candidates
   - Validate predicted contacts
   - Export results for clinical use

## 📁 Project Structure

```
SEEG_automatic_segmentation/
├── SEEG_masking/                 # Main 3D Slicer extension
│   ├── SEEG_masking.py          # Main module file
│   ├── Resources/               # UI resources and icons
│   └── Brain_mask_methods/      # Brain extraction algorithms
├── notebooks/                   # Development notebooks
├── models/                      # Pre-trained ML models
├── documentation/               # Technical documentation
└── examples/                    # Example data and usage
```

## 🔧 Advanced Usage

### Confidence-Based Workflow

The system provides graduated clinical decision support:

- **High Confidence (>60%)**: Direct clinical use
- **Medium Confidence (20-60%)**: Clinical review recommended  
- **Low Confidence (<20%)**: Manual validation required

### Trajectory Analysis

For complete electrode pathway reconstruction:

1. Enable bolt head detection
2. Configure trajectory parameters
3. Review quality scores and flagged trajectories
4. Export to clinical planning systems

### Custom Configuration

The system supports various electrode manufacturers and imaging protocols:

- **Electrode Types**: DIXI Medical (default), Medtronic, Abbott
- **CT Protocols**: Standard clinical protocols (120 kVp, 0.5-0.625mm slices)
- **Enhancement Methods**: Configurable based on image quality

## 📚 Documentation

- **[Technical Documentation](documentation/)**: Detailed implementation guide
- **[API Reference](documentation/api/)**: Complete function documentation
- **[Clinical Workflow](documentation/clinical/)**: Step-by-step clinical usage
- **[Development Guide](documentation/development/)**: For contributors

## 🧪 Validation

The system has been rigorously validated through:

- **Leave-One-Patient-Out Cross-Validation**: 8-patient development cohort
- **External Patient Validation**: Independent test cases
- **Clinical Deployment**: Real-world usage at Hospital del Mar
- **Performance Benchmarking**: Comparison with state-of-the-art methods

## 🏆 Published Research

This work is documented in a comprehensive bachelor's thesis:

**"Medical Software Module in 3D Slicer for Automatic Segmentation and Trajectory Reconstruction of SEEG Electrodes Using AI and Data Science"**

*Author*: Rocío Ávalos Morillas  
*Institution*: Universitat Politècnica de Catalunya (UPC)  
*Collaboration*: Hospital del Mar, Center for Brain and Cognition (UPF)

## 🤝 Contributing

We welcome contributions from the medical imaging and epilepsy research communities!

### How to Contribute

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/amazing-feature`)
3. **Commit changes** (`git commit -m 'Add amazing feature'`)
4. **Push to branch** (`git push origin feature/amazing-feature`)
5. **Open a Pull Request**

### Areas for Contribution

- **Algorithm Improvements**: Enhanced segmentation or classification methods
- **Clinical Validation**: Testing on additional patient cohorts
- **User Interface**: Improved clinical workflow integration
- **Documentation**: Examples, tutorials, and clinical guides
- **Performance**: Optimization for different hardware configurations

## ⚖️ Ethical Considerations

This system is designed with clinical ethics in mind:

- **Patient Safety**: Conservative confidence scoring prevents overconfidence
- **Privacy**: Data anonymization and HIPAA compliance
- **Transparency**: Interpretable confidence scores enable clinical validation
- **Human Authority**: Preserves physician decision-making in all critical decisions

## 📈 Future Development

Planned enhancements include:

- **📦 Official Extension Release**: Coming soon to 3D Slicer Extension Manager
- **Multi-Center Validation**: Expansion to additional clinical sites
- **Enhanced Electrode Support**: Additional manufacturer compatibility
- **Regulatory Compliance**: FDA Software as Medical Device (SaMD) pathway
- **Real-Time Processing**: Further performance optimization
- **Integration**: PACS and EHR system connectivity

## 🎓 Educational Use

This project serves as an educational resource for:

- **Medical Students**: Understanding SEEG electrode localization
- **Engineering Students**: Medical image processing and AI applications
- **Researchers**: Baseline for advanced electrode localization methods
- **Clinicians**: Training on automated analysis tools

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Hospital del Mar Epilepsy Unit**: Clinical collaboration and validation
- **Center for Brain and Cognition (UPF)**: Research environment and support
- **3D Slicer Community**: Platform and development tools
- **Open Source Libraries**: NumPy, scikit-learn, PyTorch, MONAI, and others

## 📞 Contact

**Rocío Ávalos Morillas**  
*Biomedical Engineering Student*  
*Universitat Politècnica de Catalunya*

- 📧 Email: rocio.avalos029@gmail.com
- 🔗 LinkedIn: [Rocío Ávalos](https://www.linkedin.com/in/roc%C3%ADo-%C3%A1valos-morillas-04a5372b1/)
- 🐙 GitHub: [@rociavl](https://github.com/rociavl)

## 📋 Citation

If you use this work in your research, please cite:

```bibtex
@misc{avalos2025seeg,
  title={Medical Software Module in 3D Slicer for Automatic Segmentation and Trajectory Reconstruction of SEEG Electrodes Using AI and Data Science},
  author={Ávalos Morillas, Rocío},
  year={2025},
  institution={Universitat Politècnica de Catalunya},
  url={https://github.com/rociavl/SEEG_automatic_segmentation}
}
```

---

**⚠️ Medical Device Notice**: This software is for research and educational purposes. Clinical use requires appropriate validation and regulatory approval in your jurisdiction.
