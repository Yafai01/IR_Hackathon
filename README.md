# 🌍 Infrared Satellite Image Colorization & Enhancement

> **ISRO Bharatiya Antariksh Hackathon 2026**

An AI-powered framework for enhancing low-resolution Thermal Infrared
(TIR) satellite imagery through **deep learning--based colorization**
and **super-resolution**. This project aims to improve the visual
interpretability of thermal satellite data for applications in disaster
management, environmental monitoring, agriculture, and geospatial
intelligence.

------------------------------------------------------------------------

## 🚀 Features

-   Thermal Infrared Image Colorization
-   Super Resolution using Deep Learning
-   UNet-based Colorization Network
-   SRResNet-based Enhancement
-   End-to-End Inference Pipeline
-   Interactive Gradio Web Interface
-   GPU Acceleration (CUDA)
-   Modular and Extensible Architecture
-   Landsat-9 Compatible

------------------------------------------------------------------------

## 🛰️ Project Overview

Thermal infrared satellite imagery contains rich information about
Earth's surface but is difficult to interpret due to its low spatial
resolution and lack of natural colors.

This project combines modern computer vision and deep learning
techniques to:

1.  Load RGB and Thermal Infrared imagery.
2.  Preprocess satellite data.
3.  Colorize thermal imagery using a UNet model.
4.  Enhance image quality using SRResNet.
5.  Generate visually interpretable outputs.
6.  Display results through a Gradio interface.

------------------------------------------------------------------------

## 🏗️ Project Architecture

``` text
                     Landsat-9 Data
                            │
            ┌───────────────┴───────────────┐
            │                               │
            ▼                               ▼
      RGB Satellite Image           Thermal Infrared
            │                               │
            └───────────────┬───────────────┘
                            ▼
                   Data Preprocessing
                            │
                            ▼
                  Feature Extraction
                            │
                            ▼
               UNet Colorization Network
                            │
                            ▼
            SRResNet Super Resolution Model
                            │
                            ▼
                  Post-processing Pipeline
                            │
                            ▼
              High Resolution Colorized Output
                            │
                            ▼
                  Gradio Visualization UI
```

------------------------------------------------------------------------

## 📂 Repository Structure

``` text
IR_Hackathon/
├── app.py
├── run_inference.py
├── requirements.txt
├── README.md
├── inference/
├── models/
│   ├── colorizer/
│   ├── srresnet/
│   └── unet/
├── utils/
├── datasets/
├── sample_input/
├── outputs/
└── assets/
```

------------------------------------------------------------------------

## 🧠 Models Used

  Model      Purpose
  ---------- ----------------------------
  UNet       Thermal Image Colorization
  SRResNet   Super Resolution
  PyTorch    Deep Learning Framework
  OpenCV     Image Processing

------------------------------------------------------------------------

## 💻 Technology Stack

-   Python
-   PyTorch
-   OpenCV
-   NumPy
-   Gradio
-   CUDA
-   Git & GitHub

------------------------------------------------------------------------

## ⚙️ Installation

``` bash
git clone https://github.com/Yafai01/IR_Hackathon.git
cd IR_Hackathon

python -m venv venv

# Windows
venv\Scripts\activate

pip install -r requirements.txt
```

------------------------------------------------------------------------

## ▶️ Run the Project

Launch the Gradio application:

``` bash
python app.py
```

Run the inference pipeline:

``` bash
python run_inference.py
```

------------------------------------------------------------------------

## 📦 Dataset

A lightweight demo dataset (`sample_00185`--`sample_00192`) is included
for quick testing.

The complete dataset is hosted on Kaggle:

**https://kaggle.com/datasets/bf114c221642290b97529cef6d882f0276d7dc463b56adba03b0d945a62a2012**

After downloading, extract the dataset into:

``` text
datasets/
sample_input/
```

> The full dataset is not included in this repository due to GitHub
> storage limitations.

------------------------------------------------------------------------

## 📊 Results

Replace the placeholders below with screenshots from your project.

-   Input RGB Image
-   Thermal Infrared Image
-   Colorized Output
-   Super-Resolved Output
-   Gradio User Interface

------------------------------------------------------------------------

## 🌍 Applications

-   Disaster Management
-   Forest Fire Monitoring
-   Flood Analysis
-   Climate Studies
-   Agriculture
-   Environmental Monitoring
-   Smart Cities
-   Remote Sensing
-   Geospatial Intelligence

------------------------------------------------------------------------

## 🔮 Future Work

-   Vision Transformers
-   Diffusion Models
-   Multi-Spectral Fusion
-   Sentinel-2 Support
-   TensorRT Optimization
-   ONNX Deployment
-   Jetson Orin Nano Deployment
-   Cloud Deployment

------------------------------------------------------------------------

## 👥 Team

**ISRO Bharatiya Antariksh Hackathon 2026**

**Project Lead:** Yafai

**Team Members:** - Pohar Jahanavi
                    Sahil Chavan

------------------------------------------------------------------------

## 🙏 Acknowledgements

-   Indian Space Research Organisation (ISRO)
-   National Remote Sensing Centre (NRSC)
-   Bharatiya Antariksh Hackathon 2026
-   PyTorch Community
-   OpenCV Community

------------------------------------------------------------------------

## 📄 License

This project is licensed under the MIT License.

------------------------------------------------------------------------

## ⭐ Support

If you found this project useful, please consider starring the
repository.

------------------------------------------------------------------------

::: {align="center"}
### Built for the ISRO Bharatiya Antariksh Hackathon 2026

**Empowering Earth Observation through Artificial Intelligence**
:::
