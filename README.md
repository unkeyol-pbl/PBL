# Elderly Care Violence Detection & Hand Pinching Recognition

Automatic detection of macro/micro violence and hand pinching gestures in elderly care environments.

## Introduction

 Identifying harmful behaviors in elderly care environments is structurally limited, especially when dealing with subtle forms of aggression such as pinching. Unlike overt violence, pinching is momentary, involves extremely small motion ranges, and produces only minimal visual changes. These micro-violence actions often occur in low-resolution surveillance settings, making them difficult to detect with conventional video analysis methods.

 This project specifically targets the detection of pinching gestures by quantitatively analyzing fine-grained joint movements. By focusing on micro-level motion dynamics, the model aims to capture subtle violent behaviors that existing algorithms fail to recognize, establishing a foundation for improved safety monitoring in nursing-home environments.

## Environment

- Google Colab (Python 3.10)
- GPU: NVIDIA Tesla T4
- Google Drive for storage and synchronization

## Libraries and Frameworks

- TensorFlow 2.15
- MediaPipe 0.10
- Ultralytics YOLOv8
- OpenCV 4.8
- NumPy, Matplotlib, tqdm
- yt-dlp (YouTube video acquisition)
- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN.git)

## Data Processing Tools

- OpenCV for video preprocessing
- MediaPipe for landmark extraction
- NumPy for numerical operations
- yt-dlp for external video collection

## Model Components

- MLP-based pinch classification model (126-dimensional landmark input)
- YOLOv8 object detection model
- MediaPipe Hands/FaceMesh for keypoint extraction

## Dataset

This video dataset used in this project is available on Google Drive : 
https://drive.google.com/file/d/17GROsHq_Ey3XYCGhaHTmupkAlt3HjHLV/view?usp=drive_link

## Video Processing Enhancement (Real-ESRGAN)

Source: https://github.com/xinntao/Real-ESRGAN

| Before | After |
| --- | --- |
|  <img width="1726" height="871" alt="image" src="https://github.com/user-attachments/assets/c8e4d64f-0316-403c-be5c-4c5efbceb6b4" /> | <img width="1823" height="871" alt="image2" src="https://github.com/user-attachments/assets/8395a1a4-49cf-42ec-b60a-139fb4d768f1" />|



## Violence Detecting

Stage 1: Coarse violence / non-violence detection

- use GCN-BiLSTM model
- Clear, large-scale violent actions (e.g., hitting, pushing, shoving) are filtered at this stage.
<img width="465" height="152" alt="3" src="https://github.com/user-attachments/assets/ea415dcb-6c40-46c7-8511-839ac6fcf3fa" />


- Only clips classified as **non-violent or containing subtle motions** are forwarded to Stage 2 for fine-grained analysis.
    
 <img width="763" height="574" alt="4" src="https://github.com/user-attachments/assets/a495f39d-5385-4b12-974d-e3a185453060" />

    

## Hand Pinching

| Case1 | Case2 |
| --- | --- |
| <img width="1387" height="652" alt="d" src="https://github.com/user-attachments/assets/890ce6ca-100c-4ee1-9dca-613877cd0d98" /> |<img width="1321" height="634" alt="b" src="https://github.com/user-attachments/assets/455eed81-c5cd-4879-92e8-27c7083837d7" /> |


Stage 2: Micro-violence detection tailored to nursing-home environments

1. Pinching behavior (PINCH)
    - MediaPipe Hands → 21 3D hand landmark coordinates
    - MediaPipe FaceMesh → facial region landmarks
    - YOLOv8 → bounding boxes for people and surrounding objects
    - These three sources are fused to analyze hand–face–object interaction context.
    - Classification model: MLP-based pinch classifier (`skeleton_pinch_v3.keras`)
 
## Usage

Run the main pipeline with:

```bash
python run.py
```


**Note:** The preprocessing step for raw videos can take a long time.

For convenience, a preprocessed sample video has been provided in the `enhanced_video/` folder.

Running `run.py` will use this enhanced video directly so you can quickly test the pipeline without waiting for preprocessing.
