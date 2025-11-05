# Environment Setup Guide
## Overview
This project is based on two core components: NeRAF and ScaffoldGS. Please follow the steps below to configure the development environment.

## 1. NeRAF Environment Setup
### 1.1 Create Conda Environment
```
conda create --name nerfstudio -y python=3.10
conda activate nerfstudio
python -m pip install --upgrade pip
```

### 1.2 Install PyTorch and CUDA
```
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
```

### 1.3 Install Dependencies
```
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
pip install nerfstudio
ns-install-cli    # Optional but recommended
```

### 1.4 Install NeRAF
```
git clone https://github.com/AmandineBtto/NeRAF.git
cd NeRAF/
pip install -e .
ns-install-cli
```

### 1.5. Data Directory Structure
Ensure your file structure follows this format:
```
├── NeRAF
│   ├── __init__.py
│   ├── NeRAF_config.py
│   ├── NeRAF_pipeline.py 
│   ├── NeRAF_model.py 
│   ├── NeRAF_field.py 
│   ├── NeRAF_datamanager.py 
│   ├── NeRAF_dataparser.py 
│   ├── NeRAF_dataset.py 
│   ├── NeRAF_helpers.py 
│   ├── NeRAF_resnet3d.py
│   ├── NeRAF_evaluator.py
├── pyproject.toml
├── data
│   ├── RAF
│   ├── SoundSpaces
```
RAF folder structure:
```
├── RAF
│   ├── images
│   ├── audio
│   ├── mix
│   │   ├── EmptyRoom
│   │   │   ├── data
│   │   │   ├── images
│   │   │   ├── metadata
│   │   │   ├── sparse
│   │   │   └── transforms.json
│   │   └── FurnishedRoom
│   │       ├── data
│   │       ├── images
│   │       ├── metadata
│   │       ├── sparse
│   │       └── transforms.json
│   └── README.md
```
## 2.NeRAF Training and Testing
### 2.1 Configuration File Modification
Modify line 55 in NeRAF_config.py point to your data path:
```
data_path = "/path/to/data/RAF/mix"  # Change to your actual path
```

### 2.2 Training Command
Important: You must be in the NeRAF main directory when executing training commands
```
NeRAF_dataset=RAF NeRAF_scene=FurnishedRoom ns-train NeRAF
```

### 2.3 Evaluation Command
```
ns-eval --load-config [CONFIG_PATH to config.yml] --output-path [OUTPUT_PATH to out_name.json] --render-output-path [RENDER_OUTPUT_PATH to folder conainting rendered images and audio in the eval set]
```



