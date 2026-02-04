# [IPIU 2026] Multimodal Analysis of YouTube Shorts View Counts

This repository provides the official implementation and research artifacts for the paper presented at IPIU 2026 (The 38th Workshop on Image Processing and Understanding).

---

## Abstract

The success of short-form content like YouTube Shorts is driven by the complex interplay between visual (thumbnails) and textual (titles) elements.
This research proposes a Multimodal Regression Framework that utilizes CLIP (ViT-L/14) to extract joint embeddings and LightGBM to predict view counts.
We emphasize the "Semantic Alignment" between modalities as a key driver of content virality.


## Setup

Follow these steps to configure the environment and reproduce the experimental results.

### 1. Prerequisites

- Python: v3.9 or higher (Recommended)
- CUDA: v11.3 or higher (Required for GPU-accelerated CLIP embedding)
- OS: Linux, Windows, or macOS

### 2. Installation

#### Clone the repository
```bash
git clone https://github.com/your-username/Shorts-Multimodal-Analysis.git
cd Shorts-Multimodal-Analysis 
```
#### Create a Virtual Environment (Recommended)
```bash
conda create -n shorts_env python=3.9
conda activate shorts_env ```
```
#### Install Dependencies
```bash
pip install -r requirements.txt
```
---


## Dataset

The dataset was constructed using the YouTube Data API v3, applying a stratified sampling strategy
to ensure a balanced representation of view counts across log-scale bins.

Repository & File Structure
```bash
.
├── src/                    # Core Implementation
│   ├── data_collector.py   # Step 1: Metadata harvesting
│   ├── image_downloader.py # Step 2: Multithreaded downloader (Local only)
│   ├── final_cleaner.py    # Step 3: Data refinement & Outlier removal
│   ├── main_training.py    # Step 4: Training & Evaluation
│   └── visual_attention.py # Step 5: CLIP Attention Rollout
├── data/                   # Data Storage (Metadata only)
│   └── youtube_shorts_data_CLEAN.csv # Final refined metadata
├── notebooks/              # Analysis & Visualization
│   └── eda_distribution.ipynb
├── requirements.txt        # Dependency list (Python >=3.9)
└── README.md
```
## Important Notice

The thumbnail folder is excluded from this repository to respect content creators' copyrights
and manage storage efficiency.
To Replicate the study, please use the provided image_downloader.py script
to fetch images to your local machine.