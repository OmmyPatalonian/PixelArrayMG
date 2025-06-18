# MedGemma CT Scan Analysis

A Python script that uses Google's MedGemma-4b-it model to analyze CT scan images and generate medical reports in proper radiology format.

## Features

- Loads CT scan images from JPEG files
- Converts between PIL Images and numpy pixel arrays
- Uses MedGemma-4b-it for medical image analysis
- Generates structured medical reports with "Findings" and "Impression" sections
- Handles both CUDA and CPU inference
- Includes proper error handling and interruption support

## Requirements

```bash
pip install torch transformers pillow numpy
```

## Setup

1. Install Hugging Face CLI and login:
```bash
pip install huggingface_hub
huggingface-cli login
```

2. Request access to the MedGemma model:
   - Visit: https://huggingface.co/google/medgemma-4b-it
   - Accept the terms and conditions

3. Place your CT scan images in the `CTscan/` folder

## Usage

```bash
python MedGemma.py
```

## Output Format

The script generates medical reports in standard radiology format:

```
Findings:
[Detailed observations of what's visible in the CT scan]

Impression:
[Clinical interpretation and diagnostic conclusions]
```

## Note

- CPU inference takes 5-10 minutes
- GPU inference is significantly faster
- Press Ctrl+C to interrupt if needed
