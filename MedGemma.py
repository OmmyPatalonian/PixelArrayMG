import os
import sys
import time
import signal
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

MODEL_ID = "google/medgemma-4b-it"
CT_PATH = "CTscan/51F55CCA-B915-4AF4-BBFD-13B05D6C3B7D.jpeg"
MAX_TOKENS = 150

def handle_interrupt(sig, frame):
    print('\nInterrupted by user')
    sys.exit(0)

def load_image(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    
    img = Image.open(path)
    pixels = np.array(img, dtype=np.uint16)
    return pixels, img

def convert_array_to_image(arr):
    if arr.dtype != np.uint8:
        normalized = (255 * (arr - np.min(arr)) / (np.ptp(arr) + 1e-8)).astype(np.uint8)
    else:
        normalized = arr
    
    if normalized.ndim == 2:
        return Image.fromarray(normalized, mode="L")
    elif normalized.ndim == 3 and normalized.shape[2] == 3:
        return Image.fromarray(normalized, mode="RGB")
    else:
        raise ValueError("Invalid array shape")

def setup_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cpu":
        print("Warning: CPU inference will be slow")
    return device

def load_model_components():
    try:
        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float32,
            device_map=None,
            token=None,
            low_cpu_mem_usage=True,
        )
        processor = AutoProcessor.from_pretrained(MODEL_ID, token=None)
        return model, processor
    except Exception as e:
        if "401" in str(e) or "Unauthorized" in str(e):
            print("Auth error: run 'huggingface-cli login'")
            print("Visit https://huggingface.co/google/medgemma-4b-it")
        raise e

def create_prompt(img):
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a radiologist writing a medical report."}]
        },
        {
            "role": "user", 
            "content": [
                {"type": "text", "text": "Analyze this CT scan and provide a medical report in the following format:\n\nFindings:\n[Describe what you observe in the scan]\n\nImpression:\n[Provide your clinical interpretation and diagnosis]"},
                {"type": "image", "image": img},
            ]
        }
    ]

def generate_analysis(model, processor, messages, device):
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(device)
    
    input_len = inputs["input_ids"].shape[-1]
    
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_TOKENS,
            do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id
        )
        result = output[0][input_len:]
    
    return processor.decode(result, skip_special_tokens=True)

def main():
    signal.signal(signal.SIGINT, handle_interrupt)
    
    start_time = time.time()
    device = setup_device()
    
    try:
        pixels, img = load_image(CT_PATH)
        print(f"Loaded CT scan: {pixels.shape}")
    except FileNotFoundError:
        print("CT scan not found, using dummy data")
        pixels = np.zeros((512, 512), dtype=np.uint16)
        img = convert_array_to_image(pixels)
    
    print("Loading model...")
    model, processor = load_model_components()
    model = model.to(device)
    
    messages = create_prompt(img)
    print("Analyzing CT scan...")
    
    analysis = generate_analysis(model, processor, messages, device)
    
    elapsed = time.time() - start_time
    print(f"\nAnalysis ({elapsed:.1f}s):")
    print("-" * 40)
    print(analysis.strip())
    print("-" * 40)

if __name__ == "__main__":
    assert os.path.exists("CTscan"), "CTscan folder missing"
    
    test_arr = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    test_img = convert_array_to_image(test_arr)
    assert test_img.size == (100, 100), "Array conversion failed"
    
    main()
