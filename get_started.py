# huggingface - SmolVLM2-2B-Instruct // How to get started
# conda env --- add by mxnseo
"""
    -- RTX 4070 PC --

    conda activate -n smolvlm python=3.10 -y
    conda activate smolvlm

    pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
    pip install transformers==4.51.3
    pip install num2words
    pip install accelerate
    pip install av

    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i cuda-keyring_1.1-1_all.deb
    sudo apt-get update
    
    sudo apt-get install -y cuda-toolkit-12-1

    echo 'export CUDA_HOME=/usr/local/cuda-12.1' >> ~/.bashrc
    echo 'export PATH=/usr/local/cuda-12.1/bin:$PATH' >> ~/.bashrc
    source ~/.bashrc

    conda activate smolvlm
    pip install flash-attn --no-build-isolation



    -- Jetson AGX Orin 64GB --
    (Jetpack 6.1, CUDA 12.6)

    conda activate -n smolvlm python=3.10 -y
    conda activate smolvlm

    pip install torch-2.5.0-cp310-cp310-linux_aarch64.whl
    pip install torchvision-0.20.0-cp310-cp310-linux_aarch64.whl

    pip install transformers==4.51.3
    pip install num2words
    pip install accelerate
    pip install av
    pip install pillow
    pip install "numpy<2"


"""

from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
import time

# Inference Time Check --- add by mxnseo
def measure_inference(func):
    def wrapper(*args, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_start = time.perf_counter()

        result = func(*args, **kwargs)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_end = time.perf_counter()

        elapsed = t_end - t_start
        print(f"\n[{func.__name__}] Inference Time: {elapsed:.3f} s")
        if torch.cuda.is_available():
            print(f"VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

        # if result == token -> token/s time
        if isinstance(result, tuple) and len(result) == 2:
            actual_result, n_tokens = result
            if isinstance(n_tokens, int):
                print(f"Tokens: {n_tokens} | {n_tokens / elapsed:.1f} tok/s")
            return actual_result

        return result
    return wrapper


model_path = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
processor = AutoProcessor.from_pretrained(model_path)
model = AutoModelForImageTextToText.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    # RTX 4070 version
    _attn_implementation="flash_attention_2"

    # jetson version
    # _attn_implementation="eager"
).to("cuda")

# ---


# Simple Inference

@measure_inference
def simple_inference():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"},
                {"type": "text", "text": "Can you describe this image?"},
            ]
        },
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device, dtype=torch.bfloat16)
    
    generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=64)

    input_len = inputs.input_ids.shape[-1]
    n_tokens = generated_ids.shape[-1] - input_len
    
    generated_texts = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )
    print(generated_texts[0])

    return generated_texts[0], n_tokens






# Video Inference

@measure_inference
def video_inference():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "path": "lucy.mp4"},
                {"type": "text", "text": "Describe this video in detail"}
            ]
        },
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device, dtype=torch.bfloat16)

    generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=64)
    
    input_len = inputs.input_ids.shape[-1]
    n_tokens = generated_ids.shape[-1] - input_len
    
    generated_texts = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )

    print(generated_texts[0])

    return generated_texts[0], n_tokens




# Multi-image Interleaved Inference


import torch

@measure_inference
def multi_image_inference():
    messages = [
        {
            "role": "user",
            "content": [
              {"type": "text", "text": "What is the similarity between these two images?"},
              {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"},
              {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg"},            
            ]
        },
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device, dtype=torch.bfloat16)
    
    generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=64)
    
    input_len = inputs.input_ids.shape[-1]
    n_tokens = generated_ids.shape[-1] - input_len
    
    generated_texts = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )
    print(generated_texts[0])
    
    return generated_texts[0], n_tokens



if __name__ == "__main__":
    simple_inference()
    video_inference()
    multi_image_inference()