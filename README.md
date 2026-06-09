# SmolVLM2-2.2B-Instruct 추론 예제

HuggingFace의 SmolVLM2-2.2B-Instruct 모델을 이용한 멀티모달 추론 예제 코드.  
가상환경 셋업부터 이미지/비디오/다중이미지 추론을 한 코드에 정리함.

---

## 스펙

| 항목 | 내용 |
|---|---|
| OS | Ubuntu 22.04 LTS |
| Python | 3.10 |
| GPU | RTX 4070 |
| CUDA | 12.1 |
| 모델 | HuggingFaceTB/SmolVLM2-2.2B-Instruct |

---

## 환경 셋업

### 1. conda 가상환경 생성

```bash
conda create -n smolvlm python=3.10 -y
conda activate smolvlm
```

### 2. PyTorch 설치 (CUDA 12.1)

```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
```

### 3. 의존성 설치

```bash
pip install transformers==4.51.3
pip install num2words
pip install accelerate
pip install av
```

### 4. CUDA Toolkit 12.1 설치

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-1
```

### 5. 환경변수 등록

```bash
echo 'export CUDA_HOME=/usr/local/cuda-12.1' >> ~/.bashrc
echo 'export PATH=/usr/local/cuda-12.1/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
```

### 6. Flash Attention 설치

```bash
conda activate smolvlm
pip install flash-attn --no-build-isolation
```

> `flash-attn`은 빌드 시간이 길 수 있음. 느려도 정상임.

---

## 모델 로드

```python
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch

model_path = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"

processor = AutoProcessor.from_pretrained(model_path)
model = AutoModelForImageTextToText.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    _attn_implementation="flash_attention_2"
).to("cuda")
```

---

## 추론 시간 측정 데코레이터

CUDA 동기화 후 `perf_counter`로 실측 시간을 재고, VRAM 사용량도 같이 출력함.

```python
import time

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
        return result
    return wrapper
```

함수에 `@measure_inference`만 달면 자동 측정됨.

---

## 추론 예제

### 단일 이미지 추론

```python
@measure_inference
def simple_inference():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": "https://...bee.jpg"},
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
    generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
    print(generated_texts[0])

simple_inference()
```

### 비디오 추론

로컬 mp4 파일을 직접 넘길 수 있음.

```python
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
    # ... (단일 이미지와 동일한 generate 흐름)

video_inference()
```

### 다중 이미지 추론

두 이미지를 동시에 넘겨 비교/분석 가능.

```python
@measure_inference
def multi_image_inference():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is the similarity between these two images?"},
                {"type": "image", "url": "https://...bee.jpg"},
                {"type": "image", "url": "https://...rabbit.jpg"},
            ]
        },
    ]
    # ... (단일 이미지와 동일한 generate 흐름)

multi_image_inference()
```

---

## 출력 예시

```
[simple_inference] Inference Time: 1.243 s
VRAM: 4.81 GB

[video_inference] Inference Time: 3.872 s
VRAM: 5.03 GB

[multi_image_inference] Inference Time: 2.015 s
VRAM: 4.94 GB
```

> 수치는 입력 길이 및 시스템 상태에 따라 달라질 수 있음.

---

## 참고

- 모델 원본: [HuggingFaceTB/SmolVLM2-2.2B-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct)
- `av` 라이브러리는 비디오 추론 시 필수
- `flash_attention_2`는 Ampere 이상 GPU에서만 동작 (RTX 30xx / 40xx)
- CPU 추론도 가능하나 `_attn_implementation` 옵션 제거 필요
