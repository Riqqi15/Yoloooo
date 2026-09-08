from __future__ import annotations

import platform

import cv2
import torch
import ultralytics
from huggingface_hub import __version__ as huggingface_hub_version


def main() -> None:
    result = torch.tensor([1.0, 2.0]) * 2
    assert result.tolist() == [2.0, 4.0]

    print(f"Python: {platform.python_version()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Ultralytics: {ultralytics.__version__}")
    print(f"OpenCV: {cv2.__version__}")
    print(f"Hugging Face Hub: {huggingface_hub_version}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print("Environment check: OK")


if __name__ == "__main__":
    main()
