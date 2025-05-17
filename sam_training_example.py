import os
import subprocess
import urllib.request
from pathlib import Path
from typing import Callable

import torch

# Attempt to import segment_anything. If unavailable, install it on demand.
try:
    from segment_anything import sam_model_registry
except ImportError:  # pragma: no cover - network is required
    subprocess.check_call([
        "pip",
        "install",
        "git+https://github.com/facebookresearch/segment-anything.git",
    ])
    from segment_anything import sam_model_registry


def download_sam_weights(model_type: str = "vit_h", dest: str = "models") -> Path:
    """Download SAM pretrained weights if they are not present.

    Args:
        model_type: One of ``"vit_h"``, ``"vit_l"`` or ``"vit_b"``.
        dest: Directory where the checkpoint is stored.

    Returns:
        Path to the downloaded checkpoint file.
    """
    urls = {
        "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    }
    dest_path = Path(dest)
    dest_path.mkdir(parents=True, exist_ok=True)
    ckpt_path = dest_path / f"sam_{model_type}.pth"
    if not ckpt_path.exists():  # pragma: no cover - requires network
        print(f"Downloading SAM weights to {ckpt_path}...")
        urllib.request.urlretrieve(urls[model_type], ckpt_path)
    return ckpt_path


def build_sam(model_type: str = "vit_h", device: str | None = None) -> torch.nn.Module:
    """Load SAM model with pretrained weights."""
    ckpt = download_sam_weights(model_type)
    sam = sam_model_registry[model_type](checkpoint=str(ckpt))
    sam.to(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    return sam


def training_step(
    sam_model: torch.nn.Module,
    image: torch.Tensor,
    loss_fn: Callable[[torch.Tensor], torch.Tensor],
    optimizer: torch.optim.Optimizer,
) -> torch.Tensor:
    """Run a single training step on ``image``.

    This function computes gradients for both the model parameters and the input
    ``image`` itself.
    """
    sam_model.train()
    image.requires_grad_(True)

    optimizer.zero_grad()
    # SAM expects a dict with the image tensor under the ``image`` key.
    outputs = sam_model([{"image": image}])
    masks = outputs[0]["masks"]  # (1, 1, H, W)

    loss = loss_fn(masks)
    loss.backward()
    optimizer.step()
    return loss.detach()


def example_usage(image_path: str) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam_model = build_sam(device=device)
    optimizer = torch.optim.Adam(sam_model.parameters(), lr=1e-5)

    img = torch.as_tensor(torch.load(image_path, map_location=device))
    if img.ndim == 3:
        img = img.unsqueeze(0)

    def custom_loss(pred_masks: torch.Tensor) -> torch.Tensor:
        return pred_masks.mean()

    loss = training_step(sam_model, img, custom_loss, optimizer)
    print("Loss:", loss.item())
    print("Gradient norm of input image:", img.grad.norm().item())


if __name__ == "__main__":  # pragma: no cover - example entry point
    import argparse

    parser = argparse.ArgumentParser(description="SAM training example")
    parser.add_argument("image", help="Path to a tensor saved with torch.save")
    args = parser.parse_args()
    example_usage(args.image)
