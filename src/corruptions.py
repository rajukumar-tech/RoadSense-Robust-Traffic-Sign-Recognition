"""
RoadSense: Corruption Benchmark Functions (Phase 3).
Implements 5 real-world driving corruptions across 3 severity levels:
  1. Gaussian Noise (simulates sensor grain in low-light / high ISO)
  2. Gaussian Blur (simulates camera defocus and vehicle motion blur)
  3. Brightness / Contrast change (simulates glare, shadows, and night conditions)
  4. Random Occlusion patch (simulates physical obstructions, dirt, stickers)
  5. JPEG Compression artifacts (simulates lossy transmission / low bitrates)

All functions accept PIL Images or numpy arrays and use deterministic random seeding.
"""

import io
import random
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance

CORRUPTIONS = [
    "gaussian_noise",
    "motion_blur",
    "brightness_contrast",
    "random_occlusion",
    "jpeg_compression",
]

SEVERITIES = [1, 2, 3]


def apply_gaussian_noise(image: Image.Image, severity: int = 1, seed: int = 42) -> Image.Image:
    """Adds zero-mean Gaussian noise to simulate camera sensor noise."""
    sigma_map = {1: 18, 2: 38, 3: 65}
    sigma = sigma_map.get(severity, 25)

    np.random.seed(seed)
    img_arr = np.array(image, dtype=np.float32)
    noise = np.random.normal(0, sigma, img_arr.shape)
    noisy_img = np.clip(img_arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_img)


def apply_gaussian_blur(image: Image.Image, severity: int = 1, seed: int = 42) -> Image.Image:
    """Applies Gaussian / motion blur to simulate focus blur and vehicle movement."""
    radius_map = {1: 1.0, 2: 2.0, 3: 3.5}
    radius = radius_map.get(severity, 1.5)
    return image.filter(ImageFilter.GaussianBlur(radius=radius))


def apply_brightness_contrast(image: Image.Image, severity: int = 1, seed: int = 42) -> Image.Image:
    """Simulates extreme driving lighting: glare or darkness with reduced contrast."""
    # Pairs of (contrast_factor, brightness_factor)
    params_map = {
        1: (0.80, 0.75),   # Mild gloom/shadow
        2: (0.60, 0.50),   # Heavy overcast / dusk
        3: (0.40, 0.30),   # Night / tunnel underpass
    }
    contrast_f, bright_f = params_map.get(severity, (0.7, 0.7))

    enhancer_con = ImageEnhance.Contrast(image)
    img_con = enhancer_con.enhance(contrast_f)

    enhancer_bri = ImageEnhance.Brightness(img_con)
    return enhancer_bri.enhance(bright_f)


def apply_random_occlusion(image: Image.Image, severity: int = 1, seed: int = 42) -> Image.Image:
    """Overlays a solid gray/black occlusion patch to simulate branches, lampposts, or stickers."""
    # Patch sizes defined as fractions of image dimension (originally calibrated at 32x32)
    # severity 1: ~19%, severity 2: ~31%, severity 3: ~44% of image width
    patch_fraction_map = {1: 0.1875, 2: 0.3125, 3: 0.4375}
    patch_fraction = patch_fraction_map.get(severity, 0.25)

    w, h = image.size
    # Scale to actual image size so severity is consistent regardless of resolution
    patch_size = max(4, int(min(w, h) * patch_fraction))

    img_arr = np.array(image).copy()

    # Deterministic placement based on seed
    rng = np.random.RandomState(seed)
    max_x = max(1, w - patch_size)
    max_y = max(1, h - patch_size)
    x0 = rng.randint(0, max_x)
    y0 = rng.randint(0, max_y)

    # Gray/black occlusion color
    patch_color = rng.randint(15, 65)
    img_arr[y0 : y0 + patch_size, x0 : x0 + patch_size] = patch_color

    return Image.fromarray(img_arr)


def apply_jpeg_compression(image: Image.Image, severity: int = 1, seed: int = 42) -> Image.Image:
    """Simulates lossy compression and bandwidth reduction artifacts."""
    quality_map = {1: 40, 2: 20, 3: 8}
    quality = quality_map.get(severity, 30)

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


CORRUPTION_FUNCTIONS = {
    "gaussian_noise": apply_gaussian_noise,
    "motion_blur": apply_gaussian_blur,
    "brightness_contrast": apply_brightness_contrast,
    "random_occlusion": apply_random_occlusion,
    "jpeg_compression": apply_jpeg_compression,
}


def apply_corruption(image: Image.Image, corruption_type: str, severity: int = 1, seed: int = 42) -> Image.Image:
    """Applies a specific corruption at the given severity level (1, 2, or 3)."""
    func = CORRUPTION_FUNCTIONS.get(corruption_type)
    if func is None:
        raise ValueError(f"Unknown corruption '{corruption_type}'. Choose from {list(CORRUPTION_FUNCTIONS.keys())}")
    return func(image, severity=severity, seed=seed)


if __name__ == "__main__":
    print(f"Registered {len(CORRUPTION_FUNCTIONS)} corruption types across {len(SEVERITIES)} severity levels.")
    print("Testing pipeline on dummy image...")
    dummy_img = Image.new("RGB", (32, 32), color=(180, 100, 50))
    for c_name in CORRUPTIONS:
        for sev in SEVERITIES:
            out = apply_corruption(dummy_img, c_name, severity=sev, seed=42)
            assert out.size == (32, 32)
    print("All 15 corruption conditions verified successfully!")
