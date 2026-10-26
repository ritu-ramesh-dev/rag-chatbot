"""Turns NASA's raw Moon maps into the small web textures the live background uses.

Inputs (download them yourself from https://svs.gsfc.nasa.gov/4720/ into tools/_nasa_raw/):
    lroc_color_poles_8k.tif   LROC colour mosaic (NASA/GSFC/Arizona State University)
    ldem_16.tif               LOLA elevation, 16 pixels per degree (NASA/GSFC/MIT/SVS)

Outputs in static/:
    moon_color_8k.jpg, moon_color_4k.jpg      the surface colour
    moon_normal_4k.jpg, moon_normal_2k.jpg    a tangent-space normal map computed from the elevation, so the real
                                              craters and mountains catch the light

Run from the project root, with isolated mode so nothing in the data folder is imported as code:
    .venv\\Scripts\\python.exe -I tools\\make_moon_textures.py
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # these maps are large on purpose

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "tools" / "_nasa_raw"
OUT = ROOT / "static"
OUT.mkdir(exist_ok=True)

MOON_RADIUS_M = 1_737_400.0
EXAGGERATION = 4.0  # real slopes are gentle; a little exaggeration makes the relief read at screen size


def open_rgb(path: Path) -> Image.Image:
    img = Image.open(path)
    print(f"  {path.name}: mode={img.mode} size={img.size}")
    if img.mode in ("I;16", "I;16B", "I;16L", "I"):  # 16-bit greyscale-style: scale down to 8-bit
        a = np.asarray(img, dtype=np.float32)
        img = Image.fromarray(np.clip(a / (a.max() or 1) * 255, 0, 255).astype(np.uint8))
    return img.convert("RGB")


def save_jpg(img: Image.Image, name: str, quality: int) -> None:
    p = OUT / name
    img.save(p, "JPEG", quality=quality, optimize=True, progressive=True)
    print(f"  wrote {name}: {img.size[0]}x{img.size[1]}, {p.stat().st_size / 1e6:.1f} MB")


def colour() -> None:
    print("colour map")
    img = open_rgb(RAW / "lroc_color_poles_8k.tif")
    if img.size != (8192, 4096):
        img = img.resize((8192, 4096), Image.LANCZOS)
    save_jpg(img, "moon_color_8k.jpg", 90)
    save_jpg(img.resize((4096, 2048), Image.LANCZOS), "moon_color_4k.jpg", 90)


def normals() -> None:
    print("normal map from LOLA elevation")
    h = Image.open(RAW / "ldem_16.tif")
    print(f"  ldem_16.tif: mode={h.mode} size={h.size}")
    a = np.asarray(h, dtype=np.float32)
    if a.ndim == 3:
        a = a[..., 0]
    lo, hi = float(np.nanmin(a)), float(np.nanmax(a))
    print(f"  raw range {lo:.3f} .. {hi:.3f}")
    metres = a * 1000.0 if max(abs(lo), abs(hi)) < 50 else a  # the file is in kilometres; be robust if it is metres
    metres = np.nan_to_num(metres, nan=0.0)

    for width, name in ((4096, "moon_normal_4k.jpg"), (2048, "moon_normal_2k.jpg")):
        height = width // 2
        hm = np.asarray(Image.fromarray(metres, mode="F").resize((width, height), Image.BOX), dtype=np.float32)
        lat = (0.5 - (np.arange(height) + 0.5) / height) * np.pi          # +90 at the top row
        dy = MOON_RADIUS_M * np.pi / height                               # metres per pixel, north-south
        dx = MOON_RADIUS_M * np.cos(lat)[:, None] * (2 * np.pi / width)   # metres per pixel, east-west (shrinks to the poles)
        dx = np.maximum(dx, dy * 0.08)
        gx = (np.roll(hm, -1, axis=1) - np.roll(hm, 1, axis=1)) / (2 * dx)  # slope toward east
        gy = (np.vstack([hm[:1], hm[:-1]]) - np.vstack([hm[1:], hm[-1:]])) / (2 * dy)  # slope toward north (rows go south)
        nx, ny, nz = -gx * EXAGGERATION, -gy * EXAGGERATION, np.ones_like(gx)
        inv = 1.0 / np.sqrt(nx * nx + ny * ny + nz * nz)
        rgb = np.stack([nx * inv, ny * inv, nz * inv], axis=-1) * 0.5 + 0.5
        save_jpg(Image.fromarray((rgb * 255).astype(np.uint8), "RGB"), name, 94)


if __name__ == "__main__":
    for needed in ("lroc_color_poles_8k.tif", "ldem_16.tif"):
        if not (RAW / needed).exists():
            sys.exit(f"Missing {RAW / needed}. Download it from https://svs.gsfc.nasa.gov/4720/ first.")
    colour()
    normals()
    print("done")
