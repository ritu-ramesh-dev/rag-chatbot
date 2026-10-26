"""Renders assets/background.jpg: a still of the live moon, from the same NASA textures and the same lighting maths
as static/moon.js. It is shown for the instant before the live moon starts and in browsers without WebGL2.

Run from the project root:   .venv\\Scripts\\python.exe -I tools\\make_fallback.py
"""
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
W, H = 1920, 1080
SPIN, TILT = 0.3, 0.12                     # same starting pose as moon.js
L = np.array([-0.88, 0.20, 0.22]); L /= np.linalg.norm(L)
rad = min(H * 0.64, W * 0.42)
cx, cy = W * 0.80, H * 0.50

Image.MAX_IMAGE_PIXELS = None
colour = np.asarray(Image.open(ROOT / "static" / "moon_color_4k.jpg").convert("RGB"), dtype=np.float32) / 255
normal = np.asarray(Image.open(ROOT / "static" / "moon_normal_4k.jpg").convert("RGB"), dtype=np.float32) / 255
TH, TW = colour.shape[:2]


def sample(tex, u, v):
    """Bilinear lookup, longitude wraps, latitude clamps (same as the shader's REPEAT / CLAMP_TO_EDGE)."""
    x = (u % 1.0) * TW - 0.5
    y = np.clip(v, 0, 1) * TH - 0.5
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = (x - x0)[..., None], (y - y0)[..., None]
    x1, y1 = (x0 + 1) % TW, np.clip(y0 + 1, 0, TH - 1)
    x0 %= TW
    y0 = np.clip(y0, 0, TH - 1)
    return (tex[y0, x0] * (1 - fx) * (1 - fy) + tex[y0, x1] * fx * (1 - fy)
            + tex[y1, x0] * (1 - fx) * fy + tex[y1, x1] * fx * fy)


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


R = rot_x(TILT) @ rot_y(SPIN)               # body -> view

yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
px = (xx + 0.5 - cx) / rad
py = ((H - 1 - yy) + 0.5 - cy) / rad        # flip: the shader's origin is bottom-left
r = np.sqrt(px * px + py * py)

rng = np.random.default_rng(7)
img = np.zeros((H, W, 3), np.float32) + np.array([0.0016, 0.0018, 0.0028], np.float32)

# stars: pin-points with a soft core, a few brighter ones
for _ in range(900):
    sx, sy = rng.uniform(0, W), rng.uniform(0, H)
    amp = (rng.random() ** 3) * 1.6 + 0.12
    sig = 0.6 + rng.random() * 0.6
    x0, x1, y0, y1 = int(sx) - 4, int(sx) + 5, int(sy) - 4, int(sy) + 5
    if x0 < 0 or y0 < 0 or x1 > W or y1 > H:
        continue
    gx, gy = np.meshgrid(np.arange(x0, x1) + 0.5 - sx, np.arange(y0, y1) + 0.5 - sy)
    tint = np.array([0.75, 0.85, 1.0]) * (1 - rng.random()) + np.array([1.0, 0.9, 0.75]) * rng.random()
    img[y0:y1, x0:x1] += (amp * np.exp(-(gx * gx + gy * gy) / (2 * sig * sig)))[..., None] * tint

# thin cool limb glow, strongest on the sunlit side
limb = np.maximum(r - 1.0, 0)
sun_side = np.maximum((px * L[0] + py * L[1]) / (np.hypot(px, py) * np.hypot(L[0], L[1]) + 1e-6), 0)
img += np.array([0.22, 0.40, 0.85]) * (np.exp(-limb * 38) * (0.04 + 0.22 * sun_side) * (r >= 1.0))[..., None]

# the moon
inside = r < 1.0
z = np.sqrt(np.clip(1 - r * r, 0, 1))
N0 = np.stack([px, py, z], -1)
b = N0 @ R                                    # row vectors times R == R^T applied to the column vector
lat = np.arcsin(np.clip(b[..., 1], -1, 1))
lon = np.arctan2(b[..., 0], b[..., 2])
u, v = lon / (2 * np.pi) + 0.5, 0.5 - lat / np.pi
alb = sample(colour, u, v)
nm = sample(normal, u, v) * 2 - 1
east = np.stack([b[..., 2], np.zeros_like(z), -b[..., 0]], -1)
east /= np.linalg.norm(east, axis=-1, keepdims=True) + 1e-6
north = np.cross(b, east)
nb = east * nm[..., :1] + north * nm[..., 1:2] + b * nm[..., 2:3]
nb /= np.linalg.norm(nb, axis=-1, keepdims=True) + 1e-6
n = nb @ R.T                                  # R applied to the column vector
mu0 = np.maximum(n @ L, 0)
mu = np.maximum(n[..., 2], 0.05)
ls = mu0 / (mu0 + mu)                         # Lommel-Seeliger
t = np.clip(((N0 @ L) + 0.03) / 0.13, 0, 1)
geo = t * t * (3 - 2 * t)
lin = alb ** 2.2
moon = lin * (ls * geo * 8.2)[..., None] + lin * np.array([0.45, 0.62, 1.0]) * 0.035
edge = (1 - np.clip((r - (1 - 1.5 / rad)) / (1.5 / rad), 0, 1))[..., None]
img = np.where(inside[..., None], img * (1 - edge) + moon * edge, img)

img = 1 - np.exp(-img * 1.15)                 # same filmic roll-off as the shader
img = np.clip(img, 0, 1) ** (1 / 2.2)
out = ROOT / "assets" / "background.jpg"
Image.fromarray((img * 255 + 0.5).astype(np.uint8)).save(out, quality=92, optimize=True, progressive=True)
print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB)")
