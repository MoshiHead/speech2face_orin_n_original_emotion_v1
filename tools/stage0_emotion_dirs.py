"""tools/stage0_emotion_dirs.py -- Stage 0 feasibility test for avatar emotion.

READ-ONLY with respect to the streaming pipeline: this script imports the
renderer classes and loads the shipped checkpoints, and writes only into
--out_dir. It starts no server and modifies no production file.

    $(cat .python_path) tools/stage0_emotion_dirs.py \
        --videos_dir emotion_demo_videos --ref_path weights/ref.jpeg \
        --renderer_path weights/renderer.ckpt \
        --distilled_renderer weights/renderer_narrow.pt \
        --out_dir stage0_out

WHAT IT ANSWERS
  1. Do the three expression videos yield emotion DIRECTIONS in the 32-D motion
     latent that are expression and not framing?  (r_null, cosine structure)
  2. How far can each direction be pushed before the NARROW (distilled,
     single-identity) renderer stops tracking the FULL IMTalker renderer?
     (alpha_max per emotion)
  3. Does adding a direction to a talking trajectory destroy lip-sync?
     (mouth-motion energy retention)

THE PROBLEM IT SOLVES
  The 32-D motion latent is a learned LIA-style code with no semantics, and
  load_ref_image() applies NO face alignment -- it just resizes to 512x512.  So
  a latent encodes expression AND head pose AND framing AND scale AND lighting,
  all entangled.  The four videos are framed differently from each other and
  from ref.jpeg, so a naive

      d_happy = mean(happy frames) - mean(neutral frames)

  is dominated by the framing difference, and would make the avatar shift and
  scale its head instead of smiling.  Three layers remove that:

    L1  geometric canonicalisation: warp each video into ref.jpeg's framing
        (per-video similarity transform, estimated from the video's MEDIAN
        frame so expression/mouth motion is marginalised out).
    L2  nuisance-subspace projection: perturb ref.jpeg by KNOWN similarity and
        lighting transforms, encode, PCA the latent deltas.  Those deltas carry
        zero expression by construction, so their span is the framing/lighting
        subspace -- project it out of every direction.
    L3  speech-subspace projection: the neutral video has constant expression,
        so its within-video temporal variance IS mouth + pose dynamics.  PCA it
        and project it out, so emotion does not fight lip-sync.

  Validation that L1+L2 worked: mean(neutral latents) and encode(ref.jpeg) are
  BOTH neutral expression, so their residual must collapse.  That ratio is
  r_null, and it is the number to read first.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
# Same convention as run.sh / run_x86.sh:  DESIGN_A_DIR=src/niloys  IMTALKER_DIR=src/imtalker
IMTALKER = Path(os.environ.get("IMTALKER_DIR", ROOT / "src" / "imtalker"))
DESIGN_A = Path(os.environ.get("DESIGN_A_DIR", ROOT / "src" / "niloys"))
for _p in (str(IMTALKER), str(DESIGN_A)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from student.renderer import build as build_narrow   # noqa: E402  src/niloys
# renderer.models (the full IMTalker renderer) is imported LAZILY inside
# load_full(), because it pulls in timm / torchvision / matplotlib and is not
# needed at all in --encoder invert mode.

EMOTIONS = ("happy", "sad", "angry")
NEUTRAL = "neutral"
RES = 512


# --------------------------------------------------------------------------- io

def load_ref_image(path):
    """Byte-for-byte the pipeline's own preprocessing (imtalker_native_passthrough.py)."""
    img = Image.open(path).convert("RGB").resize((RES, RES), Image.LANCZOS)
    arr = np.asarray(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def read_video_rgb(path):
    """-> [T, 512, 512, 3] uint8 RGB."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError("cannot open video: %s" % path)
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if bgr.shape[0] != RES or bgr.shape[1] != RES:
            bgr = cv2.resize(bgr, (RES, RES), interpolation=cv2.INTER_LANCZOS4)
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError("no frames decoded from %s" % path)
    return np.stack(frames, 0)


# ------------------------------------------------------- L1: canonicalisation

_CASCADE = None


def _face_box(gray):
    """Coarse face box via the Haar cascade bundled with opencv-python.

    None if unavailable: OpenCV 5 dropped cv2.CascadeClassifier, and the pod
    pins 4.9.0.80 where it exists.  Callers must treat None as "no coarse
    initial guess" -- estimate_warp() then relies on its multi-start ladder.
    """
    global _CASCADE
    if _CASCADE is False:
        return None
    if _CASCADE is None:
        try:
            xml = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
            _CASCADE = cv2.CascadeClassifier(str(xml))
            if _CASCADE.empty():
                _CASCADE = False
                return None
        except Exception:
            _CASCADE = False
            return None
    try:
        faces = _CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                          minSize=(80, 80))
    except Exception:
        return None
    if len(faces) == 0:
        return None
    return max(faces, key=lambda f: f[2] * f[3])        # largest


def _coarse_similarity(src_gray, dst_gray):
    """2x3 warp mapping src -> dst from face boxes; identity if either is missing."""
    W = np.eye(2, 3, dtype=np.float32)
    bs, bd = _face_box(src_gray), _face_box(dst_gray)
    if bs is None or bd is None:
        return W
    sx, sy, sw, sh = [float(v) for v in bs]
    dx, dy, dw, dh = [float(v) for v in bd]
    s = ((dw / sw) + (dh / sh)) / 2.0
    scx, scy = sx + sw / 2.0, sy + sh / 2.0
    dcx, dcy = dx + dw / 2.0, dy + dh / 2.0
    W[0, 0] = W[1, 1] = s
    W[0, 2] = dcx - s * scx
    W[1, 2] = dcy - s * scy
    return W


def _nearest_similarity(W):
    """Project a 2x3 affine onto the nearest similarity.

    Permits translation, rotation and UNIFORM scale; forbids shear and aspect
    change, which would deform the face and be read downstream as expression.
    """
    A = W[:2, :2].astype(np.float64)
    U, S, Vt = np.linalg.svd(A)
    s = float(S.mean())
    out = W.copy().astype(np.float32)
    out[:2, :2] = (U @ np.diag([s, s]) @ Vt).astype(np.float32)
    return out


def _scale_init(s):
    """2x3 similarity: uniform scale s about the image centre (FORWARD convention)."""
    return cv2.getRotationMatrix2D((RES / 2.0, RES / 2.0), 0.0, float(s)).astype(np.float32)


def estimate_warp(video_median, ref_rgb):
    """FORWARD similarity warp W with warpAffine(video_frame, W) ~ ref framing.

    Convention, because it is easy to get backwards: cv2.findTransformECC
    returns a matrix mapping TEMPLATE coordinates to INPUT coordinates, so
    resampling the input into template geometry needs WARP_INVERSE_MAP.  We
    keep everything in the forward (video -> ref) convention instead and invert
    at the ECC boundary, so every caller can use a plain warpAffine and the
    reported scale/tx/ty describe the transform actually applied.

    ECC is a local optimiser and needs a decent start when the framings differ
    a lot, so we run it from several -- the Haar face-box guess when available,
    plus a ladder of uniform scales -- and keep the best correlation.
    """
    src = cv2.cvtColor(video_median, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    dst = cv2.cvtColor(ref_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    info = {}

    starts = []                                          # all FORWARD (video -> ref)
    W0 = _coarse_similarity((src * 255).astype(np.uint8), (dst * 255).astype(np.uint8))
    if not np.allclose(W0, np.eye(2, 3)):
        starts.append(("haar", W0))
        info["coarse_scale"] = float(W0[0, 0])
        info["coarse_tx"] = float(W0[0, 2])
        info["coarse_ty"] = float(W0[1, 2])
    else:
        info["coarse_scale"] = None
    for s in (0.70, 0.80, 0.90, 1.00, 1.10, 1.25, 1.45):
        starts.append(("scale%.2f" % s, _scale_init(s)))

    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 400, 1e-6)
    best = (-2.0, None, "none")
    errs = []
    for tag, Wf in starts:
        # ECC wants template->input, i.e. the inverse of our forward warp.
        W_init = cv2.invertAffineTransform(Wf).astype(np.float32)
        try:
            cc, W_ecc = cv2.findTransformECC(dst, src, W_init, cv2.MOTION_AFFINE,
                                             crit, None, 5)
        except cv2.error as e:                           # ECC can fail to converge
            errs.append("%s:%s" % (tag, str(e).splitlines()[0][:40]))
            continue
        if np.isfinite(cc) and cc > best[0]:
            best = (float(cc), W_ecc, tag)

    if best[1] is None:
        info["ecc"] = "all starts failed -- coarse only"
        info["ecc_cc"] = float("nan")
        info["ecc_errors"] = errs[:3]
        W = _nearest_similarity(W0)
    else:
        info["ecc"] = "ok (%s)" % best[2]
        info["ecc_cc"] = best[0]
        info["ecc_starts_tried"] = len(starts)
        # back to forward (video -> ref), then constrain to a similarity
        W = _nearest_similarity(cv2.invertAffineTransform(best[1]).astype(np.float32))

    info["scale"] = float(np.sqrt(abs(np.linalg.det(W[:2, :2]))))
    info["tx"] = float(W[0, 2])
    info["ty"] = float(W[1, 2])
    return W, info


def apply_warp_and_match(frames, W, ref_rgb):
    """Warp every frame by W, then match luminance to ref over the face region."""
    out = np.empty_like(frames)
    for i in range(frames.shape[0]):
        out[i] = cv2.warpAffine(frames[i], W, (RES, RES), flags=cv2.INTER_LANCZOS4,
                                borderMode=cv2.BORDER_REPLICATE)
    # robust gain/offset from the central region (the face), on the median frame
    c = slice(RES // 6, RES - RES // 6)
    med = np.median(out, axis=0).astype(np.float32)
    a_m = med[c, c].mean()
    a_s = med[c, c].std() + 1e-6
    r = ref_rgb.astype(np.float32)
    b_m = r[c, c].mean()
    b_s = r[c, c].std() + 1e-6
    g = float(b_s / a_s)
    return np.clip((out.astype(np.float32) - a_m) * g + b_m, 0, 255).astype(np.uint8)


# ------------------------------------------------------------------- encoding

@torch.no_grad()
def encode_frames(renderer, frames, device, batch=16):
    """[T,512,512,3] uint8 -> [T,32] motion latents, via the pipeline's own encoder."""
    outs = []
    for i in range(0, frames.shape[0], batch):
        chunk = frames[i:i + batch].astype(np.float32) / 255.0
        t = torch.from_numpy(chunk).permute(0, 3, 1, 2).to(device)
        outs.append(renderer.latent_token_encoder(t).float().cpu())
    return torch.cat(outs, 0)


# -------------------------------------------------------------- L2: nuisance

def perturb(ref_rgb):
    """Known similarity + lighting perturbations of ref.jpeg.

    Zero expression change by construction, so the latent deltas these induce
    span the framing/lighting nuisance subspace.
    """
    out = []
    for dx, dy in ((-6, 0), (6, 0), (0, -6), (0, 6), (-3, -3), (3, 3), (-2, 2), (2, -2)):
        W = np.float32([[1, 0, dx], [0, 1, dy]])
        out.append(cv2.warpAffine(ref_rgb, W, (RES, RES), flags=cv2.INTER_LANCZOS4,
                                  borderMode=cv2.BORDER_REPLICATE))
    c = (RES / 2.0, RES / 2.0)
    for sc in (0.94, 0.97, 1.03, 1.06):
        W = cv2.getRotationMatrix2D(c, 0.0, sc)
        out.append(cv2.warpAffine(ref_rgb, W, (RES, RES), flags=cv2.INTER_LANCZOS4,
                                  borderMode=cv2.BORDER_REPLICATE))
    for ang in (-3.0, -1.5, 1.5, 3.0):
        W = cv2.getRotationMatrix2D(c, ang, 1.0)
        out.append(cv2.warpAffine(ref_rgb, W, (RES, RES), flags=cv2.INTER_LANCZOS4,
                                  borderMode=cv2.BORDER_REPLICATE))
    for g, b in ((0.95, 0), (1.05, 0), (1.0, -12), (1.0, 12)):
        out.append(np.clip(ref_rgb.astype(np.float32) * g + b, 0, 255).astype(np.uint8))
    return out


def pca_basis(deltas, k):
    """Orthonormal row basis [k,32] of the leading directions of `deltas`, + explained ratio."""
    if k <= 0 or deltas.shape[0] == 0:
        return torch.zeros(0, deltas.shape[-1]), np.zeros(0)
    X = deltas.double().numpy()
    U, S, Vt = np.linalg.svd(X, full_matrices=False)     # deltas are already differences
    var = S ** 2
    ratio = var / max(var.sum(), 1e-12)
    k = int(min(k, Vt.shape[0]))
    return torch.from_numpy(Vt[:k]).float(), ratio[:k]


def project_out(v, basis):
    if basis is None or basis.numel() == 0:
        return v
    return v - basis.t() @ (basis @ v)


# --------------------------------------------------------------- renderers

def load_full(renderer_path, device):
    """IMTRenderer needs only three fields off args (base_options defaults)."""
    from renderer.models import IMTRenderer          # lazy: needs timm/torchvision
    args = argparse.Namespace(num_heads=8, swin_res_threshold=128, window_size=8)
    r = IMTRenderer(args).to(device).eval()
    ckpt = torch.load(renderer_path, map_location="cpu")
    raw = ckpt.get("state_dict", ckpt.get("model", ckpt))
    cleaned = {k.replace("gen.", "", 1).replace("model.", "", 1): v for k, v in raw.items()}
    missing, unexpected = r.load_state_dict(cleaned, strict=False)
    print("[stage0] full IMTRenderer: %s" % renderer_path)
    print("[stage0]   missing=%d unexpected=%d" % (len(missing), len(unexpected)))
    for p in r.parameters():
        p.requires_grad_(False)
    return r


def load_narrow(path, device, dtype):
    pack = torch.load(path, map_location="cpu")
    D = build_narrow(**pack["cfg"]).to(device).eval()
    D.load_state_dict(pack["model"])
    D = D.to(dtype=dtype)
    for p in D.parameters():
        p.requires_grad_(False)
    print("[stage0] narrow renderer: %s" % path)
    print("[stage0]   params=%s val_PSNR=%.2f dB blink_len=%s dtype=%s" % (
        "{:,}".format(D.num_params()), pack.get("val_psnr", 0.0),
        getattr(D.cfg, "blink_len", 0), dtype))
    return D


class FullCtx:
    """Reference features for the full renderer, exactly as the live engine builds them."""

    @torch.no_grad()
    def __init__(self, r, ref, device):
        self.r = r
        self.f_r, self.g_r = r.dense_feature_encoder(ref.to(device))
        self.ref_x = r.latent_token_encoder(ref.to(device)).float()
        self.m_r = r.latent_token_decoder(r.adapt(self.ref_x, self.g_r))

    @torch.no_grad()
    def render(self, motion):
        n = motion.shape[0]
        g = self.g_r.expand(n, -1)
        m_r = tuple(m.expand(n, -1, -1, -1) for m in self.m_r)
        f_r = [f.expand(n, -1, -1, -1) for f in self.f_r]
        m_c = self.r.latent_token_decoder(self.r.adapt(motion, g))
        return self.r.decode(m_c, m_r, f_r).float().clamp(0, 1)


def invert_latents(D, targets, init, iters=240, lr=0.05, chunk=8, reg=1e-3, tag=""):
    """Find the 32-D codes whose NARROW render best matches `targets`.

    This is the fallback when weights/renderer.ckpt (the full IMTalker
    renderer, and so the only image->latent encoder) is unavailable -- it is
    not published in avatar_assets because production never loads it.

    It is not merely a substitute.  For the question "can the narrow renderer
    display this expression at all?" inversion is the more direct measurement:
    it searches the renderer's ENTIRE 32-D input space for the best match, so
    the residual it converges to is an upper bound on how well this renderer
    can represent that expression.  A high floor means no steering vector can
    ever work and Stage 3 is mandatory.

    What it cannot give is an INDEPENDENT reference for direction purity -- the
    full renderer plays that role in --encoder full mode.

    targets: [N,3,512,512] in [0,1] on `device`.  init: [32] starting code.
    Returns (codes [N,32], per-target final L1 residual [N]).
    """
    dt = next(D.parameters()).dtype
    bl = int(getattr(D.cfg, "blink_len", 0) or 0)
    out_codes, out_res = [], []
    n_total = targets.shape[0]
    for s in range(0, n_total, chunk):
        tgt = targets[s:s + chunk].to(dtype=dt)
        n = tgt.shape[0]
        m = init.view(1, -1).repeat(n, 1).clone().to(dtype=torch.float32)
        m.requires_grad_(True)
        opt = torch.optim.Adam([m], lr=lr)
        ph = None
        if bl > 0:
            # phase 0 for every target: we are matching a mean/median face, and
            # the blink table is a separate additive term we do not want to fit.
            ph = torch.zeros(n, dtype=torch.long, device=tgt.device)
        for it in range(iters):
            opt.zero_grad(set_to_none=True)
            img = D(m.to(dtype=dt), ph) if bl > 0 else D(m.to(dtype=dt))
            loss = (img.float() - tgt.float()).abs().mean()
            loss = loss + reg * (m - init.view(1, -1)).pow(2).mean()
            loss.backward()
            opt.step()
        with torch.no_grad():
            img = D(m.to(dtype=dt), ph) if bl > 0 else D(m.to(dtype=dt))
            res = (img.float() - tgt.float()).abs().mean(dim=(1, 2, 3))
        out_codes.append(m.detach().float().cpu())
        out_res.append(res.float().cpu())
        print("[stage0]   invert%s %d/%d  mean L1 residual=%.5f" % (
            (" " + tag) if tag else "", min(s + chunk, n_total), n_total,
            float(res.mean())))
    return torch.cat(out_codes, 0), torch.cat(out_res, 0)


@torch.no_grad()
def render_narrow(D, motion, phase0=0, chunk=8):
    """Chunked: the decoder's 256-channel stack at 256x256 is ~8 MB/frame of
    activations, so a 193-frame trajectory rendered in one batch would need
    well over 10 GB. Chunking keeps peak memory flat."""
    dt = next(D.parameters()).dtype
    bl = int(getattr(D.cfg, "blink_len", 0) or 0)
    outs = []
    for i in range(0, motion.shape[0], chunk):
        mo = motion[i:i + chunk].to(dtype=dt)
        if bl > 0:
            ph = (torch.arange(mo.shape[0], device=mo.device) + phase0 + i) % bl
            img = D(mo, ph)
        else:
            img = D(mo)
        outs.append(img.float().clamp(0, 1))
    return torch.cat(outs, 0)


# ----------------------------------------------------------------- reporting

def save_grid(path, rows, labels, scale=0.5):
    """Contact sheet: one row per renderer, one column per alpha."""
    h = w = int(RES * scale)
    tiles = [[cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA) for im in row]
             for row in rows]
    pad = 28
    H = len(tiles) * (h + pad)
    W = max(len(r) for r in tiles) * w
    sheet = np.full((H, W, 3), 255, np.uint8)
    for i, row in enumerate(tiles):
        y = i * (h + pad) + pad
        for j, im in enumerate(row):
            sheet[y:y + h, j * w:(j + 1) * w] = im
        cv2.putText(sheet, labels[i], (4, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))


def to_np(img):
    return (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)


def cos(a, b):
    return float(torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0))


# ---------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos_dir", default="emotion_demo_videos")
    ap.add_argument("--ref_path", default="weights/ref.jpeg")
    ap.add_argument("--renderer_path", default="weights/renderer.ckpt")
    ap.add_argument("--distilled_renderer", default="weights/renderer_narrow.pt")
    ap.add_argument("--out_dir", default="stage0_out")
    ap.add_argument("--alphas", default="0,0.25,0.5,0.75,1.0,1.25,1.5,2.0")
    ap.add_argument("--nuisance_k", type=int, default=6)
    ap.add_argument("--speech_k", type=int, default=6)
    ap.add_argument("--speech_motion", default="",
                    help="optional .pt from --dump_motion (Stage 0b): real speech motion [T,32]")
    ap.add_argument("--corr_thresh", type=float, default=0.80,
                    help="narrow-vs-full residual correlation defining alpha_max")
    ap.add_argument("--dyn_alpha", type=float, default=1.0,
                    help="alpha used for the lip-sync retention test")
    ap.add_argument("--narrow_dtype", choices=("fp32", "fp16"), default="fp32")
    ap.add_argument("--encoder", choices=("auto", "full", "invert"), default="auto",
                    help="how to map frames -> 32-D latents. full = the IMTalker "
                         "renderer's latent_token_encoder (needs --renderer_path, "
                         "which prepare_x86.sh does NOT download). invert = optimise "
                         "codes through the narrow renderer (public weights only). "
                         "auto = full if the checkpoint loads, else invert.")
    ap.add_argument("--invert_iters", type=int, default=240)
    ap.add_argument("--invert_frames", type=int, default=32,
                    help="frames subsampled per video in invert mode (inversion is "
                         "iterative, so we do not invert all 193)")
    ap.add_argument("--no_align", action="store_true", help="ablation: skip L1")
    ap.add_argument("--no_nuisance", action="store_true", help="ablation: skip L2")
    ap.add_argument("--no_speech", action="store_true", help="ablation: skip L3")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    dev = torch.device(a.device)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    rep = {"config": vars(a)}

    # ---- reference -------------------------------------------------------
    ref_t = load_ref_image(a.ref_path)
    ref_rgb = (ref_t[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    narrow = load_narrow(a.distilled_renderer, dev,
                         torch.float16 if a.narrow_dtype == "fp16" else torch.float32)

    # ---- pick the frames -> latent encoder -------------------------------
    full, ctx = None, None
    mode = a.encoder
    if mode in ("auto", "full"):
        try:
            full = load_full(a.renderer_path, dev)
            ctx = FullCtx(full, ref_t, dev)
            mode = "full"
        except Exception as e:
            if a.encoder == "full":
                raise
            print("[stage0] full renderer unavailable (%s: %s)" % (type(e).__name__, e))
            print("[stage0] -> falling back to --encoder invert "
                  "(weights/renderer.ckpt is NOT published in avatar_assets)")
            full, ctx, mode = None, None, "invert"
    rep["encoder_mode"] = mode
    print("[stage0] encoder mode: %s" % mode)

    def encode(frames_np, tag=""):
        """[T,512,512,3] uint8 -> [T,32]."""
        if mode == "full":
            return encode_frames(full, frames_np, dev)
        t = torch.from_numpy(frames_np.astype(np.float32) / 255.0)
        t = t.permute(0, 3, 1, 2).to(dev)
        codes, res = invert_latents(narrow, t, ref_x0, iters=a.invert_iters, tag=tag)
        _INV_RES[tag] = float(res.mean())
        return codes

    _INV_RES = {}
    if mode == "full":
        ref_x = ctx.ref_x[0].cpu()
        ref_x0 = ref_x
    else:
        # No image->latent encoder, so anchor on the renderer's own trained
        # centre (motion_mean travels in the checkpoint), then invert ref.jpeg.
        ref_x0 = narrow.motion_mean.detach().float().cpu()
        rt = ref_t.to(dev)
        print("[stage0] inverting ref.jpeg for the base latent")
        codes, res = invert_latents(narrow, rt, ref_x0, iters=a.invert_iters, tag="ref")
        ref_x = codes[0]
        rep["ref_inversion_residual"] = float(res.mean())
        ref_x0 = ref_x
    print("[stage0] ref_x norm=%.4f" % ref_x.norm())

    # ---- L1: load + canonicalise + encode --------------------------------
    vids = {}
    rep["videos"] = {}
    pl = None
    for name in (NEUTRAL,) + EMOTIONS:
        p = Path(a.videos_dir) / ("%s.mp4" % name)
        frames = read_video_rgb(p)
        meta = {"frames": int(frames.shape[0])}
        if a.no_align:
            aligned, winfo = frames, {"ecc": "skipped (--no_align)"}
        else:
            med = np.median(frames, axis=0).astype(np.uint8)
            W, winfo = estimate_warp(med, ref_rgb)
            aligned = apply_warp_and_match(frames, W, ref_rgb)
        meta["align"] = winfo
        if mode == "full":
            enc_frames = aligned
        else:
            # inversion is iterative, so use an even subsample
            k = max(2, min(int(a.invert_frames), aligned.shape[0]))
            idx = np.linspace(0, aligned.shape[0] - 1, k).round().astype(int)
            enc_frames = aligned[idx]
            meta["inverted_frames"] = int(k)
        lat = encode(enc_frames, tag=name)
        if mode != "full":
            meta["inversion_residual"] = _INV_RES.get(name)
        vids[name] = {"frames": aligned, "lat": lat}
        meta["latent_mean_norm"] = float(lat.mean(0).norm())
        meta["latent_temporal_std"] = float(lat.std(0).mean())
        rep["videos"][name] = meta
        print("[stage0] %-8s T=%4d align=%s scale=%.4f t=(%+.1f,%+.1f) "
              "|mean|=%.4f temporal_std=%.4f" % (
                  name, frames.shape[0], winfo.get("ecc"),
                  winfo.get("scale", float("nan")), winfo.get("tx", 0.0),
                  winfo.get("ty", 0.0), meta["latent_mean_norm"],
                  meta["latent_temporal_std"]))
        cv2.imwrite(str(out / ("aligned_median_%s.png" % name)),
                    cv2.cvtColor(np.median(aligned, axis=0).astype(np.uint8),
                                 cv2.COLOR_RGB2BGR))

    # ---- L2: nuisance subspace from known perturbations of ref -----------
    if a.no_nuisance:
        N, nvar = torch.zeros(0, 32), np.zeros(0)
    else:
        pert = perturb(ref_rgb)
        pl = encode(np.stack(pert, 0), tag="nuisance")
        N, nvar = pca_basis(pl - ref_x.unsqueeze(0), a.nuisance_k)
    rep["nuisance"] = {
        "k": int(N.shape[0]),
        "explained": [float(x) for x in nvar],
        "delta_norms_mean": None if pl is None else float((pl - ref_x).norm(dim=1).mean()),
    }
    print("[stage0] nuisance subspace k=%d explained=%s" % (
        N.shape[0], np.round(nvar, 3).tolist()))

    # ---- L3: speech subspace --------------------------------------------
    if a.no_speech:
        S, svar, ssrc = torch.zeros(0, 32), np.zeros(0), "skipped"
    elif a.speech_motion:
        pack = torch.load(a.speech_motion, map_location="cpu")
        m = pack["motion"] if isinstance(pack, dict) and "motion" in pack else pack
        m = m.float().reshape(-1, 32)
        S, svar = pca_basis(m - m.mean(0, keepdim=True), a.speech_k)
        ssrc = "dumped motion %s" % (tuple(m.shape),)
    else:
        ln = vids[NEUTRAL]["lat"]
        S, svar = pca_basis(ln - ln.mean(0, keepdim=True), a.speech_k)
        ssrc = "neutral video temporal variance (proxy)"
    rep["speech"] = {"k": int(S.shape[0]), "source": ssrc,
                     "explained": [float(x) for x in svar]}
    print("[stage0] speech subspace  k=%d src=%s explained=%s" % (
        S.shape[0], ssrc, np.round(svar, 3).tolist()))

    # ---- directions ------------------------------------------------------
    mu_n = vids[NEUTRAL]["lat"].mean(0)
    dirs, dmeta = {}, {}
    for e in EMOTIONS:
        raw = vids[e]["lat"].mean(0) - mu_n
        after_n = project_out(raw, N)
        after_s = project_out(after_n, S)
        dirs[e] = after_s
        dmeta[e] = {
            "norm_raw": float(raw.norm()),
            "norm_after_nuisance": float(after_n.norm()),
            "norm_final": float(after_s.norm()),
            "frac_removed_nuisance": float(1 - after_n.norm() / max(raw.norm(), 1e-9)),
            "frac_removed_speech": float(1 - after_s.norm() / max(after_n.norm(), 1e-9)),
        }
        print("[stage0] d_%-6s |raw|=%.4f -> |after N|=%.4f -> |final|=%.4f "
              "(N removed %.1f%%, S removed %.1f%%)" % (
                  e, dmeta[e]["norm_raw"], dmeta[e]["norm_after_nuisance"],
                  dmeta[e]["norm_final"], 100 * dmeta[e]["frac_removed_nuisance"],
                  100 * dmeta[e]["frac_removed_speech"]))
    rep["directions"] = dmeta

    # ---- CHECK 1: r_null -- both terms are neutral, so this must collapse
    null_raw = mu_n - ref_x
    null_v = project_out(project_out(null_raw, N), S)
    mean_dn = float(np.mean([dirs[e].norm().item() for e in EMOTIONS]))
    r_null = float(null_v.norm() / max(mean_dn, 1e-9))
    rep["check_r_null"] = {
        "r_null": r_null, "norm_raw": float(null_raw.norm()),
        "norm_projected": float(null_v.norm()), "mean_direction_norm": mean_dn,
        "verdict": "GOOD" if r_null < 0.35 else ("MARGINAL" if r_null < 0.6 else "FAIL"),
    }
    print("\n[stage0] CHECK 1  r_null = %.3f  (%s;  <0.35 good, >0.6 means framing "
          "was not removed)" % (r_null, rep["check_r_null"]["verdict"]))

    # ---- CHECK 2: cosine structure
    cosm = {}
    for i, x in enumerate(EMOTIONS):
        for y in EMOTIONS[i + 1:]:
            cosm["%s|%s" % (x, y)] = cos(dirs[x], dirs[y])
    allpar = all(v > 0.9 for v in cosm.values())
    rep["check_cosines"] = {"cos": cosm,
                            "verdict": "FAIL (near-parallel)" if allpar else "OK"}
    print("[stage0] CHECK 2  cosines %s  -> %s" % (
        {k: round(v, 3) for k, v in cosm.items()}, rep["check_cosines"]["verdict"]))
    print("[stage0]           expected: happy|sad < 0 (opposite valence), "
          "sad|angry > 0 (shared negative valence)")

    # ---- alpha sweep: narrow vs reference --------------------------------
    # The reference answers "what SHOULD this direction look like".
    #   full mode:   the full IMTalker renderer, which is identity-general and
    #                trained on real video, so it is range-independent.
    #   invert mode: the narrow render of the inverted expression mean, i.e.
    #                the best this renderer can do for that expression.
    alphas = [float(x) for x in a.alphas.split(",")]
    rep["sweep"] = {}
    ref_label = "FULL" if mode == "full" else "REF(inv)"
    for e in EMOTIONS:
        d = dirs[e].to(dev)
        mot = torch.stack([ref_x.to(dev) + al * d for al in alphas], 0)
        n_img = render_narrow(narrow, mot, chunk=4)
        if mode == "full":
            f_img = torch.cat([ctx.render(mot[i:i + 1]) for i in range(mot.shape[0])], 0)
            ref_resid = None
        else:
            # one reference image pair: neutral mean vs this emotion's mean
            mu_e = vids[e]["lat"].mean(0).to(dev)
            pair = render_narrow(narrow, torch.stack([mu_n.to(dev), mu_e], 0), chunk=2)
            f_img = torch.stack([pair[0]] + [pair[1]] * (len(alphas) - 1), 0)
            ref_resid = pair[1] - pair[0]

        rows = [[to_np(f_img[i]) for i in range(len(alphas))],
                [to_np(n_img[i]) for i in range(len(alphas))]]
        alab = " ".join("%g" % x for x in alphas)
        save_grid(out / ("sweep_%s.png" % e), rows,
                  ["%s   %s   (reference)" % (ref_label, e),
                   "NARROW %s   alpha = %s" % (e, alab)])

        per = []
        amax = 0.0
        for i, al in enumerate(alphas):
            fr = (f_img[i] - f_img[0]) if ref_resid is None else ref_resid
            nr = n_img[i] - n_img[0]
            c = cos(nr, fr) if al != 0 else 1.0
            fe, ne = float(fr.norm()), float(nr.norm())
            clip = float(((n_img[i] <= 1e-4) | (n_img[i] >= 1 - 1e-4)).float().mean())
            gain = (ne / fe) if fe > 1e-9 else float("nan")
            per.append({"alpha": al, "residual_corr": c, "full_energy": fe,
                        "narrow_energy": ne, "gain_ratio": gain,
                        "narrow_clip_frac": clip})
            if al > 0 and c >= a.corr_thresh:
                amax = al
            print("[stage0]   %-6s a=%4g  corr=%5.3f  E_full=%8.2f  E_narrow=%8.2f  "
                  "gain=%5.3f  clip=%.2f%%" % (e, al, c, fe, ne, gain, 100 * clip))
        rep["sweep"][e] = {"per_alpha": per, "alpha_max": amax}
        print("[stage0]   -> alpha_max(%s) = %g  (corr >= %g)\n" % (e, amax, a.corr_thresh))

    # ---- dynamic lip-sync retention -------------------------------------
    # Motion mask = the top 5% most temporally variable pixels of the baseline
    # sequence; the repo's own measurement is that ~3.8% of pixels carry 90% of
    # temporal variance, so this mask is essentially the mouth.
    traj = vids[NEUTRAL]["lat"].to(dev)
    base = render_narrow(narrow, traj)
    std = base.std(0).mean(0)
    thr = torch.quantile(std.flatten().float(), 0.95)
    mask = (std >= thr).float()
    base_e = float(((base - base.mean(0, keepdim=True)).abs().mean(1) * mask).sum())
    rep["lipsync"] = {"alpha": a.dyn_alpha, "mask_frac": float(mask.mean()),
                      "baseline_energy": base_e, "retention": {}}
    for e in EMOTIONS:
        withe = render_narrow(narrow, traj + a.dyn_alpha * dirs[e].to(dev))
        ee = float(((withe - withe.mean(0, keepdim=True)).abs().mean(1) * mask).sum())
        r = ee / max(base_e, 1e-9)
        rep["lipsync"]["retention"][e] = r
        print("[stage0] lip-sync retention %-6s alpha=%g: %.3f (%s)" % (
            e, a.dyn_alpha, r, "OK" if r >= 0.9 else "DEGRADED"))

    # ---- verdict ---------------------------------------------------------
    amax_all = {e: rep["sweep"][e]["alpha_max"] for e in EMOTIONS}
    ok_null = r_null < 0.35
    ok_cos = not allpar
    ok_range = sum(1 for e in ("happy", "sad") if amax_all[e] >= 1.0) == 2
    ok_lip = all(v >= 0.9 for v in rep["lipsync"]["retention"].values())
    if ok_null and ok_cos and ok_range and ok_lip:
        verdict = "GO -> Stage 1 (latent steering is viable as-is)"
    elif ok_null and ok_cos and max(amax_all.values()) < 0.5:
        verdict = "GO but Stage 3 FIRST (directions transfer; narrow renderer too clipped)"
    elif not (ok_null and ok_cos):
        verdict = "NO-GO (extraction failed: framing dominates -> try landmark alignment)"
    else:
        verdict = "PARTIAL (see per-emotion alpha_max / lip-sync numbers)"
    rep["verdict"] = verdict

    # ---- save ------------------------------------------------------------
    torch.save({
        "dirs": {e: dirs[e] for e in EMOTIONS},
        "ref_x": ref_x, "neutral_mean": mu_n,
        "nuisance_basis": N, "speech_basis": S,
        "alpha_max": amax_all,
        "meta": {"videos_dir": a.videos_dir, "ref_path": a.ref_path,
                 "r_null": r_null, "cosines": cosm},
    }, out / "emotion_dirs.pt")
    (out / "stage0_report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")

    print("\n" + "=" * 72)
    print("  encoder mode    %s%s" % (
        mode, "  (reference is the narrow renderer's own best fit, not an "
              "independent model)" if mode != "full" else ""))
    print("  r_null          %.3f   %s" % (r_null, rep["check_r_null"]["verdict"]))
    print("  cosines         %s" % rep["check_cosines"]["verdict"])
    print("  alpha_max       %s" % "  ".join("%s=%g" % (e, amax_all[e]) for e in EMOTIONS))
    print("  lip-sync        %s" % "  ".join(
        "%s=%.3f" % (e, rep["lipsync"]["retention"][e]) for e in EMOTIONS))
    print("  VERDICT         %s" % verdict)
    print("=" * 72)
    print("\nwrote %s/emotion_dirs.pt, stage0_report.json, sweep_*.png, "
          "aligned_median_*.png" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
