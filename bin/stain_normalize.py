"""
A tiny, self-contained normalization module:
- NormContext.build(...)  -> constructs context (fits the reference normalizer once)
- ctx.normalize_patch(chw_u8)  -> normalizes a single CHW uint8 patch (or no-op if disabled)
"""
from typing import Optional, Dict, Any, Tuple
from pathlib import Path

import numpy as np
import zarr
from tiatoolbox.tools.stainnorm import ReinhardNormalizer
from PIL import Image


class NormContext:
    """
    Holds a fitted stain normalizer and metadata.
    If 'apply_normalization' is False, normalize_patch() is a no-op.

    Typical usage:
        ctx = NormContext.build(
            apply_normalization=True,
            method="reinhard",
            ref_zarr="/path/to/ref.zarr",
            ref_roi_zarr="/path/to/ref_roi.zarr",
            ref_ds_key="0/6",
            ref_roi_key="GreenThresh/4",
            ref_size=512,
            ref_strategy="random",
        )
        patch = ctx.normalize_patch(patch)  # CHW uint8 -> CHW uint8
    """
    def __init__(
        self,
        apply_normalization: bool,
        method: str,
        normalizer: Optional[object],
        meta: Dict[str, Any]
    ):
        self.apply_normalization = apply_normalization
        self.method = method
        self._normalizer = normalizer
        self.meta = meta  # safe to write into zarr attrs for provenance

    # ---------- public API ----------
    @classmethod
    def build(
        cls,
        apply_normalization: bool,
        method: str,
        ref_zarr: str,
        ref_roi_zarr: str,
        ref_ds_key: str,
        ref_roi_key: str,
        ref_size: int = 512,
        ref_strategy: str = "random",
        rng_seed: int = 0,
    ):
        """
        Construct a normalization context. If 'apply_normalization' is False, returns a no-op context.
        - Fits a Reinhard normalizer from a reference HxWxC patch sampled within the reference ROI.
        """
        if not apply_normalization:
            return cls(apply_normalization=False, method="none", normalizer=None, meta={"apply_normalization": False})

        method_l = (method or "").lower()
        if method_l != "reinhard":
            raise ValueError("Only 'reinhard' normalization is supported currently.")

        # Load reference image (CHW uint8) and ROI mask (2D)
        zr = zarr.open(ref_zarr, mode="r")
        zmr = zarr.open(ref_roi_zarr, mode="r")

        # [T,C,Z,Y,X]
        ref_arr = zr[ref_ds_key]
        # CHW
        ref_chw = np.asarray(ref_arr[0, :, 0, :, :])
        # HWC uint8
        ref_hwc = cls._chw_to_hwc(ref_chw)
        # Make sure numeric image data is formatted with correct type
        ref_chw = cls._ensure_uint8(ref_chw)

        # 2D
        ref_mask = np.asarray(zmr[ref_roi_key][:, :])
        ref_mask = cls._nearest_resize_mask(ref_mask, ref_hwc.shape[:2])

        # Choose a reference patch inside tissue
        ref_patch = cls._pick_reference_patch(
            ref_hwc, ref_mask, ref_size=ref_size, strategy=ref_strategy, rng_seed=rng_seed
        )

        # Fit Reinhard
        # TODO: can we dump/ save parameters and pass down from ScanPrep?
        normalizer = ReinhardNormalizer()
        normalizer.fit(ref_patch)

        meta = {
            "apply_normalization": True,
            "normalization_method": "reinhard",
            "normalization_ref_zarr": ref_zarr,
            "normalization_ref_roi_zarr": ref_roi_zarr,
            "normalization_ref_ds": ref_ds_key,
            "normalization_ref_roi_key": ref_roi_key,
            "normalization_ref_size": ref_size,
            "normalization_ref_strategy": ref_strategy,
        }
        return cls(apply_normalization=True, method="reinhard", normalizer=normalizer, meta=meta)

    def normalize_patch(self, chw_u8: np.ndarray) -> np.ndarray:
        """
        Normalize a single CHW uint8 patch. Returns CHW uint8.
        No-op if self.apply_normalization is False.
        """
        if (not self.apply_normalization) or (self._normalizer is None):
            return chw_u8
        if self.method == "reinhard":
            hwc = self._chw_to_hwc(chw_u8)
            hwc_norm = self._normalizer.transform(hwc)
            return self._hwc_to_chw(hwc_norm.astype(np.uint8))
        # Future extension (e.g., macenko, vahadane) would branch here.
        return chw_u8

    # ---------- small helpers (private) ----------
    @staticmethod
    def _chw_to_hwc(chw: np.ndarray) -> np.ndarray:
        return np.transpose(chw, (1, 2, 0))

    @staticmethod
    def _hwc_to_chw(hwc: np.ndarray) -> np.ndarray:
        return np.transpose(hwc, (2, 0, 1))

    @staticmethod
    def _ensure_uint8(chw: np.ndarray) -> np.ndarray:
        if chw.dtype == np.uint8:
            return chw
        out = chw.astype(np.float32)
        for c in range(out.shape[0]):
            mn, mx = float(out[c].min()), float(out[c].max())
            if mx <= mn:
                out[c] = 0
            else:
                out[c] = (out[c] - mn) / (mx - mn) * 255.0
        return np.clip(out, 0, 255).astype(np.uint8)

    @staticmethod
    def _nearest_resize_mask(mask: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
        Ht, Wt = target_hw
        if mask.shape == (Ht, Wt):
            return mask
        ry = Ht / mask.shape[0]
        rx = Wt / mask.shape[1]
        yy = (np.arange(Ht) / ry).astype(np.int64)
        xx = (np.arange(Wt) / rx).astype(np.int64)
        yy = np.clip(yy, 0, mask.shape[0] - 1)
        xx = np.clip(xx, 0, mask.shape[1] - 1)
        return mask[yy[:, None], xx[None, :]]

    @staticmethod
    def _pick_reference_patch(
        hwc: np.ndarray,
        tissue_mask: np.ndarray,
        ref_size: int,
        strategy: str = "random",
        rng_seed: int = 0,
    ) -> np.ndarray:
        H, W, _ = hwc.shape
        yy, xx = np.where(tissue_mask)
        if yy.size == 0:
            # fallback: center crop
            y0 = max(0, H // 2 - ref_size // 2)
            x0 = max(0, W // 2 - ref_size // 2)
        else:
            if strategy == "median":
                y_c = int(np.median(yy)); x_c = int(np.median(xx))
            else:
                rng = np.random.default_rng(rng_seed)
                idx = rng.integers(0, yy.size)
                y_c, x_c = int(yy[idx]), int(xx[idx])
            y0 = max(0, y_c - ref_size // 2)
            x0 = max(0, x_c - ref_size // 2)


        y1 = min(H, y0 + ref_size)
        x1 = min(W, x0 + ref_size)
        patch = hwc[y0:y1, x0:x1]

        # pad to ref_size if we hit borders
        pad_y = ref_size - patch.shape[0]
        pad_x = ref_size - patch.shape[1]
        if pad_y > 0 or pad_x > 0:
            patch = np.pad(patch, ((0, max(0, pad_y)), (0, max(0, pad_x)), (0, 0)), mode="edge")
        return patch.astype(np.uint8)

    # ---------- Methods for saving pngs ----------
    @staticmethod
    def zarr2png(
        zarr_path,
        key: str = "0/6",
        output_name: str = "zarr2png_out.png"
    ):
        """
        Go from zarr to png.
        """
        # Load the zarr file
        z = zarr.open(zarr_path, mode='r')

        # Convert to numpy array
        arr = np.array(z[key])
        arr = np.squeeze(arr) # remove 1 dims
        arr = np.moveaxis(arr, 0, -1)

        # If the image has float values (0-1), scale to 0-255
        if arr.dtype != np.uint8:
            arr = (255 * (arr - arr.min()) / (arr.max() - arr.min())).astype(np.uint8)

        # Convert to PIL image
        img = Image.fromarray(arr)

        # Save as PNG
        img.save(output_name)

        print(f"[zarr2png] Saved image as {output_name}")

    @staticmethod
    def dump_reference_png(ref_zarr: str, ds_key: str, output_dir: str) -> str:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        ref_id = NormContext._derive_basename_from_zarr_path(ref_zarr)
        out = Path(output_dir) / f"{ref_id}__ref__{ds_key.replace('/', '-')}.png"
        if not out.exists():
            NormContext.zarr2png(ref_zarr, key=ds_key, output_name=str(out))
        return str(out)

    @staticmethod
    def dump_target_png(target_zarr: str, ds_key: str, output_dir: str) -> str:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        tid = NormContext._derive_basename_from_zarr_path(target_zarr)
        out = Path(output_dir) / f"{tid}__raw__{ds_key.replace('/', '-')}.png"
        if not out.exists():
            NormContext.zarr2png(target_zarr, key=ds_key, output_name=str(out))
        return str(out)
    
    @staticmethod
    def _derive_basename_from_zarr_path(p: str) -> str:
        name = Path(p).name
        if name.endswith(".zarr"):
            name = name[:-5]
        if name.startswith("roi_masks_"):
            name = name[len("roi_masks_"):]
        return name




