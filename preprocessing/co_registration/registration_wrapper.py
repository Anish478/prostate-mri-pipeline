import os
import subprocess
from pathlib import Path
from typing import Union, Sequence, Dict, Optional

ELASTIX_PATH = "/Users/anish/elastix-5/bin/elastix"
TRANSFORMIX_PATH = "/Users/anish/elastix-5/bin/transformix"

# -----------------------------------------------------------------------------
# Generic Elastix wrapper (rigid by default, but any parameter file works)
# -----------------------------------------------------------------------------

def run_elastix(fixed_image: Union[str, Path],
                moving_image: Union[str, Path],
                output_dir: Union[str, Path],
                param_files: Sequence[Union[str, Path]],
                fixed_mask: Optional[Union[str, Path]] = None,
                moving_mask: Optional[Union[str, Path]] = None,
                elastix_path: str = ELASTIX_PATH,
                **kwargs) -> Path:
    """Run elastix registration.

    Parameters
    ----------
    fixed_image : path to fixed/reference volume (NIfTI)
    moving_image : path to moving volume (NIfTI)
    output_dir : folder that will receive result.* + TransformParameters.txt
    param_files : one or more elastix parameter files (chain executed in order)
    fixed_mask / moving_mask : optional binary masks
    elastix_path : path to elastix executable

    Returns
    -------
    Path to the resulting TransformParameters.txt (last stage)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [elastix_path,
           "-f", str(fixed_image),
           "-m", str(moving_image),
           "-out", str(output_dir)]

    # allow arbitrary number of -p files
    for p in param_files:
        cmd.extend(["-p", str(p)])

    if fixed_mask:
        cmd.extend(["-fMask", str(fixed_mask)])
    if moving_mask:
        cmd.extend(["-mMask", str(moving_mask)])

    # allow override options via kwargs -> "-threads", etc.
    for k, v in kwargs.items():
        cmd.extend([f"-{k}", str(v)])

    print("[elastix]", " ".join(cmd))
    subprocess.run(cmd, check=True)

    # By convention elastix writes TransformParameters.0.txt, .1.txt …
    # grab the last one (highest index)
    tp_files = sorted(output_dir.glob("TransformParameters.*.txt"))
    if not tp_files:
        raise RuntimeError("No TransformParameters file produced by elastix")
    return tp_files[-1]


def apply_transform(
    moving_image: Union[str, Path],
    transform_param: Union[str, Path],
    output_dir: Union[str, Path],
    *,
    preserve_moving_grid: bool = False,
    transformix_path: str = TRANSFORMIX_PATH,
) -> Path:
    """Warp *moving_image* using *transform_param* via transformix.

    Parameters
    ----------
    preserve_moving_grid : If ``True`` the result image keeps the *moving* voxel
        spacing and size (handy when the reference/fixed image has much thicker
        slices).  Implemented by copying the transform parameter file and
        appending the two elastix keys::

            (ResultImagePixelSpacing  -999)
            (ResultImageSize         -999)
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tp_to_use = Path(transform_param)
    if preserve_moving_grid:
        # create a local copy with grid directives appended once
        tp_mod = output_dir / (tp_to_use.stem + "_preserve.txt")
        if not tp_mod.exists():
            with tp_to_use.open("r") as fin, tp_mod.open("w") as fout:
                fout.write(fin.read())
                fout.write("\n(ResultImagePixelSpacing -999)\n")
                fout.write("(ResultImageSize -999)\n")
        tp_to_use = tp_mod

    cmd = [transformix_path,
           "-in", str(moving_image),
           "-out", str(output_dir),
           "-tp", str(tp_to_use)]

    print("[transformix]", " ".join(cmd))
    subprocess.run(cmd, check=True)

    # result image is result.nii or result.mhd etc.
    for ext in ("result.nii", "result.nii.gz", "result.mhd"):
        res = output_dir / ext
        if res.exists():
            return res
    raise RuntimeError("transformix did not produce a result image") 