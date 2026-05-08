from pathlib import Path

import tecplot as tp
from tecplot.constant import ExportRegion


DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "out_files"


def _resolve_output_root(output_root=None):
    root = DEFAULT_OUTPUT_ROOT if output_root is None else Path(output_root).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _ensure_subdir(output_root, subdir):
    path = output_root / subdir
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_png(
    filename="plot.png",
    output_root=None,
    width=2000,
    supersample=3,
    region=ExportRegion.AllFrames,
):
    root = _resolve_output_root(output_root)
    out_path = _ensure_subdir(root, "png") / filename
    tp.export.save_png(
        str(out_path),
        width=width,
        supersample=supersample,
        region=region,
    )
    return out_path


def save_eps(filename="plot.eps", output_root=None, region=ExportRegion.AllFrames):
    root = _resolve_output_root(output_root)
    out_path = _ensure_subdir(root, "eps") / filename
    tp.export.save_eps(str(out_path), region=region)
    return out_path


def save_layout(filename="plot.lay", output_root=None, region=ExportRegion.AllFrames):
    root = _resolve_output_root(output_root)
    out_path = _ensure_subdir(root, "lay") / filename
    tp.save_layout(str(out_path))
    return out_path


def save_all(base_name="plot", output_root=None, region=ExportRegion.AllFrames):
    png_path = save_png(
        filename=f"{base_name}.png",
        output_root=output_root,
        region=region,
    )
    eps_path = save_eps(
        filename=f"{base_name}.eps",
        output_root=output_root,
        region=region,
    )
    lay_path = save_layout(
        filename=f"{base_name}.lay",
        output_root=output_root,
        region=region,
    )
    return {"png": png_path, "eps": eps_path, "lay": lay_path}
