#!/usr/bin/env python3
"""Compare reproduced PDFs with FIG-10-10 using Poppler text and raster output."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated", type=Path, default=ROOT / "generated")
    parser.add_argument("--reference", type=Path, default=ROOT / "reference_figures")
    parser.add_argument("--dpi", type=int, default=144)
    parser.add_argument("--output", type=Path, default=ROOT / "generated/validation/figure_comparison.json")
    args = parser.parse_args()
    results = []
    with tempfile.TemporaryDirectory() as temporary:
        tmp = Path(temporary)
        for reference in sorted(args.reference.rglob("*.pdf")):
            relative = reference.relative_to(args.reference)
            generated = args.generated / relative
            if not generated.is_file():
                raise FileNotFoundError(generated)
            texts, arrays = [], []
            for index, path in enumerate((reference, generated)):
                texts.append(subprocess.check_output(["pdftotext", "-layout", str(path), "-"]))
                prefix = tmp / str(index)
                subprocess.run(["pdftoppm", "-singlefile", "-r", str(args.dpi), "-png",
                                str(path), str(prefix)], check=True, capture_output=True)
                with Image.open(str(prefix) + ".png") as image:
                    arrays.append(np.array(image.convert("RGB")))
            same_shape = arrays[0].shape == arrays[1].shape
            row = {"file": str(relative), "dpi": args.dpi,
                   "text_layout_identical": texts[0] == texts[1], "raster_dimensions_identical": same_shape,
                   "reference_shape": list(arrays[0].shape), "generated_shape": list(arrays[1].shape)}
            if same_shape:
                delta = np.abs(arrays[0].astype(np.int16) - arrays[1].astype(np.int16))
                row.update(raster_identical=bool(np.all(delta == 0)),
                           differing_pixel_fraction=float(np.any(delta != 0, axis=2).mean()),
                           mean_absolute_channel_error=float(delta.mean()), max_channel_error=int(delta.max()))
            results.append(row)
            print(json.dumps(row), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
