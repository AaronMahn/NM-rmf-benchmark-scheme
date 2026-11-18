#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fit_run import FIT_REFERENCES, evaluate_fit
import rmf
import vpaths

HERE = Path(__file__).resolve().parent
MODELS = {
    "gaussian": (r"$\sigma^6$", "#386CA7", "-"),
    "gaussian_lambda": (r"$\sigma^6+c_4$", "#E09E1A", "-"),
    "tm1": ("TM1", "#222222", "--"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot M-R curves for TM1 and the Table 3 Gaussian fits.")
    parser.add_argument("--out", type=Path, default=HERE / "sigma6_tm1_mr.png", help="Output figure path.")
    args = parser.parse_args()

    curves = {case: evaluate_fit(case)[1] for case in FIT_REFERENCES}
    baseline = json.loads((vpaths.SIGMA6 / "model.json").read_text(encoding="utf-8"))
    curves["tm1"], _, _ = rmf.curve(rmf.vector(baseline["params"]), fine=True)

    # Use the manuscript's figure style without requiring a local style file.
    plt.rcParams.update({
        "font.family": "STIXGeneral", "mathtext.fontset": "stix",
        "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "xtick.minor.visible": True, "ytick.minor.visible": True,
        "legend.fontsize": 7.5, "legend.frameon": False, "pdf.fonttype": 42,
    })
    fig, axis = plt.subplots(figsize=(3.8, 2.65))
    for case, (label, color, style) in MODELS.items():
        curve = curves[case]
        stable = curve[:curve[:, 1].argmax() + 1]
        axis.plot(stable[:, 2], stable[:, 1], style, color=color, lw=1.3, label=label)

    axis.set(xlim=(10, 16), ylim=(0.8, 2.7), xlabel=r"$R$ [km]", ylabel=r"$M$ [$M_\odot$]")
    axis.axhline(2.08, color="#7F7F7F", ls=":", lw=0.7)
    axis.text(10.1, 2.10, r"$2.08\,M_\odot$", fontsize=7, color="#7F7F7F")
    axis.legend(loc="lower left")
    fig.tight_layout()
    output = args.out.expanduser()
    if not output.is_absolute():
        output = HERE / output
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
