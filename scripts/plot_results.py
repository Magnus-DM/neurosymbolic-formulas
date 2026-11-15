"""Draw docs/accuracy_vs_length.png from the test accuracies in the project report (Table 5).

Values are copied from the report, not recomputed: 3-image results are the mean of
3 runs, 5-image results are single runs; test set of 1,000 expressions each.
"""

from pathlib import Path

import matplotlib.pyplot as plt

REPORTED = {  # model: (3 images, 5 images), test accuracy in %
    "Neural baseline": (86.03, 7.10),
    "Logic in the architecture": (97.83, 98.70),
    "Logic in the loss": (96.47, 28.90),
}


def main():
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=150)
    groups = ["3 images (e.g. 7 + 2)", "5 images (e.g. 7 + 2 - 4)"]
    width = 0.26
    for i, (name, accs) in enumerate(REPORTED.items()):
        xs = [g + (i - 1) * width for g in range(len(groups))]
        bars = ax.bar(xs, accs, width, label=name)
        ax.bar_label(bars, labels=[f"{a:.1f}" for a in accs], padding=2, fontsize=8)
    ax.set_xticks(range(len(groups)), groups)
    ax.set_ylim(0, 112)
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Accuracy versus formula length")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=8, frameon=False, ncols=3)
    fig.tight_layout()
    out = Path(__file__).resolve().parent.parent / "docs" / "accuracy_vs_length.png"
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
