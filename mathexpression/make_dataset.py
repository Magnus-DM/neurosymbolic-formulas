"""Build the MathExpression JSON files from a folder of symbol images.

The course provided the expression splits used for the reported numbers; those
files are not redistributed here. This script builds equivalent splits:

* from the public Kaggle "Handwritten math symbols" images (full experiments), or
* from synthetic glyph images it renders itself (--synthetic, for the smoke test).

Images live in mathexpression/data/hwf/Handwritten_Math_Symbols/<symbol>/*.jpg,
with symbol folders 0-9, +, -, times and div. Each symbol's images are split
into disjoint train/val/test pools, so no test image is seen during training.

    python mathexpression/make_dataset.py                 # real images, all four files
    python mathexpression/make_dataset.py --synthetic     # small synthetic sample
"""

import argparse
import json
import os
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

DATA_DIR = Path(os.environ.get("MATHEXPR_DATA", Path(__file__).resolve().parent / "data" / "hwf"))
SYMBOL_DIR = "Handwritten_Math_Symbols"
DIGITS = [str(d) for d in range(10)]
TOKEN_TO_FOLDER = {"+": "+", "-": "-", "*": "times", "/": "div", **{d: d for d in DIGITS}}
SPLIT_FRACTIONS = {"train": 0.8, "val": 0.1, "test": 0.1}


def evaluate(expression, mod10):
    """Value of an expression such as '8 - 3 * 5'.

    plusminus task: ordinary integer arithmetic.
    mod10 task: x and / bind tighter than + and -, / rounds down, result modulo 10.
    """
    value = eval(expression.replace("/", "//"))  # only digits and + - * / reach this point
    return value % 10 if mod10 else value


def random_expression(rng, nr_images, mod10):
    operators = ["+", "-", "*", "/"] if mod10 else ["+", "-"]
    tokens = [rng.choice(DIGITS)]
    for _ in range(nr_images // 2):
        op = rng.choice(operators)
        digit = rng.choice(DIGITS[1:] if op == "/" else DIGITS)  # no division by zero
        tokens += [op, digit]
    return tokens


def split_pools(rng):
    pools = {split: {} for split in SPLIT_FRACTIONS}
    for folder in sorted(set(TOKEN_TO_FOLDER.values())):
        files = sorted(p.name for p in (DATA_DIR / SYMBOL_DIR / folder).glob("*.jpg"))
        if not files:
            raise FileNotFoundError(
                f"No images in {DATA_DIR / SYMBOL_DIR / folder} (see README, Data)"
            )
        rng.shuffle(files)
        n_train = int(len(files) * SPLIT_FRACTIONS["train"])
        n_val = int(len(files) * SPLIT_FRACTIONS["val"])
        pools["train"][folder] = files[:n_train]
        pools["val"][folder] = files[n_train : n_train + n_val]
        pools["test"][folder] = files[n_train + n_val :]
    return pools


def build_file(rng, pools, nr_images, mod10, sizes):
    out = {}
    for split, size in sizes.items():
        examples = []
        for _ in range(size):
            tokens = random_expression(rng, nr_images, mod10)
            paths = []
            for token in tokens:
                folder = TOKEN_TO_FOLDER[token]
                paths.append(f"{SYMBOL_DIR}/{folder}/{rng.choice(pools[split][folder])}")
            expression = " ".join(tokens)
            examples.append(
                {"paths": paths, "expression": expression, "result": evaluate(expression, mod10)}
            )
        out[split] = examples
    return out


def draw_symbol(rng, folder, font_cache):
    """Render one 45x45 'handwritten-looking' symbol: random size, position, angle, stroke."""
    img = Image.new("L", (45, 45), 255)
    draw = ImageDraw.Draw(img)
    cx, cy = 22 + rng.randint(-3, 3), 22 + rng.randint(-3, 3)
    width = rng.randint(2, 4)
    r = rng.randint(9, 14)
    if folder in DIGITS:
        size = rng.randint(26, 38)
        font = font_cache.setdefault(size, ImageFont.load_default(size=size))
        draw.text((cx, cy), folder, font=font, fill=0, anchor="mm", stroke_width=width // 2)
    elif folder in ("+", "-"):
        draw.line((cx - r, cy, cx + r, cy), fill=0, width=width)
        if folder == "+":
            draw.line((cx, cy - r, cx, cy + r), fill=0, width=width)
    elif folder == "times":
        draw.line((cx - r, cy - r, cx + r, cy + r), fill=0, width=width)
        draw.line((cx - r, cy + r, cx + r, cy - r), fill=0, width=width)
    elif folder == "div":
        draw.line((cx - r, cy, cx + r, cy), fill=0, width=width)
        for dy in (-r + 2, r - 2):
            draw.ellipse((cx - 2, cy + dy - 2, cx + 2, cy + dy + 2), fill=0)
    img = img.rotate(rng.uniform(-15, 15), fillcolor=255)
    return img.filter(ImageFilter.GaussianBlur(rng.uniform(0.0, 0.8)))


def render_synthetic(rng, per_symbol):
    font_cache = {}
    for folder in sorted(set(TOKEN_TO_FOLDER.values())):
        target = DATA_DIR / SYMBOL_DIR / folder
        target.mkdir(parents=True, exist_ok=True)
        for i in range(per_symbol):
            draw_symbol(rng, folder, font_cache).save(target / f"synthetic_{i:05d}.jpg", quality=90)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--synthetic", action="store_true", help="render synthetic symbols first")
    parser.add_argument("--per-symbol", type=int, default=300, help="synthetic images per symbol")
    parser.add_argument("--train", type=int, default=None)
    parser.add_argument("--val", type=int, default=None)
    parser.add_argument("--test", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    if args.synthetic:
        render_synthetic(rng, args.per_symbol)
    default_sizes = (1000, 200, 200) if args.synthetic else (50000, 2000, 2000)
    sizes = {
        "train": args.train or default_sizes[0],
        "val": args.val or default_sizes[1],
        "test": args.test or default_sizes[2],
    }

    pools = split_pools(rng)
    for nr_images in (3, 5):
        for mod10 in (False, True):
            name = f"math_expression_examples_size_{nr_images}{'_task_3' if mod10 else ''}.json"
            data = build_file(rng, pools, nr_images, mod10, sizes)
            (DATA_DIR / name).write_text(json.dumps(data))
            print(
                f"wrote {DATA_DIR / name}: " + ", ".join(f"{k}={len(v)}" for k, v in data.items())
            )


if __name__ == "__main__":
    main()
