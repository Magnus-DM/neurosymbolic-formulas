"""Evaluate a checkpoint written by train.py on the test split.

python mathexpression/evaluate.py --model architecture --images 5
"""

import argparse
import sys
from pathlib import Path

import pytorch_lightning as pl
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import dataset  # noqa: E402
from train import TASKS, build_model, checkpoint_path  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", choices=sorted(TASKS), required=True)
    parser.add_argument("--images", type=int, choices=[3, 5], default=3)
    parser.add_argument("--task", choices=["plusminus", "mod10"], default="plusminus")
    parser.add_argument("--test-examples", type=int, default=1000)
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args()

    dataset.TASK_3 = args.task == "mod10"
    task_test = TASKS[args.model](
        train=False, split="test", nr_images_per_example=args.images, nr_examples=args.test_examples
    )
    model = build_model(
        args.model, args.images, getattr(task_test, "program", None), mod10=dataset.TASK_3
    )
    ckpt = args.checkpoint or checkpoint_path(args.model, args.images, args.task)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(state["state_dict"])

    trainer = pl.Trainer(accelerator="cpu", logger=False)
    trainer.test(model=model, dataloaders=task_test.dataloader(batch_size=8))


if __name__ == "__main__":
    main()
