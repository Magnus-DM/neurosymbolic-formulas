"""Train and test one model on the MathExpression task.

Examples (run from the repository root):

    # Headline 5-image comparison (30,000 training examples, full early stopping)
    python mathexpression/train.py --model baseline --images 5
    python mathexpression/train.py --model architecture --images 5
    python mathexpression/train.py --model loss --images 5

    # New task: modulo-10 arithmetic with x and / and operator precedence
    python mathexpression/train.py --model architecture --images 3 --task mod10

    # Smoke test on the synthetic sample (a few minutes on a laptop CPU)
    python mathexpression/train.py --model architecture --images 3 \
        --train-examples 500 --val-examples 100 --test-examples 200 --max-steps 300

The model, logic program and training settings are the ones used for the report:
AdamW (lr 0.001), batch size 8, early stopping on validation accuracy
(patience 5, min delta 0.001), validation every 1,000 steps for the baseline
and every 500 steps for the two neurosymbolic models.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import dataset  # noqa: E402
from dataset import (  # noqa: E402
    MathExpressionTaskLogicInArchitecture,
    MathExpressionTaskLogicInLoss,
    MathExpressionTaskNeural,
)

from nesy.logic import ForwardChaining  # noqa: E402
from nesy.model import (  # noqa: E402
    FeatureExtractor,
    NeSyModelLogicInArchitecture,
    NeSyModelLogicInLoss,
    NetworkHeadLarge,
    NetworkHeadSmall,
    NeuralBaseline,
    NeuralModel,
    SymbolCNN,
)
from nesy.semantics import SumProductSemiring  # noqa: E402

TASKS = {
    "baseline": MathExpressionTaskNeural,
    "architecture": MathExpressionTaskLogicInArchitecture,
    "loss": MathExpressionTaskLogicInLoss,
}
VAL_CHECK_INTERVAL = {"baseline": 1000, "architecture": 500, "loss": 500}
CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"


def operator_encoding(mod10):
    if mod10:
        return {"plus": 0, "minus": 1, "multiply": 2, "divide": 3}
    return {"plus": 0, "minus": 1}


def build_model(model_name, nr_images, program, mod10=False):
    """Build the network(s) exactly as in the report (Tables 1-4)."""
    size_output_layer = 28 if nr_images == 3 else 46
    target_shift = 9 if nr_images == 3 else 18
    operators = operator_encoding(mod10)

    if model_name == "baseline":
        network = NeuralBaseline(n_inputs=nr_images, m_outputs=size_output_layer, softmax=False)
        return NeuralModel(neural_network=network)

    if model_name == "architecture":
        neural_predicates = torch.nn.ModuleDict(
            {
                "digit_nn": SymbolCNN(10),
                "operator_nn": SymbolCNN(len(operators), output_encoding=operators),
            }
        )
        return NeSyModelLogicInArchitecture(
            program=program,
            logic_engine=ForwardChaining(),
            neural_predicates=neural_predicates,
            label_semantics=SumProductSemiring(),
        )

    if model_name == "loss":
        feature_extractor = FeatureExtractor(size_output_vector=128)
        result_encoding = {str(i - target_shift): i for i in range(size_output_layer)}
        neural_predicates = torch.nn.ModuleDict(
            {
                "result_head": NetworkHeadLarge(
                    feature_extractor, nr_images, size_output_layer, output_encoding=result_encoding
                ),
                "digit_head": NetworkHeadSmall(
                    feature_extractor, 1, 10, output_encoding={str(i): i for i in range(10)}
                ),
                "operator_head": NetworkHeadSmall(
                    feature_extractor, 1, len(operators), output_encoding=operators
                ),
            }
        )
        return NeSyModelLogicInLoss(
            program=program,
            logic_engine=ForwardChaining(),
            neural_predicates=neural_predicates,
            label_semantics=SumProductSemiring(),
        )

    raise ValueError(f"Unknown model {model_name}")


def checkpoint_path(model_name, nr_images, task):
    return CHECKPOINT_DIR / f"{model_name}_{task}_{nr_images}imgs.ckpt"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", choices=sorted(TASKS), required=True)
    parser.add_argument("--images", type=int, choices=[3, 5], default=3)
    parser.add_argument(
        "--task",
        choices=["plusminus", "mod10"],
        default="plusminus",
        help="plusminus: the course task; mod10: the new task with x and / (modulo 10)",
    )
    parser.add_argument("--train-examples", type=int, default=30000)
    parser.add_argument("--val-examples", type=int, default=2000)
    parser.add_argument("--test-examples", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=-1, help="-1 = until early stopping")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="torch CPU threads (small networks run fastest on few threads)",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    pl.seed_everything(args.seed, workers=True)
    dataset.TASK_3 = args.task == "mod10"
    task_cls = TASKS[args.model]

    print(f"Loading data ({args.train_examples} train / {args.val_examples} val examples)...")
    task_train = task_cls(
        train=True,
        split="train",
        nr_images_per_example=args.images,
        nr_examples=args.train_examples,
    )
    task_val = task_cls(
        train=False, split="val", nr_images_per_example=args.images, nr_examples=args.val_examples
    )
    task_test = task_cls(
        train=False, split="test", nr_images_per_example=args.images, nr_examples=args.test_examples
    )
    train_loader = task_train.dataloader(
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    val_loader = task_val.dataloader(batch_size=args.batch_size, num_workers=args.num_workers)
    test_loader = task_test.dataloader(batch_size=args.batch_size, num_workers=args.num_workers)

    program = getattr(task_train, "program", None)
    model = build_model(args.model, args.images, program, mod10=dataset.TASK_3)

    val_check_interval = VAL_CHECK_INTERVAL[args.model]
    if 0 < args.max_steps < val_check_interval:
        val_check_interval = args.max_steps
    early_stopping = EarlyStopping(
        monitor="val_acc",
        mode="max",
        patience=5,
        min_delta=0.001,
        verbose=True,
        check_on_train_epoch_end=False,
    )
    trainer = pl.Trainer(
        max_epochs=1000,
        max_steps=args.max_steps,
        accelerator="cpu",
        callbacks=[early_stopping],
        val_check_interval=val_check_interval,
        check_val_every_n_epoch=None,
        enable_checkpointing=False,
        logger=False,
    )

    start = time.time()
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    training_time = time.time() - start

    ckpt = checkpoint_path(args.model, args.images, args.task)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(ckpt)

    test_acc = trainer.test(model=model, dataloaders=test_loader)[0]["test_acc"]
    summary = {
        "model": args.model,
        "task": args.task,
        "images": args.images,
        "train_examples": args.train_examples,
        "seed": args.seed,
        "steps": trainer.global_step,
        "training_time_s": round(training_time, 1),
        "test_examples": len(task_test),
        "test_acc": round(100 * test_acc, 2),
        "checkpoint": str(
            ckpt.relative_to(Path.cwd()) if ckpt.is_relative_to(Path.cwd()) else ckpt
        ),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
