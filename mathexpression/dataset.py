# Provided by the course Learning and Reasoning (KU Leuven, 2025-2026).
# Added by Magnus De Maerteleire: the modulo-10 task (mult_and_div_* logic programs and
# the TASK_3 switches) and the MATHEXPR_DATA override for the data folder.

import json
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset, default_collate

from nesy.parser import parse_program

# Data lives next to this file in data/hwf/ unless MATHEXPR_DATA points elsewhere (see README).
DATA_PATH = (
    os.environ.get("MATHEXPR_DATA", str(Path(__file__).resolve().parent / "data" / "hwf")) + "/"
)

# Set to True (train.py --task mod10) for the new modulo-10 task with x and / operators.
TASK_3 = False


def custom_collate(batch):
    batch = tuple(zip(*batch))
    return default_collate(batch[0]), batch[1], default_collate(batch[2])


def custom_collate_train_logic_in_loss(batch):
    batch = tuple(zip(*batch))
    return (
        default_collate(batch[0]),
        batch[1],
        default_collate(batch[2]),
        default_collate(batch[3]),
        default_collate(batch[4]),
    )


# Define a transform: convert to grayscale (1 channel), resize to 28×28, then to tensor [0,1]
transform = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),  # adjust if you want original size
        transforms.ToTensor(),  # → tensor, shape [C,H,W], values in [0,1]
    ]
)


def mult_and_div_3imgs_program_logic_in_architecture():
    program_string = """
    result(tensor(images, 0), tensor(images, 1), tensor(images, 2), N) :- digit(tensor(images, 0), N1), operator(tensor(images, 1), plus), digit(tensor(images, 2), N2), add(N1, N2, N).
    result(tensor(images, 0), tensor(images, 1), tensor(images, 2), N) :- digit(tensor(images, 0), N1), operator(tensor(images, 1), minus), digit(tensor(images, 2), N2), subtract(N1, N2, N).
    result(tensor(images, 0), tensor(images, 1), tensor(images, 2), N) :- digit(tensor(images, 0), N1), operator(tensor(images, 1), multiply), digit(tensor(images, 2), N2), multiply(N1, N2, N).
    result(tensor(images, 0), tensor(images, 1), tensor(images, 2), N) :- digit(tensor(images, 0), N1), operator(tensor(images, 1), divide), digit(tensor(images, 2), N2), divide(N1, N2, N).
    """
    program_string += "\n".join(
        [f"add({x}, {y}, {(x + y) % 10})." for x in range(10) for y in range(10)]
    )
    program_string += "\n"
    program_string += "\n".join(
        [f"subtract({x}, {y}, {(x - y) % 10})." for x in range(10) for y in range(10)]
    )
    program_string += "\n"
    program_string += "\n".join(
        [f"multiply({x}, {y}, {(x * y) % 10})." for x in range(10) for y in range(10)]
    )
    program_string += "\n"
    program_string += "\n".join(
        [f"divide({x}, {y}, {(x // y) % 10})." for x in range(10) for y in range(1, 10)]
    )
    program_string += "\n"
    program_string += "\n".join(
        [
            f"nn(digit_nn, tensor(images, 0), {x}) :: digit(tensor(images, 0),{x})."
            for x in range(10)
        ]
    )
    program_string += "\n"
    program_string += (
        "nn(operator_nn, tensor(images, 1), plus) :: operator(tensor(images, 1), plus).\n"
    )
    program_string += (
        "nn(operator_nn, tensor(images, 1), minus) :: operator(tensor(images, 1), minus).\n"
    )
    program_string += (
        "nn(operator_nn, tensor(images, 1), multiply) :: operator(tensor(images, 1), multiply).\n"
    )
    program_string += (
        "nn(operator_nn, tensor(images, 1), divide) :: operator(tensor(images, 1), divide).\n"
    )
    program_string += "\n"
    program_string += "\n".join(
        [
            f"nn(digit_nn, tensor(images, 2), {x}) :: digit(tensor(images, 2),{x})."
            for x in range(10)
        ]
    )
    return program_string


def mult_and_div_3imgs_program_logic_in_loss():
    """Program for logic-in-loss: uses 'correct_computation' as heading, result_head, digit_head, operator_head"""
    program_string = """
    correct_computation(tensor(images, 0), tensor(images, 1), tensor(images, 2), N) :- result(tensor(images, 0), tensor(images, 1), tensor(images, 2), N), digit(tensor(images, 0), N1), operator(tensor(images, 1), plus), digit(tensor(images, 2), N2), add(N1, N2, N).
    correct_computation(tensor(images, 0), tensor(images, 1), tensor(images, 2), N) :- result(tensor(images, 0), tensor(images, 1), tensor(images, 2), N), digit(tensor(images, 0), N1), operator(tensor(images, 1), minus), digit(tensor(images, 2), N2), subtract(N1, N2, N).
    correct_computation(tensor(images, 0), tensor(images, 1), tensor(images, 2), N) :- result(tensor(images, 0), tensor(images, 1), tensor(images, 2), N), digit(tensor(images, 0), N1), operator(tensor(images, 1), multiply), digit(tensor(images, 2), N2), multiply(N1, N2, N).
    correct_computation(tensor(images, 0), tensor(images, 1), tensor(images, 2), N) :- result(tensor(images, 0), tensor(images, 1), tensor(images, 2), N), digit(tensor(images, 0), N1), operator(tensor(images, 1), divide), digit(tensor(images, 2), N2), divide(N1, N2, N).
    """
    program_string += "\n".join(
        [f"add({x}, {y}, {(x + y) % 10})." for x in range(10) for y in range(10)]
    )
    program_string += "\n"
    program_string += "\n".join(
        [f"subtract({x}, {y}, {(x - y) % 10})." for x in range(10) for y in range(10)]
    )
    program_string += "\n"
    program_string += "\n".join(
        [f"multiply({x}, {y}, {(x * y) % 10})." for x in range(10) for y in range(10)]
    )
    program_string += "\n"
    program_string += "\n".join(
        [f"divide({x}, {y}, {(x // y) % 10})." for x in range(10) for y in range(1, 10)]
    )
    program_string += "\n"
    program_string += "\n".join(
        [
            f"nn(result_head, tensor(images, 0), tensor(images, 1), tensor(images, 2), {x}) :: result(tensor(images, 0), tensor(images, 1), tensor(images, 2),{x})."
            for x in range(10)
        ]
    )
    program_string += "\n"
    program_string += "\n".join(
        [
            f"nn(digit_head, tensor(images, 0), {x}) :: digit(tensor(images, 0),{x})."
            for x in range(10)
        ]
    )
    program_string += (
        "nn(operator_head, tensor(images, 1), plus) :: operator(tensor(images, 1), plus).\n"
    )
    program_string += (
        "nn(operator_head, tensor(images, 1), minus) :: operator(tensor(images, 1), minus).\n"
    )
    program_string += (
        "nn(operator_head, tensor(images, 1), multiply) :: operator(tensor(images, 1), multiply).\n"
    )
    program_string += (
        "nn(operator_head, tensor(images, 1), divide) :: operator(tensor(images, 1), divide).\n"
    )
    program_string += "\n"
    program_string += "\n".join(
        [
            f"nn(digit_head, tensor(images, 2), {x}) :: digit(tensor(images, 2),{x})."
            for x in range(10)
        ]
    )
    return program_string


def mult_and_div_5imgs_program_logic_in_architecture():
    """Program for logic-in-architecture: uses 'result' as heading, digit_nn and operator_nn"""
    program_string = """
    result(tensor(images, 0), tensor(images, 1), tensor(images, 2), tensor(images, 3), tensor(images, 4), N) :- digit(tensor(images, 0), N1), operator(tensor(images, 1), plus), digit(tensor(images, 2), N2), operator(tensor(images, 3), plus), digit(tensor(images, 4), N3), add_all(N1, N2, N3, N).
    result(tensor(images, 0), tensor(images, 1), tensor(images, 2), tensor(images, 3), tensor(images, 4), N) :- digit(tensor(images, 0), N1), operator(tensor(images, 1), plus), digit(tensor(images, 2), N2), operator(tensor(images, 3), minus), digit(tensor(images, 4), N3), add_then_subtract(N1, N2, N3, N).
    result(tensor(images, 0), tensor(images, 1), tensor(images, 2), tensor(images, 3), tensor(images, 4), N) :- digit(tensor(images, 0), N1), operator(tensor(images, 1), plus), digit(tensor(images, 2), N2), operator(tensor(images, 3), multiply), digit(tensor(images, 4), N3), add_then_multiply(N1, N2, N3, N).
    result(tensor(images, 0), tensor(images, 1), tensor(images, 2), tensor(images, 3), tensor(images, 4), N) :- digit(tensor(images, 0), N1), operator(tensor(images, 1), plus), digit(tensor(images, 2), N2), operator(tensor(images, 3), divide), digit(tensor(images, 4), N3), add_then_divide(N1, N2, N3, N).
    result(tensor(images, 0), tensor(images, 1), tensor(images, 2), tensor(images, 3), tensor(images, 4), N) :- digit(tensor(images, 0), N1), operator(tensor(images, 1), minus), digit(tensor(images, 2), N2), operator(tensor(images, 3), plus), digit(tensor(images, 4), N3), subtract_then_add(N1, N2, N3, N).
    result(tensor(images, 0), tensor(images, 1), tensor(images, 2), tensor(images, 3), tensor(images, 4), N) :- digit(tensor(images, 0), N1), operator(tensor(images, 1), minus), digit(tensor(images, 2), N2), operator(tensor(images, 3), minus), digit(tensor(images, 4), N3), subtract_all(N1, N2, N3, N).
    result(tensor(images, 0), tensor(images, 1), tensor(images, 2), tensor(images, 3), tensor(images, 4), N) :- digit(tensor(images, 0), N1), operator(tensor(images, 1), minus), digit(tensor(images, 2), N2), operator(tensor(images, 3), multiply), digit(tensor(images, 4), N3), subtract_then_multiply(N1, N2, N3, N).
    result(tensor(images, 0), tensor(images, 1), tensor(images, 2), tensor(images, 3), tensor(images, 4), N) :- digit(tensor(images, 0), N1), operator(tensor(images, 1), minus), digit(tensor(images, 2), N2), operator(tensor(images, 3), divide), digit(tensor(images, 4), N3), subtract_then_divide(N1, N2, N3, N).
    result(tensor(images, 0), tensor(images, 1), tensor(images, 2), tensor(images, 3), tensor(images, 4), N) :- digit(tensor(images, 0), N1), operator(tensor(images, 1), multiply), digit(tensor(images, 2), N2), operator(tensor(images, 3), plus), digit(tensor(images, 4), N3), multiply_then_add(N1, N2, N3, N).
    result(tensor(images, 0), tensor(images, 1), tensor(images, 2), tensor(images, 3), tensor(images, 4), N) :- digit(tensor(images, 0), N1), operator(tensor(images, 1), multiply), digit(tensor(images, 2), N2), operator(tensor(images, 3), minus), digit(tensor(images, 4), N3), multiply_then_subtract(N1, N2, N3, N).
    result(tensor(images, 0), tensor(images, 1), tensor(images, 2), tensor(images, 3), tensor(images, 4), N) :- digit(tensor(images, 0), N1), operator(tensor(images, 1), multiply), digit(tensor(images, 2), N2), operator(tensor(images, 3), multiply), digit(tensor(images, 4), N3), multiply_all(N1, N2, N3, N).
    result(tensor(images, 0), tensor(images, 1), tensor(images, 2), tensor(images, 3), tensor(images, 4), N) :- digit(tensor(images, 0), N1), operator(tensor(images, 1), multiply), digit(tensor(images, 2), N2), operator(tensor(images, 3), divide), digit(tensor(images, 4), N3), multiply_then_divide(N1, N2, N3, N).
    result(tensor(images, 0), tensor(images, 1), tensor(images, 2), tensor(images, 3), tensor(images, 4), N) :- digit(tensor(images, 0), N1), operator(tensor(images, 1), divide), digit(tensor(images, 2), N2), operator(tensor(images, 3), plus), digit(tensor(images, 4), N3), divide_then_add(N1, N2, N3, N).
    result(tensor(images, 0), tensor(images, 1), tensor(images, 2), tensor(images, 3), tensor(images, 4), N) :- digit(tensor(images, 0), N1), operator(tensor(images, 1), divide), digit(tensor(images, 2), N2), operator(tensor(images, 3), minus), digit(tensor(images, 4), N3), divide_then_subtract(N1, N2, N3, N).
    result(tensor(images, 0), tensor(images, 1), tensor(images, 2), tensor(images, 3), tensor(images, 4), N) :- digit(tensor(images, 0), N1), operator(tensor(images, 1), divide), digit(tensor(images, 2), N2), operator(tensor(images, 3), multiply), digit(tensor(images, 4), N3), divide_then_multiply(N1, N2, N3, N).
    result(tensor(images, 0), tensor(images, 1), tensor(images, 2), tensor(images, 3), tensor(images, 4), N) :- digit(tensor(images, 0), N1), operator(tensor(images, 1), divide), digit(tensor(images, 2), N2), operator(tensor(images, 3), divide), digit(tensor(images, 4), N3), divide_all(N1, N2, N3, N).
    """
    # Add facts for all arithmetic operations
    program_string += "\n".join(
        [
            f"add_all({x}, {y}, {z}, {(x + y + z) % 10})."
            for x in range(10)
            for y in range(10)
            for z in range(10)
        ]
    )
    program_string += "\n"
    program_string += "\n".join(
        [
            f"add_then_subtract({x}, {y}, {z}, {(x + y - z) % 10})."
            for x in range(10)
            for y in range(10)
            for z in range(10)
        ]
    )
    program_string += "\n"
    program_string += "\n".join(
        [
            f"add_then_multiply({x}, {y}, {z}, {(x + y * z) % 10})."
            for x in range(10)
            for y in range(10)
            for z in range(10)
        ]
    )
    program_string += "\n"
    program_string += "\n".join(
        [
            f"add_then_divide({x}, {y}, {z}, {(x + y // z) % 10})."
            for x in range(10)
            for y in range(10)
            for z in range(1, 10)
        ]
    )
    program_string += "\n"
    program_string += "\n".join(
        [
            f"subtract_then_add({x}, {y}, {z}, {(x - y + z) % 10})."
            for x in range(10)
            for y in range(10)
            for z in range(10)
        ]
    )
    program_string += "\n"
    program_string += "\n".join(
        [
            f"subtract_all({x}, {y}, {z}, {(x - y - z) % 10})."
            for x in range(10)
            for y in range(10)
            for z in range(10)
        ]
    )
    program_string += "\n"
    program_string += "\n".join(
        [
            f"subtract_then_multiply({x}, {y}, {z}, {(x - y * z) % 10})."
            for x in range(10)
            for y in range(10)
            for z in range(10)
        ]
    )
    program_string += "\n"
    program_string += "\n".join(
        [
            f"subtract_then_divide({x}, {y}, {z}, {(x - y // z) % 10})."
            for x in range(10)
            for y in range(10)
            for z in range(1, 10)
        ]
    )
    program_string += "\n"
    program_string += "\n".join(
        [
            f"multiply_then_add({x}, {y}, {z}, {(x * y + z) % 10})."
            for x in range(10)
            for y in range(10)
            for z in range(10)
        ]
    )
    program_string += "\n"
    program_string += "\n".join(
        [
            f"multiply_then_subtract({x}, {y}, {z}, {(x * y - z) % 10})."
            for x in range(10)
            for y in range(10)
            for z in range(10)
        ]
    )
    program_string += "\n"
    program_string += "\n".join(
        [
            f"multiply_all({x}, {y}, {z}, {(x * y * z) % 10})."
            for x in range(10)
            for y in range(10)
            for z in range(10)
        ]
    )
    program_string += "\n"
    program_string += "\n".join(
        [
            f"multiply_then_divide({x}, {y}, {z}, {(x * y // z) % 10})."
            for x in range(10)
            for y in range(10)
            for z in range(1, 10)
        ]
    )
    program_string += "\n"
    program_string += "\n".join(
        [
            f"divide_then_add({x}, {y}, {z}, {(x // y + z) % 10})."
            for x in range(10)
            for y in range(1, 10)
            for z in range(10)
        ]
    )
    program_string += "\n"
    program_string += "\n".join(
        [
            f"divide_then_subtract({x}, {y}, {z}, {(x // y - z) % 10})."
            for x in range(10)
            for y in range(1, 10)
            for z in range(10)
        ]
    )
    program_string += "\n"
    program_string += "\n".join(
        [
            f"divide_then_multiply({x}, {y}, {z}, {(x // y * z) % 10})."
            for x in range(10)
            for y in range(1, 10)
            for z in range(10)
        ]
    )
    program_string += "\n"
    program_string += "\n".join(
        [
            f"divide_all({x}, {y}, {z}, {(x // y // z) % 10})."
            for x in range(10)
            for y in range(1, 10)
            for z in range(1, 10)
        ]
    )
    program_string += "\n"
    # Neural network definitions for digits
    program_string += "\n".join(
        [
            f"nn(digit_nn, tensor(images, 0), {x}) :: digit(tensor(images, 0),{x})."
            for x in range(10)
        ]
    )
    program_string += "\n"
    program_string += "\n".join(
        [
            f"nn(digit_nn, tensor(images, 2), {x}) :: digit(tensor(images, 2),{x})."
            for x in range(10)
        ]
    )
    program_string += "\n"
    program_string += "\n".join(
        [
            f"nn(digit_nn, tensor(images, 4), {x}) :: digit(tensor(images, 4),{x})."
            for x in range(10)
        ]
    )
    program_string += "\n"
    # Neural network definitions for operators at position 1
    program_string += (
        "nn(operator_nn, tensor(images, 1), plus) :: operator(tensor(images, 1), plus).\n"
    )
    program_string += (
        "nn(operator_nn, tensor(images, 1), minus) :: operator(tensor(images, 1), minus).\n"
    )
    program_string += (
        "nn(operator_nn, tensor(images, 1), multiply) :: operator(tensor(images, 1), multiply).\n"
    )
    program_string += (
        "nn(operator_nn, tensor(images, 1), divide) :: operator(tensor(images, 1), divide).\n"
    )
    program_string += "\n"
    # Neural network definitions for operators at position 3
    program_string += (
        "nn(operator_nn, tensor(images, 3), plus) :: operator(tensor(images, 3), plus).\n"
    )
    program_string += (
        "nn(operator_nn, tensor(images, 3), minus) :: operator(tensor(images, 3), minus).\n"
    )
    program_string += (
        "nn(operator_nn, tensor(images, 3), multiply) :: operator(tensor(images, 3), multiply).\n"
    )
    program_string += (
        "nn(operator_nn, tensor(images, 3), divide) :: operator(tensor(images, 3), divide).\n"
    )
    program_string += "\n"
    return program_string


def mult_and_div_5imgs_program_logic_in_loss():
    """Program for logic-in-loss: uses 'correct_computation' as heading, result_head, digit_head, operator_head"""
    program_string = """
    correct_computation(tensor(images, 0), tensor(images, 1), tensor(images, 2), tensor(images, 3), tensor(images, 4), N) :- result(tensor(images, 0), tensor(images, 1), tensor(images, 2), tensor(images, 3), tensor(images, 4), N), digit(tensor(images, 0), N1), operator(tensor(images, 1), plus), digit(tensor(images, 2), N2), operator(tensor(images, 3), plus), digit(tensor(images, 4), N3), add_all(N1, N2, N3, N).
    correct_computation(tensor(images, 0), tensor(images, 1), tensor(images, 2), tensor(images, 3), tensor(images, 4), N) :- result(tensor(images, 0), tensor(images, 1), tensor(images, 2), tensor(images, 3), tensor(images, 4), N), digit(tensor(images, 0), N1), operator(tensor(images, 1), plus), digit(tensor(images, 2), N2), operator(tensor(images, 3), minus), digit(tensor(images, 4), N3), add_then_subtract(N1, N2, N3, N).
    correct_computation(tensor(images, 0), tensor(images, 1), tensor(images, 2), tensor(images, 3), tensor(images, 4), N) :- result(tensor(images, 0), tensor(images, 1), tensor(images, 2), tensor(images, 3), tensor(images, 4), N), digit(tensor(images, 0), N1), operator(tensor(images, 1), plus), digit(tensor(images, 2), N2), operator(tensor(images, 3), multiply), digit(tensor(images, 4), N3), add_then_multiply(N1, N2, N3, N).
    correct_computation(tensor(images, 0), tensor(images, 1), tensor(images, 2), tensor(images, 3), tensor(images, 4), N) :- result(tensor(images, 0), tensor(images, 1), tensor(images, 2), tensor(images, 3), tensor(images, 4), N), digit(tensor(images, 0), N1), operator(tensor(images, 1), plus), digit(tensor(images, 2), N2), operator(tensor(images, 3), divide), digit(tensor(images, 4), N3), add_then_divide(N1, N2, N3, N).
    correct_computation(tensor(images, 0), tensor(images, 1), tensor(images, 2), tensor(images, 3), tensor(images, 4), N) :- result(tensor(images, 0), tensor(images, 1), tensor(images, 2), tensor(images, 3), tensor(images, 4), N), digit(tensor(images, 0), N1), operator(tensor(images, 1), minus), digit(tensor(images, 2), N2), operator(tensor(images, 3), plus), digit(tensor(images, 4), N3), subtract_then_add(N1, N2, N3, N).
    correct_computation(tensor(images, 0), tensor(images, 1), tensor(images, 2), tensor(images, 3), tensor(images, 4), N) :- result(tensor(images, 0), tensor(images, 1), tensor(images, 2), tensor(images, 3), tensor(images, 4), N), digit(tensor(images, 0), N1), operator(tensor(images, 1), minus), digit(tensor(images, 2), N2), operator(tensor(images, 3), minus), digit(tensor(images, 4), N3), subtract_all(N1, N2, N3, N).
    correct_computation(tensor(images, 0), tensor(images, 1), tensor(images, 2), tensor(images, 3), tensor(images, 4), N) :- result(tensor(images, 0), tensor(images, 1), tensor(images, 2), tensor(images, 3), tensor(images, 4), N), digit(tensor(images, 0), N1), operator(tensor(images, 1), minus), digit(tensor(images, 2), N2), operator(tensor(images, 3), multiply), digit(tensor(images, 4), N3), subtract_then_multiply(N1, N2, N3, N).
    correct_computation(tensor(images, 0), tensor(images, 1), tensor(images, 2), tensor(images, 3), tensor(images, 4), N) :- result(tensor(images, 0), tensor(images, 1), tensor(images, 2), tensor(images, 3), tensor(images, 4), N), digit(tensor(images, 0), N1), operator(tensor(images, 1), minus), digit(tensor(images, 2), N2), operator(tensor(images, 3), divide), digit(tensor(images, 4), N3), subtract_then_divide(N1, N2, N3, N).
    correct_computation(tensor(images, 0), tensor(images, 1), tensor(images, 2), tensor(images, 3), tensor(images, 4), N) :- result(tensor(images, 0), tensor(images, 1), tensor(images, 2), tensor(images, 3), tensor(images, 4), N), digit(tensor(images, 0), N1), operator(tensor(images, 1), multiply), digit(tensor(images, 2), N2), operator(tensor(images, 3), plus), digit(tensor(images, 4), N3), multiply_then_add(N1, N2, N3, N).
    correct_computation(tensor(images, 0), tensor(images, 1), tensor(images, 2), tensor(images, 3), tensor(images, 4), N) :- result(tensor(images, 0), tensor(images, 1), tensor(images, 2), tensor(images, 3), tensor(images, 4), N), digit(tensor(images, 0), N1), operator(tensor(images, 1), multiply), digit(tensor(images, 2), N2), operator(tensor(images, 3), minus), digit(tensor(images, 4), N3), multiply_then_subtract(N1, N2, N3, N).
    correct_computation(tensor(images, 0), tensor(images, 1), tensor(images, 2), tensor(images, 3), tensor(images, 4), N) :- result(tensor(images, 0), tensor(images, 1), tensor(images, 2), tensor(images, 3), tensor(images, 4), N), digit(tensor(images, 0), N1), operator(tensor(images, 1), multiply), digit(tensor(images, 2), N2), operator(tensor(images, 3), multiply), digit(tensor(images, 4), N3), multiply_all(N1, N2, N3, N).
    correct_computation(tensor(images, 0), tensor(images, 1), tensor(images, 2), tensor(images, 3), tensor(images, 4), N) :- result(tensor(images, 0), tensor(images, 1), tensor(images, 2), tensor(images, 3), tensor(images, 4), N), digit(tensor(images, 0), N1), operator(tensor(images, 1), multiply), digit(tensor(images, 2), N2), operator(tensor(images, 3), divide), digit(tensor(images, 4), N3), multiply_then_divide(N1, N2, N3, N).
    correct_computation(tensor(images, 0), tensor(images, 1), tensor(images, 2), tensor(images, 3), tensor(images, 4), N) :- result(tensor(images, 0), tensor(images, 1), tensor(images, 2), tensor(images, 3), tensor(images, 4), N), digit(tensor(images, 0), N1), operator(tensor(images, 1), divide), digit(tensor(images, 2), N2), operator(tensor(images, 3), plus), digit(tensor(images, 4), N3), divide_then_add(N1, N2, N3, N).
    correct_computation(tensor(images, 0), tensor(images, 1), tensor(images, 2), tensor(images, 3), tensor(images, 4), N) :- result(tensor(images, 0), tensor(images, 1), tensor(images, 2), tensor(images, 3), tensor(images, 4), N), digit(tensor(images, 0), N1), operator(tensor(images, 1), divide), digit(tensor(images, 2), N2), operator(tensor(images, 3), minus), digit(tensor(images, 4), N3), divide_then_subtract(N1, N2, N3, N).
    correct_computation(tensor(images, 0), tensor(images, 1), tensor(images, 2), tensor(images, 3), tensor(images, 4), N) :- result(tensor(images, 0), tensor(images, 1), tensor(images, 2), tensor(images, 3), tensor(images, 4), N), digit(tensor(images, 0), N1), operator(tensor(images, 1), divide), digit(tensor(images, 2), N2), operator(tensor(images, 3), multiply), digit(tensor(images, 4), N3), divide_then_multiply(N1, N2, N3, N).
    correct_computation(tensor(images, 0), tensor(images, 1), tensor(images, 2), tensor(images, 3), tensor(images, 4), N) :- result(tensor(images, 0), tensor(images, 1), tensor(images, 2), tensor(images, 3), tensor(images, 4), N), digit(tensor(images, 0), N1), operator(tensor(images, 1), divide), digit(tensor(images, 2), N2), operator(tensor(images, 3), divide), digit(tensor(images, 4), N3), divide_all(N1, N2, N3, N).
    """
    # Add facts for all arithmetic operations
    program_string += "\n".join(
        [
            f"add_all({x}, {y}, {z}, {(x + y + z) % 10})."
            for x in range(10)
            for y in range(10)
            for z in range(10)
        ]
    )
    program_string += "\n"
    program_string += "\n".join(
        [
            f"add_then_subtract({x}, {y}, {z}, {(x + y - z) % 10})."
            for x in range(10)
            for y in range(10)
            for z in range(10)
        ]
    )
    program_string += "\n"
    program_string += "\n".join(
        [
            f"add_then_multiply({x}, {y}, {z}, {(x + y * z) % 10})."
            for x in range(10)
            for y in range(10)
            for z in range(10)
        ]
    )
    program_string += "\n"
    program_string += "\n".join(
        [
            f"add_then_divide({x}, {y}, {z}, {(x + y // z) % 10})."
            for x in range(10)
            for y in range(10)
            for z in range(1, 10)
        ]
    )
    program_string += "\n"
    program_string += "\n".join(
        [
            f"subtract_then_add({x}, {y}, {z}, {(x - y + z) % 10})."
            for x in range(10)
            for y in range(10)
            for z in range(10)
        ]
    )
    program_string += "\n"
    program_string += "\n".join(
        [
            f"subtract_all({x}, {y}, {z}, {(x - y - z) % 10})."
            for x in range(10)
            for y in range(10)
            for z in range(10)
        ]
    )
    program_string += "\n"
    program_string += "\n".join(
        [
            f"subtract_then_multiply({x}, {y}, {z}, {(x - y * z) % 10})."
            for x in range(10)
            for y in range(10)
            for z in range(10)
        ]
    )
    program_string += "\n"
    program_string += "\n".join(
        [
            f"subtract_then_divide({x}, {y}, {z}, {(x - y // z) % 10})."
            for x in range(10)
            for y in range(10)
            for z in range(1, 10)
        ]
    )
    program_string += "\n"
    program_string += "\n".join(
        [
            f"multiply_then_add({x}, {y}, {z}, {(x * y + z) % 10})."
            for x in range(10)
            for y in range(10)
            for z in range(10)
        ]
    )
    program_string += "\n"
    program_string += "\n".join(
        [
            f"multiply_then_subtract({x}, {y}, {z}, {(x * y - z) % 10})."
            for x in range(10)
            for y in range(10)
            for z in range(10)
        ]
    )
    program_string += "\n"
    program_string += "\n".join(
        [
            f"multiply_all({x}, {y}, {z}, {(x * y * z) % 10})."
            for x in range(10)
            for y in range(10)
            for z in range(10)
        ]
    )
    program_string += "\n"
    program_string += "\n".join(
        [
            f"multiply_then_divide({x}, {y}, {z}, {(x * y // z) % 10})."
            for x in range(10)
            for y in range(10)
            for z in range(1, 10)
        ]
    )
    program_string += "\n"
    program_string += "\n".join(
        [
            f"divide_then_add({x}, {y}, {z}, {(x // y + z) % 10})."
            for x in range(10)
            for y in range(1, 10)
            for z in range(10)
        ]
    )
    program_string += "\n"
    program_string += "\n".join(
        [
            f"divide_then_subtract({x}, {y}, {z}, {(x // y - z) % 10})."
            for x in range(10)
            for y in range(1, 10)
            for z in range(10)
        ]
    )
    program_string += "\n"
    program_string += "\n".join(
        [
            f"divide_then_multiply({x}, {y}, {z}, {(x // y * z) % 10})."
            for x in range(10)
            for y in range(1, 10)
            for z in range(10)
        ]
    )
    program_string += "\n"
    program_string += "\n".join(
        [
            f"divide_all({x}, {y}, {z}, {(x // y // z) % 10})."
            for x in range(10)
            for y in range(1, 10)
            for z in range(1, 10)
        ]
    )
    program_string += "\n"
    program_string += "\n".join(
        [
            f"nn(result_head, tensor(images, 0), tensor(images, 1), tensor(images, 2), tensor(images, 3), tensor(images, 4), {x}) :: result(tensor(images, 0), tensor(images, 1), tensor(images, 2), tensor(images, 3), tensor(images, 4) ,{x})."
            for x in range(10)
        ]
    )
    program_string += "\n"
    program_string += "\n".join(
        [
            f"nn(digit_head, tensor(images, 0), {x}) :: digit(tensor(images, 0),{x})."
            for x in range(10)
        ]
    )
    program_string += "\n"
    program_string += "\n".join(
        [
            f"nn(digit_head, tensor(images, 2), {x}) :: digit(tensor(images, 2),{x})."
            for x in range(10)
        ]
    )
    program_string += "\n"
    program_string += "\n".join(
        [
            f"nn(digit_head, tensor(images, 4), {x}) :: digit(tensor(images, 4),{x})."
            for x in range(10)
        ]
    )
    program_string += "\n"
    program_string += (
        "nn(operator_head, tensor(images, 1), plus) :: operator(tensor(images, 1), plus).\n"
    )
    program_string += (
        "nn(operator_head, tensor(images, 1), minus) :: operator(tensor(images, 1), minus).\n"
    )
    program_string += (
        "nn(operator_head, tensor(images, 1), multiply) :: operator(tensor(images, 1), multiply).\n"
    )
    program_string += (
        "nn(operator_head, tensor(images, 1), divide) :: operator(tensor(images, 1), divide).\n"
    )
    program_string += "\n"
    program_string += (
        "nn(operator_head, tensor(images, 3), plus) :: operator(tensor(images, 3), plus).\n"
    )
    program_string += (
        "nn(operator_head, tensor(images, 3), minus) :: operator(tensor(images, 3), minus).\n"
    )
    program_string += (
        "nn(operator_head, tensor(images, 3), multiply) :: operator(tensor(images, 3), multiply).\n"
    )
    program_string += (
        "nn(operator_head, tensor(images, 3), divide) :: operator(tensor(images, 3), divide).\n"
    )
    program_string += "\n"
    return program_string


class MathExpressionTaskNeural(Dataset):
    def __init__(self, train=True, split="train", nr_images_per_example=3, nr_examples=None):
        # Load the example images
        self.examples = []
        self.split = split
        self.train = train
        self.nr_images_per_example = nr_images_per_example
        if nr_images_per_example == 3:
            self.target_shift = 9
            file_name = (
                "math_expression_examples_size_3.json"
                if not TASK_3
                else "math_expression_examples_size_3_task_3.json"
            )
        elif nr_images_per_example == 5:
            self.target_shift = 18
            file_name = (
                "math_expression_examples_size_5.json"
                if not TASK_3
                else "math_expression_examples_size_5_task_3.json"
            )
        else:
            raise ValueError("nr_images_per_example must be 3 or 5")
        with open(DATA_PATH + file_name, "r", encoding="utf-8") as f:
            examples_json = json.load(f)
        if nr_examples is not None:
            if nr_examples > len(examples_json[split]):
                raise ValueError(
                    "nr_examples exceeds to number of available examples in this dataset"
                )
        else:
            nr_examples = len(examples_json[split])
        self.nr_examples = nr_examples
        for example in examples_json[split]:
            if len(self.examples) >= self.nr_examples:
                break
            images = [transform(Image.open(DATA_PATH + p).convert("RGB")) for p in example["paths"]]
            self.examples.append(
                (torch.cat(images, dim=0), example["result"], example["expression"])
            )

    def __getitem__(self, index):
        images = self.examples[index][0]
        target = self.examples[index][1]

        return images, target + self.target_shift  # Shift target to be non-negative

    def dataloader(self, batch_size=2, shuffle=None, num_workers=0):
        if shuffle is None:
            shuffle = self.split == "train"
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=default_collate,
            num_workers=num_workers,
        )

    def __len__(self):
        return len(self.examples)


class MathExpressionTaskLogicInArchitecture(Dataset):
    def __init__(self, train=True, split="train", nr_images_per_example=3, nr_examples=None):
        # Load the example images
        self.examples = []
        self.split = split
        self.train = train
        self.nr_images_per_example = nr_images_per_example

        if nr_images_per_example == 3:
            self.target_shift = 9
            file_name = (
                "math_expression_examples_size_3.json"
                if not TASK_3
                else "math_expression_examples_size_3_task_3.json"
            )
        elif nr_images_per_example == 5:
            self.target_shift = 18
            file_name = (
                "math_expression_examples_size_5.json"
                if not TASK_3
                else "math_expression_examples_size_5_task_3.json"
            )
        else:
            raise ValueError("nr_images_per_example must be 3 or 5")
        with open(DATA_PATH + file_name, "r", encoding="utf-8") as f:
            examples_json = json.load(f)

        if nr_examples is not None:
            if nr_examples > len(examples_json[split]):
                raise ValueError(
                    "nr_examples exceeds to number of available examples in this dataset"
                )
        else:
            nr_examples = len(examples_json[split])
        self.nr_examples = nr_examples

        for example in examples_json[split]:
            if len(self.examples) >= self.nr_examples:
                break
            images = [transform(Image.open(DATA_PATH + p).convert("RGB")) for p in example["paths"]]
            self.examples.append((torch.stack(images), example["result"], example["expression"]))

        if TASK_3:
            if nr_images_per_example == 3:
                self.program = parse_program(mult_and_div_3imgs_program_logic_in_architecture())
            else:
                self.program = parse_program(mult_and_div_5imgs_program_logic_in_architecture())
            return

        # Define program
        if nr_images_per_example == 3:
            program_string = """
            result(tensor(images, 0), tensor(images, 1), tensor(images, 2), N) :- digit(tensor(images, 0), N1), operator(tensor(images, 1), plus), digit(tensor(images, 2), N2), add(N1, N2, N).
            result(tensor(images, 0), tensor(images, 1), tensor(images, 2), N) :- digit(tensor(images, 0), N1), operator(tensor(images, 1), minus), digit(tensor(images, 2), N2), subtract(N1, N2, N).
            """
            program_string += "\n".join(
                [f"add({x}, {y}, {x + y})." for x in range(10) for y in range(10)]
            )
            program_string += "\n"
            program_string += "\n".join(
                [f"subtract({x}, {y}, {x - y})." for x in range(10) for y in range(10)]
            )
            program_string += "\n"
            program_string += "\n".join(
                [
                    f"nn(digit_nn, tensor(images, 0), {x}) :: digit(tensor(images, 0),{x})."
                    for x in range(10)
                ]
            )
            program_string += (
                "nn(operator_nn, tensor(images, 1), plus) :: operator(tensor(images, 1), plus).\n"
            )
            program_string += (
                "nn(operator_nn, tensor(images, 1), minus) :: operator(tensor(images, 1), minus).\n"
            )
            program_string += "\n"
            program_string += "\n".join(
                [
                    f"nn(digit_nn, tensor(images, 2), {x}) :: digit(tensor(images, 2),{x})."
                    for x in range(10)
                ]
            )

        elif nr_images_per_example == 5:
            program_string = """
            result(tensor(images, 0), tensor(images, 1), tensor(images, 2), tensor(images, 3), tensor(images, 4), N) :- digit(tensor(images, 0), N1), operator(tensor(images, 1), plus), digit(tensor(images, 2), N2), operator(tensor(images, 3), plus), digit(tensor(images, 4), N3), add_all(N1, N2, N3, N).
            result(tensor(images, 0), tensor(images, 1), tensor(images, 2), tensor(images, 3), tensor(images, 4), N) :- digit(tensor(images, 0), N1), operator(tensor(images, 1), plus), digit(tensor(images, 2), N2), operator(tensor(images, 3), minus), digit(tensor(images, 4), N3), add_then_subtract(N1, N2, N3, N).
            result(tensor(images, 0), tensor(images, 1), tensor(images, 2), tensor(images, 3), tensor(images, 4), N) :- digit(tensor(images, 0), N1), operator(tensor(images, 1), minus), digit(tensor(images, 2), N2), operator(tensor(images, 3), plus), digit(tensor(images, 4), N3), subtract_then_add(N1, N2, N3, N).
            result(tensor(images, 0), tensor(images, 1), tensor(images, 2), tensor(images, 3), tensor(images, 4), N) :- digit(tensor(images, 0), N1), operator(tensor(images, 1), minus), digit(tensor(images, 2), N2), operator(tensor(images, 3), minus), digit(tensor(images, 4), N3), subtract_all(N1, N2, N3, N).
            """
            program_string += "\n".join(
                [
                    f"add_all({x}, {y}, {z}, {x + y + z})."
                    for x in range(10)
                    for y in range(10)
                    for z in range(10)
                ]
            )
            program_string += "\n"
            program_string += "\n".join(
                [
                    f"add_then_subtract({x}, {y}, {z}, {x + y - z})."
                    for x in range(10)
                    for y in range(10)
                    for z in range(10)
                ]
            )
            program_string += "\n"
            program_string += "\n".join(
                [
                    f"subtract_then_add({x}, {y}, {z}, {x - y + z})."
                    for x in range(10)
                    for y in range(10)
                    for z in range(10)
                ]
            )
            program_string += "\n"
            program_string += "\n".join(
                [
                    f"subtract_all({x}, {y}, {z}, {x - y - z})."
                    for x in range(10)
                    for y in range(10)
                    for z in range(10)
                ]
            )
            program_string += "\n"
            program_string += "\n".join(
                [
                    f"nn(digit_nn, tensor(images, 0), {x}) :: digit(tensor(images, 0),{x})."
                    for x in range(10)
                ]
            )
            program_string += "\n"
            program_string += "\n".join(
                [
                    f"nn(digit_nn, tensor(images, 2), {x}) :: digit(tensor(images, 2),{x})."
                    for x in range(10)
                ]
            )
            program_string += "\n"
            program_string += "\n".join(
                [
                    f"nn(digit_nn, tensor(images, 4), {x}) :: digit(tensor(images, 4),{x})."
                    for x in range(10)
                ]
            )
            program_string += "\n"
            program_string += (
                "nn(operator_nn, tensor(images, 1), plus) :: operator(tensor(images, 1), plus).\n"
            )
            program_string += (
                "nn(operator_nn, tensor(images, 1), minus) :: operator(tensor(images, 1), minus).\n"
            )
            program_string += "\n"
            program_string += (
                "nn(operator_nn, tensor(images, 3), plus) :: operator(tensor(images, 3), plus).\n"
            )
            program_string += (
                "nn(operator_nn, tensor(images, 3), minus) :: operator(tensor(images, 3), minus).\n"
            )
        self.program = parse_program(program_string)

    def __getitem__(self, index):
        images = self.examples[index][0]
        target = self.examples[index][1]

        if self.train:
            # In MathExpression, training for a single example (a tuple of images) requires a query for the correct result.
            # The probability of that query is maximized during training.
            if self.nr_images_per_example == 3:
                query = parse_program(
                    "result(tensor(images, 0), tensor(images,1), tensor(images,2), {}).".format(
                        target
                    )
                )[0].term
            elif self.nr_images_per_example == 5:
                query = parse_program(
                    "result(tensor(images, 0), tensor(images,1), tensor(images,2), tensor(images,3), tensor(images,4), {}).".format(
                        target
                    )
                )[0].term
            tensor_sources = {"images": images}
            return tensor_sources, query, torch.tensor([1.0])

        else:
            # In MathExpression, testing for a single example (a tuple of images) requires queries for all possible sums and subtractions of two digits.
            # In this way, we can compute the most probable result.
            # For TASK_3, results are modulo 10 (0-9), otherwise range is -9 to 18 for 3 images, -18 to 27 for 5 images.
            if TASK_3:
                # TASK_3: modulo 10 arithmetic, so only 0-9 are possible
                if self.nr_images_per_example == 3:
                    queries = [
                        parse_program(
                            "result(tensor(images, 0), tensor(images,1), tensor(images,2), {}).".format(
                                z
                            )
                        )[0].term
                        for z in range(10)
                    ]
                elif self.nr_images_per_example == 5:
                    queries = [
                        parse_program(
                            "result(tensor(images, 0), tensor(images,1), tensor(images,2), tensor(images,3), tensor(images,4), {}).".format(
                                z
                            )
                        )[0].term
                        for z in range(10)
                    ]
                tensor_sources = {"images": images}
                return tensor_sources, queries, target  # No shift needed for TASK_3 (0-9)
            else:
                # Original task: plus/minus only
                if self.nr_images_per_example == 3:
                    queries = [
                        parse_program(
                            "result(tensor(images, 0), tensor(images,1), tensor(images,2), {}).".format(
                                z
                            )
                        )[0].term
                        for z in range(-9, 19)
                    ]
                elif self.nr_images_per_example == 5:
                    queries = [
                        parse_program(
                            "result(tensor(images, 0), tensor(images,1), tensor(images,2), tensor(images,3), tensor(images,4), {}).".format(
                                z
                            )
                        )[0].term
                        for z in range(-18, 28)
                    ]
                tensor_sources = {"images": images}
                return (
                    tensor_sources,
                    queries,
                    target + self.target_shift,
                )  # Shift target to be non-negative due to encoding validation step

    def dataloader(self, batch_size=2, shuffle=None, num_workers=0):
        if shuffle is None:
            shuffle = self.split == "train"

        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=custom_collate,
            num_workers=num_workers,
        )

    def __len__(self):
        return len(self.examples)


class MathExpressionTaskLogicInLoss(Dataset):
    def __init__(self, train=True, split="train", nr_images_per_example=3, nr_examples=None):
        # Load the example images
        self.examples = []
        self.split = split
        self.train = train
        self.nr_images_per_example = nr_images_per_example

        if nr_images_per_example == 3:
            self.target_shift = 9
            file_name = (
                "math_expression_examples_size_3.json"
                if not TASK_3
                else "math_expression_examples_size_3_task_3.json"
            )
        elif nr_images_per_example == 5:
            self.target_shift = 18
            file_name = (
                "math_expression_examples_size_5.json"
                if not TASK_3
                else "math_expression_examples_size_5_task_3.json"
            )
        else:
            raise ValueError("nr_images_per_example must be 3 or 5")
        with open(DATA_PATH + file_name, "r", encoding="utf-8") as f:
            examples_json = json.load(f)

        if nr_examples is not None:
            if nr_examples > len(examples_json[split]):
                raise ValueError(
                    "nr_examples exceeds to number of available examples in this dataset"
                )
        else:
            nr_examples = len(examples_json[split])
        self.nr_examples = nr_examples

        for example in examples_json[split]:
            if len(self.examples) >= self.nr_examples:
                break
            images = [transform(Image.open(DATA_PATH + p).convert("RGB")) for p in example["paths"]]
            self.examples.append((torch.stack(images), example["result"], example["expression"]))

        # Define program
        if TASK_3:
            if nr_images_per_example == 3:
                self.program = parse_program(mult_and_div_3imgs_program_logic_in_loss())
            else:
                self.program = parse_program(mult_and_div_5imgs_program_logic_in_loss())
            return

        if nr_images_per_example == 3:
            program_string = """
            correct_computation(tensor(images, 0), tensor(images, 1), tensor(images, 2), N) :- result(tensor(images, 0), tensor(images, 1), tensor(images, 2), N), digit(tensor(images, 0), N1), operator(tensor(images, 1), plus), digit(tensor(images, 2), N2), add(N1, N2, N).
            correct_computation(tensor(images, 0), tensor(images, 1), tensor(images, 2), N) :- result(tensor(images, 0), tensor(images, 1), tensor(images, 2), N), digit(tensor(images, 0), N1), operator(tensor(images, 1), minus), digit(tensor(images, 2), N2), subtract(N1, N2, N).
            """
            program_string += "\n".join(
                [f"add({x}, {y}, {x + y})." for x in range(10) for y in range(10)]
            )
            program_string += "\n"
            program_string += "\n".join(
                [f"subtract({x}, {y}, {x - y})." for x in range(10) for y in range(10)]
            )
            program_string += "\n"
            program_string += "\n".join(
                [
                    f"nn(result_head, tensor(images, 0), tensor(images, 1), tensor(images, 2), {x}) :: result(tensor(images, 0), tensor(images, 1), tensor(images, 2),{x})."
                    for x in range(-9, 19)
                ]
            )
            program_string += "\n"
            program_string += "\n".join(
                [
                    f"nn(digit_head, tensor(images, 0), {x}) :: digit(tensor(images, 0),{x})."
                    for x in range(10)
                ]
            )
            program_string += (
                "nn(operator_head, tensor(images, 1), plus) :: operator(tensor(images, 1), plus).\n"
            )
            program_string += "nn(operator_head, tensor(images, 1), minus) :: operator(tensor(images, 1), minus).\n"
            program_string += "\n"
            program_string += "\n".join(
                [
                    f"nn(digit_head, tensor(images, 2), {x}) :: digit(tensor(images, 2),{x})."
                    for x in range(10)
                ]
            )
        elif nr_images_per_example == 5:
            program_string = """
            correct_computation(tensor(images, 0), tensor(images, 1), tensor(images, 2), tensor(images, 3), tensor(images, 4), N) :- result(tensor(images, 0), tensor(images, 1), tensor(images, 2), tensor(images, 3), tensor(images, 4), N), digit(tensor(images, 0), N1), operator(tensor(images, 1), plus), digit(tensor(images, 2), N2), operator(tensor(images, 3), plus), digit(tensor(images, 4), N3), add_all(N1, N2, N3, N).
            correct_computation(tensor(images, 0), tensor(images, 1), tensor(images, 2), tensor(images, 3), tensor(images, 4), N) :- result(tensor(images, 0), tensor(images, 1), tensor(images, 2), tensor(images, 3), tensor(images, 4), N), digit(tensor(images, 0), N1), operator(tensor(images, 1), plus), digit(tensor(images, 2), N2), operator(tensor(images, 3), minus), digit(tensor(images, 4), N3), add_then_subtract(N1, N2, N3, N).
            correct_computation(tensor(images, 0), tensor(images, 1), tensor(images, 2), tensor(images, 3), tensor(images, 4), N) :- result(tensor(images, 0), tensor(images, 1), tensor(images, 2), tensor(images, 3), tensor(images, 4), N), digit(tensor(images, 0), N1), operator(tensor(images, 1), minus), digit(tensor(images, 2), N2), operator(tensor(images, 3), plus), digit(tensor(images, 4), N3), subtract_then_add(N1, N2, N3, N).
            correct_computation(tensor(images, 0), tensor(images, 1), tensor(images, 2), tensor(images, 3), tensor(images, 4), N) :- result(tensor(images, 0), tensor(images, 1), tensor(images, 2), tensor(images, 3), tensor(images, 4), N), digit(tensor(images, 0), N1), operator(tensor(images, 1), minus), digit(tensor(images, 2), N2), operator(tensor(images, 3), minus), digit(tensor(images, 4), N3), subtract_all(N1, N2, N3, N).
            """
            program_string += "\n".join(
                [
                    f"add_all({x}, {y}, {z}, {x + y + z})."
                    for x in range(10)
                    for y in range(10)
                    for z in range(10)
                ]
            )
            program_string += "\n"
            program_string += "\n".join(
                [
                    f"add_then_subtract({x}, {y}, {z}, {x + y - z})."
                    for x in range(10)
                    for y in range(10)
                    for z in range(10)
                ]
            )
            program_string += "\n"
            program_string += "\n".join(
                [
                    f"subtract_then_add({x}, {y}, {z}, {x - y + z})."
                    for x in range(10)
                    for y in range(10)
                    for z in range(10)
                ]
            )
            program_string += "\n"
            program_string += "\n".join(
                [
                    f"subtract_all({x}, {y}, {z}, {x - y - z})."
                    for x in range(10)
                    for y in range(10)
                    for z in range(10)
                ]
            )
            program_string += "\n"
            program_string += "\n".join(
                [
                    f"nn(result_head, tensor(images, 0), tensor(images, 1), tensor(images, 2), tensor(images, 3), tensor(images, 4), {x}) :: result(tensor(images, 0), tensor(images, 1), tensor(images, 2), tensor(images, 3), tensor(images, 4) ,{x})."
                    for x in range(-18, 28)
                ]
            )
            program_string += "\n"
            program_string += "\n".join(
                [
                    f"nn(digit_head, tensor(images, 0), {x}) :: digit(tensor(images, 0),{x})."
                    for x in range(10)
                ]
            )
            program_string += "\n"
            program_string += "\n".join(
                [
                    f"nn(digit_head, tensor(images, 2), {x}) :: digit(tensor(images, 2),{x})."
                    for x in range(10)
                ]
            )
            program_string += "\n"
            program_string += "\n".join(
                [
                    f"nn(digit_head, tensor(images, 4), {x}) :: digit(tensor(images, 4),{x})."
                    for x in range(10)
                ]
            )
            program_string += "\n"
            program_string += (
                "nn(operator_head, tensor(images, 1), plus) :: operator(tensor(images, 1), plus).\n"
            )
            program_string += "nn(operator_head, tensor(images, 1), minus) :: operator(tensor(images, 1), minus).\n"
            program_string += "\n"
            program_string += (
                "nn(operator_head, tensor(images, 3), plus) :: operator(tensor(images, 3), plus).\n"
            )
            program_string += "nn(operator_head, tensor(images, 3), minus) :: operator(tensor(images, 3), minus).\n"
        self.program = parse_program(program_string)

    def __getitem__(self, index):
        images = self.examples[index][0]
        target = self.examples[index][1]

        if self.train:
            # In MathExpression, training for a single example (a tuple of images) requires a query for the correct result.
            # The probability of that query is maximized during training.
            if self.nr_images_per_example == 3:
                query = parse_program(
                    "correct_computation(tensor(images, 0), tensor(images,1), tensor(images,2), {}).".format(
                        target
                    )
                )[0].term
            elif self.nr_images_per_example == 5:
                query = parse_program(
                    "correct_computation(tensor(images, 0), tensor(images,1), tensor(images,2), tensor(images,3), tensor(images,4), {}).".format(
                        target
                    )
                )[0].term
            tensor_sources = {"images": images}
            return (
                tensor_sources,
                query,
                torch.tensor([1.0]),
                images,
                {"result_head": target + self.target_shift},
            )

        else:
            return images, {
                "result_head": target + self.target_shift
            }  # Shift target to be non-negative due to encoding validation step

    def dataloader(self, batch_size=2, shuffle=None, num_workers=0):
        if self.train:
            shuffle = True if shuffle is None else shuffle
            return DataLoader(
                self,
                batch_size=batch_size,
                shuffle=shuffle,
                collate_fn=custom_collate_train_logic_in_loss,
                num_workers=num_workers,
            )
        else:
            shuffle = False if shuffle is None else shuffle
            return DataLoader(
                self,
                batch_size=batch_size,
                shuffle=shuffle,
                collate_fn=default_collate,
                num_workers=num_workers,
            )

    def __len__(self):
        return len(self.examples)


def visualize_examples(task, nr_examples=3):
    examples = task.examples
    nr_images_per_example = task.nr_images_per_example
    n = min(max(1, int(nr_examples)), len(examples))

    # Pick a diverse subset (random sample) so repeated calls don’t always show the first ones
    idxs = random.sample(range(len(examples)), n) if len(examples) > n else list(range(n))

    fig, axes = plt.subplots(
        n, nr_images_per_example + 1, figsize=(nr_images_per_example + 1, 1 * n)
    )
    if n == 1:
        axes = axes.reshape(1, nr_images_per_example + 1)

    bg_color = "#e8e8e8"  # light gray background

    for row, idx in enumerate(idxs):
        images, result, expression = examples[idx]  # images: (nr_images_per_example, C, H, W)

        for col in range(nr_images_per_example):
            ax = axes[row, col]
            img = images[col]
            if isinstance(img, torch.Tensor):
                img = img.squeeze().cpu().numpy()
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            ax.set_facecolor(bg_color)
            ax.axis("off")

        # Display the numerical result in the last column
        ax_result = axes[row, nr_images_per_example]
        ax_result.set_facecolor(bg_color)
        ax_result.text(
            0.5, 0.5, str(result), fontsize=20, ha="center", va="center", fontweight="bold"
        )
        ax_result.axis("off")

    fig.patch.set_facecolor(bg_color)
    plt.tight_layout()
    plt.show()
