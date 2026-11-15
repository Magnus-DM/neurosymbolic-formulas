# Provided by the course Learning and Reasoning (KU Leuven, 2025-2026). Unchanged.

from abc import ABC, abstractmethod
import torch


class Semantics(ABC):

    @abstractmethod
    def conjunction(self, a, b):
        pass

    @abstractmethod
    def disjunction(self, a, b):
        pass

    @abstractmethod
    def negation(self, a):
        pass


class SumProductSemiring(Semantics):

    def conjunction(self, a, b):
        return a * b

    def reduce_conjunction(self, a, dim):
        return torch.prod(a, dim)

    def disjunction(self, a, b):
        return a + b

    def reduce_disjunction(self, a, dim):
        return torch.sum(a, dim)

    def negation(self, a):
        return 1 - a