import operator
from typing import Callable
import numpy as np
import copy 
import random

class OUNoise:

    def __init__(
        self, 
        size: int, 
        mu: float = 0.0, 
        theta: float = 0.15, 
        sigma: float = 0.2,
    ):
        """Initialize parameters and noise process."""
        self.state = np.float64(0.0)
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self) -> np.ndarray:
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array(
            [random.random() for _ in range(len(x))]
        )
        self.state = x + dx
        return self.state

class SegmentTree:
    """ Segment Tree for Prioritized Experience Replay"""

    def __init__(self, capacity: int, operation: Callable, init_value: float):
        assert (
            capacity > 0 and capacity & (capacity - 1) == 0
        ), "capacity must be positive and a power 2."

        self.capacity = capacity
        self.tree = [init_value for _ in range(2 * capacity)]
        self.operation = operation

    def _operate_helper(self, start: int, end: int, node: int, node_start: int, node_end: int) -> float:
        """ Return the result of the operator in segment"""
        if start == node_start and end == node_end:
            return self.tree[node]
        mid = (node_start + node_end) // 2
        if end <= mid:
            return self._operate_helper(start, end, 2 * node, node_start, mid)
        else:
            if mid + 1 <= start:
                return self._operate_helper(start, end, 2 * node + 1, mid + 1, node_end)
            return self.operation(
                self._operate_helper(start, mid, 2 * node, node_start, mid),
                self._operate_helper(mid + 1, end, 2 * node + 1, mid + 1, node_end)
            )

    def operate(self, start: int = 0, end: int = 0) -> float:
        """ retruns result after applying operation"""
        if end <= 0:
            end += self.capacity
        end -= 1

        return self._operate_helper(start, end, 1, 0, self.capacity - 1)

    def __setitem__(self, idx: int, val: float):
        """ Set value in a tree"""
        idx += self.capacity
        self.tree[idx] = val

        idx //= 2
        while idx >= 1:
            self.tree[idx] = self.operation(self.tree[2 * idx], self.tree[2 * idx +1])
            idx //= 2

    def __getitem__(self, idx: int)-> float:
        """ Get real value in the leaf of a tree"""
        assert 0 <= idx < self.capacity

        return self.tree[self.capacity + idx]


class SumSegmentTree(SegmentTree):
    """ Create Sum Segment Tree"""


    def __init__(self, capacity: int):
        super(SumSegmentTree, self).__init__(capacity = capacity, operation = operator.add, init_value = 0.0)

    def sum(self, start: int = 0, end: int = 0) -> float:
        return super(SumSegmentTree, self).operate(start, end)

    def retrieve(self, upperbound: float) -> int:
        assert 0 <= upperbound <= self.sum() + 1e-5, "upperbound: {}".format(upperbound)

        idx = 1

        while idx < self.capacity:
            left = 2 * idx
            right = left + 1
            if self.tree[left] > upperbound:
                idx = 2 * idx
            else:
                upperbound -= self.tree[left]
                idx = right

        return idx - self.capacity

class MinSegmentTree(SegmentTree):
    """ Create Min Segment tree"""

    def __init__(self, capacity: int):
        super(MinSegmentTree, self).__init__(capacity = capacity, operation = min, init_value=float("inf"))

    def min(self, start: int = 0, end: int = 0) -> float:
        return super(MinSegmentTree, self).operate(start, end)
