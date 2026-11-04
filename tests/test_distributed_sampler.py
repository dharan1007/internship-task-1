"""Verify that DDP padding never changes the number of contributing images."""

import math

import pytest

from drvim.train import ExactContributionDistributedSampler


@pytest.mark.parametrize("count,replicas,batch_size", [
    (1, 4, 2), (3, 4, 2), (7, 4, 2), (17, 4, 2), (35, 4, 2),
    (17, 2, 4), (35, 8, 2),
])
def test_distributed_padding_has_zero_contribution(count, replicas, batch_size):
    samplers = [ExactContributionDistributedSampler(range(count), num_replicas=replicas,
                                                    rank=rank, seed=2026)
                for rank in range(replicas)]
    per_rank = [list(sampler) for sampler in samplers]
    assert len({len(rows) for rows in per_rank}) == 1
    contributing = [index for rows in per_rank for index, weight in rows if weight == 1.0]
    assert sorted(contributing) == list(range(count))
    assert sum(weight for rows in per_rank for _, weight in rows) == count
    assert all(weight in (0.0, 1.0) for rows in per_rank for _, weight in rows)

    steps = math.ceil(len(per_rank[0]) / batch_size)
    for first_step in range(0, steps, 2):
        last_step = min(first_step + 2, steps)
        actual = sum(weight for rows in per_rank
                     for _, weight in rows[first_step * batch_size:last_step * batch_size])
        first_global = first_step * batch_size * replicas
        capacity = (last_step - first_step) * batch_size * replicas
        assert actual == min(capacity, count - first_global)
