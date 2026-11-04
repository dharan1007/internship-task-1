"""Checkpoint RNG state must survive restricted loading and exact restoration."""

import io
import random

import numpy as np
import torch

from drvim.train import _restore_rng, _rng_state


def test_rng_state_round_trip_with_restricted_loader() -> None:
    state = _rng_state()
    stream = io.BytesIO()
    torch.save(state, stream)
    stream.seek(0)
    restored_state = torch.load(stream, weights_only=True)
    expected = (random.random(), float(np.random.random()), torch.rand(1))
    _restore_rng(restored_state)
    observed = (random.random(), float(np.random.random()), torch.rand(1))
    assert observed[:2] == expected[:2]
    torch.testing.assert_close(observed[2], expected[2], rtol=0, atol=0)
    _restore_rng(state)
