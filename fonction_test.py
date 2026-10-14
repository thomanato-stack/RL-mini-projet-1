import gymnasium as gym
import numpy as np
import pytest

from bias import discounted_returns, keep, rollout


def test_returns_by_hand():
    # r[t] = r_{t+1}: G_3 = 4, G_2 = 3 + .9*4, G_1 = 2 + .9*6.6, G_0 = 1 + .9*7.94
    g = discounted_returns(np.array([1.0, 2.0, 3.0, 4.0]), 0.9)
    assert g == pytest.approx([8.146, 7.94, 6.6, 4.0])


def test_returns_on_terminated_episode():
    env = gym.make("LunarLanderContinuous-v3")
    S, A, R, trunc = rollout(env, lambda o: np.zeros(2, dtype=np.float32), seed=0)
    T, gm = len(R), 0.98
    assert not trunc and abs(R[-1]) == 100  # crash or landing: terminal reward included in R

    # Replaying the actions gives back the same (s_t, r_{t+1}) pairs.
    obs, _ = env.reset(seed=0)
    for t in range(T):
        assert np.array_equal(obs, S[t])
        obs, r, *_ = env.step(A[t])
        assert r == R[t]

    g = discounted_returns(R, gm)
    assert g[T - 1] == pytest.approx(R[T - 1])
    assert g[T - 2] == pytest.approx(R[T - 2] + gm * R[T - 1])
    assert g[T - 3] == pytest.approx(R[T - 3] + gm * R[T - 2] + gm**2 * R[T - 1])
    assert g[T - 4] == pytest.approx(R[T - 4] + gm * R[T - 3] + gm**2 * R[T - 2] + gm**3 * R[T - 1])
    assert g[0] == pytest.approx(sum(gm**k * R[k] for k in range(T)))


def test_selection_depends_on_t_only():
    assert keep(np.arange(1000), 1000, 500).sum() == 501  # truncated: T - t >= 500
    assert keep(np.arange(300), 1000, 500).all()  # short terminated episode kept whole
