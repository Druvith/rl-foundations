import pytest
import torch
import numpy as np
from gymnasium.spaces import Box, Discrete

from src.core import MLPActorCritic

@pytest.fixture
def simple_box_space():
    return Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

@pytest.fixture
def simple_discrete_space():
    return Discrete(5)

@pytest.fixture
def make_observation(shape):
    def _make(obs_shape):
        return torch.as_tensor(np.random.randn(*obs_shape).astype(np.float32))
    return _make

def test_mlp_actor_critic_step_continuous_action_space(simple_box_space, mocker):
    obs = torch.as_tensor(np.random.randn(3).astype(np.float32))
    mocker.patch("torch.no_grad", side_effect=lambda: (yield))
    model = MLPActorCritic(observation_space=simple_box_space, action_space=simple_box_space)
    a, v, logp = model.step(obs)
    assert isinstance(a, np.ndarray)
    assert isinstance(v, np.ndarray)
    assert isinstance(logp, np.ndarray)
    assert a.shape == (3,)
    assert v.shape == ()
    assert logp.shape == ()

def test_mlp_actor_critic_step_discrete_action_space(simple_box_space, simple_discrete_space, mocker):
    obs = torch.as_tensor(np.random.randn(3).astype(np.float32))
    mocker.patch("torch.no_grad", side_effect=lambda: (yield))
    model = MLPActorCritic(observation_space=simple_box_space, action_space=simple_discrete_space)
    a, v, logp = model.step(obs)
    # For discrete action, action is a single scalar - can be int or 0d ndarray
    assert isinstance(a, np.ndarray) or np.isscalar(a)
    assert v.shape == ()
    assert logp.shape == ()

@pytest.mark.parametrize("action_space_fixture,expected_shape",
    [("simple_box_space", (3,)), ("simple_discrete_space", ())])
def test_mlp_actor_critic_act_action_output(request, simple_box_space, simple_discrete_space, action_space_fixture, expected_shape, mocker):
    obs = torch.as_tensor(np.random.randn(3).astype(np.float32))
    mocker.patch("torch.no_grad", side_effect=lambda: (yield))
    action_space = request.getfixturevalue(action_space_fixture)
    model = MLPActorCritic(observation_space=simple_box_space, action_space=action_space)
    a = model.act(obs)
    if isinstance(a, np.ndarray):
        assert a.shape == expected_shape
    else:
        assert np.isscalar(a)

def test_mlp_actor_critic_unsupported_action_space(simple_box_space):
    class DummySpace:
        pass
    with pytest.raises(Exception):
        MLPActorCritic(observation_space=simple_box_space, action_space=DummySpace())

@pytest.mark.parametrize("shape", [(8, 3), (2, 3)])
def test_mlp_actor_critic_step_with_batched_observations(simple_box_space, shape, mocker):
    obs = torch.as_tensor(np.random.randn(*shape).astype(np.float32))
    mocker.patch("torch.no_grad", side_effect=lambda: (yield))
    model = MLPActorCritic(observation_space=simple_box_space, action_space=simple_box_space)
    a, v, logp = model.step(obs)
    assert isinstance(a, np.ndarray)
    assert a.shape == shape
    assert v.shape == (shape[0],)
    assert logp.shape == (shape[0],)

@pytest.mark.parametrize("obs_input,should_raise", [
    (torch.empty(0), True),                   # Empty tensor
    (torch.as_tensor([[1.0, 2.0]]), True),    # Incorrect shape (should be shape of obs, not (batch, obs_dim))
    (torch.as_tensor([1.0, 2.0, 3.0]), False) # Correct shape
])
def test_mlp_actor_critic_invalid_observation_handling(simple_box_space, obs_input, should_raise, mocker):
    mocker.patch("torch.no_grad", side_effect=lambda: (yield))
    model = MLPActorCritic(observation_space=simple_box_space, action_space=simple_box_space)
    if should_raise:
        with pytest.raises(Exception):
            model.step(obs_input)
    else:
        a, v, logp = model.step(obs_input)
        assert isinstance(a, np.ndarray)