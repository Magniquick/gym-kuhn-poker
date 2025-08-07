# type: ignore

import pytest

from gym_kuhn_poker.envs.kuhn_poker_env import ActionType, KuhnPokerEnv


def run_sequence(env, actions):
    """
    Helper to step through a sequence of action indices.
    Returns: rewards, done flag, truncated flag, info dict from the last step.
    """
    rewards = None
    done = False
    truncated = None
    info = None
    for a in actions:
        _, rewards, done, truncated, info = env.step(a)
    return rewards, done, truncated, info


@pytest.fixture
def env():
    return KuhnPokerEnv()

@pytest.mark.parametrize("seed", list(range(100)))
def test_all_pass(env, seed):
    # Two consecutive PASS actions should terminate via all_players_passed()
    env.reset(seed=seed)
    rewards, done, truncated, info = run_sequence(env, [ActionType.PASS.value, ActionType.PASS.value])
    assert done, "Env did not terminate after two PASS actions"
    # Zero-sum: one positive, one negative
    assert sum(rewards) == 0, f"Rewards not zero-sum: {rewards}"
    assert any(r > 0 for r in rewards) and any(r < 0 for r in rewards)
    assert not truncated
    assert info == {}

@pytest.mark.parametrize("seed", list(range(100)))
def test_bet_fold(env, seed):
    # First BET, then opponent PASS => fold, terminates via betting_is_over()
    env.reset(seed=seed)
    rewards, done, truncated, info = run_sequence(env, [ActionType.BET.value, ActionType.PASS.value])
    assert done, "Env did not terminate after BET followed by PASS"
    assert sum(rewards) == 0, f"Rewards not zero-sum: {rewards}"
    assert rewards[0] > 0 and rewards[1] < 0, f"Expected player 0 to win when player 1 folds: {rewards}"
    assert not truncated
    assert info == {}

@pytest.mark.parametrize("seed", list(range(100)))
def test_bet_call(env, seed):
    # Both players BET => terminates via betting_is_over() when each has bet once
    env.reset(seed=seed)
    rewards, done, truncated, info = run_sequence(env, [ActionType.BET.value, ActionType.BET.value])
    assert done, "Env did not terminate after both players BET"
    assert sum(rewards) == 0, f"Rewards not zero-sum: {rewards}"
    # Both have contributed more than the ante, so rewards should be non-zero
    assert any(r != 0 for r in rewards), f"Expected non-zero rewards when both BET: {rewards}"
    assert not truncated
    assert info == {}

@pytest.mark.parametrize("seed", list(range(100)))
def test_winner_always_non_negative_and_matches_winner_attribute(env, seed):
    """
    Across several random plays, ensure that:
      1. The env.winner attribute corresponds to the index of the highest reward.
      2. That reward is never negative.
    """
    env.reset(seed=seed)
    # Play until terminal state with random actions
    rewards = None
    done = False
    while not done:
        action = env.np_random.integers(len(ActionType))
        _, rewards, done, truncated, info = env.step(action)
    # Determine index of highest reward
    max_idx = rewards.index(max(rewards))
    # Check winner attribute matches
    assert env.winner == max_idx, f"env.winner={env.winner} but highest reward at index {max_idx}: {rewards}"
    # Check that winner's reward is non-negative
    assert rewards[env.winner] >= 0, f"Winner has negative reward: {rewards}"
