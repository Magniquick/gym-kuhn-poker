# tests/test_kuhn_poker_env.py
import pytest

from gym_kuhn_poker.envs.kuhn_poker_env import KuhnPokerEnv, ActionType

# ---------- Basic construction & spaces ----------

def test_env_constructs_and_spaces_match():
    env = KuhnPokerEnv(number_of_players=2, deck_size=3, betting_rounds=2, ante=1)
    obs, info = env.reset(seed=123)

    # observation should match the declared observation_space
    assert env.observation_space.contains(obs), "Initial obs must be in observation_space"
    # action space is Discrete(2)
    assert env.action_space.n == 2

    # check shapes/dtypes that matter
    assert obs["card"].shape == (env.cfg.deck_size,)
    assert obs["history_actions"].shape == (env.cfg.max_actions, 1 + env.action_space.n)
    assert obs["pots"].shape == (env.cfg.n_players,)


# ---------- Deterministic dealing with seed ----------

def test_reset_is_deterministic_with_seed():
    env = KuhnPokerEnv()
    obs1, _ = env.reset(seed=777)
    hands1 = env.state.hands[:]  # type: ignore

    # New instance, same seed -> identical deal
    env2 = KuhnPokerEnv()
    obs2, _ = env2.reset(seed=777)
    hands2 = env2.state.hands[:]  # type: ignore

    assert hands1 == hands2, "Same seed should produce same unique deal"


# ---------- Observation correctness ----------

def test_obs_card_one_hot_matches_internal_hand():
    env = KuhnPokerEnv()
    # Fix the deal to known cards
    env._deal = lambda: [0, 2]  # P0 gets 0, P1 gets 2
    obs, _ = env.reset()
    assert obs["player_id"] == 0
    # Player 0's one-hot must reflect card 0
    assert obs["card"][0] == 1 and obs["card"].sum() == 1

    # After one step, next obs belongs to player 1 and should reflect card 2
    obs_next, r, terminated, truncated, info = env.step(ActionType.PASS)
    assert obs_next["player_id"] == 1
    assert obs_next["card"][2] == 1 and obs_next["card"].sum() == 1


def test_history_one_hot_encoding_for_pass_pass():
    env = KuhnPokerEnv()
    env._deal = lambda: [0, 2]
    obs, _ = env.reset()

    # P0: PASS, P1: PASS -> terminal (all checked once with 2p game)
    obs1, r0, term0, trunc0, info0 = env.step(ActionType.PASS)
    obs2, r1, term1, trunc1, info1 = env.step(ActionType.PASS)

    assert env.state.done is True  # type: ignore
    # First two rows should be PASS one-hots; remaining rows EMPTY
    hist = obs2["history_actions"]
    # columns: [EMPTY, PASS, BET]
    EMPTY, PASS, BET = 0, 1, 2

    # Row 0: PASS
    assert hist[0, PASS] == 1 and hist[0].sum() == 1
    # Row 1: PASS
    assert hist[1, PASS] == 1 and hist[1].sum() == 1
    # Remaining rows: EMPTY
    for i in range(2, env.cfg.max_actions):
        assert hist[i, EMPTY] == 1 and hist[i].sum() == 1


# ---------- Flow/termination logic ----------

ANTE = 1  # matches default in env


def _assert_fold_terminal(env, last_scalar_reward, info, bettor_id):
    """Common checks for any fold-termination line in 2p Kuhn."""
    # Terminal & winner
    assert env.state.done is True  # type: ignore
    assert env.state.winner == bettor_id  # type: ignore

    # Pot math (2 players): antes + 1-chip bet; folder pays nothing extra
    # Start: [1,1]; after bet by bettor: +1 to bettor -> [2,1]; total pot = 3
    assert env.state.pots == [2, 1] if bettor_id == 0 else [1, 2]  # type: ignore

    rewards_vec = info["rewards"]
    assert isinstance(rewards_vec, list) and len(rewards_vec) == 2
    assert sum(rewards_vec) == 0  # zero-sum

    # The scalar reward returned by step() equals the entry for the acting player.
    # The last actor was the folder.
    folder_id = 1 - bettor_id
    assert last_scalar_reward == rewards_vec[folder_id]


# ---------- Variant A: Direct bet → fold (no prior checks) ----------

def test_p0_bet_then_p1_fold_immediate_end():
    env = KuhnPokerEnv()
    # cards irrelevant for fold; fix deal for determinism
    env._deal = lambda: [0, 2]
    env.reset()

    # P0: BET -> live bet
    env.step(ActionType.BET)

    # P1: PASS (facing bet) -> FOLD -> terminal, bettor=P0 wins
    _, r1, term, trunc, info = env.step(ActionType.PASS)
    assert term is True and trunc is False
    _assert_fold_terminal(env, r1, info, bettor_id=0)

# ---------- Variant B: Check → bet → fold ----------

def test_p0_check_p1_bet_p0_fold():
    env = KuhnPokerEnv()
    env._deal = lambda: [0, 2]
    env.reset()

    # P0: PASS (check)
    env.step(ActionType.PASS)

    # P1: BET -> live bet; pots become [1, 2]
    env.step(ActionType.BET)

    # P0: PASS facing bet -> FOLD -> terminal, bettor=P1 wins
    _, r0, term, trunc, info = env.step(ActionType.PASS)
    assert term is True and trunc is False
    _assert_fold_terminal(env, r0, info, bettor_id=1)

# ---------- Errors & validation ----------

def test_invalid_action_raises_value_error():
    env = KuhnPokerEnv()
    env.reset()
    with pytest.raises(ValueError):
        # 2 is not a valid ActionType; enum conversion should fail
        env.step(2)


def test_config_validation_asserions():
    with pytest.raises(AssertionError):
        KuhnPokerEnv(number_of_players=1)
    with pytest.raises(AssertionError):
        KuhnPokerEnv(deck_size=1, number_of_players=2)
    with pytest.raises(AssertionError):
        KuhnPokerEnv(betting_rounds=0)
    with pytest.raises(AssertionError):
        KuhnPokerEnv(ante=0)
