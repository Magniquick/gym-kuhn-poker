import enum
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from typing import TypedDict
from numpy.typing import NDArray

field_dtype = np.int64


class ObsDict(TypedDict):
    player_id: field_dtype
    card: NDArray[field_dtype]  # shape: (deck_size,)
    history_actions: NDArray[field_dtype]  # shape: (max_actions, bins)
    pots: NDArray[field_dtype]  # shape: (n_players,)


class ActionType(enum.IntEnum):
    PASS = 0  # check / call
    BET = 1  # bet / raise


@dataclass(slots=True)
class GameConfig:
    n_players: int = 2
    deck_size: int = 3
    betting_rounds: int = 2  # upper-bound for how many moves we’ll encode in obs
    ante: int = 1

    def validate(self):
        assert self.n_players >= 2, "Game must be played with at least 2 players"
        assert self.deck_size >= self.n_players, "Deck must have at least one card per player"
        assert self.betting_rounds >= 1, "Need at least 1 betting round"
        assert self.ante >= 1, "Minimum ante must be one"

    @property
    def max_actions(self) -> int:
        # Worst-case with 2-action Kuhn-style flow is one lap of checks,
        # or one bet and the remaining players respond once.
        # We preserve your older bound of n_players * betting_rounds.
        return self.n_players * self.betting_rounds


@dataclass(slots=True)
class GameState:
    current_player: int
    hands: List[int]  # card index per player (0..deck_size-1)
    pots: List[int]  # ante + additional bets/calls per player
    history: List[Tuple[int, ActionType]] = field(default_factory=list)
    first_to_bet: Optional[int] = None
    eligible_players: List[int] = field(default_factory=list)
    done: bool = False
    winner: Optional[int] = None


class KuhnPokerEnv(gym.Env):
    """
    A cleaner Kuhn Poker environment (2 actions) with readable internal state.
    We still squeeze multi-player into a single-agent Gym step: each `step()` is
    from the perspective of the current player, and we return that player's obs.
    Rewards are a vector (one per player) at terminal.
    """

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, number_of_players=2, deck_size=3, betting_rounds=2, ante=1):
        super().__init__()
        self.cfg = GameConfig(
            n_players=number_of_players,
            deck_size=deck_size,
            betting_rounds=betting_rounds,
            ante=ante,
        )
        self.cfg.validate()

        # Spaces
        self.action_space = spaces.Discrete(2)
        self._hist_bins: int = 1 + int(self.action_space.n)

        single_obs = spaces.Dict(
            {
                "player_id": spaces.Discrete(self.cfg.n_players),
                "card": spaces.MultiBinary(self.cfg.deck_size),  # one-hot
                # History as 2 aligned tracks:
                # - actions: 0=empty, 1=PASS, 2=BET
                # - actors:  which player took the action (0..n_players-1), or 0 if empty
                "history_actions": spaces.MultiBinary((self.cfg.max_actions, self._hist_bins)),
                # "history_actors": spaces.MultiDiscrete([self.cfg.n_players] * self.cfg.max_actions), # TODO: test multi agent (> 2) games
                # Each player's contribution: 0..ante+rounds
                "pots": spaces.MultiDiscrete(
                    [self.cfg.ante + self.cfg.betting_rounds + 1] * self.cfg.n_players, dtype=field_dtype
                ),
            }
        )

        # Single-agent turn-based: each step/reset returns the dict for the current player
        self.observation_space = single_obs

        # Will be set in reset()
        self.state: Optional[GameState] = None
        self.np_random: np.random.Generator = np.random.default_rng()

    # ---------- Core helpers ----------

    def _deal(self) -> List[int]:
        # Unique cards to each player
        return list(self.np_random.choice(self.cfg.deck_size, size=self.cfg.n_players, replace=False))

    def _one_hot(self, idx: int, size: int) -> np.ndarray:
        v = np.zeros(size, dtype=field_dtype)
        v[idx] = 1
        return v

    def _build_obs_for(self, player_id: int) -> ObsDict:
        assert self.state is not None
        s = self.state

        # 2D one-hot: rows = action index, cols = [EMPTY, PASS, BET]
        hist_oh = np.zeros((self.cfg.max_actions, self._hist_bins), dtype=field_dtype)
        for i in range(self.cfg.max_actions):
            if i < len(s.history):
                _, act = s.history[i]  # PASS_=0, BET=1
                bin_idx = 1 + int(act)  # 0=EMPTY, 1=PASS, 2=BET
            else:
                bin_idx = 0  # EMPTY
            hist_oh[i, bin_idx] = 1

        # actors = np.zeros(self.cfg.max_actions, dtype=field_type)
        # for i, (pid, _) in enumerate(s.history[: self.cfg.max_actions]):
        #     actors[i] = pid

        return {
            "player_id": field_dtype(player_id),
            "card": self._one_hot(s.hands[player_id], self.cfg.deck_size),
            "history_actions": hist_oh,  # now 2D
            # "history_actors": actors,
            "pots": np.array(s.pots, dtype=field_dtype),
        }

    def _advance_player(self):
        assert self.state is not None
        self.state.current_player = (self.state.current_player + 1) % self.cfg.n_players

    def _count_since_first_bet(self) -> int:
        assert self.state is not None
        if self.state.first_to_bet is None:
            return 0
        for i, (_, a) in enumerate(self.state.history):
            if a == ActionType.BET:
                return len(self.state.history) - (i + 1)
        return 0

    def _all_checked_once(self) -> bool:
        assert self.state is not None
        return (
            self.state.first_to_bet is None
            and len(self.state.history) >= self.cfg.n_players
            and all(a == ActionType.PASS for _, a in self.state.history[: self.cfg.n_players])
        )

    def _betting_over_after_bet(self) -> bool:
        # After a first bet, once every *other* player has acted once, stop.
        assert self.state is not None
        return self.state.first_to_bet is not None and self._count_since_first_bet() >= (self.cfg.n_players - 1)

    def _maybe_finish(self):
        assert self.state is not None
        if self._all_checked_once() or self._betting_over_after_bet():
            self.state.done = True
            self.state.winner = self._compute_winner()

    def _compute_winner(self) -> int:
        assert self.state is not None
        s = self.state
        # Winner = max card among eligible players
        elig = s.eligible_players or list(range(self.cfg.n_players))
        best_card = -1
        best_player = elig[0]
        for p in elig:
            if s.hands[p] > best_card:
                best_card = s.hands[p]
                best_player = p
        return best_player

    def _rewards(self) -> List[int]:
        assert self.state is not None and self.state.done and self.state.winner is not None
        total = sum(self.state.pots)
        w = self.state.winner
        return [(total - self.state.pots[i]) if i == w else -self.state.pots[i] for i in range(self.cfg.n_players)]

    def get_terminal_state(self):
        return self._rewards()

    # ---------- Gym API ----------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        hands = self._deal()
        pots = [self.cfg.ante] * self.cfg.n_players

        self.state = GameState(
            current_player=0,
            hands=hands,
            pots=pots,
            history=[],
            first_to_bet=None,
            eligible_players=list(range(self.cfg.n_players)),
            done=False,
            winner=None,
        )

        # Just reset and return initial observation
        obs = self._build_obs_for(self.state.current_player)
        return obs, {}

    def step(self, action):
        assert self.state is not None
        move = ActionType(int(action))
        s = self.state
        pid = s.current_player

        # Apply action to history
        s.history.append((pid, move))

        if s.first_to_bet is None:
            # No live bet yet: PASS=check, BET=bet
            if move == ActionType.PASS:
                # check: no pot change, remain eligible for showdown
                if pid not in s.eligible_players:
                    s.eligible_players.append(pid)
            else:  # move == ActionType.BET
                # first bet: bettor pays 1, mark live bet, clear eligibles (we'll add responders)
                s.first_to_bet = pid
                s.eligible_players = [pid]  # bettor is eligible by definition
                s.pots[pid] += 1  # bet costs 1
        else:
            # There is a live bet: PASS=fold, BET=call
            if move == ActionType.PASS:
                # fold: hand ends immediately, bettor wins without showdown
                s.done = True
                # winner is the first bettor (could also be last aggressor if you extend rounds)
                s.winner = s.first_to_bet
            else:  # move == ActionType.BET  -> CALL
                s.pots[pid] += 1  # call pays 1 to match the bet
                if pid not in s.eligible_players:
                    s.eligible_players.append(pid)

        # If not already ended by a fold, advance & maybe finish by rule
        if not s.done:
            self._advance_player()
            # End conditions:
            # - all checked once (no bet occurred)
            # - after first bet, once every *other* player has acted once, stop -> showdown
            self._maybe_finish()

        # Rewards only at terminal
        terminated = s.done
        rewards = self._rewards() if terminated else [0] * self.cfg.n_players
        truncated = False
        info = {"rewards": rewards}

        # Observation for the next player (or arbitrary at terminal)
        next_pid = s.current_player
        obs_next = self._build_obs_for(next_pid)

        # Return scalar reward for the actor (pid), vector lives in info
        return obs_next, rewards[pid], terminated, truncated, info

    # ---------- Utils ----------

    def render(self, mode="human"):
        assert self.state is not None
        s = self.state
        card_names = ("Jack", "Queen", "King")

        if len(s.history) == 0:
            print("Dealt cards:")
            for p in range(self.cfg.n_players):
                c = s.hands[p]
                if self.cfg.deck_size == 3:
                    print(f"  Player {p}: {card_names[c]} (#{c})", end="")
                else:
                    print(f"  Player {p}: Card #{c}", end="")
            print()
        else:
            mover = (s.current_player - 1) % self.cfg.n_players
            last_act = s.history[-1][1].name.replace("_", "")
            print(f"Step {len(s.history)}: Player {mover} → {last_act}")

        print("Pots:", s.pots)
        print()
