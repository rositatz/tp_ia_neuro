import pickle
from pathlib import Path

import numpy as np
from artifact_paths import MONTE_CARLO_TABLE_PATH

from config import (
    HORIZON,
    SHIELD_COST,
    STORE_DIE_COST,
    get_new_dice_cost,
    get_upgrade_cost,
)
from env import (
    PASS,
    SCORE,
    BUY_DICE,
    UPGRADE,
    BUY_SHIELD,
    STORE_BEST_DIE,
    N_ACTIONS,
)


NUM_DICE_CAP = 8
DICE_BONUS_CAP = 8
SHIELD_CAP = 2

GOLD_THRESHOLDS = (
    1, 4, 5, 8, 9, 14, 18, 24,
    32, 42, 55, 72, 95, 130, 180, 250,
)


def gold_bucket(gold):
    return sum(int(gold) >= threshold for threshold in GOLD_THRESHOLDS)


def encode_state(obs):
    return (
        HORIZON - int(obs["turn"]),
        min(int(obs["num_dice"]), NUM_DICE_CAP),
        min(int(obs["dice_bonus"]), DICE_BONUS_CAP),
        min(int(obs["shields"]), SHIELD_CAP),
        gold_bucket(obs["gold"]),
    )


def valid_action_mask(obs):
    gold = int(obs["gold"])
    num_dice = int(obs["num_dice"])
    dice_bonus = int(obs["dice_bonus"])
    roll_max = int(obs.get("roll_max", 0))

    mask = np.zeros(N_ACTIONS, dtype=bool)
    mask[PASS] = True
    mask[SCORE] = gold > 0
    mask[BUY_DICE] = gold >= get_new_dice_cost(num_dice)
    mask[UPGRADE] = gold >= get_upgrade_cost(dice_bonus)
    mask[BUY_SHIELD] = gold >= SHIELD_COST
    mask[STORE_BEST_DIE] = (gold >= STORE_DIE_COST) and (roll_max > 0)
    return mask


# --------------------------------------------------------------------------- #
# Agent
# --------------------------------------------------------------------------- #
DEFAULT_Q_PATH = MONTE_CARLO_TABLE_PATH


class MonteCarloAgent:

    def __init__(self, q_path=DEFAULT_Q_PATH, Q=None, epsilon=0.0, seed=None):
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)
        if Q is not None:
            self.Q = Q
        else:
            self.Q = self.load_q(q_path)

    # -- persistence ------------------------------------------------------- #
    @staticmethod
    def load_q(q_path):
        path = Path(q_path) if q_path else None
        if path and path.is_file():
            with path.open("rb") as f:
                return pickle.load(f)
        return {}

    def save_q(self, q_path=DEFAULT_Q_PATH):
        with Path(q_path).open("wb") as f:
            pickle.dump(self.Q, f)

    # -- action selection -------------------------------------------------- #
    def _greedy_action(self, state, mask):
        """Best valid action for `state`; None if the state is unknown."""
        q = self.Q.get(state)
        if q is None:
            return None
        masked = np.where(mask, q, -np.inf)
        if not np.isfinite(masked).any():
            return None
        return int(np.argmax(masked))

    def _fallback_action(self, obs, mask):
        turns_left = HORIZON - int(obs["turn"])
        num_dice = int(obs["num_dice"])
        dice_bonus = int(obs["dice_bonus"])
        shields = int(obs["shields"])

        if turns_left == 0:
            return SCORE
        # value of one more die / upgrade over the remaining turns vs its cost
        if mask[BUY_DICE] and (3.5 + dice_bonus) * turns_left > get_new_dice_cost(num_dice) * 1.5:
            return BUY_DICE
        if mask[UPGRADE] and num_dice * turns_left > get_upgrade_cost(dice_bonus) * 1.5:
            return UPGRADE
        if mask[BUY_SHIELD] and shields == 0 and turns_left > 3:
            return BUY_SHIELD
        if mask[SCORE]:
            return SCORE
        return PASS

    def act(self, obs, env=None):
        mask = valid_action_mask(obs)
        turns_left = HORIZON - int(obs["turn"])

        # Last turn: any gold left would be wasted, so always lock it in.
        if turns_left == 0:
            gold = int(obs["gold"])
            return (SCORE, gold) if gold > 0 else (PASS, None)

        # Optional exploration (used during training via act()).
        if self.epsilon > 0.0 and self.rng.random() < self.epsilon:
            action = int(self.rng.choice(np.flatnonzero(mask)))
        else:
            action = self._greedy_action(encode_state(obs), mask)
            if action is None:
                action = self._fallback_action(obs, mask)

        score_amount = int(obs["gold"]) if action == SCORE else None
        return action, score_amount
