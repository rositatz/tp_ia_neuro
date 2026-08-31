import numpy as np

from config import (
    HORIZON,
    SHIELD_COST,
    STORE_DIE_COST,
    get_new_dice_cost,
    get_upgrade_cost,
)


NUM_DICE_CAP = 10
DICE_BONUS_CAP = 10
SHIELD_CAP = 2
ROLL_MAX_CAP = 12


def gold_level(gold):
    """
    Agrupa el oro en una escala logaritmica.

    Mantiene mas precision para cantidades pequenas y agrupa mas
    las cantidades grandes, evitando un estado distinto por moneda.
    """
    gold = int(gold)

    if gold <= 0:
        return 0

    return int(np.log(gold + 1) / np.log(1.5))


def get_affordability(obs):
    """
    Indica que inversiones puede pagar el agente en este estado.

    Orden:
        STORE_BEST_DIE
        BUY_SHIELD
        UPGRADE
        BUY_DICE
    """
    gold = int(obs["gold"])
    num_dice = int(obs["num_dice"])
    dice_bonus = int(obs["dice_bonus"])
    roll_max = int(obs["roll_max"])

    return (
        int(gold >= STORE_DIE_COST and roll_max > 0),
        int(gold >= SHIELD_COST),
        int(gold >= get_upgrade_cost(dice_bonus)),
        int(gold >= get_new_dice_cost(num_dice)),
    )


def encode_tabular_state(obs):
    """
    Representacion de estado compartida por Q-Learning, SARSA
    y Monte Carlo.
    """
    turns_left = max(
        0,
        HORIZON - int(obs["turn"]) + 1,
    )

    return (
        turns_left,
        gold_level(obs["gold"]),
        get_affordability(obs),
        min(int(obs["num_dice"]), NUM_DICE_CAP),
        min(int(obs["dice_bonus"]), DICE_BONUS_CAP),
        min(int(obs["shields"]), SHIELD_CAP),
        min(int(obs["roll_max"]), ROLL_MAX_CAP),
    )
