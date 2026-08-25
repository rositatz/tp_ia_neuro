import numpy as np

from config import (
    HORIZON,
    DICE_FACES,
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
)



class RandomLegalAgent:
    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def act(self, obs, env):
        action = int(self.rng.choice(env.get_valid_actions()))
        score_amount = None

        if action == SCORE:
            score_amount = int(self.rng.choice(env.get_valid_score_amounts()))

        return action, score_amount


class SimpleExpectancyAgent:

    def act(self, obs, env=None):
        turn = obs["turn"]
        gold = obs["gold"]
        num_dice = obs["num_dice"]
        dice_bonus = obs["dice_bonus"]
        shields = obs["shields"]

        turns_left = HORIZON - turn

        if turns_left == 0:
            return SCORE, gold

        if shields == 0 and gold >= SHIELD_COST:
            return BUY_SHIELD, None

        best_action = PASS
        best_value = 0.0

        dice_cost = get_new_dice_cost(num_dice)
        if gold >= dice_cost:
            buy_dice_value = (float(np.mean(DICE_FACES)) + dice_bonus) * turns_left - dice_cost
            if buy_dice_value > best_value:
                best_value = buy_dice_value
                best_action = BUY_DICE

        upgrade_cost = get_upgrade_cost(dice_bonus)
        if gold >= upgrade_cost:
            upgrade_value = num_dice * turns_left - upgrade_cost
            if upgrade_value > best_value:
                best_value = upgrade_value
                best_action = UPGRADE

        return best_action, None


def get_state(obs):
    turns_left = HORIZON - obs["turn"] + 1  # Cuantos turnos quedan, incluyendo el actual


    # Estas variables pueden crecer para siempre pero para Q learning se necesita que la cantidad de estados sea finita, sino cada estado se ve una sola vez y no se aprende
    # Ponemos un limite a cada variable para solucionar esto con min(valor, limite)
    gold_lvl = gold_level(obs["gold"], obs["num_dice"], obs["dice_bonus"])
    dice_lvl = min(obs["num_dice"], 6)
    bonus_lvl = min(obs["dice_bonus"], 6)
    shield_lvl = min(obs["shields"], 2)
    stored_lvl = min(obs["stored_value"], 10)
    max_lvl = min(obs["roll_max"], 12)

    return (turns_left, gold_lvl, dice_lvl, bonus_lvl, shield_lvl, stored_lvl, max_lvl)

def gold_level(gold, num_dice, dice_bonus):
    # Calcula cuantas de las cosas se pueden comprar con el oro que se tiene en el momento
    thresholds = sorted(set([0, STORE_DIE_COST, SHIELD_COST, get_new_dice_cost(num_dice), get_upgrade_cost(dice_bonus)]))

    lvl = 0
    for t in thresholds:
        # Recorre cada costo de menor a mayor y se fija si el oro le alcanza, cada vez que si suma 1
        if gold >= t:
            lvl += 1
    return lvl

# SCORE en el ambiente recibe un monto (score_amount) que puede ser cualquier número entre 0 y el oro actual
# Como no podemos meter una acción continua en una Q-table, la reemplazamos por unas jugadas de puntuar un porcentaje fijo del oro que se tiene en ese momento
SCORE_CUTS = [0.25, 0.5, 0.75, 1.0]

MOVES = (["PASS", "BUY_DICE", "UPGRADE", "BUY_SHIELD", "STORE_BEST_DIE"] + [f"SCORE_{int(f * 100)}" for f in SCORE_CUTS])

N_MOVES = len(MOVES)

def move_to_action(move, obs, env):
    """
    Traduce la jugada a lo que realmete entiende el env: (accion, score_amount)
    Si la jugada no es calida en este estado, se hace un PASS
    """
    name = MOVES[move]
    valid = env.get_valid_actions()

    if name == "PASS":
        return PASS, None
    
    if name == "BUY_DICE":
        if BUY_DICE in valid:
            return (BUY_DICE, None)
        else:
            return (PASS, None)

    if name == "BUY_SHIELD":
        if BUY_SHIELD in valid:
            return (BUY_SHIELD, None)
        else:
            return (PASS, None)

    if name == "STORE_BEST_DIE":
            if STORE_BEST_DIE in valid:
                return (STORE_BEST_DIE, None)
            else:
                return (PASS, None)

    # Los nombres son del estilo "SCORE_50". Tienen el porcentaje escrito adentro del string. Lo sacamos para saber cuánto oro puntuar
    frac = int(name.split("_")[1]) / 100.0
    amount = int(obs["gold"]*frac)
    return SCORE, amount

def greedy_action(Q, state):
    return int(np.argmax(Q[state]))

class QLearningAgent:
    """
    Este agente NO entrena, solo juega. Recibe una Q-table ya entrenada
    (armada en train_qlearning.py) y juega siempre greedy (nada de
    exploración), que es como se supone que tiene que jugar el agente
    ya entrenado.
    """
    def __init__(self, Q):
        self.Q = Q

    def act(self, obs, env):
        state = get_state(obs)
        move = greedy_action(self.Q, state)

        return move_to_action(move, obs, env)

    