import numpy as np
import torch
import torch.nn as nn

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
    N_ACTIONS
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
    turn_lvl = turn_bucket(turns_left)

    # Estas variables pueden crecer para siempre pero para Q learning se necesita que la cantidad de estados sea finita, sino cada estado se ve una sola vez y no se aprende
    # Ponemos un limite a cada variable para solucionar esto con min(valor, limite)
    gold_lvl = gold_level(obs["gold"])
    dice_lvl = min(obs["num_dice"], 10)
    bonus_lvl = min(obs["dice_bonus"], 10)
    shield_lvl = min(obs["shields"], 2)
    stored_lvl = min(obs["stored_value"], 6)
    max_lvl = min(obs["roll_max"], 8)

    return (turn_lvl, gold_lvl, dice_lvl, bonus_lvl, shield_lvl, stored_lvl, max_lvl)


def turn_bucket(turns_left):
    if turns_left <= 6:
        return turns_left
    return 6 + (turns_left - 6) // 4


# los turnos lejanos los agrupamos de a 4 (no hace falta tanta precisión cuando queda mucha partida), pero los últimos 6 los dejamos exactos, porque ahí sí importa el turno justo
def turn_bucket(turns_left):
    if turns_left <= 6:
        return turns_left
    return 6 + (turns_left - 6) // 4

def gold_level(gold):
    if gold <= 0:
        return 0
    return int(np.log(gold + 1) / np.log(1.5))

# Dejamos solo la opción de puntuar todo el oro de una, para que no le convenga ir puntuando de a poquito en medio de la partida
MOVES = (["PASS", "BUY_DICE", "UPGRADE", "BUY_SHIELD", "STORE_BEST_DIE", "SCORE_ALL"])

N_MOVES = len(MOVES)

def move_to_action(move, obs, env):
    """
    Traduce la jugada a lo que realmete entiende el env: (accion, score_amount)
    Si la jugada no es valida en este estado, se hace un PASS
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
        
    if name == "UPGRADE":
        if UPGRADE in valid:
            return (UPGRADE, None)
        else:
            return (PASS, None)
        
    if name == "STORE_BEST_DIE":
            if STORE_BEST_DIE in valid:
                return (STORE_BEST_DIE, None)
            else:
                return (PASS, None)

    return SCORE, obs["gold"]

def greedy_action(Q, state):
    if state not in Q:
        # si nunca vio este estado, asegura el oro como puntos antes que arriesgarse con PASS y perderlo todo si es tarde en la partida
        return MOVES.index("SCORE_ALL")
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



N_FEATURES = 8

def state_vector(obs):
    # divide cada variable por un numero parecido a su maximo para que a la red le lleguen todos numeros chicos y parecidos entre si
    return np.array([
        obs["turn"] / HORIZON,
        obs["gold"] / 100.0,
        obs["num_dice"] / 10.0,
        obs["dice_bonus"] / 10.0,
        obs["shields"] / 3.0,
        obs["stored_value"] / 15.0,
        obs["roll_sum"] / 50.0,
        obs["roll_max"] / 15.0,
    ], dtype=np.float32)


# la red recibe el estado (8 números) y devuelve un valor por cada jugada posible (N_MOVES números). 
# Tiene dos capas ocultas 
class QNet(nn.Module):
    def __init__(self, n_features, n_moves):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_moves),
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    """
    Como QLearningAgent, pero en vez de una tabla usa una red neuronal
    ya entrenada para estimar el valor de cada jugada.
    """

    def __init__(self, net):
        self.net = net
        self.net.eval()  

    def act(self, obs, env):
        x = torch.from_numpy(state_vector(obs)).unsqueeze(0)
        with torch.no_grad():
            q_values = self.net(x).squeeze(0).numpy()
        move = int(np.argmax(q_values))
        return move_to_action(move, obs, env)

from mc_agent import MonteCarloAgent
def get_state_tuple(obs):
    return (
        int(obs["turn"]),
        int(obs["gold"]),
        int(obs["num_dice"]),
        int(obs["dice_bonus"]),
        int(obs["shields"]),
        int(obs["roll_max"])
    )
def get_q_values(Q, state):
    if state not in Q:
        q = np.zeros(N_ACTIONS, dtype=float)
        q[SCORE] = 0.1
        return q

    return Q[state]


class SARSAAgent:

    def __init__(self, Q):
        self.Q = Q

    def act(self, obs, env):

        state = get_state(obs)

        valid_actions = env.get_valid_actions()

        if state not in self.Q:
                    if SCORE in valid_actions:
                        return SCORE, obs["gold"]

                    return PASS, None

        q_values = self.Q[state]
        best_action = max(
            valid_actions,
            key=lambda a: q_values[a]
        )

        if best_action == SCORE:
            return SCORE, obs["gold"]

        return best_action, None