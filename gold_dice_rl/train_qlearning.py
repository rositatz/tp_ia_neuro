from collections import defaultdict
import numpy as np
import pickle

from env import GoldDiceEnv
from artifact_paths import QLEARNING_TABLE_PATH
from agents import (
    get_state,
    get_valid_move_mask,
    move_to_action,
    N_MOVES,
    QLearningAgent,
)

# Q es un diccionario, para cada estado guarda un array con el valor de cada una de las N_MOVES jugadas posibles. 
# Si un estado nunca se visitó, defaultdict lo crea solo con ceros (Q(s,a) inicializado arbitrariamente)
def empty_q():
    return defaultdict(lambda: np.zeros(N_MOVES, dtype=float))


# Ver cuántas veces se visitó cada (estado, jugada) y tener una idea de si la discretización quedó razonable o hay estados casi sin visitar
def empty_visits():
    return defaultdict(lambda: np.zeros(N_MOVES, dtype=int))

def epsilon_greedy(Q, state, valid_move_mask, epsilon, rng):
    valid_moves = np.flatnonzero(valid_move_mask)

    # jugada al azar, si no la de mayor Q para ese estado
    if rng.random() < epsilon:
        return int(rng.choice(valid_moves))

    q = Q[state]
    best_value = q[valid_moves].max()
    best_moves = valid_moves[q[valid_moves] == best_value]
    return int(rng.choice(best_moves))


# epsilon arranca alto (mucha exploración al principio, cuando la Q todavía no significa nada) y baja de forma lineal hasta un piso, para no dejar de explorar nunca del todo
def epsilon_schedule(episode, n_episodes, eps_start=1.0, eps_end=0.05):
    fraction = episode / max(1, n_episodes - 1)
    return eps_start + fraction * (eps_end - eps_start)


# guarda el promedio de recompensas de las últimas partidas, para poder graficar después cómo fue mejorando el agente
def record_checkpoint(history, episode_rewards, episode, reward_window=500):
    start = max(0, len(episode_rewards) - reward_window)
    recent_reward = np.mean(episode_rewards[start:])
    history["episode"].append(episode)
    history["train_reward"].append(recent_reward)



def train_q_learning(n_episodes=100000, alpha=0.05,
    gamma=1.0,          # sin descuento: el retorno total = puntos finales, que es justo lo que queremos maximizar
    eps_start=1.0, eps_end=0.05, eval_every=2500,
    seed=0,
    # evaluate_agents.py evalúa con semillas 0..999. Si entrenaramos con esas mismas semillas, el agente podría terminar memorizando esas partidas puntuales en vez de
    # aprender a jugar en general. Por eso entreno en un rango de semillas separado.
    train_seed_offset=100000,):
    Q = empty_q()
    visits = empty_visits()
    rewards = []
    history = {"episode": [], "train_reward": []}

    rng = np.random.default_rng(seed)

    for ep in range(n_episodes):
        epsilon = epsilon_schedule(ep, n_episodes, eps_start, eps_end)

        train_seed = train_seed_offset + ep
        env = GoldDiceEnv(obs_mode="dict", seed=train_seed, track_history=False)
        obs = env.reset(seed=train_seed)
        state = get_state(obs)
        done = False
        total_reward = 0.0

        while not done:
            valid_move_mask = get_valid_move_mask(obs, env)
            move = epsilon_greedy(
                Q,
                state,
                valid_move_mask,
                epsilon,
                rng,
            )
            action, score_amount = move_to_action(move, obs, env)

            next_obs, reward, done, info = env.step(action, score_amount=score_amount)
            next_state = get_state(next_obs)

            visits[state][move] += 1
            total_reward += reward

            # uso el máximo de Q en el próximo estado, no la jugada que realmente voy a tomar ahí
            if done:
                target = reward  # no hay estado siguiente, el objetivo es directamente la recompensa
            else:
                next_valid_move_mask = get_valid_move_mask(next_obs, env)
                next_valid_moves = np.flatnonzero(next_valid_move_mask)
                target = reward + gamma * np.max(Q[next_state][next_valid_moves])

            Q[state][move] += alpha * (target - Q[state][move])

            obs, state = next_obs, next_state

        rewards.append(total_reward)

        # cada tantos episodios, guardo un checkpoint y mostramos por pantalla cómo viene el entrenamiento
        if (ep + 1) % eval_every == 0 or ep == n_episodes - 1:
            record_checkpoint(history, rewards, ep + 1)
            print(f"Episodio {ep+1}/{n_episodes}  eps={epsilon:.3f}  "
                  f"reward_reciente={history['train_reward'][-1]:.2f}  |Q|={len(Q)}")

    return Q, visits, rewards, history


if __name__ == "__main__":
    Q, visits, rewards, history = train_q_learning(n_episodes=1000000)

    # guardo la Q como dict común para poder cargarla después sin tener que reentrenar
    with QLEARNING_TABLE_PATH.open("wb") as file:
        pickle.dump(dict(Q), file)
    print(f"Guardado {len(Q)} estados en {QLEARNING_TABLE_PATH}")
