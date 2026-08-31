import numpy as np
from collections import defaultdict
import pickle
from env import GoldDiceEnv, N_ACTIONS
import itertools
from agents import SARSAAgent, get_state
from evaluate_agents import evaluate
from env import (
    SCORE,
)

def new_q():
    return defaultdict(
        lambda: np.zeros(N_ACTIONS, dtype=float)
    )

def epsilon_greedy_valid(Q, state, valid_actions, epsilon, rng):
    if rng.random() < epsilon:
        return rng.choice(valid_actions)
    q_values = Q[state]
    best_action = max(
        valid_actions,
        key=lambda a: q_values[a]
    )
    return best_action

def epsilon_schedule(
    episode,
    n_episodes,
    eps_start=1.0,
    eps_end=0.05,
):
    fraction = episode / max(1, n_episodes - 1)
    return eps_start + fraction * (eps_end - eps_start)

def train_sarsa(
    env,
    n_episodes=100_000,
    alpha=0.05,
    gamma=1.0,
    eps_start=1.0,
    eps_end=0.05,
    seed=42,
    train_seed_offset=100000,  
    eval_every=2_500,
):
    rng = np.random.default_rng(seed)
    Q = new_q()
    rewards = []

    for ep in range(n_episodes):

        epsilon = epsilon_schedule(
            ep,
            n_episodes,
            eps_start,
            eps_end
        )

        train_seed = train_seed_offset + ep
        obs = env.reset(seed=train_seed)

        state = get_state(obs)

        # Elegimos la primera acción A
        action = epsilon_greedy_valid(
            Q,
            state,
            env.get_valid_actions(),
            epsilon,
            rng
        )

        done = False
        total_reward = 0.0

        while not done:
            # SCORE necesita score_amount
            if action == SCORE:
                score_amount = obs["gold"]
            else:
                score_amount = None

            # Ejecutamos A
            next_obs, reward, done, info = env.step(
                action,
                score_amount=score_amount
            )

            total_reward += reward

            # S'
            next_state = get_state(next_obs)

            if done:

                # Si terminó el episodio no existe A'
                target = reward

            else:

                # Elegimos A' en S'
                next_action = epsilon_greedy_valid(
                    Q,
                    next_state,
                    env.get_valid_actions(),
                    epsilon,
                    rng
                )

                # SARSA:
                # R + gamma * Q(S', A')
                target = (
                    reward
                    + gamma * Q[next_state][next_action]
                )

            # Actualización Q(S,A)
            Q[state][action] += alpha * (
                target - Q[state][action]
            )

            if not done:

                # S <- S'
                state = next_state

                # A <- A'
                action = next_action

                # obs <- obs'
                obs = next_obs

        rewards.append(total_reward)

        if (ep + 1) % eval_every == 0 or ep == n_episodes - 1:
            start = max(0, len(rewards) - 500)
            recent_reward = np.mean(rewards[start:])
            print(f"Episodio {ep+1}/{n_episodes}  eps={epsilon:.3f}  "
                  f"reward_reciente={recent_reward:.2f}  |Q|={len(Q)}")

    return Q


def tune_sarsa_hyperparameters(env, n_train_episodes=20000, n_eval_episodes=1000):
    """
    Busca la mejor combinación de alpha y gamma .
    """
    # 1. Definimos los valores que queremos probar
    # Alpha: Tasa de aprendizaje (qué tanto reemplaza el valor viejo por el nuevo)
    alphas = [0.1, 0.05, 0.01] 
    
    # Gamma: Factor de descuento (1.0 es ideal para juegos con fin definido, pero probamos otros)
    gammas = [1.0, 0.99, 0.95, 0.9] 
    
    best_score = -np.inf
    best_params = None
    best_q_table = None

    # Generamos todas las combinaciones posibles
    combinations = list(itertools.product(alphas, gammas))
    print(f"Total de combinaciones a probar: {len(combinations)}\n")

    for idx, (alpha, gamma) in enumerate(combinations):
        print(f"[{idx+1}/{len(combinations)}] Entrenando con alpha={alpha}, gamma={gamma}...")
        
        # 2. Entrenamos el agente (usamos menos episodios para que la búsqueda no tarde días)
        Q_trained = train_sarsa(
            env=env, 
            n_episodes=n_train_episodes, 
            alpha=alpha, 
            gamma=gamma, 
            eps_start=1.0, 
            eps_end=0.05,
            seed=42
        )
        
        # 3. Lo evaluamos 100% greedy usando tu agente y función de evaluación
        agent = SARSAAgent(Q_trained)
        
        eval_score = evaluate(
            agent,
            n_episodes=n_eval_episodes,
            seed=10_000,
        )
        print(
            f"Puntaje de validacion: {eval_score['mean']:.3f} | "
            f"Estados explorados: {len(Q_trained)}\n"
        )
        
        # 4. Guardamos si es el mejor hasta ahora
        if eval_score["mean"] > best_score:
            best_score = eval_score["mean"]
            best_params = {
                "alpha": alpha, 
                "gamma": gamma
            }
            best_q_table = Q_trained

    print("-" * 30)
    print(f"Mejor puntaje: {best_score:.2f}")
    print(f"Mejores parámetros: {best_params}")
    
    return best_params, best_q_table

if __name__ == "__main__":
    print("Entrenando agente SARSA...")
    env = GoldDiceEnv(obs_mode="dict", seed=123, track_history=False)

    best_params, best_Q = tune_sarsa_hyperparameters(env, n_train_episodes=20_000)
    

    print(f"\nEntrenando el modelo final con 100,000 episodios y los mejores parámetros: {best_params}")
    final_Q = train_sarsa(
        env,
        n_episodes=100_000,
        alpha=best_params["alpha"],
        gamma=best_params["gamma"],
        eps_start=1.0,
        eps_end=0.05
    )
    with open("q_table_sarsa.pkl", "wb") as f: pickle.dump(dict(final_Q), f)
        
    print("¡Entrenamiento terminado y tabla guardada!")
