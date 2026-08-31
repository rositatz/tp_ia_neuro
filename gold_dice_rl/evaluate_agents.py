import argparse
import pickle

import numpy as np
import torch

from env import GoldDiceEnv
from agents import (
    DQNAgent,
    N_FEATURES,
    N_MOVES,
    QLearningAgent,
    QNet,
    RandomLegalAgent,
    SARSAAgent,
    SimpleExpectancyAgent,
)
from artifact_paths import (
    DQN_WEIGHTS_PATH,
    MONTE_CARLO_TABLE_PATH,
    QLEARNING_TABLE_PATH,
    SARSA_TABLE_PATH,
)
from mc_agent import MonteCarloAgent


def evaluate(agent, n_episodes=1000, seed=0, obs_mode="dict"):
    if n_episodes <= 0:
        raise ValueError("n_episodes must be greater than zero.")

    scores = []
    env = GoldDiceEnv(obs_mode=obs_mode, seed=seed, track_history=False)

    for ep in range(n_episodes):
        obs = env.reset(seed=seed + ep)
        done = False

        while not done:
            action, score_amount = agent.act(obs, env)
            obs, reward, done, info = env.step(action, score_amount=score_amount)

        scores.append(env.points)

    scores = np.array(scores)

    return {
        "mean": float(scores.mean()),
        "std": float(scores.std()),
        "min": int(scores.min()),
        "p25": float(np.percentile(scores, 25)),
        "median": float(np.percentile(scores, 50)),
        "p75": float(np.percentile(scores, 75)),
        "max": int(scores.max()),
    }


def require_artifacts(paths):
    missing = [path for path in paths if not path.is_file()]
    if missing:
        missing_names = ", ".join(path.name for path in missing)
        raise FileNotFoundError(
            f"Missing trained artifacts: {missing_names}. "
            "Run the corresponding training scripts first."
        )


def load_pickle(path):
    with path.open("rb") as file:
        return pickle.load(file)


def load_agents():
    artifact_paths = (
        QLEARNING_TABLE_PATH,
        SARSA_TABLE_PATH,
        MONTE_CARLO_TABLE_PATH,
        DQN_WEIGHTS_PATH,
    )
    require_artifacts(artifact_paths)

    q_learning_table = load_pickle(QLEARNING_TABLE_PATH)
    sarsa_table = load_pickle(SARSA_TABLE_PATH)

    dqn_net = QNet(N_FEATURES, N_MOVES)
    dqn_net.load_state_dict(
        torch.load(
            DQN_WEIGHTS_PATH,
            map_location="cpu",
            weights_only=True,
        )
    )
    dqn_net.eval()

    return {
        "RandomLegal": RandomLegalAgent(seed=123),
        "SimpleExpectancy": SimpleExpectancyAgent(),
        "QLearning": QLearningAgent(q_learning_table),
        "DQN": DQNAgent(dqn_net),
        "SARSA": SARSAAgent(sarsa_table),
        "MonteCarlo": MonteCarloAgent(q_path=MONTE_CARLO_TABLE_PATH),
    }


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate every agent on the same episode seeds."
    )
    parser.add_argument("--episodes", type=positive_int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    for name, agent in load_agents().items():
        result = evaluate(
            agent,
            n_episodes=args.episodes,
            seed=args.seed,
        )
        print(name, result)


if __name__ == "__main__":
    main()
