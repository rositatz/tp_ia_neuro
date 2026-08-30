import argparse
import time

import numpy as np

from config import HORIZON
from env import GoldDiceEnv, SCORE, N_ACTIONS
from mc_agent import (
    MonteCarloAgent,
    encode_state,
    valid_action_mask,
    DEFAULT_Q_PATH,
)


def greedy_from_q(Q, state, mask, rng):
    q = Q.get(state)
    valid = np.flatnonzero(mask)
    if q is None:
        return int(rng.choice(valid))
    masked = np.where(mask, q, -np.inf)
    best = masked.max()
    candidates = np.flatnonzero((masked == best) & mask)
    return int(rng.choice(candidates))


def run_episode(env, Q, epsilon, rng, train=True, alpha=0.05):
    obs = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
    done = False
    trajectory = []  # (state, action, reward)

    while not done:
        turns_left = HORIZON - int(obs["turn"])
        mask = valid_action_mask(obs)
        state = encode_state(obs)

        if turns_left == 0:
            action = SCORE
            score_amount = int(obs["gold"])
        elif train and rng.random() < epsilon:
            action = int(rng.choice(np.flatnonzero(mask)))
            score_amount = int(obs["gold"]) if action == SCORE else None
        else:
            action = greedy_from_q(Q, state, mask, rng)
            score_amount = int(obs["gold"]) if action == SCORE else None

        obs, reward, done, _ = env.step(action, score_amount=score_amount)
        trajectory.append((state, action, reward))

    if train:
        G = 0.0
        for state, action, reward in reversed(trajectory):
            G += reward
            q = Q.get(state)
            if q is None:
                q = np.zeros(N_ACTIONS, dtype=np.float64)
                Q[state] = q
            q[action] += alpha * (G - q[action])

    return trajectory[-1] and env.points


def evaluate(Q, n_episodes=500, seed=0):
    agent = MonteCarloAgent(Q=Q, epsilon=0.0)
    scores = np.empty(n_episodes)
    for ep in range(n_episodes):
        env = GoldDiceEnv(obs_mode="dict", seed=seed + ep, track_history=False)
        obs = env.reset(seed=seed + ep)
        done = False
        while not done:
            a, sc = agent.act(obs, env)
            obs, _, done, _ = env.step(a, score_amount=sc)
        scores[ep] = env.points
    return scores.mean(), scores.std()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=600_000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--eps-start", type=float, default=1.0)
    parser.add_argument("--eps-end", type=float, default=0.05)
    parser.add_argument("--eps-decay-frac", type=float, default=0.8,
                        help="fraction of training over which epsilon decays")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default=DEFAULT_Q_PATH)
    parser.add_argument("--eval-every", type=int, default=50_000)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    env = GoldDiceEnv(obs_mode="dict", seed=args.seed, track_history=False)
    Q = {}

    decay_episodes = max(1, int(args.episodes * args.eps_decay_frac))
    best_mean = -np.inf
    best_Q = None

    t0 = time.time()
    for ep in range(args.episodes):
        frac = min(1.0, ep / decay_episodes)
        epsilon = args.eps_start + frac * (args.eps_end - args.eps_start)
        run_episode(env, Q, epsilon, rng, train=True, alpha=args.alpha)

        if (ep + 1) % args.eval_every == 0 or (ep + 1) == args.episodes:
            mean, std = evaluate(Q, n_episodes=500, seed=10_000)  # held-out seeds
            elapsed = time.time() - t0
            print(f"ep {ep + 1:>8,} | eps {epsilon:.3f} | states {len(Q):>7,} "
                  f"| eval mean {mean:7.1f} +/- {std:5.1f} | {elapsed:6.1f}s")
            if mean > best_mean:
                best_mean = mean
                best_Q = {s: q.copy() for s, q in Q.items()}

    final = best_Q if best_Q is not None else Q
    MonteCarloAgent(Q=final).save_q(args.out)
    print(f"\nSaved {len(final):,} states to {args.out} | best eval mean {best_mean:.1f}")


if __name__ == "__main__":
    main()
