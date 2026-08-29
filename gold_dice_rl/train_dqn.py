import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from env import GoldDiceEnv
from agents import state_vector, move_to_action, N_MOVES, QNet, DQNAgent
from train_qlearning import epsilon_schedule 


def epsilon_greedy_dqn(net, x, epsilon, rng):
    if rng.random() < epsilon:
        return int(rng.integers(N_MOVES))
    with torch.no_grad():
        q_values = net(torch.from_numpy(x).unsqueeze(0)).squeeze(0).numpy()
    return int(np.argmax(q_values))


def train_dqn(
    n_episodes=20000,
    lr=1e-3,             # learning rate del optimizador
    gamma=1.0,
    eps_start=1.0,
    eps_end=0.05,
    buffer_size=50000,
    min_buffer_size=1000,
    batch_size=64,
    target_sync_every=500,  # cada cuántas actualizaciones sincroniza la target network
    eval_every=1000,
    seed=0,
    train_seed_offset=100000,
):

    random.seed(seed)      
    torch.manual_seed(seed) 

    q_net = QNet(8, N_MOVES)
    target_net = QNet(8, N_MOVES)
    target_net.load_state_dict(q_net.state_dict()) 
    target_net.eval()

    optimizer = optim.Adam(q_net.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    memory = deque(maxlen=buffer_size)
    rng = np.random.default_rng(seed)

    rewards = []
    history = {"episode": [], "train_reward": []}
    train_steps = 0

    for ep in range(n_episodes):
        epsilon = epsilon_schedule(ep, n_episodes, eps_start, eps_end)

        train_seed = train_seed_offset + ep
        env = GoldDiceEnv(obs_mode="dict", seed=train_seed, track_history=False)
        obs = env.reset(seed=train_seed)
        x = state_vector(obs)
        done = False
        total_reward = 0.0

        while not done:
            move = epsilon_greedy_dqn(q_net, x, epsilon, rng)
            action, score_amount = move_to_action(move, obs, env)

            next_obs, reward, done, info = env.step(action, score_amount=score_amount)
            next_x = state_vector(next_obs)

            # guarda la transición en la memoria en vez de entrenar ya mismo
            memory.append((x, move, reward, next_x, done))
            total_reward += reward
            obs, x = next_obs, next_x

            # entrena con un lote al azar de la memoria 
            if len(memory) >= min_buffer_size:
                batch = random.sample(memory, batch_size)
                states, moves, batch_rewards, next_states, dones = zip(*batch)

                states = torch.from_numpy(np.array(states))
                next_states = torch.from_numpy(np.array(next_states))
                moves_t = torch.tensor(moves, dtype=torch.long)
                rewards_t = torch.tensor(batch_rewards, dtype=torch.float32)
                dones_t = torch.tensor(dones, dtype=torch.float32)

                # Q(S,A) estimado por la red, para las jugadas que realmente se tomaron
                q_pred = q_net(states).gather(1, moves_t.unsqueeze(1)).squeeze(1)

                with torch.no_grad():
                    q_next = target_net(next_states).max(dim=1).values
                    target = rewards_t + gamma * q_next * (1.0 - dones_t)


                loss = loss_fn(q_pred, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_steps += 1
                if train_steps % target_sync_every == 0:
                    target_net.load_state_dict(q_net.state_dict())

        rewards.append(total_reward)

        if (ep + 1) % eval_every == 0 or ep == n_episodes - 1:
            start = max(0, len(rewards) - 500)
            recent_reward = np.mean(rewards[start:])
            history["episode"].append(ep + 1)
            history["train_reward"].append(recent_reward)
            print(f"Episodio {ep+1}/{n_episodes}  eps={epsilon:.3f}  reward_reciente={recent_reward:.2f}")

    return q_net, rewards, history


if __name__ == "__main__":
    q_net, rewards, history = train_dqn(n_episodes=20000)
    torch.save(q_net.state_dict(), "dqn_weights.pt")
    print("Guardado dqn_weights.pt")