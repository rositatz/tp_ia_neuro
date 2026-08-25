from train_qlearning import train_q_learning

Q, visits, rewards, history = train_q_learning(n_episodes=2000, eval_every=500)
print("listo, |Q| =", len(Q))