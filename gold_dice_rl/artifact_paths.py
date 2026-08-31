from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent

QLEARNING_TABLE_PATH = PROJECT_DIR / "q_table.pkl"
SARSA_TABLE_PATH = PROJECT_DIR / "q_table_sarsa.pkl"
MONTE_CARLO_TABLE_PATH = PROJECT_DIR / "mc_q.pkl"
DQN_WEIGHTS_PATH = PROJECT_DIR / "dqn_weights.pt"
EVALUATION_RESULTS_PATH = PROJECT_DIR / "evaluation_results.json"
