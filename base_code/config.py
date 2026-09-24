"""Shared tunable constants."""

# --- Reproducibility ---
SEED = 42

# --- Simulation sizes ---
TRAIN_POPULATION_SIZE = 1000  # used whenever a training population size isn't explicitly given
TEST_POPULATION_SIZE = 20_000  # used whenever a test population size isn't explicitly given

# --- Replication ---
N_REPS = 50  # training/evaluation replications averaged into each learner's mean ATE bias
