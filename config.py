# =============================================================================
# Project Hilbert-Link — Centralized Configuration
# =============================================================================
# Single source of truth for ALL notebooks and scripts.
#   • data_preparation.ipynb
#   • Copy_of_Projecet_Hilbert_Link_Mark_1.ipynb  (model + training)
#   • analysis.ipynb
#
# Usage (in any notebook / script):
#   import sys; sys.path.append('/content/drive/MyDrive/MolGAN_Experiments/')
#   from config import *          # pulls every constant into the namespace
#   # — or —
#   import config as cfg          # keeps constants namespaced as cfg.BASE_DIR, etc.
# =============================================================================


# ── 1. PATHS ──────────────────────────────────────────────────────────────────

BASE_DIR         = 'data'

# Sub-directories (created automatically where needed)
GENERATED_DIR    = BASE_DIR + 'Generated_Molecules/'

# Key files
PREPARED_DATA_PATH   = BASE_DIR + '/prepared_data.pkl'
MASTER_REGISTRY_PATH = BASE_DIR + '/master_unified_registry_v27.csv'
RESULTS_PATH         = BASE_DIR + '/master_results.csv'   # used by analysis.ipynb


# ── 2. REPRODUCIBILITY ────────────────────────────────────────────────────────

SEED       = 42                    # numpy / random
TORCH_SEED = 2284704428437296257   # torch.manual_seed
CUDA_SEED  = 5570312346629078      # torch.cuda.manual_seed_all

# Hardware determinism (slows training slightly but guarantees reproducibility)
CUDNN_DETERMINISTIC = True
CUDNN_BENCHMARK     = False


# ── 3. DATA & FEATURIZER ──────────────────────────────────────────────────────

DATASET      = 'tox21'
DATASET_SIZE = 6258                          # molecules after loading

NUM_ATOMS    = 12                            # max atoms per molecule
ATOM_LABELS  = [0, 5, 6, 7, 8, 9, 11, 12, 13, 14]


# ── 4. EXPERIMENT IDENTITY ────────────────────────────────────────────────────
# ← Change these for every new run

EXPERIMENT_NOTES_SHORT = ''

EXPERIMENT_NOTES = """
Quantum Hybrid MolGAN — Project Hilbert-Link
• 9 qubits + AmplitudeEmbedding (64 → 512)
• PauliZ measurement on qubit 4
• WGAN-GP with Gradient Penalty
• 50 epochs, generator_steps=0.2
"""


# ── 5. TRAINING ───────────────────────────────────────────────────────────────

# Learning-rate schedule  →  ExponentialDecay(LR_INIT, LR_DECAY_RATE, LR_DECAY_STEPS)
LR_INIT        = 0.001
LR_DECAY_RATE  = 0.9
LR_DECAY_STEPS = 5000

EPOCHS          = 300
BATCH_SIZE      = 256
GENERATOR_STEPS = 0.5 #0.2    # fraction of discriminator steps used for generator updates

# MolGAN graph dimensions (must be consistent with featurizer)
GAN_VERTICES = NUM_ATOMS   # == 12
GAN_NODES    = 5
GAN_EDGES    = 5


# ── 6. GENERATION ─────────────────────────────────────────────────────────────

NUM_GENERATED = 1000   # molecules sampled from the trained generator per evaluation


# ── 7. LOSS / RL ─────────────────────────────────────────────────────────────

# Generator total loss = LAMBDA_WGAN * L_WGAN  +  (1 - LAMBDA_WGAN) * L_RL
LAMBDA_WGAN = 0.7

# Comma-separated chemical properties used as RL reward signal
REWARD_METRICS = 'qed,sa,logp'

# Dimension of the latent noise vector fed to the generator
Z_DIM = 10

# WGAN gradient penalty coefficient (λ in the GP term)
GRADIENT_PENALTY_WEIGHT = 10.0

# Human-readable tags stored in the run registry
GRADIENT_PENALTY = 'WGAN-GP'
ENTROPY_REG      = 'Not used'
N_TRIALS = 1

# ── 8. QUANTUM CIRCUIT ────────────────────────────────────────────────────────

N_QUBITS           = 9
N_LAYERS           = 3    # StronglyEntanglingLayers depth
MEASURED_QUBIT_IDX = 4    # index of the PauliZ-measured qubit
DIFF_METHOD        = 'backprop'   # PennyLane differentiation method

# Derived weight shape for qml.qnn.TorchLayer
WEIGHT_SHAPES = {'weights': (N_LAYERS, N_QUBITS, 3)}

# Bridge layer: classical → quantum input dimension  (2 ** N_QUBITS)
QUANTUM_INPUT_DIM = 2 ** N_QUBITS   # 512


# ── 9. SANITY CHECK (optional, run once) ──────────────────────────────────────

if __name__ == '__main__':
    import os
    print('=== Project Hilbert-Link — Config ===')
    print(f'  BASE_DIR          : {BASE_DIR}')
    print(f'  MODEL_TYPE        : {MODEL_TYPE}')
    print(f'  DATASET           : {DATASET}  ({DATASET_SIZE} molecules)')
    print(f'  NUM_ATOMS         : {NUM_ATOMS}')
    print(f'  ATOM_LABELS       : {ATOM_LABELS}')
    print(f'  EPOCHS            : {EPOCHS}')
    print(f'  BATCH_SIZE        : {BATCH_SIZE}')
    print(f'  GENERATOR_STEPS   : {GENERATOR_STEPS}')
    print(f'  LR_INIT           : {LR_INIT}')
    print(f'  LAMBDA_WGAN       : {LAMBDA_WGAN}')
    print(f'  REWARD_METRICS    : {REWARD_METRICS}')
    print(f'  Z_DIM             : {Z_DIM}')
    print(f'  N_QUBITS          : {N_QUBITS}  (input_dim = {QUANTUM_INPUT_DIM})')
    print(f'  N_LAYERS          : {N_LAYERS}')
    print(f'  MEASURED_QUBIT    : {MEASURED_QUBIT_IDX}')
    print(f'  DIFF_METHOD       : {DIFF_METHOD}')
    print(f'  TORCH_SEED        : {TORCH_SEED}')
    print(f'  CUDA_SEED         : {CUDA_SEED}')
    os.makedirs(BASE_DIR, exist_ok=True)
    os.makedirs(GENERATED_DIR, exist_ok=True)
    print('\nDirectories verified / created.')
