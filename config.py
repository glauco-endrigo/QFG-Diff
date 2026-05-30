"""Configuration for QFG-Diff experiments."""

from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
BASE_DIR = PROJECT_DIR / "data"
GENERATED_DIR = BASE_DIR / "generated_molecules"

PREPARED_DATA_PATH = BASE_DIR / "prepared_data.pkl"
RESULTS_PATH = BASE_DIR / "master_results.csv"

SEED = 42
TORCH_SEED = 2284704428437296257
CUDA_SEED = 5570312346629078
CUDNN_DETERMINISTIC = True
CUDNN_BENCHMARK = False

DATASET = "QM9"
NUM_ATOMS = 12
ATOM_LABELS = [0, 6, 7, 8, 9]

EPOCHS = 300
BATCH_SIZE = 256
NUM_GENERATED = 1000


def ensure_directories() -> None:
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    ensure_directories()
    print("QFG-Diff configuration")
    print(f"  dataset: {DATASET}")
    print(f"  base_dir: {BASE_DIR}")
    print(f"  prepared_data: {PREPARED_DATA_PATH}")
    print(f"  results: {RESULTS_PATH}")
    print(f"  num_atoms: {NUM_ATOMS}")
    print(f"  atom_labels: {ATOM_LABELS}")
    print(f"  epochs: {EPOCHS}")
    print(f"  batch_size: {BATCH_SIZE}")
