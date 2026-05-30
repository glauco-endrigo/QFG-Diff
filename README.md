# QFG-Diff

This folder contains the code used for the QFG-Diff molecular generation experiments.

QFG-Diff is a discrete graph diffusion framework for generating small molecular graphs from QM9. The project compares three variants:

- `train_diffusion_mol_baseline.py`: baseline discrete molecular graph diffusion.
- `train_diffusion_mol_fragment_classic.py`: classical fragment-guided diffusion using BRICS fragments.
- `train_diffusion_mol_orca.py`: ORCA-guided diffusion using photonic fragment signatures.

Supporting files:

- `data_preparation_qm9.ipynb`: prepares and filters QM9 molecules, then saves graph tensors.
- `mol_metrics.py`: computes molecular generation metrics such as validity, uniqueness, novelty, internal diversity, scaffold similarity, fragment similarity, QED, SAS, LogP, and FCD.
- `config.py`: central configuration for paths, dataset settings, atom labels, batch size, and training epochs.

Basic workflow:

1. Run `data_preparation_qm9.ipynb` to create `data/prepared_data.pkl`.
2. Run one of the training scripts.
3. Check generated results in `data/master_results.csv`.

The experiments focus on whether fragment guidance improves scaffold and fragment preservation while keeping generated molecules valid, novel, and diverse.
