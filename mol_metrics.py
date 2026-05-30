"""Molecular evaluation metrics for generated SMILES."""

from __future__ import annotations

import math
import logging
from typing import List, Optional, Sequence

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, BRICS, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from scipy.stats import wasserstein_distance

try:
    from fcd_torch import FCD as _FCDMetric
    _FCD_AVAILABLE = True
except ImportError:
    _FCD_AVAILABLE = False

RDLogger.logger().setLevel(RDLogger.CRITICAL)
logging.getLogger("deepchem").setLevel(logging.ERROR)

def _build_medchem_catalog() -> FilterCatalog:
    params = FilterCatalogParams()
    for catalog in (
        FilterCatalogParams.FilterCatalogs.PAINS,
        FilterCatalogParams.FilterCatalogs.BRENK,
        FilterCatalogParams.FilterCatalogs.NIH,
        FilterCatalogParams.FilterCatalogs.ZINC,
    ):
        params.AddCatalog(catalog)
    return FilterCatalog(params)

def _morgan_fps(mols: List[Chem.Mol], radius: int = 2, n_bits: int = 2048):
    return [AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=n_bits) for m in mols]

def _internal_diversity(fps) -> float:
    if len(fps) < 2:
        return 0.0
    distances = []
    for i in range(len(fps)):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        distances.extend(1.0 - s for s in sims)
    return float(np.mean(distances))

def _fragment_similarity(gen_mols: List[Chem.Mol], ref_mols: List[Chem.Mol]) -> float:
    def frags(mol_list):
        result = set()
        for m in mol_list:
            result.update(BRICS.BRICSDecompose(m))
        return result

    gen_f, ref_f = frags(gen_mols), frags(ref_mols)
    if not gen_f:
        return 0.0
    return len(gen_f & ref_f) / len(gen_f)

def _scaffold_similarity(gen_mols: List[Chem.Mol], ref_mols: List[Chem.Mol]) -> float:
    def scaffolds(mol_list):
        return {MurckoScaffold.MurckoScaffoldSmiles(mol=m) for m in mol_list}

    gen_s, ref_s = scaffolds(gen_mols), scaffolds(ref_mols)
    if not gen_s:
        return 0.0
    return len(gen_s & ref_s) / len(gen_s)

def _wasserstein_distances(
    gen_mols: List[Chem.Mol],
    ref_mols: List[Chem.Mol],
) -> dict:
    props = {
        "logp": Descriptors.MolLogP,
        "mw":   Descriptors.MolWt,
        "tpsa": Descriptors.TPSA,
    }
    return {
        name: wasserstein_distance(
            [fn(m) for m in gen_mols],
            [fn(m) for m in ref_mols],
        )
        for name, fn in props.items()
    }

def compute_sas(mol: Optional[Chem.Mol]) -> float:
    """Synthetic Accessibility Score (simplified proxy)."""
    if mol is None:
        return 10.0
    n_atoms  = mol.GetNumAtoms()
    n_spiro  = AllChem.CalcNumSpiroAtoms(mol)
    n_bridge = AllChem.CalcNumBridgeheadAtoms(mol)
    return n_atoms * 1.005 + math.log10(n_spiro + 1) + math.log10(n_bridge + 1)

def compute_qed(mol: Optional[Chem.Mol]) -> float:
    from rdkit.Chem import QED
    return QED.qed(mol) if mol is not None else 0.0

def compute_logp(mol: Optional[Chem.Mol]) -> float:
    from rdkit.Chem import Crippen
    return Crippen.MolLogP(mol) if mol is not None else -4.0

def evaluate_molecules(
    generated_smiles: Sequence[str],
    reference_smiles: Sequence[str],
    *,
    fcd_device: str = "cpu",
    fcd_n_jobs: int = 1,
) -> dict:
    """
    Compute the full evaluation suite for a set of generated molecules.

    Parameters
    ----------
    generated_smiles:
        SMILES strings produced by the generative model (may include duplicates
        and invalid strings – the function handles filtering internally).
    reference_smiles:
        Training / reference SMILES used to compute novelty, FCD, Wasserstein
        distances, fragment similarity, and scaffold similarity.
    fcd_device:
        Device passed to FCDMetric (``"cpu"`` or ``"cuda"``).
    fcd_n_jobs:
        Number of parallel workers for FCD computation.

    Returns
    -------
    dict
        Keys
        ----
        valid_ratio, uniqueness, novelty, int_div,
        medchem_pass_rate,
        fragment_similarity, scaffold_similarity,
        avg_sas, avg_qed, avg_logp, avg_mw,
        wasserstein_logp, wasserstein_mw, wasserstein_tpsa,
        fcd_value
    """

    all_mols = [Chem.MolFromSmiles(s) for s in generated_smiles]

    valid_mols = [
        m for m in all_mols
        if m is not None
        and m.GetNumHeavyAtoms() > 1
        and m.GetNumBonds() > 0
        and len(Chem.GetMolFrags(m)) == 1
    ]

    if not valid_mols:
        return {"error": "No valid molecules in generated_smiles."}

    valid_smiles   = [Chem.MolToSmiles(m) for m in valid_mols]
    unique_smiles  = list(dict.fromkeys(valid_smiles))
    unique_mols    = [Chem.MolFromSmiles(s) for s in unique_smiles]

    ref_mols = [Chem.MolFromSmiles(s) for s in reference_smiles if s]
    ref_mols = [m for m in ref_mols if m is not None]
    ref_set  = set(reference_smiles)

    uniqueness   = len(unique_smiles) / len(valid_mols)
    novel_smiles = [s for s in unique_smiles if s not in ref_set]
    novelty      = len(novel_smiles) / len(unique_smiles) if unique_smiles else 0.0

    fps     = _morgan_fps(unique_mols)
    int_div = _internal_diversity(fps)

    catalog         = _build_medchem_catalog()
    medchem_pass    = sum(1 for m in unique_mols if not catalog.HasMatch(m))
    medchem_pass_rate = medchem_pass / len(unique_mols)

    frag_sim   = _fragment_similarity(unique_mols, ref_mols) if ref_mols else 0.0
    scaff_sim  = _scaffold_similarity(unique_mols, ref_mols) if ref_mols else 0.0
    w_dist     = _wasserstein_distances(unique_mols, ref_mols) if ref_mols else {"logp": 0.0, "mw": 0.0, "tpsa": 0.0}

    sas_scores = [compute_sas(m)             for m in unique_mols]
    qed_scores = [compute_qed(m)             for m in unique_mols]
    logp_scores = [compute_logp(m)           for m in unique_mols]
    mw_scores   = [Descriptors.MolWt(m)      for m in unique_mols]

    fcd_value = 100.0
    if _FCD_AVAILABLE and ref_mols and unique_smiles:
        try:
            fcd_metric = _FCDMetric(device=fcd_device, n_jobs=fcd_n_jobs)
            fcd_value  = float(fcd_metric(unique_smiles, list(reference_smiles)))
            print('Sucess')
        except Exception as exc:
            logging.getLogger(__name__).warning("FCD computation failed: %s", exc)

    return {
        "uniqueness":          round(uniqueness,   4),
        "novelty":             round(novelty,      4),
        "int_div":             round(int_div,      4),
        "medchem_pass_rate":   round(medchem_pass_rate, 4),
        "avg_sas":             round(float(np.mean(sas_scores)),  4),
        "avg_qed":             round(float(np.mean(qed_scores)),  4),
        "avg_logp":            round(float(np.mean(logp_scores)), 4),
        "avg_mw":              round(float(np.mean(mw_scores)),   4),
        "fragment_similarity": round(frag_sim,  4),
        "scaffold_similarity": round(scaff_sim, 4),
        "wasserstein_logp":    round(w_dist["logp"], 4),
        "wasserstein_mw":      round(w_dist["mw"],   4),
        "wasserstein_tpsa":    round(w_dist["tpsa"],  4),
        "fcd_value":           round(fcd_value, 4),
    }
