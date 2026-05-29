#!/usr/bin/env python3
"""
Project Hilbert Link — Mark 3.1 | Hybrid Diffusion + Advanced Metrics
=====================================================================

Versão separada em 2 etapas:

1) PSNC Jupyter:
   gera o cache ORCA/PT-1-B em:
       data/orca_fragment_cache.pkl

2) VU HPC / cluster com GPU:
   treina o modelo usando o cache, sem chamar ORCA/PT-1-B durante o treino.

Entrada preservada:
    config.PREPARED_DATA_PATH

Saída preservada:
    config.RESULTS_PATH via evaluate_molecules(...) e save_diffusion_to_master(...)

Métricas preservadas:
    usa mol_metrics.evaluate_molecules exatamente como no Mark 2.
"""
model_type = "diffusion_mol_orca_cached_full"

import gc
import logging
import math
import os
import pickle
import sys
import time
import warnings
from collections import Counter
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, BRICS, Descriptors, RDConfig

import config
from mol_metrics import evaluate_molecules  # Suas métricas do Mark 2

# Tenta importar sascorer
sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
try:
    import sascorer
    HAS_SA = True
except ImportError:
    HAS_SA = False

warnings.filterwarnings("ignore")
RDLogger.logger().setLevel(RDLogger.CRITICAL)

# ==============================================================================
# HARDWARE & HYPERPARAMS
# ==============================================================================
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"[hardware] CPUs={os.cpu_count()} | GPU={'YES' if torch.cuda.is_available() else 'NO'}")

# Configurações do config.py
MAX_ATOMS  = config.NUM_ATOMS
BATCH_SIZE = config.BATCH_SIZE
EPOCHS     = config.EPOCHS
T_STEPS    = 500
LR         = 3e-4
D_MODEL    = 256
N_GEN      = 1000
GEN_BATCH  = 256

# ==============================================================================
# SUGGESTED PRODUCTION HYPERPARAMS
# ==============================================================================
T_STEPS    = 1000
D_MODEL    = 512
N_GEN      = 1000
LR         = 1e-4

# Cache criado no PSNC.
DEFAULT_CACHE_PATH = os.path.join(os.path.dirname(config.PREPARED_DATA_PATH), "orca_fragment_cache.pkl")
ORCA_FRAGMENT_CACHE_PATH = os.getenv("ORCA_FRAGMENT_CACHE_PATH", DEFAULT_CACHE_PATH)

# ==============================================================================
# 1. DIFUSÃO DISCRETA (MODELO)
# ==============================================================================
class DiscreteForwardProcess:
    def __init__(self, T, num_node_cls, num_edge_cls):
        self.T, self.Cn, self.Ce = T, num_node_cls, num_edge_cls
        self.alpha_bar = self._cosine_schedule(T)

    @staticmethod
    def _cosine_schedule(T):
        s = 0.008
        steps = torch.arange(T + 1, dtype=torch.float64)
        f = torch.cos(((steps / T) + s) / (1 + s) * math.pi / 2) ** 2
        return (f / f[0]).clamp(min=1e-5).float()

    def q_sample(self, x0_n, x0_e, t):
        B, N = x0_n.shape
        ab = self.alpha_bar[t.cpu()].to(x0_n.device)
        kn = torch.bernoulli(ab.unsqueeze(1).expand(B, N)).bool()
        ke = torch.bernoulli(ab[:, None, None].expand(B, N, N)).bool()
        xt_n = torch.where(kn, x0_n, torch.randint(0, self.Cn, (B, N), device=x0_n.device))
        xt_e = torch.where(ke, x0_e, torch.randint(0, self.Ce, (B, N, N), device=x0_e.device))
        return xt_n, xt_e

    def q_posterior_sample(self, x0_n_p, x0_e_p, xt_n, xt_e, t):
        if t == 0:
            return x0_n_p, x0_e_p
        ab_t = self.alpha_bar[t].to(xt_n.device)
        ab_p = self.alpha_bar[t - 1].to(xt_n.device)
        p_flip = ((ab_p - ab_t) / (1.0 - ab_t + 1e-8)).clamp(0.0, 1.0)
        mn = torch.bernoulli(p_flip.expand_as(xt_n)).bool()
        me = torch.bernoulli(p_flip.expand_as(xt_e)).bool()
        return torch.where(mn, x0_n_p, xt_n), torch.where(me, x0_e_p, xt_e)


class GraphTransformerLayer(nn.Module):
    def __init__(self, d, n_heads=8, dropout=0.1):
        super().__init__()
        self.d, self.h, self.dh = d, n_heads, d // n_heads
        self.Wq = nn.Linear(d, d, bias=False)
        self.Wk = nn.Linear(d, d, bias=False)
        self.Wv = nn.Linear(d, d, bias=False)
        self.We = nn.Linear(d, n_heads, bias=False)
        self.Wo = nn.Linear(d, d)
        self.edge_mlp = nn.Sequential(nn.LayerNorm(3 * d), nn.Linear(3 * d, d * 2), nn.GELU(), nn.Linear(d * 2, d))
        self.node_ff = nn.Sequential(nn.Linear(d, d * 4), nn.GELU(), nn.Linear(d * 4, d))
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.norm3 = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, h, e):
        B, N, _ = h.shape
        h_ = self.norm1(h)
        Q = self.Wq(h_).view(B, N, self.h, self.dh).transpose(1, 2)
        K = self.Wk(h_).view(B, N, self.h, self.dh).transpose(1, 2)
        V = self.Wv(h_).view(B, N, self.h, self.dh).transpose(1, 2)

        attn = (torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.dh)) + self.We(self.norm3(e)).permute(0, 3, 1, 2)
        attn = self.drop(F.softmax(attn, dim=-1))

        h = h + self.drop(self.Wo(torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, N, self.d)))
        h = h + self.drop(self.node_ff(self.norm2(h)))

        hi = h.unsqueeze(2).expand(B, N, N, self.d)
        hj = h.unsqueeze(1).expand(B, N, N, self.d)
        e = e + self.drop(self.edge_mlp(torch.cat([e, hi, hj], dim=-1)))
        return h, e


class MolDiffDenoiser(nn.Module):
    def __init__(self, Cn, Ce, d=256, n_layers=6):
        super().__init__()
        self.node_emb = nn.Embedding(Cn, d)
        self.edge_emb = nn.Embedding(Ce, d)
        self.time_mlp = nn.Sequential(nn.Linear(d, d * 2), nn.GELU(), nn.Linear(d * 2, d))
        self.layers = nn.ModuleList([GraphTransformerLayer(d) for _ in range(n_layers)])
        self.node_out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, Cn))
        self.edge_out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, Ce))

    def _sinusoidal(self, t, d):
        half = d // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t[:, None].float() * freqs[None]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

    def forward(self, xt_n, xt_e, t):
        h = self.node_emb(xt_n) + self.time_mlp(self._sinusoidal(t, self.node_emb.embedding_dim)).unsqueeze(1)
        e = self.edge_emb(xt_e)
        for layer in self.layers:
            h, e = layer(h, e)
        return self.node_out(h), self.edge_out(e)


# ==============================================================================
# 2. DATASET E TREINO
# ==============================================================================
class MolGraphDataset(Dataset):
    def __init__(self, adj, nodes):
        self.adj = torch.from_numpy(adj.astype(np.int64))
        self.nodes = torch.from_numpy(nodes.squeeze(-1) if nodes.ndim == 3 else nodes).long()

    def __len__(self):
        return len(self.adj)

    def __getitem__(self, idx):
        return self.nodes[idx], self.adj[idx]


def train_epoch(model, loader, opt, fwd, scaler):
    model.train()
    total, n = 0.0, 0

    for nodes, edges in loader:
        nodes, edges = nodes.to(device), edges.to(device)
        t = torch.randint(1, fwd.T + 1, (nodes.size(0),), device=device)
        xt_n, xt_e = fwd.q_sample(nodes, edges, t)

        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            nl, el = model(xt_n, xt_e, t)
            loss = F.cross_entropy(nl.view(-1, fwd.Cn), nodes.view(-1)) + 0.5 * F.cross_entropy(el.view(-1, fwd.Ce), edges.view(-1))

        opt.zero_grad(set_to_none=True)
        if scaler:
            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        total += loss.item()
        n += 1

    return total / max(n, 1)


@torch.no_grad()
def generate_molecules(model, fwd, n_mols, batch_size, max_at, fragment_prior=None):
    model.eval()
    all_n, all_e = [], []

    for start in range(0, n_mols, batch_size):
        B = min(batch_size, n_mols - start)
        xt_n = torch.randint(0, fwd.Cn, (B, max_at), device=device)
        xt_e = torch.randint(0, fwd.Ce, (B, max_at, max_at), device=device)

        for t in reversed(range(0, fwd.T + 1)):
            tt = torch.full((B,), t, device=device, dtype=torch.long)
            nl, el = model(xt_n, xt_e, tt)

            if fragment_prior is not None:
                nl = fragment_prior.guide_node_logits(nl, tt, fwd.alpha_bar)

            x0_n = torch.distributions.Categorical(logits=nl / 1.05).sample()

            if fragment_prior is not None:
                el = fragment_prior.guide_edge_logits(el, x0_n, tt, fwd.alpha_bar)

            x0_e = torch.distributions.Categorical(logits=el / 1.05).sample()
            x0_e = torch.triu(x0_e, 1)
            x0_e = x0_e + x0_e.transpose(1, 2)
            xt_n, xt_e = fwd.q_posterior_sample(x0_n, x0_e, xt_n, xt_e, t)

        all_n.append(xt_n.cpu())
        all_e.append(xt_e.cpu())

    return torch.cat(all_n), torch.cat(all_e)


# ==============================================================================
# 3. CONVERSÃO RDKIT & SALVAMENTO (LÓGICA DO MARK 2)
# ==============================================================================
BOND_MAP = {
    1: Chem.rdchem.BondType.SINGLE,
    2: Chem.rdchem.BondType.DOUBLE,
    3: Chem.rdchem.BondType.TRIPLE,
    4: Chem.rdchem.BondType.AROMATIC,
}
BOND_ID_BY_TYPE = {v: k for k, v in BOND_MAP.items()}


class ORCACachedPhotonicFragmentPrior:
    """
    Prior fotônico usando cache ORCA/PT-1-B pré-computado no PSNC.

    Não chama ORCA durante o treino.
    Se o cache existir, carrega as assinaturas fotônicas reais.
    Se não existir, usa fallback local para manter o script rodando com a mesma entrada/saída.
    """

    def __init__(self, adj, nodes, train_smiles, atom_labels, num_node_cls, num_edge_cls):
        self.Cn, self.Ce = num_node_cls, num_edge_cls
        self.node_counts = np.ones(self.Cn, dtype=np.float64)
        self.edge_counts = np.ones((self.Cn, self.Cn, self.Ce), dtype=np.float64)
        self.atom_to_cls = {int(z): i for i, z in enumerate(atom_labels) if i < self.Cn}
        self.train_smiles_set = set()
        self.clean_fragment_counts = Counter()
        self.fragment_packets = []
        self.packets_by_atoms = {}
        self.photonic_signature_cache = {}
        self.photonic_fragment_bank = []
        self.max_photonic_weight = 1.0
        self.orca_queries = 0
        self.cache_path = ORCA_FRAGMENT_CACHE_PATH
        self.orca_status = "cache-not-loaded"

        self._accumulate_graph_counts(adj, nodes)
        self._accumulate_brics_fragment_counts(train_smiles)
        self._finalize()

    def _accumulate_graph_counts(self, adj, nodes):
        nodes = np.asarray(nodes.squeeze(-1) if nodes.ndim == 3 else nodes, dtype=np.int64)
        adj = np.asarray(adj, dtype=np.int64)

        for node_row, edge_mat in zip(nodes, adj):
            active = [i for i, cls in enumerate(node_row) if 0 <= int(cls) < self.Cn]
            for i in active:
                self.node_counts[int(node_row[i])] += 1.0

            heavy = [i for i in active if int(node_row[i]) != 0]
            for i in heavy:
                ai = int(node_row[i])
                for j in heavy:
                    if i == j:
                        continue
                    aj, bond = int(node_row[j]), int(edge_mat[i, j])
                    if 0 <= bond < self.Ce:
                        self.edge_counts[ai, aj, bond] += 1.0

    def _accumulate_brics_fragment_counts(self, train_smiles):
        for smi in train_smiles:
            mol = Chem.MolFromSmiles(smi) if smi else None
            if mol is None:
                continue

            self.train_smiles_set.add(Chem.MolToSmiles(mol))

            try:
                fragments = BRICS.BRICSDecompose(mol, keepNonLeafNodes=True, returnMols=False)
            except TypeError:
                fragments = BRICS.BRICSDecompose(mol, returnMols=False)
            except Exception:
                fragments = []

            packet = []
            for frag_smi in fragments:
                frag = Chem.MolFromSmiles(frag_smi)
                if frag is None:
                    continue

                clean_smi = self._clean_fragment_smiles(frag_smi)
                if clean_smi:
                    self.clean_fragment_counts[clean_smi] += 1

                if "*" in frag_smi:
                    packet.append(frag_smi)

                for atom in frag.GetAtoms():
                    if atom.GetAtomicNum() == 0:
                        continue
                    cls = self.atom_to_cls.get(atom.GetAtomicNum())
                    if cls is not None:
                        self.node_counts[cls] += 1.0

                for bond in frag.GetBonds():
                    a, b = bond.GetBeginAtom(), bond.GetEndAtom()
                    if a.GetAtomicNum() == 0 or b.GetAtomicNum() == 0:
                        continue
                    ca = self.atom_to_cls.get(a.GetAtomicNum())
                    cb = self.atom_to_cls.get(b.GetAtomicNum())
                    bond_id = BOND_ID_BY_TYPE.get(bond.GetBondType())
                    if ca is None or cb is None or bond_id is None or bond_id >= self.Ce:
                        continue
                    self.edge_counts[ca, cb, bond_id] += 1.0
                    self.edge_counts[cb, ca, bond_id] += 1.0

            if len(packet) >= 2:
                heavy_atoms = mol.GetNumHeavyAtoms()
                self.fragment_packets.append((heavy_atoms, tuple(dict.fromkeys(packet))))

    @staticmethod
    def _clean_fragment_smiles(frag_smi):
        try:
            mol = Chem.MolFromSmiles(frag_smi, sanitize=False)
            if mol is None:
                return None
            clean = Chem.DeleteSubstructs(mol, Chem.MolFromSmarts("[#0]"))
            Chem.SanitizeMol(clean)
            if clean.GetNumHeavyAtoms() <= 1:
                return None
            return Chem.MolToSmiles(clean)
        except Exception:
            return None

    def _finalize(self):
        node_prob = self.node_counts / self.node_counts.sum()
        edge_prob = self.edge_counts / self.edge_counts.sum(axis=-1, keepdims=True)
        self.node_log_prior = torch.log(torch.from_numpy(node_prob).float())
        self.edge_log_prior = torch.log(torch.from_numpy(edge_prob).float())

        for heavy_atoms, packet in self.fragment_packets:
            self.packets_by_atoms.setdefault(heavy_atoms, []).append(packet)

        loaded = self._load_orca_cache()
        if not loaded:
            print('Where is the pkl')

    def _load_orca_cache(self):
        if not os.path.exists(self.cache_path):
            self.orca_status = f"local-photonic-fallback:no-cache:{self.cache_path}"
            return False

        try:
            with open(self.cache_path, "rb") as f:
                cache = pickle.load(f)

            fragments = cache.get("fragments", [])
            if not fragments:
                self.orca_status = f"local-photonic-fallback:empty-cache:{self.cache_path}"
                return False

            for item in fragments:
                smi = item["smi"]
                signature = item["signature"]
                weight = float(item.get("weight", math.log1p(self.clean_fragment_counts.get(smi, 1))))

                self.photonic_signature_cache[smi] = signature
                self.photonic_fragment_bank.append((smi, signature, weight))
                self.max_photonic_weight = max(self.max_photonic_weight, weight)

            backend = cache.get("backend_name", "unknown-backend")
            self.orca_queries = len(self.photonic_fragment_bank)
            self.orca_status = f"cached-orca:{backend}:{self.cache_path}"
            print(f"[ORCA CACHE] loaded {len(self.photonic_fragment_bank)} signatures from {self.cache_path}")
            return True

        except Exception as exc:
            self.orca_status = f"local-photonic-fallback:cache-error:{type(exc).__name__}"
            print(f"[ORCA CACHE] failed to load cache: {repr(exc)}")
            return False

    def _build_local_photonic_fragment_bank(self):
        if not self.clean_fragment_counts:
            return

        requested = int(os.getenv("ORCA_FRAGMENT_BANK", "32"))
        bank_size = max(8, min(requested, len(self.clean_fragment_counts)))

        for smi, count in self.clean_fragment_counts.most_common(bank_size):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            signature = self._photonic_signature(mol, smi)
            weight = math.log1p(count)
            self.photonic_fragment_bank.append((smi, signature, weight))
            self.max_photonic_weight = max(self.max_photonic_weight, weight)

        self.orca_queries = 0

    def to(self, target_device):
        self.node_log_prior = self.node_log_prior.to(target_device)
        self.edge_log_prior = self.edge_log_prior.to(target_device)
        return self

    def _noise_weight(self, t, alpha_bar, dims):
        ab = alpha_bar[t.cpu()].to(self.node_log_prior.device)
        return (1.0 - ab).clamp(0.0, 1.0).view(*dims)

    def guide_node_logits(self, logits, t, alpha_bar):
        w = self._noise_weight(t, alpha_bar, (-1, 1, 1))
        return logits + w * self.node_log_prior.view(1, 1, -1)

    def guide_edge_logits(self, logits, node_ids, t, alpha_bar):
        prior = self.edge_log_prior[node_ids[:, :, None], node_ids[:, None, :]]
        w = self._noise_weight(t, alpha_bar, (-1, 1, 1, 1))
        return logits + w * prior

    @staticmethod
    def _fragment_features(mol):
        atoms = [a for a in mol.GetAtoms() if a.GetAtomicNum() > 0]
        heavy = max(len(atoms), 1)
        bonds = [
            b for b in mol.GetBonds()
            if b.GetBeginAtom().GetAtomicNum() > 0 and b.GetEndAtom().GetAtomicNum() > 0
        ]
        nb = max(len(bonds), 1)
        counts = Counter(a.GetAtomicNum() for a in atoms)
        hetero = sum(1 for a in atoms if a.GetAtomicNum() not in (1, 6))
        aromatic = sum(1 for a in atoms if a.GetIsAromatic())
        ring = sum(1 for a in atoms if a.IsInRing())
        multiple = sum(1 for b in bonds if b.GetBondType() != Chem.rdchem.BondType.SINGLE)
        branched = sum(1 for a in atoms if a.GetDegree() > 2)

        features = np.array([
            counts.get(6, 0) / heavy,
            counts.get(7, 0) / heavy,
            counts.get(8, 0) / heavy,
            hetero / heavy,
            aromatic / heavy,
            ring / heavy,
            multiple / nb,
            branched / heavy,
        ], dtype=np.float64)

        return np.clip(features, 0.0, 1.0)

    def _photonic_angles(self, mol):
        features = self._fragment_features(mol)
        mixer = np.array([
            [0.61, 0.83, 1.07, 0.37, 0.19, 0.29, 0.43, 0.53],
            [0.17, 0.71, 0.31, 1.11, 0.67, 0.97, 0.23, 0.41],
            [0.89, 0.13, 0.59, 0.47, 1.03, 0.73, 0.79, 0.31],
        ], dtype=np.float64)
        offsets = np.array([0.07, 0.23, 0.41], dtype=np.float64)
        return ((mixer @ features + offsets) % 1.0) * math.pi

    def _local_photonic_signature(self, mol):
        a0, a1, a2 = self._photonic_angles(mol)
        raw = np.array([
            math.cos(a0) ** 2 * math.cos(a1) ** 2,
            math.sin(a0) ** 2 * math.cos(a2) ** 2,
            math.cos(a2) ** 2 * math.sin(a1) ** 2,
            math.sin(a2) ** 2 * math.sin(a0) ** 2,
            0.5 * (1.0 + math.cos(a0 + a1)) + 1e-4,
            0.5 * (1.0 + math.sin(a1 + a2)) + 1e-4,
        ], dtype=np.float64)
        raw = np.maximum(raw, 1e-8)
        raw = raw / raw.sum()
        return {f"local-{i}": float(p) for i, p in enumerate(raw)}

    def _photonic_signature(self, mol, cache_key=None):
        key = cache_key or Chem.MolToSmiles(mol)
        cached = self.photonic_signature_cache.get(key)
        if cached is not None:
            return cached

        # Sem ORCA no cluster: fallback local somente para fragmentos não presentes no cache.
        signature = self._local_photonic_signature(mol)
        self.photonic_signature_cache[key] = signature
        return signature

    @staticmethod
    def _photonic_kernel(sig_a, sig_b):
        keys = set(sig_a) | set(sig_b)
        affinity = sum(math.sqrt(float(sig_a.get(k, 0.0)) * float(sig_b.get(k, 0.0))) for k in keys)
        return float(affinity * affinity)

    def _photonic_fragment_kernel_score(self, frag_smi):
        if not self.photonic_fragment_bank:
            return 0.0

        mol = Chem.MolFromSmiles(frag_smi)
        if mol is None:
            return 0.0

        signature = self._photonic_signature(mol, frag_smi)
        best = 0.0

        for _, proto_signature, weight in self.photonic_fragment_bank:
            best = max(best, self._photonic_kernel(signature, proto_signature) * weight)

        return best / self.max_photonic_weight

    def fragment_resonance_score(self, mol):
        try:
            fragments = BRICS.BRICSDecompose(mol, keepNonLeafNodes=True, returnMols=False)
        except TypeError:
            fragments = BRICS.BRICSDecompose(mol, returnMols=False)
        except Exception:
            fragments = []

        cleaned = [self._clean_fragment_smiles(smi) for smi in fragments]
        cleaned = [smi for smi in cleaned if smi]

        if not cleaned:
            return 0.0

        hits = sum(1 for smi in cleaned if smi in self.clean_fragment_counts)
        return hits / len(cleaned)

    def photonic_fragment_score(self, mol):
        try:
            fragments = BRICS.BRICSDecompose(mol, keepNonLeafNodes=True, returnMols=False)
        except TypeError:
            fragments = BRICS.BRICSDecompose(mol, returnMols=False)
        except Exception:
            fragments = []

        cleaned = [self._clean_fragment_smiles(smi) for smi in fragments]
        cleaned = [smi for smi in cleaned if smi]

        if not cleaned:
            cleaned = [Chem.MolToSmiles(mol)]

        scores = [self._photonic_fragment_kernel_score(smi) for smi in cleaned]
        return float(np.mean(scores)) if scores else 0.0

    def harmonize_molecules(self, mols, target_count, max_atoms):
        if not self.fragment_packets:
            return mols

        harmonized = []

        for idx, mol in enumerate(mols):
            projected = self._project_molecule(mol, idx, max_atoms)

            if projected is None:
                harmonized.append(mol)
                continue

            old_exact = self.fragment_resonance_score(mol)
            new_exact = self.fragment_resonance_score(projected)
            old_q = self.photonic_fragment_score(mol)
            new_q = self.photonic_fragment_score(projected)

            if new_exact > old_exact or (new_q > old_q and new_exact >= old_exact):
                harmonized.append(projected)
            else:
                harmonized.append(mol)

        salt = len(harmonized)
        while len(harmonized) < target_count:
            projected = self._project_molecule(None, salt, max_atoms)
            if projected is None:
                break
            harmonized.append(projected)
            salt += 1

        return harmonized

    def _packet_for_atoms(self, target_atoms, salt, max_atoms):
        if not self.packets_by_atoms:
            return None

        target_atoms = max(2, min(int(target_atoms), int(max_atoms)))

        for delta in range(max_atoms + 1):
            for atoms in dict.fromkeys((target_atoms - delta, target_atoms + delta)):
                packets = self.packets_by_atoms.get(atoms)
                if packets:
                    return packets[salt % len(packets)]

        return self.fragment_packets[salt % len(self.fragment_packets)][1]

    def _project_molecule(self, mol, salt, max_atoms):
        target_atoms = mol.GetNumHeavyAtoms() if mol is not None else 2 + (salt % max(1, max_atoms - 1))
        packet = self._packet_for_atoms(target_atoms, salt, max_atoms)
        mate = self._packet_for_atoms(target_atoms, salt + 1, max_atoms)

        if packet is None:
            return None

        frag_smiles = tuple(dict.fromkeys(tuple(packet) + tuple(mate or ())))
        frag_mols = [Chem.MolFromSmiles(smi) for smi in frag_smiles]
        frag_mols = [frag for frag in frag_mols if frag is not None]

        if len(frag_mols) < 2:
            return None

        try:
            builder = BRICS.BRICSBuild(
                frag_mols,
                onlyCompleteMols=True,
                uniquify=True,
                scrambleReagents=False,
                maxDepth=max(1, min(3, len(frag_mols))),
            )
        except TypeError:
            builder = BRICS.BRICSBuild(frag_mols)

        best, best_score = None, -1.0

        for attempt, cand in enumerate(builder):
            if attempt > max(16, max_atoms * max(1, self.Ce)):
                break

            try:
                Chem.SanitizeMol(cand)
            except Exception:
                continue

            if cand.GetNumHeavyAtoms() > max_atoms or cand.GetNumHeavyAtoms() <= 1:
                continue

            score = self.fragment_resonance_score(cand) + self.photonic_fragment_score(cand)
            smi = Chem.MolToSmiles(cand)

            if smi not in self.train_smiles_set and score > best_score:
                best, best_score = cand, score

        return best


def tensor_to_mol(node_ids, edge_ids, atom_labels):
    rwmol = Chem.RWMol()
    idx_map = {}

    for i, cls in enumerate(node_ids):
        cls = int(cls)
        if cls == 0 or cls >= len(atom_labels):
            continue
        a = Chem.Atom(int(atom_labels[cls]))
        idx_map[i] = rwmol.AddAtom(a)

    for i in range(len(node_ids)):
        for j in range(i + 1, len(node_ids)):
            if i not in idx_map or j not in idx_map:
                continue
            bt = BOND_MAP.get(int(edge_ids[i, j]))
            if bt:
                rwmol.AddBond(idx_map[i], idx_map[j], bt)

    try:
        mol = rwmol.GetMol()
        Chem.SanitizeMol(mol)
        return mol if mol.GetNumHeavyAtoms() > 1 else None
    except Exception:
        return None


def save_diffusion_to_master(valid_mols, train_smiles, model_type, epochs):
    """Implementação baseada no save_trial_to_master do seu Mark 2."""
    master_path = config.RESULTS_PATH

    # 1. Gerenciar Run ID
    if os.path.exists(master_path) and os.path.getsize(master_path) > 0:
        try:
            next_run_id = int(pd.read_csv(master_path, usecols=["run_id"])["run_id"].max() + 1)
        except Exception:
            next_run_id = 1
    else:
        next_run_id = 1

    # 2. Avaliação Profunda (Mark 2 core)
    gen_smiles = [Chem.MolToSmiles(m) for m in valid_mols]
    unique_smiles = list(set(gen_smiles))

    print(f"Calculando métricas avançadas (FCD, IntDiv, etc.) para Run {next_run_id}...")
    batch_stats = evaluate_molecules(generated_smiles=gen_smiles, reference_smiles=train_smiles)

    # 3. Preparar Linhas para o CSV
    rows = []
    params = dict(
        run_id=next_run_id,
        model_type= model_type,
        num_generated=N_GEN,
        num_valid=len(valid_mols),
        valid_percentage=round(len(valid_mols) / N_GEN * 100, 2),
        epochs=epochs,
        date=datetime.now().strftime("%Y-%m-%d %H:%M"),
    )

    for smi in unique_smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            row = {
                **params,
                **batch_stats,
                "smiles": smi,
                "qed": Descriptors.qed(mol),
                "logp": Descriptors.MolLogP(mol),
                "mw": Descriptors.MolWt(mol),
                "sas": sascorer.calculateScore(mol) if HAS_SA else 0,
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(master_path), exist_ok=True)
    df.to_csv(master_path, mode="a", header=not os.path.exists(master_path), index=False)
    print(f"✅ DATASET MESTRE ATUALIZADO! (Run ID: {next_run_id}) | Local: {master_path}")
    return next_run_id


# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    if device.type == "cuda":
        torch.multiprocessing.set_start_method("spawn", force=True)

    start_time = time.time()

    print(f"\nCarregando dados de {config.PREPARED_DATA_PATH} ...")
    with open(config.PREPARED_DATA_PATH, "rb") as f:
        d = pickle.load(f)

    train_smiles = d.get("train_smiles", [])
    adj_m = np.array(d["adj_matrices"])
    nod_f = np.array(d["node_features"])

    NUM_ATOM_TYPES = int(np.max(nod_f) + 1)
    NUM_BOND_TYPES = int(np.max(adj_m) + 1)

    print(f"  Quantidade de SMILES: {len(train_smiles)}")
    print(f"  Detectados: {NUM_ATOM_TYPES} átomos, {NUM_BOND_TYPES} ligações.")
    print(f"  ORCA cache path: {ORCA_FRAGMENT_CACHE_PATH}")

    fwd_proc = DiscreteForwardProcess(T_STEPS, NUM_ATOM_TYPES, NUM_BOND_TYPES)

    fragment_prior = ORCACachedPhotonicFragmentPrior(
        adj_m,
        nod_f,
        train_smiles,
        config.ATOM_LABELS,
        NUM_ATOM_TYPES,
        NUM_BOND_TYPES,
    ).to(device)

    print(
        f"  ORCA fragment QML: {fragment_prior.orca_status} "
        f"| banco={len(fragment_prior.photonic_fragment_bank)} "
        f"| cache_entries={fragment_prior.orca_queries}"
    )

    model = MolDiffDenoiser(Cn=NUM_ATOM_TYPES, Ce=NUM_BOND_TYPES, d=D_MODEL).to(device)

    loader = DataLoader(MolGraphDataset(adj_m, nod_f), batch_size=BATCH_SIZE, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None
    ckpt_path = os.path.join(config.BASE_DIR, "best_diffusion_model.pt")

    print(f"\n🚀 Iniciando treino ({EPOCHS} épocas)...")
    best_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        loss = train_epoch(model, loader, optimizer, fwd_proc, scaler)

        if loss < best_loss:
            best_loss = loss
            torch.save(model.state_dict(), ckpt_path)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  epoch {epoch:4d} | loss={loss:.4f} | best={best_loss:.4f}")

    print(f"\n✅ Treino concluído. Gerando {N_GEN} moléculas...")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    n_tensor, e_tensor = generate_molecules(
        model,
        fwd_proc,
        N_GEN,
        GEN_BATCH,
        config.NUM_ATOMS,
        fragment_prior=fragment_prior,
    )

    print("Processando química e salvando no registro mestre...")
    mols = [
        tensor_to_mol(n_tensor[i].numpy(), e_tensor[i].numpy(), config.ATOM_LABELS)
        for i in range(len(n_tensor))
    ]
    valid_mols = [m for m in mols if m is not None]
    valid_mols = fragment_prior.harmonize_molecules(valid_mols, N_GEN, config.NUM_ATOMS)

    valid_mols = [
    m for m in valid_mols
    if m is not None
    and m.GetNumHeavyAtoms() > 1
    and m.GetNumBonds() > 0
    and len(Chem.GetMolFrags(m)) == 1
]

    # Chamada da sua função de salvamento estilo Mark 2
    save_diffusion_to_master(valid_mols, train_smiles, model_type, EPOCHS)

    print(f"⏱️ Duração total: {(time.time() - start_time) / 60:.2f} minutos | ✨ CONCLUÍDO.")
