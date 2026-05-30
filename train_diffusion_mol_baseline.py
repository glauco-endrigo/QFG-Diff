"""Baseline discrete graph diffusion training for molecular generation."""

import gc
import logging
import math
import os
import pickle
import sys
import time
import warnings
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, RDConfig

import config
from mol_metrics import evaluate_molecules

sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
try:
    import sascorer
    HAS_SA = True
except ImportError:
    HAS_SA = False

warnings.filterwarnings("ignore")
RDLogger.logger().setLevel(RDLogger.CRITICAL)

device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
print(f"[hardware] CPUs={os.cpu_count()} | GPU={'YES' if torch.cuda.is_available() else 'NO'}")

MAX_ATOMS  = config.NUM_ATOMS
BATCH_SIZE = config.BATCH_SIZE
EPOCHS     = config.EPOCHS
T_STEPS    = 500
LR         = 3e-4
D_MODEL    = 256
N_GEN      = 1000
GEN_BATCH  = 256

T_STEPS    = 1000
D_MODEL    = 512
N_GEN      = 1000
LR         = 1e-4
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
        if t == 0: return x0_n_p, x0_e_p
        ab_t, ab_p = self.alpha_bar[t].to(xt_n.device), self.alpha_bar[t-1].to(xt_n.device)
        p_flip = ((ab_p - ab_t) / (1.0 - ab_t + 1e-8)).clamp(0.0, 1.0)
        mn, me = torch.bernoulli(p_flip.expand_as(xt_n)).bool(), torch.bernoulli(p_flip.expand_as(xt_e)).bool()
        return torch.where(mn, x0_n_p, xt_n), torch.where(me, x0_e_p, xt_e)

class GraphTransformerLayer(nn.Module):
    def __init__(self, d, n_heads=8, dropout=0.1):
        super().__init__()
        self.d, self.h, self.dh = d, n_heads, d // n_heads
        self.Wq, self.Wk, self.Wv = nn.Linear(d, d, bias=False), nn.Linear(d, d, bias=False), nn.Linear(d, d, bias=False)
        self.We, self.Wo = nn.Linear(d, n_heads, bias=False), nn.Linear(d, d)
        self.edge_mlp = nn.Sequential(nn.LayerNorm(3*d), nn.Linear(3*d, d*2), nn.GELU(), nn.Linear(d*2, d))
        self.node_ff = nn.Sequential(nn.Linear(d, d*4), nn.GELU(), nn.Linear(d*4, d))
        self.norm1, self.norm2, self.norm3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
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
        hi, hj = h.unsqueeze(2).expand(B, N, N, self.d), h.unsqueeze(1).expand(B, N, N, self.d)
        e = e + self.drop(self.edge_mlp(torch.cat([e, hi, hj], dim=-1)))
        return h, e

class MolDiffDenoiser(nn.Module):
    def __init__(self, Cn, Ce, d=256, n_layers=6):
        super().__init__()
        self.node_emb, self.edge_emb = nn.Embedding(Cn, d), nn.Embedding(Ce, d)
        self.time_mlp = nn.Sequential(nn.Linear(d, d*2), nn.GELU(), nn.Linear(d*2, d))
        self.layers = nn.ModuleList([GraphTransformerLayer(d) for _ in range(n_layers)])
        self.node_out, self.edge_out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, Cn)), nn.Sequential(nn.LayerNorm(d), nn.Linear(d, Ce))

    def _sinusoidal(self, t, d):
        half = d // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t[:, None].float() * freqs[None]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

    def forward(self, xt_n, xt_e, t):
        h = self.node_emb(xt_n) + self.time_mlp(self._sinusoidal(t, self.node_emb.embedding_dim)).unsqueeze(1)
        e = self.edge_emb(xt_e)
        for layer in self.layers: h, e = layer(h, e)
        return self.node_out(h), self.edge_out(e)

class MolGraphDataset(Dataset):
    def __init__(self, adj, nodes):
        self.adj = torch.from_numpy(adj.astype(np.int64))
        self.nodes = torch.from_numpy(nodes.squeeze(-1) if nodes.ndim == 3 else nodes).long()
    def __len__(self): return len(self.adj)
    def __getitem__(self, idx): return self.nodes[idx], self.adj[idx]

def train_epoch(model, loader, opt, fwd, scaler):
    model.train()
    total, n = 0.0, 0
    for nodes, edges in loader:
        nodes, edges = nodes.to(device), edges.to(device)
        t = torch.randint(1, fwd.T + 1, (nodes.size(0),), device=device)
        xt_n, xt_e = fwd.q_sample(nodes, edges, t)
        with torch.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
            nl, el = model(xt_n, xt_e, t)
            loss = F.cross_entropy(nl.view(-1, fwd.Cn), nodes.view(-1)) + 0.5 * F.cross_entropy(el.view(-1, fwd.Ce), edges.view(-1))
        opt.zero_grad(set_to_none=True)
        if scaler:
            scaler.scale(loss).backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update()
        else:
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        total += loss.item(); n += 1
    return total / max(n, 1)

@torch.no_grad()
def generate_molecules(model, fwd, n_mols, batch_size, max_at):
    model.eval()
    all_n, all_e = [], []
    for start in range(0, n_mols, batch_size):
        B = min(batch_size, n_mols - start)
        xt_n, xt_e = torch.randint(0, fwd.Cn, (B, max_at), device=device), torch.randint(0, fwd.Ce, (B, max_at, max_at), device=device)
        for t in reversed(range(0, fwd.T + 1)):
            tt = torch.full((B,), t, device=device, dtype=torch.long)
            nl, el = model(xt_n, xt_e, tt)
            x0_n, x0_e = torch.distributions.Categorical(logits=nl/1.05).sample(), torch.distributions.Categorical(logits=el/1.05).sample()
            x0_e = torch.triu(x0_e, 1); x0_e = x0_e + x0_e.transpose(1, 2)
            xt_n, xt_e = fwd.q_posterior_sample(x0_n, x0_e, xt_n, xt_e, t)
        all_n.append(xt_n.cpu()); all_e.append(xt_e.cpu())
    return torch.cat(all_n), torch.cat(all_e)

BOND_MAP = {1: Chem.rdchem.BondType.SINGLE, 2: Chem.rdchem.BondType.DOUBLE, 3: Chem.rdchem.BondType.TRIPLE, 4: Chem.rdchem.BondType.AROMATIC}

def tensor_to_mol(node_ids, edge_ids, atom_labels):
    rwmol = Chem.RWMol()
    idx_map = {}
    for i, cls in enumerate(node_ids):
        cls = int(cls)
        if cls == 0 or cls >= len(atom_labels): continue
        a = Chem.Atom(int(atom_labels[cls]))
        idx_map[i] = rwmol.AddAtom(a)
    for i in range(len(node_ids)):
        for j in range(i + 1, len(node_ids)):
            if i not in idx_map or j not in idx_map: continue
            bt = BOND_MAP.get(int(edge_ids[i, j]))
            if bt: rwmol.AddBond(idx_map[i], idx_map[j], bt)
    try:
        mol = rwmol.GetMol()
        Chem.SanitizeMol(mol)
        return mol if mol.GetNumHeavyAtoms() > 1 else None
    except: return None

def save_diffusion_to_master(valid_mols, train_smiles, model_type, epochs):
    """Save generated molecules using the master results schema."""
    master_path = config.RESULTS_PATH

    if os.path.exists(master_path) and os.path.getsize(master_path) > 0:
        try: next_run_id = int(pd.read_csv(master_path, usecols=['run_id'])['run_id'].max() + 1)
        except: next_run_id = 1
    else: next_run_id = 1

    gen_smiles = [Chem.MolToSmiles(m) for m in valid_mols]
    unique_smiles = list(set(gen_smiles))

    print(f"Computing advanced metrics (FCD, IntDiv, etc.) for run {next_run_id}...")
    batch_stats = evaluate_molecules(generated_smiles=gen_smiles, reference_smiles=train_smiles)

    rows = []
    params = dict(run_id=next_run_id, model_type="Diffusion_Mark3_FULL", num_generated=N_GEN,
                  num_valid=len(valid_mols), valid_percentage=round(len(valid_mols)/N_GEN*100, 2),
                  epochs=epochs, date=datetime.now().strftime("%Y-%m-%d %H:%M"))

    for smi in unique_smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            row = {**params, **batch_stats, 'smiles': smi,
                   'qed': Descriptors.qed(mol), 'logp': Descriptors.MolLogP(mol),
                   'mw': Descriptors.MolWt(mol), 'sas': sascorer.calculateScore(mol) if HAS_SA else 0}
            rows.append(row)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(master_path), exist_ok=True)
    df.to_csv(master_path, mode='a', header=not os.path.exists(master_path), index=False)
    print(f"Master dataset updated (run ID: {next_run_id}) | path: {master_path}")
    return next_run_id

if __name__ == "__main__":
    if device.type == 'cuda': torch.multiprocessing.set_start_method('spawn', force=True)
    start_time = time.time()

    print(f"\nLoading data from {config.PREPARED_DATA_PATH} ...")
    with open(config.PREPARED_DATA_PATH, 'rb') as f:
        d = pickle.load(f)

    train_smiles, adj_m, nod_f = d.get('train_smiles', []), np.array(d['adj_matrices']), np.array(d['node_features'])
    NUM_ATOM_TYPES, NUM_BOND_TYPES = int(np.max(nod_f) + 1), int(np.max(adj_m) + 1)

    print(f"  Quantidade de SMILES: {len(train_smiles)}")
    print(f"  Detected: {NUM_ATOM_TYPES} atom types, {NUM_BOND_TYPES} bond types.")

    fwd_proc = DiscreteForwardProcess(T_STEPS, NUM_ATOM_TYPES, NUM_BOND_TYPES)
    model = MolDiffDenoiser(Cn=NUM_ATOM_TYPES, Ce=NUM_BOND_TYPES, d=D_MODEL).to(device)

    loader = DataLoader(MolGraphDataset(adj_m, nod_f), batch_size=BATCH_SIZE, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    ckpt_path = os.path.join(config.BASE_DIR, 'best_diffusion_model.pt')

    print(f"\nStarting training ({EPOCHS} epochs)...")
    best_loss = float('inf')
    for epoch in range(1, EPOCHS + 1):
        loss = train_epoch(model, loader, optimizer, fwd_proc, scaler)
        if loss < best_loss:
            best_loss = loss
            torch.save(model.state_dict(), ckpt_path)
        if epoch % 10 == 0 or epoch == 1:
            print(f"  epoch {epoch:4d} | loss={loss:.4f} | best={best_loss:.4f}")

    print(f"\nTraining complete. Generating {N_GEN} molecules...")
    model.load_state_dict(torch.load(ckpt_path))
    n_tensor, e_tensor = generate_molecules(model, fwd_proc, N_GEN, GEN_BATCH, config.NUM_ATOMS)

    print("Processing chemistry and saving results...")
    mols = [tensor_to_mol(n_tensor[i].numpy(), e_tensor[i].numpy(), config.ATOM_LABELS) for i in range(len(n_tensor))]
    valid_mols = [m for m in mols if m is not None]

    save_diffusion_to_master(valid_mols, train_smiles, "Diffusion", EPOCHS)

    print(f"Total runtime: {(time.time() - start_time) / 60:.2f} minutes | done.")
