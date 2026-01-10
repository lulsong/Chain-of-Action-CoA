#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_skillformer_nogoal.py
SkillFormer (No-Goal) 训练版：
  - 输入：instruction + start_image + proprio state
  - 输出：skill token sequence
  - 保留完整训练/验证指标 (Loss, EM, Levenshtein, F1_2gram)
"""

import os, json, glob, math, random
import numpy as np
from typing import List, Dict, Any, Tuple
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ------------------- Utils -------------------
def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

# ------------------- Dataset -------------------
class EpisNPZ(Dataset):
    def __init__(self, npz_dir: str, max_out_len=32):
        self.paths = sorted(glob.glob(os.path.join(npz_dir, "*.npz")))
        self.max_out_len = max_out_len
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        z = np.load(self.paths[i], allow_pickle=True)
        seq = z["skill_seq"].astype(np.int64).tolist()
        seq = seq[:self.max_out_len-1]
        st = z["state"]
        st = st.astype(np.float32) if st.size>0 else np.array([], dtype=np.float32)
        return {
            "H_txt": z["H_txt"].astype(np.float32),
            "H_vs":  z["H_vis_start"].astype(np.float32),
            "state": st,
            "seq":   seq,
        }

def pad_nd(arrs, pad_val=0.0):
    max0 = max(a.shape[0] for a in arrs)
    D = arrs[0].shape[1]
    out = np.full((len(arrs), max0, D), pad_val, dtype=np.float32)
    lengths = []
    for i,a in enumerate(arrs):
        L = a.shape[0]; out[i,:L,:] = a; lengths.append(L)
    return out, np.array(lengths, dtype=np.int32)

def collate(batch, K, max_out_len, state_dim, state_mean, state_std):
    H_txts = [b["H_txt"] for b in batch]
    H_vss  = [b["H_vs"]  for b in batch]
    X_txt, L_txt = pad_nd(H_txts)
    X_vs,  L_vs  = pad_nd(H_vss)

    # proprio state
    S = np.zeros((len(batch), state_dim), dtype=np.float32)
    for i,b in enumerate(batch):
        st = b["state"]
        if st.size == 0:
            S[i,:] = 0.0
        else:
            m = min(state_dim, st.shape[0])
            tmp = np.zeros(state_dim, dtype=np.float32)
            tmp[:m] = st[:m]
            if state_mean is not None and state_std is not None:
                tmp = (tmp - state_mean) / (state_std + 1e-6)
            S[i,:] = tmp

    # targets
    Y = []
    for b in batch:
        y = b["seq"] + [K+1]  # EOS
        Y.append(np.array(y, dtype=np.int64))
    L = max(len(y) for y in Y)
    Yin = np.full((len(batch), L), K+2, dtype=np.int64)
    Yout= np.full((len(batch), L), K+2, dtype=np.int64)
    for i,y in enumerate(Y):
        Yout[i,:len(y)] = y
        Yin[i,0] = K
        if len(y)>1: Yin[i,1:len(y)] = y[:-1]

    return {
        "H_txt": torch.from_numpy(X_txt), "L_txt": torch.from_numpy(L_txt),
        "H_vs":  torch.from_numpy(X_vs),  "L_vs":  torch.from_numpy(L_vs),
        "state": torch.from_numpy(S),
        "Yin": torch.from_numpy(Yin), "Yout": torch.from_numpy(Yout)
    }

# ------------------- Model -------------------
class CondProject(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.pos = nn.Parameter(torch.randn(1, 2048, d)*0.01)
    def forward(self, H_txt, H_vs, H_state):
        mem = torch.cat([H_txt, H_vs, H_state], dim=1)
        Tm = mem.size(1)
        return self.ln(mem + self.pos[:, :Tm, :])

class StateMLP(nn.Module):
    def __init__(self, d_state, d, hidden=2):
        super().__init__()
        layers=[]
        in_dim=d_state
        for _ in range(hidden):
            layers += [nn.Linear(in_dim, d), nn.GELU()]
            in_dim = d
        self.net = nn.Sequential(*layers)
    def forward(self, s):
        return self.net(s).unsqueeze(1)

class SkillFormer(nn.Module):
    def __init__(self, K, d=1024, n_layers=4, n_heads=4, d_ff=4096, max_len=32, dropout=0.1, d_state=8):
        super().__init__()
        self.K = K; self.d = d
        self.embed_out = nn.Embedding(K+3, d)
        self.pos_out   = nn.Parameter(torch.randn(1, max_len, d)*0.01)
        layer = nn.TransformerDecoderLayer(d, n_heads, d_ff, dropout=dropout, batch_first=True)
        self.dec = nn.TransformerDecoder(layer, num_layers=n_layers)
        self.fc  = nn.Linear(d, K+3)
        self.cond= CondProject(d)
        self.state_mlp = StateMLP(d_state, d) if d_state>0 else None
    def forward(self, H_txt, H_vs, Yin, S=None):
        B, L = Yin.size()
        if self.state_mlp is not None and S is not None:
            H_state = self.state_mlp(S)
        else:
            H_state = torch.zeros((B,1,self.d), device=Yin.device, dtype=torch.float32)
        mem = self.cond(H_txt, H_vs, H_state)
        x   = self.embed_out(Yin) + self.pos_out[:,:L,:]
        causal = torch.triu(torch.full((L,L), float("-inf"), device=Yin.device), diagonal=1)
        out = self.dec(x, mem, tgt_mask=causal)
        return self.fc(out)

# ------------------- Constraint Decoding -------------------
@torch.no_grad()
def greedy_with_constraints(model, H_txt, H_vs, S, K, trans_mask, max_len=32, temperature=1.0):
    B = H_txt.size(0)
    BOS, EOS, PAD = K, K+1, K+2
    Yin = torch.full((B,1), BOS, dtype=torch.long, device=H_txt.device)
    prev = None
    outs = [[] for _ in range(B)]
    for _ in range(max_len):
        lg = model(H_txt, H_vs, Yin, S)[:, -1, :] / max(1e-6, temperature)
        lg[:, PAD] = -1e9
        if prev is not None:
            for i, p in enumerate(prev.tolist()):
                if p is None or p>=K: continue
                lg[i, :K][~trans_mask[p]] = -1e9
        nxt = lg.argmax(-1)
        Yin = torch.cat([Yin, nxt[:,None]], dim=1)
        new_prev = []
        for i, t in enumerate(nxt.tolist()):
            if t==EOS: new_prev.append(EOS); continue
            if t<K: outs[i].append(t); new_prev.append(t)
            else: new_prev.append(None)
        prev = torch.tensor([(x if isinstance(x,int) else K+2) for x in new_prev], device=H_txt.device)
    return outs

# ------------------- Main -------------------
def main(data_dir: str, out_dir: str, epochs=15, batch_size=128, lr=3e-4, d=1024, max_out_len=32, seed=42):
    set_seed(seed)
    os.makedirs(out_dir, exist_ok=True)
    meta = json.load(open(os.path.join(data_dir, "meta.json")))
    K = int(meta["K"])
    state_dim = int(meta.get("state_dim", 0))
    state_mean = np.array(meta["state_mean"], dtype=np.float32) if meta.get("state_mean") is not None else None
    state_std  = np.array(meta["state_std"],  dtype=np.float32) if meta.get("state_std")  is not None else None

    epi_dir = os.path.join(data_dir, meta.get("episode_dir", "episodes"))
    ds = EpisNPZ(epi_dir, max_out_len=max_out_len)
    idx = np.arange(len(ds)); np.random.shuffle(idx)
    n_val = max(100, int(0.1*len(ds)))
    val_idx = set(idx[:n_val])
    train_items = [ds[i] for i in range(len(ds)) if i not in val_idx]
    val_items   = [ds[i] for i in range(len(ds)) if i in val_idx]
    coll = lambda b: collate(b, K, max_out_len, state_dim, state_mean, state_std)
    tr = DataLoader(train_items, batch_size=batch_size, shuffle=True,  collate_fn=coll, num_workers=2)
    va = DataLoader(val_items,   batch_size=batch_size, shuffle=False, collate_fn=coll, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SkillFormer(K, d=d, n_layers=4, n_heads=4, d_ff=4*d, max_len=max_out_len, dropout=0.1, d_state=state_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss(ignore_index=K+2, label_smoothing=0.05)

    trans = np.load(os.path.join(data_dir, "transition.npy"))
    trans_mask = torch.from_numpy((trans>0)).to(device)

    def to_dev(batch):
        return {k: v.to(device) if torch.is_tensor(v) else v for k,v in batch.items()}

    best_em = -1.0
    for ep in range(1, epochs+1):
        # ---- train ----
        model.train(); tot=0.0; N=0
        for batch in tr:
            batch=to_dev(batch)
            logits=model(batch["H_txt"],batch["H_vs"],batch["Yin"],batch["state"])
            loss=ce(logits.reshape(-1,logits.size(-1)),batch["Yout"].reshape(-1))
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
            tot+=loss.item()*logits.size(0); N+=logits.size(0)
        tr_loss=tot/max(1,N)

        # ---- val ----
        model.eval(); em=0; lev=0; f1=0; M=0; vloss=0.0; VN=0
        with torch.no_grad():
            for batch in va:
                batch=to_dev(batch)
                logits=model(batch["H_txt"],batch["H_vs"],batch["Yin"],batch["state"])
                vloss+=ce(logits.reshape(-1,logits.size(-1)),batch["Yout"].reshape(-1)).item()*logits.size(0); VN+=logits.size(0)
                preds=greedy_with_constraints(model,batch["H_txt"],batch["H_vs"],batch["state"],K,trans_mask,max_len=max_out_len)
                for i,p in enumerate(preds):
                    gold=[int(x) for x in batch["Yout"][i].tolist() if x!=K+2 and x!=K+1]
                    em+=int(p==gold)
                    n=len(p); m=len(gold)
                    dp=[[0]*(m+1) for _ in range(n+1)]
                    for a in range(n+1): dp[a][0]=a
                    for b in range(m+1): dp[0][b]=b
                    for a in range(1,n+1):
                        for b in range(1,m+1):
                            cost=0 if p[a-1]==gold[b-1] else 1
                            dp[a][b]=min(dp[a-1][b]+1,dp[a][b-1]+1,dp[a-1][b-1]+cost)
                    lev+=dp[n][m]
                    def ngram_f1(pred,gold,n=2):
                        from collections import Counter
                        def grams(x):
                            return Counter([tuple(x[i:i+n]) for i in range(len(x)-n+1)]) if len(x)>=n else Counter()
                        P=grams(pred);G=grams(gold)
                        inter=sum((P&G).values());p_=sum(P.values());g_=sum(G.values())
                        if p_==0 or g_==0:return 0.0
                        prec,rec=inter/p_,inter/g_
                        return 2*prec*rec/(prec+rec+1e-9)
                    f1+=ngram_f1(p,gold,2);M+=1
        metrics={"loss":vloss/max(1,VN),"EM":em/max(1,M),"Lev":lev/max(1,M),"F1_2gram":f1/max(1,M)}
        print(f"[ep {ep:02d}] train {tr_loss:.4f} | val loss {metrics['loss']:.4f} EM {metrics['EM']:.3f} Lev {metrics['Lev']:.2f} F1 {metrics['F1_2gram']:.3f}")

        # ---- save ----
        torch.save({"model":model.state_dict(),"K":K,"cfg":{"d":d,"max_out_len":max_out_len,"state_dim":state_dim}},
                   os.path.join(out_dir,"last.pt"))
        if metrics["EM"]>best_em:
            best_em=metrics["EM"]
            torch.save({"model":model.state_dict(),"K":K,"cfg":{"d":d,"max_out_len":max_out_len,"state_dim":state_dim}},
                       os.path.join(out_dir,"best.pt"))
            print("  ↳ new best saved")

# ------------------- CLI -------------------
if __name__ == "__main__":
    import argparse
    ap=argparse.ArgumentParser("Train SkillFormer (No-Goal)")
    ap.add_argument("--data_dir",required=True)
    ap.add_argument("--out_dir",default="./plan_ckpt_nogoal")
    ap.add_argument("--epochs",type=int,default=15)
    ap.add_argument("--batch_size",type=int,default=128)
    ap.add_argument("--lr",type=float,default=3e-4)
    ap.add_argument("--d",type=int,default=1024)
    ap.add_argument("--max_out_len",type=int,default=32)
    ap.add_argument("--seed",type=int,default=42)
    args=ap.parse_args()
    main(**vars(args))

