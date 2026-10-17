"""Lightweight SASRec-style next-item model (PyTorch)."""
import numpy as np
from .config import CONFIG

class SASRecRecommender:
    def __init__(self, epochs=CONFIG.sasrec_epochs, max_len=CONFIG.max_sequence_length, seed=CONFIG.seed):
        self.epochs, self.max_len, self.seed = epochs, max_len, seed
    def fit(self, data):
        try:
            import torch
            from torch import nn
            from torch.utils.data import Dataset, DataLoader
        except ImportError as e: raise RuntimeError("SASRec требует PyTorch") from e
        self.torch, self.nn = torch, nn
        torch.manual_seed(self.seed); np.random.seed(self.seed)
        d = data[data.rating >= CONFIG.rating_threshold].sort_values(["user_id", "timestamp", "movie_id"])
        raw = d.groupby("user_id").movie_id.apply(lambda s: list(map(int, s))).to_dict()
        raw = {int(u): seq for u, seq in raw.items() if len(seq) >= 2}
        items = sorted({i for seq in raw.values() for i in seq})
        self.item_map = {item: idx + 1 for idx, item in enumerate(items)}
        self.known_items = set(self.item_map)
        sequences = {u: [self.item_map[x] for x in seq] for u, seq in raw.items()}
        self.histories = {u: seq[-self.max_len:] for u, seq in sequences.items()}
        n_items = len(self.item_map)
        if n_items < 2: raise ValueError("SASRec: недостаточно фильмов с положительными оценками")
        rows=[]; rng=np.random.default_rng(self.seed)
        for u, seq in sequences.items():
            seen=set(seq)
            for t in range(1, len(seq)):
                neg=0
                for _ in range(100):
                    q=int(rng.integers(1,n_items+1))
                    if q not in seen: neg=q; break
                if neg: rows.append((seq[:t][-self.max_len:], seq[t], neg))
        if not rows: raise ValueError("SASRec: пустой набор обучающих последовательностей")
        class SeqDataset(Dataset):
            def __len__(self): return len(rows)
            def __getitem__(self, idx):
                seq, pos, neg = rows[idx]; x=np.zeros(self_len,dtype=np.int64); x[-len(seq):]=seq
                return torch.tensor(x), torch.tensor(pos), torch.tensor(neg)
        self_len=self.max_len
        class Net(nn.Module):
            def __init__(self):
                super().__init__(); dim=64
                self.item=nn.Embedding(n_items+1,dim,padding_idx=0); self.pos=nn.Embedding(self_len,dim)
                layer=nn.TransformerEncoderLayer(dim,2,dim*4,dropout=.2,activation="gelu",batch_first=True,norm_first=True)
                self.encoder=nn.TransformerEncoder(layer,2); self.norm=nn.LayerNorm(dim)
            def vector(self,x):
                b,l=x.shape; p=torch.arange(l,device=x.device).unsqueeze(0).expand(b,l)
                mask=torch.triu(torch.ones(l,l,device=x.device,dtype=torch.bool),diagonal=1)
                out=self.encoder(self.item(x)+self.pos(p),mask=mask,src_key_padding_mask=x.eq(0))
                # last non-padding position, not the padded final slot
                lengths=x.ne(0).sum(1).clamp(min=1)-1
                return self.norm(out[torch.arange(b,device=x.device), lengths])
            def loss(self,x,p,n):
                v=self.vector(x); pos=(v*self.item(p)).sum(-1); neg=(v*self.item(n)).sum(-1)
                return (torch.nn.functional.softplus(-pos)+torch.nn.functional.softplus(neg)).mean()
        self.model=Net().to("cuda" if torch.cuda.is_available() else "cpu")
        self.device=next(self.model.parameters()).device
        loader=DataLoader(SeqDataset(),batch_size=256,shuffle=True)
        optimizer=torch.optim.Adam(self.model.parameters(),lr=1e-3)
        for epoch in range(self.epochs):
            self.model.train(); losses=[]
            for x,p,n in loader:
                x,p,n=x.to(self.device),p.to(self.device),n.to(self.device); optimizer.zero_grad()
                loss=self.model.loss(x,p,n); loss.backward(); optimizer.step(); losses.append(float(loss.detach().cpu()))
            print(f"SASRec epoch {epoch+1}/{self.epochs}: loss={np.mean(losses):.4f}")
        self.model.eval(); return self
    def score(self, user_id, item_ids):
        torch=self.torch; x=np.zeros(self.max_len,dtype=np.int64); hist=self.histories.get(int(user_id),[])
        if hist: x[-len(hist):]=hist
        ids=np.asarray([self.item_map.get(int(i),0) for i in item_ids]); valid=ids>0
        out=np.full(len(item_ids),-1e9,dtype=np.float32)
        if valid.any():
            with torch.no_grad():
                xx=torch.tensor(x,device=self.device).unsqueeze(0); vec=self.model.vector(xx)
                emb=self.model.item(torch.tensor(ids[valid],device=self.device)); out[valid]=(vec@emb.T).squeeze(0).cpu().numpy()
        return out
