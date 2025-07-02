import torch, os, numpy as np, pandas as pd
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    precision_score, recall_score, precision_recall_curve, log_loss
)
from scipy.special import expit  # sigmoid
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression

from training.utils import CustomDataset, load_config
from modelos.feature_extractor import CNNFeatureExtractor
from modelos.sequence_model import SequenceModel
from modelos.attention_layer import AttentionBlock
from modelos.drl_agent import PPOAgent
from torch.utils.data import DataLoader

# ╭───────────────────── CONFIG USER ─────────────────────╮
CAPITAL_PER_TRADE = 200       # USD margen
LEVERAGE          = 25        # 25×
POSITION_SIZE     = CAPITAL_PER_TRADE * LEVERAGE
TAKE_PROFIT_PCT   = 0.004     # 0.4 %
STOP_LOSS_PCT     = 0.002     # 0.2 %
FEE_PCT_ROUND     = 0.0004    # 0.04 % ida‑vuelta
CHECKPOINT        = "checkpoints/model_last.pt"
CONFIG_YAML       = "data/config/config.yaml"
# ╰────────────────────────────────────────────────────────╯

cfg     = load_config(CONFIG_YAML)
device  = torch.device(cfg["training"]["device"])

# ─── Dataset de test (último 10 %) ───────────────────────
ds      = CustomDataset(cfg)
n_total = len(ds)
test_id = list(range(int(n_total*0.9), n_total))
test_ds = torch.utils.data.Subset(ds, test_id)
dl      = DataLoader(test_ds, batch_size=cfg["training"]["batch_size"], shuffle=False)

# ─── Cargar modelo ───────────────────────────────────────
fx  = CNNFeatureExtractor(cfg).to(device)
seq = SequenceModel(cfg).to(device)
att = AttentionBlock(cfg).to(device)
agt = PPOAgent(cfg).to(device)

ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
fx.load_state_dict(ckpt["feature_extractor"])
seq.load_state_dict(ckpt["sequence_model"])
att.load_state_dict(ckpt["attention"])
agt.load_state_dict(ckpt["agent"])

for m in (fx, seq, att, agt):
    m.eval()

all_logits, all_probs, all_lbls = [], [], []

with torch.no_grad():
    for X, y in dl:
        X, y = X.to(device), y.to(device)
        logits = agt(att(seq(fx(X)))).squeeze()
        probs  = torch.sigmoid(logits)

        logits_np = logits.cpu().numpy()
        probs_np  = probs.cpu().numpy()
        y_np      = y.cpu().numpy()
        # Si es escalar, usar append. Si es array, usar extend.
        for arr, store in zip([logits_np, probs_np, y_np], [all_logits, all_probs, all_lbls]):
            if np.isscalar(arr):
                store.append(arr)
            else:
                arr_list = arr.tolist()
                if isinstance(arr_list, (float, int)):
                    store.append(arr_list)
                else:
                    store.extend(arr_list)

all_logits, all_probs, all_lbls = map(np.array, (all_logits, all_probs, all_lbls))

# ─── Calibración de temperatura ──────────────────────────
def temp_loss(t):
    return log_loss(all_lbls, expit(all_logits / t[0]))
optT = minimize(temp_loss, [1.0], bounds=[(0.05,10)]).x[0]
print(f"🔥 Temperatura óptima: {optT:.4f}")
if optT >= 2.0:
    # Platt scaling (sigmoid fit) como fallback
    lr_platt = LogisticRegression(max_iter=1000, solver='lbfgs')
    lr_platt.fit(all_logits.reshape(-1,1), all_lbls)
    all_probs = lr_platt.predict_proba(all_logits.reshape(-1,1))[:,1]
    print("🔄 Usando Platt scaling para calibrar.")
else:
    all_probs = expit(all_logits / optT)
    print("✅ Aplicando temperatura.")

# ─── Mejor umbral por F1 en todo el rango 0.3–0.9 ───
best_f1, best_thr = 0, 0.50
for t in np.linspace(0.30, 0.90, 121):
    pr = (all_probs >= t).astype(int)
    f1 = f1_score(all_lbls, pr, zero_division='warn')
    if f1 > best_f1:
        best_f1, best_thr = f1, t

print(f"🎯 Mejor umbral por F1 = {best_thr:.3f} (F1={best_f1:.4f})")
thr_bal = best_thr
preds   = (all_probs >= thr_bal).astype(int)

# ─── Métricas clásicas ───────────────────────────────────
print("\n📊 MÉTRICAS CLASIFICACIÓN:")
print(classification_report(all_lbls, preds, digits=4))
print("📉 Matriz de Confusión:\n", confusion_matrix(all_lbls, preds))

# ─── Simulación económica ────────────────────────────────
trade_results = []
for p,l in zip(all_probs, all_lbls):
    if p < thr_bal:
        continue            # no trade
    is_win = (l==1)
    pnl = (TAKE_PROFIT_PCT if is_win else -STOP_LOSS_PCT) * POSITION_SIZE
    pnl -= FEE_PCT_ROUND * POSITION_SIZE
    trade_results.append(pnl)

trades = len(trade_results)
gross  = np.sum(trade_results)
roi    = gross / (CAPITAL_PER_TRADE * trades) * 100 if trades else 0
win_rt = (np.array(trade_results) > 0).mean()*100 if trades else 0

print("\n💸 SIMULACIÓN ECONÓMICA")
print(f"Trades ejecutados:     {trades}")
print(f"Win rate real:         {win_rt:.2f}%")
print(f"PNL bruto total:       ${gross:,.2f}")
print(f"ROI sobre capital/trade: {roi:.2f}%")
