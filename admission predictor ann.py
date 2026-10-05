"""
Graduate Admission Predictor - Multi-Layer Perceptron (ANN) from scratch
Soft Computing Mini Project
Author : Aakansha Kasana
Roll/ID: 2026246809

Predicts whether a student gets admitted (1) or not (0) from:
  x1 = GRE score (260-340), x2 = CGPA (6-10), x3 = Research experience (0/1)
Architecture: 3 inputs -> 6 hidden (ReLU) -> 4 hidden (ReLU) -> 1 output (Sigmoid)
Learning: forward pass, binary cross-entropy loss, backpropagation, gradient descent.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

# ---------------- 1. Dataset (synthetic, reproducible) ----------------
def make_data(n=500):
    gre = rng.normal(310, 12, n).clip(260, 340)
    cgpa = (6 + (gre - 260) / 80 * 3.5 + rng.normal(0, 0.4, n)).clip(6, 10)
    research = (rng.random(n) < 0.5).astype(float)
    score = 0.45 * (gre - 260) / 80 + 0.40 * (cgpa - 6) / 4 + 0.15 * research
    prob = 1 / (1 + np.exp(-12 * (score - 0.55)))
    y = (rng.random(n) < prob).astype(float)
    X = np.column_stack([gre, cgpa, research])
    return X, y.reshape(-1, 1)

X, y = make_data()
split = int(0.8 * len(X))
X_raw_tr, X_raw_te, y_tr, y_te = X[:split], X[split:], y[:split], y[split:]

# normalise using training statistics only
mu, sd = X_raw_tr.mean(0), X_raw_tr.std(0)
X_tr, X_te = (X_raw_tr - mu) / sd, (X_raw_te - mu) / sd

# ---------------- 2. Network ----------------
sizes = [3, 6, 4, 1]
W = [rng.normal(0, np.sqrt(2 / sizes[i]), (sizes[i], sizes[i + 1])) for i in range(3)]
b = [np.zeros((1, sizes[i + 1])) for i in range(3)]

relu = lambda z: np.maximum(0, z)
sigmoid = lambda z: 1 / (1 + np.exp(-z))

def forward(X):
    z1 = X @ W[0] + b[0]; a1 = relu(z1)
    z2 = a1 @ W[1] + b[1]; a2 = relu(z2)
    z3 = a2 @ W[2] + b[2]; a3 = sigmoid(z3)
    return z1, a1, z2, a2, a3

def loss_fn(y, p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

# ---------------- 3. Training (backpropagation) ----------------
lr, epochs = 0.1, 1500
losses, accs = [], []
for ep in range(epochs):
    z1, a1, z2, a2, p = forward(X_tr)
    n = len(X_tr)
    d3 = (p - y_tr) / n                       # dL/dz3 (sigmoid + BCE)
    dW3, db3 = a2.T @ d3, d3.sum(0, keepdims=True)
    d2 = (d3 @ W[2].T) * (z2 > 0)
    dW2, db2 = a1.T @ d2, d2.sum(0, keepdims=True)
    d1 = (d2 @ W[1].T) * (z1 > 0)
    dW1, db1 = X_tr.T @ d1, d1.sum(0, keepdims=True)
    for i, (dw, db_) in enumerate([(dW1, db1), (dW2, db2), (dW3, db3)]):
        W[i] -= lr * dw
        b[i] -= lr * db_
    losses.append(loss_fn(y_tr, p))
    accs.append(((p >= 0.5) == y_tr).mean())
    if ep % 300 == 0 or ep == epochs - 1:
        print(f"Epoch {ep:4d} | Loss: {losses[-1]:.4f} | Train Acc: {accs[-1]:.3f}")

# ---------------- 4. Evaluation ----------------
pred = (forward(X_te)[-1] >= 0.5).astype(float)
acc = (pred == y_te).mean()
tp = int(((pred == 1) & (y_te == 1)).sum()); tn = int(((pred == 0) & (y_te == 0)).sum())
fp = int(((pred == 1) & (y_te == 0)).sum()); fn = int(((pred == 0) & (y_te == 1)).sum())
print(f"\nTest accuracy: {acc:.3f}  (TP={tp} TN={tn} FP={fp} FN={fn})")

def predict_student(gre, cgpa, research):
    x = (np.array([[gre, cgpa, research]]) - mu) / sd
    p = forward(x)[-1][0, 0]
    return p, "Admit" if p >= 0.5 else "Reject"

for s in [(335, 9.5, 1), (320, 8.5, 0), (300, 7.0, 0), (290, 6.8, 1)]:
    p, label = predict_student(*s)
    print(f"GRE={s[0]}, CGPA={s[1]}, Research={s[2]} -> {p:.2f} ({label})")

# ---------------- 5. Plots ----------------
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(losses, color="tab:red"); ax[0].set_title("Training Loss (BCE)")
ax[0].set_xlabel("Epoch"); ax[0].set_ylabel("Loss")
ax[1].plot(accs, color="tab:green"); ax[1].set_title("Training Accuracy")
ax[1].set_xlabel("Epoch"); ax[1].set_ylabel("Accuracy")
plt.tight_layout(); plt.savefig("training_curves.png", dpi=150); plt.close()

# decision map for research = 1 (and 0), GRE vs CGPA
fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
g, c = np.meshgrid(np.linspace(260, 340, 200), np.linspace(6, 10, 200))
for r, a in zip([0, 1], axs):
    grid = (np.column_stack([g.ravel(), c.ravel(), np.full(g.size, r)]) - mu) / sd
    z = forward(grid)[-1].reshape(g.shape)
    a.contourf(g, c, z, levels=20, cmap="RdYlGn", alpha=0.7)
    a.contour(g, c, z, levels=[0.5], colors="k", linestyles="--")
    m = X_raw_te[:, 2] == r
    a.scatter(X_raw_te[m & (y_te[:, 0] == 1), 0], X_raw_te[m & (y_te[:, 0] == 1), 1], c="green", marker="o", edgecolor="k", label="Admitted")
    a.scatter(X_raw_te[m & (y_te[:, 0] == 0), 0], X_raw_te[m & (y_te[:, 0] == 0), 1], c="red", marker="^", edgecolor="k", label="Rejected")
    a.set_title(f"Decision map (Research = {r})"); a.set_xlabel("GRE"); a.set_ylabel("CGPA"); a.legend()
plt.tight_layout(); plt.savefig("decision_map.png", dpi=150); plt.close()
