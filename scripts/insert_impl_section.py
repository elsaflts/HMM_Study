import json
from pathlib import Path


def insert_impl_math_section(nb_path: Path) -> bool:
    nb = json.loads(nb_path.read_text())
    cells = nb.get("cells", [])

    # Find the index of the "## 4. Algorithm Outline" markdown cell
    insert_idx = None
    for i, c in enumerate(cells):
        if c.get("cell_type") == "markdown":
            text = "".join(c.get("source", []))
            if text.strip().startswith("## 4. Algorithm Outline"):
                insert_idx = i + 1
                break

    if insert_idx is None:
        return False

    md = r"""
## 5. Implementation & Mathematics

This notebook uses two coupled approaches:
- Kalman Filter (KF) + Rauch–Tung–Striebel (RTS) smoother for level estimation.
- Rao–Blackwellized Gibbs sampling to estimate change-point probabilities.

Model:
- Change indicator: $C_k \in \{0,1\}$ with prior $C_k \sim \text{Bernoulli}(p)$.
- State transition (piecewise-constant with resets):
  $$ W_{k+1} = A(C_k)\, W_k + R(C_k)\,U_k, \quad U_k \sim \mathcal{N}(0,1). $$
  We use
  $$ A(0)=1, \ R(0)=q0\_std, \quad A(1)=0, \ R(1)=R\_std, $$
  i.e., no-change keeps the level with tiny drift, and a change resets to a new draw.
- Observation: $$ Y_k = W_k + S\,V_k, \quad V_k \sim \mathcal{N}(0,1). $$

Conditional linear–Gaussian inference (given $C$):
- With $A_k=A(C_k)$ and $R_k=R(C_k)$, the Kalman filter computes forward moments $(m_{k|k}, P_{k|k})$ and the RTS smoother refines them to $(m_{k|n}, P_{k|n})$.
- KF prediction from time $k-1$ to $k$:
  $$ m_{k|k-1} = A_{k-1} m_{k-1|k-1}, \quad P_{k|k-1} = A_{k-1} P_{k-1|k-1} A_{k-1} + R_{k-1}^2. $$
- KF update with observation $Y_k$ (here observation matrix $B=1$):
  $$ e_k = Y_k - m_{k|k-1}, \quad G_k = P_{k|k-1} + S^2, \quad K_k = P_{k|k-1}/G_k, $$
  $$ m_{k|k} = m_{k|k-1} + K_k e_k, \quad P_{k|k} = (1 - K_k) P_{k|k-1}. $$
- RTS backward smoothing (define $J_k = P_{k|k} A_k / P_{k+1|k}$):
  $$ m_{k|n} = m_{k|k} + J_k (m_{k+1|n} - m_{k+1|k}), $$
  $$ P_{k|n} = P_{k|k} + J_k (P_{k+1|n} - P_{k+1|k}) J_k. $$

Sampling $W$ given $C, Y$ (FFBS):
- We draw a full trajectory $W_{0:n-1}$ with Forward-Filter Backward-Sample using the KF moments; then draw $W_n$ from the last transition using $(A_{n-1}, R_{n-1})$.

Gibbs updates for change indicators $C$ given $W$:
- Site-wise update uses the transition density and Bernoulli prior:
  $$\log p(C_k=0|W) \propto \log(1-p) + \log \mathcal{N}(W_{k+1}; W_k, q0\_std^2), $$
  $$\log p(C_k=1|W) \propto \log p + \log \mathcal{N}(W_{k+1}; 0, R\_std^2). $$
- We normalize these two log-scores to get $p(C_k=1|W)$ and sample $C_k \in \{0,1\}$.

Algorithms used in this notebook:
- KF + RTS for the smoothed level $m_{k|n}$.
- Rao–Blackwellized Gibbs sampler alternating $W|C,Y$ (via FFBS) and $C|W$ (Bernoulli site updates).
- Optionally, an online Particle Filter variant can approximate change probabilities with fixed lag.
"""

    new_cell = {
        "cell_type": "markdown",
        "id": "impl-math-section",
        "metadata": {},
        "source": md.splitlines(keepends=True),
    }

    cells.insert(insert_idx, new_cell)
    nb["cells"] = cells
    nb_path.write_text(json.dumps(nb, indent=1))
    return True


if __name__ == "__main__":
    path = Path("notebooks/Study_sample.ipynb")
    ok = insert_impl_math_section(path)
    if not ok:
        raise SystemExit("Could not locate Algorithm Outline section to insert after.")
    print("Inserted Implementation & Mathematics section.")

