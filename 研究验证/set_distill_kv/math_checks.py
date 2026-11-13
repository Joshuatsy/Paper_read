"""CPU checks for the SetDistill-KV proposal; no pretrained model is evaluated.

Run: python 研究验证/set_distill_kv/math_checks.py
Checks the frozen-query deletion identity against recomputed softmax, a
non-nested optimum counterexample, and the proposed small critic's gradient.
"""
from itertools import combinations, product
from pathlib import Path
import json
import numpy as np


def softmax(x):
    y = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return y / y.sum(axis=-1, keepdims=True)


def deletion_checks():
    rng = np.random.default_rng(20261113)
    max_vector_error = max_energy_error = 0.0
    count = 0
    n_visual, n_fixed, dim = 7, 3, 5
    for _ in range(50):
        logits = rng.normal(size=n_visual + n_fixed)
        values = rng.normal(size=(n_visual + n_fixed, dim))
        attention = softmax(logits)
        full_output = attention @ values
        centered = attention[:n_visual, None] * (values[:n_visual] - full_output)
        gram = centered @ centered.T
        for bits in product((0.0, 1.0), repeat=n_visual):
            mask = np.array(bits)
            keep = np.concatenate((mask == 0, np.ones(n_fixed, dtype=bool)))
            # Independent reference: recompute softmax over the remaining logits.
            direct_output = softmax(logits[keep]) @ values[keep]
            surviving_mass = 1.0 - mask @ attention[:n_visual]
            predicted_change = -(mask @ centered) / surviving_mass
            direct_change = direct_output - full_output
            max_vector_error = max(max_vector_error, float(np.max(np.abs(direct_change - predicted_change))))
            predicted_energy = float(mask @ gram @ mask / surviving_mass**2)
            max_energy_error = max(max_energy_error, abs(float(direct_change @ direct_change) - predicted_energy))
            count += 1
    assert max_vector_error < 1e-11
    assert max_energy_error < 1e-10
    return {"cases": count, "max_vector_absolute_error": max_vector_error,
            "max_squared_error_absolute_error": max_energy_error}


def non_nested_check():
    # A, B, C are visual values. The fourth entry is always retained text.
    values = np.array([2.0, 4.0, 0.0, 1.0])
    logits = np.zeros(4)
    full_output = float(softmax(logits) @ values)
    names = "ABC"
    records, best = [], {}
    for budget in (1, 2, 3):
        costs = []
        for subset in combinations(range(3), budget):
            idx = list(subset) + [3]
            output = float(softmax(logits[idx]) @ values[idx])
            cost = (output - full_output)**2
            label = "".join(names[j] for j in subset)
            records.append({"budget": budget, "retained": label, "output": output, "squared_error": cost})
            costs.append((cost, label))
        best[budget] = min(costs)[1]
    assert best[1] == "A" and best[2] == "BC"
    assert not set(best[1]).issubset(best[2])
    return {"full_output": full_output, "rows": records, "best_sets": best}


def critic(mask, factors, mass, diagonal, temperature=0.2, floor=1e-6):
    """Positive low-rank-plus-diagonal ratio critic and analytic gradient.

    Shapes: mask [N], factors [T,N,r], mass/diagonal [T,N].
    It is a proposed surrogate, not an exact average-over-queries identity.
    """
    pooled = np.einsum("tnr,n->tr", factors, mask)
    numerator = (pooled**2).sum(axis=1) + (diagonal * mask[None, :]**2).sum(axis=1)
    raw_denom = 1.0 - mass @ mask
    denom = np.maximum(raw_denom, floor)
    risks = numerator / denom**2
    grad_num = 2 * (np.einsum("tnr,tr->tn", factors, pooled) + diagonal * mask[None, :])
    grad_risk = grad_num / denom[:, None]**2
    grad_risk += ((raw_denom > floor) * 2 * numerator / denom**3)[:, None] * mass
    weights = softmax(risks / temperature)
    z = risks / temperature
    total = temperature * (z.max() + np.log(np.exp(z - z.max()).sum()))
    return float(total), weights @ grad_risk


def project_capped_simplex(vector, total, iterations=70):
    """Euclidean projection onto 0 <= x <= 1, sum(x) = total."""
    lo, hi = vector.min() - 1.0, vector.max()
    for _ in range(iterations):
        midpoint = (lo + hi) / 2
        if np.clip(vector - midpoint, 0, 1).sum() > total:
            lo = midpoint
        else:
            hi = midpoint
    return np.clip(vector - (lo + hi) / 2, 0, 1)


def critic_checks():
    rng = np.random.default_rng(17)
    n, tasks, rank = 9, 4, 3
    factors = rng.normal(scale=0.05, size=(tasks, n, rank))
    mass = softmax(rng.normal(size=(tasks, n + 1)))[:, 1:]
    diagonal = rng.uniform(0.001, 0.01, size=(tasks, n))
    mask = rng.uniform(0.2, 0.8, size=n)
    _, analytic = critic(mask, factors, mass, diagonal)
    numeric = np.empty(n)
    delta = 1e-6
    for i in range(n):
        step = np.zeros(n); step[i] = delta
        numeric[i] = (critic(mask + step, factors, mass, diagonal)[0]
                      - critic(mask - step, factors, mass, diagonal)[0]) / (2 * delta)
    error = float(np.max(np.abs(analytic - numeric)))
    assert error < 1e-7
    delete_count = 6
    initial = np.zeros(n); initial[:delete_count] = 1
    candidates = [initial.copy()]
    relaxed = initial.copy()
    projection_error = 0.0
    for _ in range(4):
        _, grad = critic(relaxed, factors, mass, diagonal)
        relaxed = project_capped_simplex(relaxed - 0.3 * grad, delete_count)
        projection_error = max(projection_error, abs(float(relaxed.sum()) - delete_count))
        rounded = np.zeros(n)
        rounded[np.argsort(relaxed)[-delete_count:]] = 1
        candidates.append(rounded)
    costs = [critic(candidate, factors, mass, diagonal)[0] for candidate in candidates]
    chosen = candidates[int(np.argmin(costs))]
    assert chosen.sum() == delete_count and projection_error < 1e-10
    assert min(costs) <= costs[0] + 1e-12
    return {"max_gradient_absolute_error": error, "max_projection_budget_error": projection_error,
            "initial_predicted_risk": costs[0], "selected_predicted_risk": min(costs),
            "interpretation": "Only a surrogate-optimization check; not downstream VLM quality."}


def plot_counterexample(result, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font = FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    cost = {(r["budget"], r["retained"]): r["squared_error"] for r in result["rows"]}
    fig, ax = plt.subplots(figsize=(8.0, 4.2), constrained_layout=True)
    budgets = np.arange(1, 4)
    for ranking, label, color in [("ABC", "固定排名 A→B→C", "#e08b35"),
                                  ("BCA", "固定排名 B→C→A", "#7463b8")]:
        ys = [cost[(b, "".join(sorted(ranking[:b])))] for b in budgets]
        ax.plot(budgets, ys, "o-", label=label, color=color, linewidth=2, markersize=7)
    optimal = [min(r["squared_error"] for r in result["rows"] if r["budget"] == b) for b in budgets]
    ax.plot(budgets, optimal, "s--", color="#208a78", linewidth=2,
            label="各预算独立最优：{A}、{B,C}、{A,B,C}")
    ax.set_xticks(budgets)
    ax.set_xlabel("保留的视觉 KV 数量", fontproperties=font)
    ax.set_ylabel("单个注意力头的输出平方误差", fontproperties=font)
    ax.set_title("固定排名无法同时得到所有预算的最优集合", fontproperties=font, fontsize=13)
    ax.text(0.01, 0.96, "合成数学反例；不是 VLM 实验结果", transform=ax.transAxes,
            va="top", fontproperties=font, fontsize=10)
    ax.set_ylim(-0.03, 0.68)
    ax.grid(alpha=0.2)
    ax.legend(prop=font, loc="upper right", framealpha=0.95)
    for suffix in ("png", "svg"):
        fig.savefig(output_dir / f"non_nested_counterexample.{suffix}", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    output_dir = Path(__file__).resolve().parent
    result = {"status": "synthetic mathematical checks only",
              "deletion_identity": deletion_checks(), "non_nested": non_nested_check(),
              "critic": critic_checks()}
    plot_counterexample(result["non_nested"], output_dir)
    (output_dir / "math_results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "non_nested"}, ensure_ascii=False, indent=2))
