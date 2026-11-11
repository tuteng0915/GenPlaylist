"""DDBC optimal assignment, with explicit similarity and distance directions."""
import numpy as np
from scipy.optimize import linear_sum_assignment


def normalized(x):
    x = np.asarray(x, dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite embedding')
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    if np.any(norm <= 1e-12):
        raise ValueError('Zero embedding')
    return x / norm


def assignment_oas(predicted, target):
    """Return one score per list and target position matched to each prediction.

    Exactly five predictions and five targets; preserve duplicate occurrences.
    This is 1 - DDBC evaluator.hgd_dist (cosine), not best-of-k generation.
    """
    p, t = np.asarray(predicted), np.asarray(target)
    if p.shape != t.shape or p.ndim != 3 or p.shape[1] != 5 or len(p) == 0:
        raise ValueError('Expected equal nonempty [lists, 5, embedding_dim] arrays')
    p, t = normalized(p), normalized(t)
    sims = np.clip(np.einsum('nif,njf->nij', p, t), -1, 1)
    scores, assignments = [], []
    for s in sims:
        row, col = linear_sum_assignment(1 - s)
        assert np.array_equal(row, np.arange(5)) and len(set(col)) == 5
        scores.append(s[row, col].mean())
        assignments.append(col)
    scores = np.asarray(scores)
    return dict(similarity=scores, distance=1-scores,
                target_positions=np.asarray(assignments), pairwise_cosine=sims)


def interval(values, seed=42, repetitions=10000):
    values = np.asarray(values, dtype=np.float64)
    assert values.ndim == 1 and len(values) and np.isfinite(values).all()
    rng = np.random.default_rng(seed)
    means = np.concatenate([values[rng.integers(len(values), size=(min(200, repetitions-i), len(values)))].mean(1)
                            for i in range(0, repetitions, 200)])
    return np.quantile(means, [.025, .975]).tolist()


def summary(values):
    values = np.asarray(values)
    return dict(n_lists=len(values), mean=float(values.mean()), ci95=interval(values))
