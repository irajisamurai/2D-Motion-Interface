def compose_loss(loss_terms: dict, weights: dict):
    total = None
    used = {}
    for name, w in weights.items():
        if w == 0:
            continue
        if name not in loss_terms:
            raise KeyError(f"Unknown loss term: {name}. Available: {list(loss_terms.keys())}")
        term = loss_terms[name] * float(w)
        used[name] = term
        total = term if total is None else (total + term)
    if total is None:
        raise ValueError("All loss weights are zero; total loss is undefined.")
    return total, used
