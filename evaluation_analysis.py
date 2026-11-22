"""Comparaciones descriptivas sobre las métricas por consulta ya evaluadas."""

import math


MODELS = ("Jaccard", "TF-IDF", "BM25", "Embeddings")
TOLERANCE = 1e-12


def same_score(left, right):
    return math.isclose(left, right, rel_tol=0.0, abs_tol=TOLERANCE)


def build_evaluation_analysis(per_query, classified, groups, profiles, case_limit=5):
    # Los perfiles globales se publican solo al terminar un modelo. No comparar
    # la muestra parcial de embeddings con todas las consultas de los otros.
    completed = [name for name in MODELS if name in profiles and name in per_query]
    by_group = {}
    for dimension in ("type", "length"):
        rows = []
        for label, query_ids in groups.get(dimension, {}).items():
            ids = [qid for qid in query_ids if qid in classified]
            scores = {
                name: sum(per_query[name][qid]["AP"] for qid in ids) / len(ids)
                for name in completed
                if ids and all(qid in per_query[name] for qid in ids)
            }
            best = max(scores.values()) if scores else None
            rows.append({
                "group": label, "count": len(ids), "scores": scores,
                "winners": [name for name, value in scores.items() if same_score(value, best)],
            })
        by_group[dimension] = rows

    best_types = {}
    for name in completed:
        candidates = [row for row in by_group["type"] if name in row["scores"]]
        if candidates:
            best = max(row["scores"][name] for row in candidates)
            best_types[name] = {
                "MAP": best,
                "groups": [row["group"] for row in candidates if same_score(row["scores"][name], best)],
            }

    comparisons = {}
    for baseline in ("BM25", "TF-IDF"):
        if "Embeddings" not in completed or baseline not in completed:
            continue
        ids = sorted(set(classified) & set(per_query["Embeddings"]) & set(per_query[baseline]))
        cases = []
        for qid in ids:
            dense_ap = per_query["Embeddings"][qid]["AP"]
            baseline_ap = per_query[baseline][qid]["AP"]

            def first_relevant(name):
                relevant = profiles[name].get(qid, {}).get("relevant", [])
                return relevant[0][0] if relevant else None

            cases.append({
                "query_id": qid, "text": classified[qid]["text"],
                "query_type": classified[qid]["query_type"],
                "dense_ap": dense_ap, "baseline_ap": baseline_ap,
                "delta_ap": dense_ap - baseline_ap,
                "first_relevant_dense": first_relevant("Embeddings"),
                "first_relevant_baseline": first_relevant(baseline),
            })
        wins = [case for case in cases if case["delta_ap"] > TOLERANCE]
        losses = [case for case in cases if case["delta_ap"] < -TOLERANCE]
        comparisons[baseline] = {
            "count": len(cases), "wins": len(wins), "losses": len(losses),
            "ties": len(cases) - len(wins) - len(losses),
            "mean_delta_ap": sum(case["delta_ap"] for case in cases) / len(cases) if cases else 0.0,
            "improvements": sorted(wins, key=lambda case: (-case["delta_ap"], case["query_id"]))[:case_limit],
            "regressions": sorted(losses, key=lambda case: (case["delta_ap"], case["query_id"]))[:case_limit],
        }
    return {
        "completed_models": completed,
        "by_group": by_group, "best_types": best_types,
        "semantic": {"ready": len(comparisons) == 2, "comparisons": comparisons},
    }
