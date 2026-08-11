import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))


class ScoreFuser:
    def __init__(self, k: int = 60):
        self.k = k


    def fuse_rrf(self, model_results: list[tuple[str, list[dict]]]) -> list[dict]:
        if len(model_results) == 1:
            return model_results[0][1]

        scores_map = {}
        item_map = {}

        for model_name, results in model_results:
            for rank, item in enumerate(results):
                f_idx = item["faiss_idx"]

                if f_idx not in scores_map:
                    scores_map[f_idx] = 0.0
                    item_map[f_idx] = item

            scores_map[f_idx] += 1.0 / (self.k + rank + 1)

        fused_results = []

        for f_idx, score in scores_map.items():
            item = item_map[f_idx].copy()
            item["score"] = score

            fused_results.append(item)

        fused_results.sort(key = lambda x: x["score"], reverse = True)

        return fused_results
        