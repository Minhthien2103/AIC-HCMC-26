import sys
from pathlib import Path

import polars as pl

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config


class ObjectFilter:
    def __init__(self, object_path: str):
        self.object_path = object_path
        self.object_df = None

        self._load_store()


    def _load_store(self):
        print("Loading Object Store")

        try:
            if self.object_path.exists():
                self.object_df = pl.read_parquet(self.object_path)
            else:
                print(f"    Object Path not exist: {self.object_path}")

        except Exception as e:
            print(f"    Loading Object store Error: {e}")
            
    def get_all_labels(self) -> list[str]:
        """Returns a list of all unique object labels found in the dataset."""
        if self.object_df is not None:
            return self.object_df["object_label"].unique().to_list()
        return []
            
    def filter_candidates(self, candidates: list[dict], required_objects: list[str], mode: str = "boost") -> list[dict]:
        if not required_objects or not candidates:
            return candidates

        if self.object_df is None:
            return candidates

        faiss_indices = [c["faiss_idx"] for c in candidates]
        cand_objects = self.object_df.filter(pl.col("faiss_idx").is_in(faiss_indices))

        req_objects_lower = [obj.strip().lower() for obj in required_objects if obj.strip()]

        cand_objects = cand_objects.filter(pl.col("object_label").str.to_lowercase().is_in(req_objects_lower))

        if cand_objects.height > 0:
            matched_map = cand_objects.group_by("faiss_idx").agg(pl.col("object_label").str.to_lowercase().unique().alias("found_objects")).to_dicts()

            lookup = {row["faiss_idx"]: set(row["found_objects"]) for row in matched_map}

        else:
            lookup = {}

        req_set = set(req_objects_lower)
        filtered_candidates = []

        for c in candidates:
            f_idx = c["faiss_idx"]
            found = lookup.get(f_idx, set())
            overlap = len(req_set.intersection(found))

            if mode == "hard":
                if overlap == len(req_set):
                    filtered_candidates.append(c)

            else:
                boost_amount = (overlap / len(req_set)) * config.OBJECT_BOOST_WEIGHT
                if "clip_score" not in c:
                    c["clip_score"] = c["score"]

                c["score"] = (c["clip_score"] * config.CLIP_WEIGHT) + boost_amount
                c["matched_objects"] = list(found)

                filtered_candidates.append(c)

        if mode == "boost":
            filtered_candidates.sort(key = lambda x: x["score"], reverse = True)

        return filtered_candidates
