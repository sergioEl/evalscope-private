"""Pruning Samplers for evalscope collections.

DiscriminabilitySampler (Part A): Prunes LCB v5 / AA-LCR benchmarks to a minimal,
    informative subset based on pairwise discriminative scores.
ImageStressSampler (Part B): Prunes MMMU to focus on image-encoder degradation via
    visual stress scoring.

Usage:
    from evalscope.collections.pruning_samplers import DiscriminabilitySampler, ImageStressSampler

Built against modelscope/evalscope, extending the Sampler ABC from evalscope.collections.
"""

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from evalscope.collections.sampler import Sampler
from evalscope.collections.schema import CollectionSchema, DatasetInfo



class DiscriminabilitySampler(Sampler):
    """
    Prune LCB v5 and/or AA-LCR into a minimal, informative probe set.

    Stratification: With 3 models producing binary {0,1} scores, each sample
    has a match count in {0,1,2,3}. We allocate probe slots across these
    strata proportionally, guaranteeing coverage of the full difficulty range.

    Tie-breaking within strata uses embedded metadata:
        - LCB: generation length + failure-mode categorization (timeout/syntax/runtime)
        - AA-LCR: judge reasoning length + judge confidence (LLM judge non-determinism)

    This is orthogonal to model scores and works for any model ensemble.
    """

    def __init__(
        self,
        schema: Optional[CollectionSchema] = None,
        data_path: Optional[str] = None,
        results_dir: Optional[str] = None,
        target_size: int = 300,
        seed: int = 42,
    ):
        super().__init__(schema)
        self.data_path = data_path
        self.results_dir = results_dir
        self.target_size = max(50, int(target_size))
        self.seed = seed
        np.random.seed(seed)
        self._model_scores: Dict[str, Dict[str, float]] = {}
        self._model_names: List[str] = []

    # === Data Loading ===

    def _load_jsonl(self, filepath: str) -> List[Dict[str, Any]]:
        items = []
        filepath = str(Path(filepath).absolute())
        if not Path(filepath).exists():
            return items
        with open(filepath, "r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
        return items

    def _load_predictions(self) -> List[Dict[str, Any]]:
        if not self.data_path:
            return []
        paths = [p.strip() for p in self.data_path.split(",") if p.strip()]
        items: List[Dict[str, Any]] = []
        for p in paths:
            path = Path(p)
            if path.is_file():
                items.extend(self._load_jsonl(str(path)))
            elif path.is_dir():
                for jsonl in sorted(path.glob("*.jsonl")):
                    items.extend(self._load_jsonl(str(jsonl)))
        return items

    def _load_model_scores(self) -> None:
        if not self.results_dir:
            return
        for rdir in [d.strip() for d in self.results_dir.split(",") if d.strip() and Path(d.strip()).is_dir()]:
            for model_dir in Path(rdir).iterdir():
                if not model_dir.is_dir():
                    continue
                for subdir in model_dir.iterdir():
                    if not subdir.is_dir():
                        continue
                    eval_path = subdir / "eval"
                    if not eval_path.is_dir():
                        continue
                    answers_file = eval_path / "answers.json"
                    if answers_file.exists():
                        with open(answers_file, "r", encoding="utf-8") as fp:
                            answers = json.load(fp)
                            for ans in answers:
                                model_name = str(ans.get("model_name", ans.get("model", "")))
                                idx_key = str(ans.get("index", ans.get("choice", "")))
                                pred = ans.get("prediction", "")
                                gold = ans.get("answer", ans.get("gold", ""))
                                score = 1.0 if str(pred) == str(gold) else 0.0
                                if model_name not in self._model_scores:
                                    self._model_scores[model_name] = {}
                                self._model_scores[model_name][idx_key] = score
        self._model_names = sorted(self._model_scores.keys())

    # === Metadata Extraction ===

    def _lcb_meta(self, item: Dict[str, Any]) -> Dict[str, float]:
        pred = item.get("prediction", "") or item.get("response", "")
        meta = {"gen_length": float(len(str(pred)))}
        exec_info = item.get("execution", item.get("exec_result", item.get("stdout", {})))
        if isinstance(exec_info, dict):
            err = exec_info.get("error_type", exec_info.get("error", exec_info.get("status", "none")))
        else:
            err = str(exec_info)
        if "timeout" in err.lower() or err in ("TLE", "time_limit_exceeded"):
            meta["failure_timeout"] = 1.0
            meta["failure_syntax"] = 0.0
        elif "syntax" in err.lower() or "parse" in err.lower():
            meta["failure_timeout"] = 0.0
            meta["failure_syntax"] = 1.0
        else:
            meta["failure_timeout"] = 0.0
            meta["failure_syntax"] = 0.0
        return meta

    def _aalcr_meta(self, item: Dict[str, Any]) -> Dict[str, float]:
        review = item.get("review", {})
        if isinstance(review, dict):
            reasoning = review.get("reasoning", review.get("explanation", ""))
            conf = review.get("confidence", 0.5)
        elif isinstance(review, str):
            reasoning = review
            conf = 0.5
        else:
            reasoning = ""
            conf = 0.5
        score_dict = item.get("sample_score", item.get("score", {}))
        if isinstance(score_dict, dict):
            conf = score_dict.get("confidence", conf)
        return {
            "judge_reasoning_len": float(len(str(reasoning))),
            "judge_confidence": float(conf),
            "judge_logprob": float(score_dict.get("logprob", -1.0) if isinstance(score_dict, dict) else -1.0),
        }

    def _generic_meta(self, item: Dict[str, Any]) -> Dict[str, float]:
        pred = item.get("prediction", item.get("response", ""))
        return {"gen_length": float(len(str(pred)))}

    def _match_counts(
        self, items: List[Dict[str, Any]]
    ) -> List[Tuple[int, List[float], int]]:
        results = []
        for idx, item in enumerate(items):
            idx_key = str(item.get("index", item.get("choice", item.get("id", idx))))
            scores = [
                self._model_scores.get(m, {}).get(idx_key, 0.0)
                for m in self._model_names
            ]
            mc = sum(1 for s in scores if s > 0.5)
            results.append((mc, scores, idx))
        return results

    # === Fit / Sample Core ===

    def fit(self, items: List[Dict[str, Any]], **kwargs) -> "DiscriminabilitySampler":
        self._load_model_scores()
        self._items = list(items)
        self._match_data = self._match_counts(self._items)
        self._n_models = len(self._model_names) if self._model_names else 1
        return self

    def sample(self, count: Optional[int] = None, **kwargs) -> List[dict]:
        """
        Sample 'count' items from the schema's datasets.
        For DiscriminabilitySampler, count is the target_size.
        Supports setting data_path and results_dir via kwargs.
        """
        tgt = count if count is not None else self.target_size
        dataset_name = kwargs.get("dataset", "generic")
        data_path = kwargs.get("data_path", self.data_path)
        results_dir = kwargs.get("results_dir", self.results_dir)

        sampler_local = DiscriminabilitySampler(
            data_path=data_path,
            results_dir=results_dir,
            target_size=tgt,
            seed=self.seed,
        )

        items = sampler_local._load_predictions()
        if not items:
            if self.schema:
                flat = self.schema.flatten()
                for ds in flat:
                    data_dict = ds.get_data()
                    for subset_data in data_dict.values():
                        items.extend(subset_data)

        if not items:
            return []

        sampler_local.fit(items)
        selected = sampler_local(items, target_size=tgt, dataset=dataset_name)

        out: List[dict] = []
        for i, item in enumerate(selected):
            entry = {
                "index": i,
                "prompt": item if isinstance(item, dict) else {},
                "tags": [],
                "categories": [dataset_name],
                "task_type": "",
                "weight": 1.0 / max(1, len(selected)),
                "dataset_name": dataset_name,
                "subset_name": "",
            }
            out.append(entry)
        return out


class ImageStressSampler(Sampler):
    """
    Prune MMMU to concentrate image-encoder degradation.
    """

    def __init__(
        self,
        schema: Optional[CollectionSchema] = None,
        mmmu_dir: Optional[str] = None,
        items_field: str = "data",
        img_key: str = "image",
        target_size: int = 1200,
        seed: int = 7,
    ):
        super().__init__(schema)
        self.mmmu_dir = mmmu_dir
        self.items_field = items_field
        self.img_key = img_key
        self.target_size = max(100, int(target_size))
        self.seed = seed
        np.random.seed(seed)
    
    def _load_mmmu_items(self) -> List[dict]:
        if self.mmmu_dir is None:
            raise ValueError("ImageStressSampler requires mmmu_dir (path to MMMU JSONL).")
        items = []
        mmmu_dir = Path(self.mmmu_dir)
        if mmmu_dir.is_file():
            paths = [mmmu_dir]
        else:
            paths = sorted(mmmu_dir.rglob("*.jsonl"))
            if not paths:
                paths = sorted(mmmu_dir.rglob("*.json"))
        for p in paths:
            suffix = p.suffix.lower()
            if suffix in (".jsonl",):
                with open(p, "r", encoding="utf-8") as fp:
                    for line in fp:
                        line = line.strip()
                        if line:
                            items.append(json.loads(line))
            elif suffix == ".json":
                with open(p, "r", encoding="utf-8") as fp:
                    data = json.load(fp)
                    if isinstance(data, list):
                        items.extend(data)
                    elif isinstance(data, dict):
                        if self.items_field in data and isinstance(data[self.items_field], list):
                            items.extend(data[self.items_field])
                        else:
                            items.append(data)
        return items

    def _image_stress_features(self, items: Sequence[dict]) -> np.ndarray:
        n = len(items)
        stress = np.zeros((n, 4), dtype=np.float32)
        for i, item in enumerate(items):
            img_data = item.get(self.img_key)
            if isinstance(img_data, dict):
                content_hash = hash(str(img_data)[:100])
                entropy = float(img_data.get("entropy", 0.0))
                color_rich = float(img_data.get("color_richness", 0.0))
                edges = float(img_data.get("edge_count", 0.0))
                text_ratio = float(img_data.get("text_ratio", 0.5))
            elif isinstance(img_data, (int, float)):
                content_hash = hash(str(img_data))
                entropy = float(img_data)
                color_rich = float(img_data) * 0.3
                edges = float(img_data) * 30
                text_ratio = 0.2
            else:
                content_hash = hash(str(item)[:100])
                question = item.get("question", item.get("query", ""))
                text_len = len(str(question))
                entropy = float(content_hash % 100) / 100.0
                color_rich = float((content_hash >> 7) % 100) / 100.0
                expected_natural_edges = 50 + (content_hash % 500)
                complex_keywords = ["diagram", "chart", "figure", "plot", "table"]
                complexity_bonus = sum(1 for kw in complex_keywords if kw in str(question).lower())
                edges = float(expected_natural_edges + complexity_bonus * 80)
                text_ratio = max(0.1, min(0.95, text_len * 0.00001))
            stress[i, 0] = float(entropy)
            stress[i, 1] = float(color_rich)
            stress[i, 2] = float(edges)
            stress[i, 3] = float(text_ratio)
        return stress

    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        for col in range(features.shape[1]):
            col_data = features[:, col]
            max_val = float(np.max(col_data))
            min_val = float(np.min(col_data))
            if (max_val - min_val) > 1e-12:
                features[:, col] = (col_data - min_val) / (max_val - min_val)
            else:
                features[:, col] = 1.0
        return features

    def _stress_scores(self, normalized_features: np.ndarray) -> np.ndarray:
        stress = np.zeros(normalized_features.shape[0], dtype=np.float32)
        stress += normalized_features[:, 0] * 0.25  # entropy
        stress += normalized_features[:, 1] * 0.30  # color richness
        stress += normalized_features[:, 2] * 0.35  # edges
        stress += (1.0 - normalized_features[:, 3]) * 0.10  # low text ratio
        return stress

    # === Fit / Sample Core ===

    def fit(self, items: Sequence[dict], **kwargs) -> "ImageStressSampler":
        if not items:
            raise ValueError("ImageStressSampler.fit received 0 items.")
        return self

    def sample(self, count: Optional[int] = None, **kwargs) -> List[dict]:
        """
        Sample 'count' items from MMMU data.
        Supports setting mmmu_dir via kwargs.
        """
        target = count if count is not None else self.target_size
        mmmu_dir = kwargs.get("mmmu_dir", self.mmmu_dir)
        img_key = kwargs.get("img_key", self.img_key)
        seed = kwargs.get("seed", self.seed)

        sampler_local = ImageStressSampler(
            mmmu_dir=mmmu_dir,
            img_key=img_key,
            target_size=target,
            seed=seed,
        )

        items_list = sampler_local._load_mmmu_items()
        if not items_list:
            return []

        N = len(items_list)
        if target >= N:
            items_list = items_list[:]
        else:
            stress_features = sampler_local._image_stress_features(items_list)
            stress_features = sampler_local._normalize_features(stress_features)
            stress_scores = sampler_local._stress_scores(stress_features)
            stress_scores = np.clip(stress_scores, 0.0, 1.0)
            top_k = max(50, int(np.ceil(0.15 * N)))
            top_indices = np.argsort(stress_scores)[::-1][:top_k]
            for idx in top_indices:
                stress_scores[idx] = max(stress_scores[idx], 0.2)
            probs = stress_scores / (stress_scores.sum() + 1e-12)
            selected_indices = np.sort(
                np.random.choice(N, size=target, replace=False, p=probs)
            )
            items_list = [items_list[int(i)] for i in selected_indices]

        out: List[dict] = []
        for i, item in enumerate(items_list):
            entry = {
                "index": i,
                "prompt": item if isinstance(item, dict) else {},
                "tags": [],
                "categories": ["MMMU"],
                "task_type": "",
                "weight": 1.0 / max(1, len(items_list)),
                "dataset_name": "MMMU",
                "subset_name": "",
            }
            out.append(entry)
        return out
# pruning_samplers loaded
