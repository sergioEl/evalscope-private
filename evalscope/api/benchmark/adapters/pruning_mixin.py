from typing import Dict, Any
from evalscope.api.dataset import DatasetDict
from evalscope.collections.pruning_samplers import DiscriminabilitySampler, ImageStressSampler
from evalscope.utils.logger import get_logger

logger = get_logger()

class PruningAdapterMixin:
    """
    Universal mixin to intercept evalscope's load_dataset() and apply 
    offline disk-based pruning samplers (Discriminability / ImageStress) 
    to the in-memory DatasetDict.
    """
    def __init__(self, **kwargs):
        # 1. Initialize parent DataAdapter to populate self.task_config and self.name
        super().__init__(**kwargs)
        
        # 2. Extract dataset_args directly from task_config, NOT kwargs
        task_config = kwargs.get('task_config')
        raw_args = getattr(task_config, 'dataset_args', {}) if task_config else {}
        
        dataset_name = getattr(self, 'name', '')
        
        # 3. Handle evalscope's nested vs flat CLI dictionary formats
        if dataset_name in raw_args and isinstance(raw_args[dataset_name], dict):
            self.prune_config = raw_args[dataset_name]
        else:
            self.prune_config = raw_args
            
        self.pruning_strategy = self.prune_config.get('pruning_strategy')
        self.prune_ratio = float(self.prune_config.get('prune_ratio', 1.0))
        
        # 4. Resolve paths (CLI overrides > Class defaults > None)
        self.pruner_data_path = self.prune_config.get(
            'data_path', getattr(self, 'default_pruner_data_path', None)
        )
        self.pruner_results_dir = self.prune_config.get(
            'results_dir', getattr(self, 'default_pruner_results_dir', None)
        )
        self.pruner_mmmu_dir = self.prune_config.get(
            'mmmu_dir', getattr(self, 'default_pruner_mmmu_dir', None)
        )

    def load_dataset(self) -> DatasetDict:

        # Temporary debug print
        print(f"\n--- DEBUG MIXIN ---")
        print(f"Strategy received: {self.pruning_strategy}")
        print(f"Ratio received: {self.prune_ratio}")
        print(f"Data path received: {self.pruner_data_path}")
        print(f"Results dir received: {self.pruner_results_dir}")
        print(f"MMMU dir received: {self.pruner_mmmu_dir}")
        print(f"-------------------\n")

        # 1. Load the full in-memory dataset via the parent adapter
        dataset_dict = super().load_dataset()

        # 2. Pass-through if no pruning strategy is configured
        if not self.pruning_strategy or self.prune_ratio >= 1.0:
            return dataset_dict

        # 3. Calculate target size mathematically
        total_samples = sum(len(dataset) for dataset in dataset_dict.values())
        target_size = int(total_samples * self.prune_ratio)
        
        kept_indices = set()

        

        # 4. Execute the appropriate disk-based sampler
        if self.pruning_strategy == "discriminability":
            logger.info(f"Applying DiscriminabilitySampler (target_size={target_size})")
            sampler = DiscriminabilitySampler(
                target_size=target_size,
                data_path=self.pruner_data_path,
                results_dir=self.pruner_results_dir
            )
            # The sample() method reads from disk files, ignoring the in-memory dict
            pruned_items = sampler.sample()
            kept_indices = {item["index"] for item in pruned_items}
            
        elif self.pruning_strategy == "image_stress":
            logger.info(f"Applying ImageStressSampler (target_size={target_size})")
            sampler = ImageStressSampler(
                target_size=target_size,
                mmmu_dir=self.pruner_mmmu_dir
            )
            pruned_items = sampler.sample()
            kept_indices = {item["index"] for item in pruned_items}
        else:
            logger.warning(f"Unknown pruning_strategy: {self.pruning_strategy}")
            return dataset_dict

        # 5. Filter the in-memory evalscope Sample objects by mapping numerical indices
        for subset_name, dataset in dataset_dict.items():
            # Assume 1:1 numerical index alignment between JSONL files and loaded Dataset
            pruned_samples = [s for i, s in enumerate(dataset) if i in kept_indices]
            
            if hasattr(dataset, 'clear'):
                dataset.clear()
                dataset.extend(pruned_samples)
            else:
                dataset_dict[subset_name] = pruned_samples
                
        return dataset_dict