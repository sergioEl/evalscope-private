# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from evalscope.utils.import_utils import _LazyModule

if TYPE_CHECKING:
    from .sampler import DatasetEntry, StratifiedSampler, UniformSampler, WeightedSampler
    from .schema import CollectionSchema, DatasetInfo
    from .pruning_samplers import DiscriminabilitySampler, ImageStressSampler
else:
    _import_structure = {
        'sampler': ['StratifiedSampler', 'UniformSampler', 'WeightedSampler', 'DatasetEntry'],
        'schema': [
            'CollectionSchema',
            'DatasetInfo',
        ],
        'pruning_samplers': [
            'DiscriminabilitySampler',
            'ImageStressSampler',
        ],
    }
    import sys
    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
