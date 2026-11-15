import base64
from typing import Any

import numpy as np


def serialize_array(arr: np.ndarray) -> dict[str, Any]:
    """Преобразует numpy array в словарь с base64 данными и метаданными."""
    return {
        'data': base64.b64encode(arr.tobytes()).decode('utf-8'),
        'dtype': str(arr.dtype),
        'shape': arr.shape,
    }
