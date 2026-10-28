"""Safe, portable NumPy model serialization without pickle or external dependencies."""

import json
import numpy as np


def save_model(path, model):
    """Store nested dicts/lists/tuples and numeric ndarrays into a compressed .npz archive.

    Uses json for metadata and pure NumPy arrays with allow_pickle=False for maximum
    portability and security across Python versions and environments.
    """
    arrays = {}

    def encode(obj):
        if isinstance(obj, np.ndarray):
            key = f"arr_{len(arrays)}"
            arrays[key] = np.ascontiguousarray(obj)
            return {"__ndarray__": key}
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, tuple):
            return {"__tuple__": [encode(v) for v in obj]}
        if isinstance(obj, list):
            return [encode(v) for v in obj]
        if isinstance(obj, dict):
            return {"__dict__": [[encode(k), encode(v)] for k, v in obj.items()]}
        return obj

    manifest = encode(model)
    np.savez_compressed(path, __manifest__=np.array(json.dumps(manifest)), **arrays)


def load_model(path):
    """Load model dictionary from a compressed .npz archive with allow_pickle=False."""
    with np.load(path, allow_pickle=False) as archive:

        def decode(obj):
            if isinstance(obj, list):
                return [decode(v) for v in obj]
            if isinstance(obj, dict):
                if "__ndarray__" in obj:
                    return archive[obj["__ndarray__"]].copy()
                if "__tuple__" in obj:
                    return tuple(decode(v) for v in obj["__tuple__"])
                if "__dict__" in obj:
                    return {decode(k): decode(v) for k, v in obj["__dict__"]}
            return obj

        manifest_str = str(archive["__manifest__"])
        return decode(json.loads(manifest_str))
