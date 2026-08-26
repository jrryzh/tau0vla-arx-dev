"""Safe wire serialization for the inference server.

``/act`` and ``/act_lerobot_bytes`` transport nested inference payloads.
Deserializing a request body with ``pickle.loads`` is unauthenticated remote
code execution, so payloads travel as a small, strictly validated ``.npz``
archive. NumPy arrays are archive entries and the container structure is a
tagged JSON tree; neither side ever enables NumPy object deserialization.
"""

from __future__ import annotations

import json
import zipfile
from io import BytesIO
from typing import Any, Dict

import numpy as np

# The server enforces MAX_BODY_BYTES while streaming the request. The decoder
# repeats that check for non-HTTP callers and separately limits the sum of the
# archive entries' uncompressed sizes to prevent compressed zip bombs.
MAX_BODY_BYTES = 256 * 1024 * 1024
MAX_DECODED_BYTES = MAX_BODY_BYTES

_JSON_ENTRY = "__json__"
_ARR_PREFIX = "__arr"
_MAX_ARRAYS = 1024
_ALLOWED_ARRAY_KINDS = frozenset("biuf")
_ALLOWED_COMPRESSION = frozenset((zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED))


class PayloadEncodingError(ValueError):
    """The request body is not a valid safe-wire payload."""


def _validate_array(array: np.ndarray) -> None:
    if array.dtype.hasobject or array.dtype.kind not in _ALLOWED_ARRAY_KINDS:
        raise TypeError(
            f"wire arrays must have a bool, integer, unsigned-integer, or floating dtype; got {array.dtype}"
        )


def pack_payload(payload: Dict[str, Any]) -> bytes:
    """Encode an inference payload as ``.npz`` bytes without pickle."""
    if not isinstance(payload, dict):
        raise TypeError(f"payload root must be a dict, got {type(payload).__name__}")

    arrays: list[np.ndarray] = []

    def encode(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            array = np.asarray(value)
            _validate_array(array)
            if len(arrays) >= _MAX_ARRAYS:
                raise ValueError(f"payload contains more than {_MAX_ARRAYS} arrays")
            arrays.append(array)
            return ["ndarray", len(arrays) - 1]
        if isinstance(value, np.generic):
            return encode(value.item())
        if isinstance(value, dict):
            items = []
            for key, item in value.items():
                if not isinstance(key, str):
                    raise TypeError(f"wire dictionary keys must be strings, got {type(key).__name__}")
                items.append([key, encode(item)])
            return ["dict", items]
        if isinstance(value, list):
            return ["list", [encode(item) for item in value]]
        if isinstance(value, tuple):
            return ["tuple", [encode(item) for item in value]]
        if value is None or isinstance(value, (bool, int, float, str)):
            return ["scalar", value]
        raise TypeError(f"unsupported wire value type: {type(value).__name__}")

    tree = encode(payload)
    try:
        envelope = json.dumps(tree, allow_nan=False, separators=(",", ":"), ensure_ascii=False)
    except (TypeError, ValueError) as error:
        raise TypeError(f"payload contains a non-JSON scalar: {error}") from error

    buf = BytesIO()
    np.savez(
        buf,
        **{_JSON_ENTRY: envelope},
        **{f"{_ARR_PREFIX}{index}": array for index, array in enumerate(arrays)},
    )
    data = buf.getvalue()
    if len(data) > MAX_BODY_BYTES:
        raise ValueError(f"encoded payload is {len(data)} bytes; limit is {MAX_BODY_BYTES}")
    return data


def _archive_array_indices(archive: zipfile.ZipFile) -> list[int]:
    infos = archive.infolist()
    if len(infos) > _MAX_ARRAYS + 1:
        raise PayloadEncodingError("archive contains too many entries")

    names = [info.filename for info in infos]
    if len(names) != len(set(names)):
        raise PayloadEncodingError("archive contains duplicate entries")
    if f"{_JSON_ENTRY}.npy" not in names:
        raise PayloadEncodingError("archive is missing its JSON envelope")

    decoded_size = 0
    indices = []
    for info in infos:
        if info.is_dir() or info.flag_bits & 0x1:
            raise PayloadEncodingError("archive directories and encryption are not supported")
        if info.compress_type not in _ALLOWED_COMPRESSION:
            raise PayloadEncodingError("archive uses an unsupported compression method")
        decoded_size += info.file_size
        if decoded_size > MAX_DECODED_BYTES:
            raise PayloadEncodingError(f"decoded payload exceeds the {MAX_DECODED_BYTES}-byte limit")

        if info.filename == f"{_JSON_ENTRY}.npy":
            continue
        prefix = f"{_ARR_PREFIX}"
        suffix = ".npy"
        if not info.filename.startswith(prefix) or not info.filename.endswith(suffix):
            raise PayloadEncodingError(f"unexpected archive entry: {info.filename!r}")
        index_text = info.filename[len(prefix) : -len(suffix)]
        if not index_text.isascii() or not index_text.isdecimal():
            raise PayloadEncodingError(f"invalid array entry: {info.filename!r}")
        indices.append(int(index_text))

    indices.sort()
    if indices != list(range(len(indices))):
        raise PayloadEncodingError("array entries must be numbered contiguously from zero")
    return indices


def _decode_tree(tree: Any, arrays: list[np.ndarray]) -> Dict[str, Any]:
    used_arrays: list[int] = []

    def decode(node: Any) -> Any:
        if not isinstance(node, list) or len(node) != 2 or not isinstance(node[0], str):
            raise PayloadEncodingError("invalid JSON node")
        tag, value = node
        if tag == "ndarray":
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < len(arrays):
                raise PayloadEncodingError("invalid ndarray reference")
            used_arrays.append(value)
            return arrays[value]
        if tag in ("list", "tuple"):
            if not isinstance(value, list):
                raise PayloadEncodingError(f"invalid {tag} node")
            items = [decode(item) for item in value]
            return items if tag == "list" else tuple(items)
        if tag == "dict":
            if not isinstance(value, list):
                raise PayloadEncodingError("invalid dict node")
            result = {}
            for pair in value:
                if not isinstance(pair, list) or len(pair) != 2 or not isinstance(pair[0], str):
                    raise PayloadEncodingError("invalid dictionary item")
                key, item = pair
                if key in result:
                    raise PayloadEncodingError(f"duplicate dictionary key: {key!r}")
                result[key] = decode(item)
            return result
        if tag == "scalar":
            if value is not None and not isinstance(value, (bool, int, float, str)):
                raise PayloadEncodingError("invalid scalar node")
            return value
        raise PayloadEncodingError(f"unknown JSON node tag: {tag!r}")

    result = decode(tree)
    if not isinstance(result, dict):
        raise PayloadEncodingError("payload root must be a dict")
    if sorted(used_arrays) != list(range(len(arrays))):
        raise PayloadEncodingError("each archive array must be referenced exactly once")
    return result


def unpack_payload(data: bytes) -> Dict[str, Any]:
    """Decode a ``pack_payload`` body. Never unpickles anything."""
    if len(data) > MAX_BODY_BYTES:
        raise PayloadEncodingError(f"payload body {len(data)} bytes exceeds {MAX_BODY_BYTES} limit")

    try:
        source = BytesIO(data)
        with zipfile.ZipFile(source) as archive:
            indices = _archive_array_indices(archive)

        source.seek(0)
        with np.load(source, allow_pickle=False) as npz:
            expected_entries = [_JSON_ENTRY, *(f"{_ARR_PREFIX}{index}" for index in indices)]
            if sorted(npz.files) != sorted(expected_entries):
                raise PayloadEncodingError("archive entry names do not match the wire schema")

            envelope = npz[_JSON_ENTRY]
            if envelope.shape != () or envelope.dtype.kind not in ("U", "S"):
                raise PayloadEncodingError("JSON envelope must be a scalar string")
            tree = json.loads(envelope.item())

            arrays = []
            decoded_array_bytes = 0
            for index in indices:
                array = np.asarray(npz[f"{_ARR_PREFIX}{index}"])
                _validate_array(array)
                decoded_array_bytes += array.nbytes
                if decoded_array_bytes > MAX_DECODED_BYTES:
                    raise PayloadEncodingError("decoded arrays exceed the size limit")
                arrays.append(array)
        return _decode_tree(tree, arrays)
    except PayloadEncodingError:
        raise
    except Exception as error:
        raise PayloadEncodingError("invalid payload encoding") from error
