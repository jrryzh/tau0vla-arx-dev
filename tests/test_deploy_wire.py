from __future__ import annotations

import io
import json
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from fastapi import HTTPException

from deploy.server import _read_payload
from deploy.wire import PayloadEncodingError, pack_payload, unpack_payload


class _WriteOnUnpickle:
    def __init__(self, path: Path):
        self.path = path

    def __reduce__(self):
        return Path.write_text, (self.path, "pickle executed")


class _Request:
    def __init__(self, chunks, *, content_length=None):
        self.headers = {}
        if content_length is not None:
            self.headers["content-length"] = str(content_length)
        self._chunks = chunks
        self.chunks_read = 0

    async def stream(self):
        for chunk in self._chunks:
            self.chunks_read += 1
            yield chunk


class WireFormatTest(unittest.TestCase):
    def test_round_trip_nested_payload(self):
        payload = {
            "prompt": "pick up the blue cup",
            "images": {
                "front": np.arange(3 * 4 * 3, dtype=np.uint8).reshape(3, 4, 3),
                "wrist": np.zeros((2, 2, 3), dtype=np.uint8),
            },
            "state": np.array([1.5, -2.0], dtype=np.float32),
            "meta": {
                "episode": np.int64(7),
                "confidence": np.float32(0.5),
                "flags": [True, None, "ready"],
                "tuple": (1, 2),
            },
        }

        decoded = unpack_payload(pack_payload(payload))

        self.assertEqual(decoded["prompt"], payload["prompt"])
        np.testing.assert_array_equal(decoded["images"]["front"], payload["images"]["front"])
        np.testing.assert_array_equal(decoded["images"]["wrist"], payload["images"]["wrist"])
        np.testing.assert_array_equal(decoded["state"], payload["state"])
        self.assertEqual(decoded["state"].dtype, np.dtype(np.float32))
        self.assertEqual(
            decoded["meta"],
            {"episode": 7, "confidence": 0.5, "flags": [True, None, "ready"], "tuple": (1, 2)},
        )

    def test_marker_shaped_user_dictionary_round_trips(self):
        payload = {"meta": {"__nd__": 0}, "array": np.array([9], dtype=np.int16)}
        decoded = unpack_payload(pack_payload(payload))
        self.assertEqual(decoded["meta"], {"__nd__": 0})
        np.testing.assert_array_equal(decoded["array"], payload["array"])

    def test_rejects_unsafe_or_ambiguous_client_values(self):
        with self.assertRaises(TypeError):
            pack_payload({"array": np.array([object()], dtype=object)})
        with self.assertRaises(TypeError):
            pack_payload({1: "non-string key"})
        with self.assertRaises(TypeError):
            pack_payload({"complex": np.array([1j], dtype=np.complex64)})

    def test_rejects_pickle_and_garbage(self):
        for body in (pickle.dumps({"prompt": "unsafe"}), b"not an npz archive"):
            with self.subTest(body=body[:10]):
                with self.assertRaises(PayloadEncodingError):
                    unpack_payload(body)

    def test_rejecting_pickle_does_not_execute_reduce(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "executed"
            body = pickle.dumps(_WriteOnUnpickle(marker))
            with self.assertRaises(PayloadEncodingError):
                unpack_payload(body)
            self.assertFalse(marker.exists())

    def test_rejects_body_over_wire_limit(self):
        with mock.patch("deploy.wire.MAX_BODY_BYTES", 8):
            with self.assertRaisesRegex(PayloadEncodingError, "payload body"):
                unpack_payload(b"123456789")

    def test_rejects_unexpected_archive_entry(self):
        buf = io.BytesIO()
        np.savez(buf, __json__=json.dumps(["dict", []]), unexpected=np.array([1]))
        with self.assertRaisesRegex(PayloadEncodingError, "unexpected archive entry"):
            unpack_payload(buf.getvalue())

    def test_rejects_non_contiguous_array_entries(self):
        buf = io.BytesIO()
        tree = ["dict", [["array", ["ndarray", 1]]]]
        np.savez(buf, __json__=json.dumps(tree), __arr1=np.array([1], dtype=np.int8))
        with self.assertRaisesRegex(PayloadEncodingError, "numbered contiguously"):
            unpack_payload(buf.getvalue())

    def test_rejects_compressed_archive_over_decoded_limit(self):
        buf = io.BytesIO()
        tree = ["dict", [["array", ["ndarray", 0]]]]
        np.savez_compressed(buf, __json__=json.dumps(tree), __arr0=np.zeros(4096, dtype=np.uint8))
        self.assertLess(len(buf.getvalue()), 1024)
        with mock.patch("deploy.wire.MAX_DECODED_BYTES", 1024):
            with self.assertRaisesRegex(PayloadEncodingError, "decoded payload exceeds"):
                unpack_payload(buf.getvalue())


class RequestLimitTest(unittest.IsolatedAsyncioTestCase):
    async def test_valid_chunked_request(self):
        body = pack_payload({"state": np.array([1.0], dtype=np.float32)})
        request = _Request([body[:17], body[17:]], content_length=len(body))
        decoded = await _read_payload(request)
        np.testing.assert_array_equal(decoded["state"], np.array([1.0], dtype=np.float32))

    async def test_content_length_rejected_before_streaming(self):
        request = _Request([b"must not be read"], content_length=9)
        with mock.patch("deploy.server.MAX_BODY_BYTES", 8):
            with self.assertRaises(HTTPException) as caught:
                await _read_payload(request)
        self.assertEqual(caught.exception.status_code, 413)
        self.assertEqual(request.chunks_read, 0)

    async def test_chunked_body_stops_at_limit(self):
        request = _Request([b"1234", b"56789", b"must not be read"])
        with mock.patch("deploy.server.MAX_BODY_BYTES", 8):
            with self.assertRaises(HTTPException) as caught:
                await _read_payload(request)
        self.assertEqual(caught.exception.status_code, 413)
        self.assertEqual(request.chunks_read, 2)

    async def test_invalid_content_length_and_encoding_return_400(self):
        request = _Request([b"garbage"], content_length="invalid")
        with self.assertRaises(HTTPException) as caught:
            await _read_payload(request)
        self.assertEqual(caught.exception.status_code, 400)

        request = _Request([b"garbage"])
        with self.assertRaises(HTTPException) as caught:
            await _read_payload(request)
        self.assertEqual(caught.exception.status_code, 400)


if __name__ == "__main__":
    unittest.main()
