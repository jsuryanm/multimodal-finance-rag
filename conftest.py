"""
Root conftest.py — applied before any test collection.

Sets PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python so that the installed
protobuf 7.x is compatible with the older opentelemetry-proto package
that chromadb depends on (opentelemetry-proto requires protobuf <7 for its
compiled C extension but works fine with the pure-Python implementation).
"""
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
