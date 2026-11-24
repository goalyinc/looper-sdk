"""The SDK as installed: its version, and looper.v1's generated messages on the
protobuf its range allows. looper-isolated also runs this file outside the
checkout, against the built wheel at the lowest versions its ranges allow."""

from __future__ import annotations

import importlib.metadata

from google.protobuf import json_format

import looper_sdk
from looper_sdk.proto.v1 import values_pb2


def test_the_distribution_carries_the_package_version() -> None:
    assert importlib.metadata.version("goaly-looper") == looper_sdk.__version__


def test_a_version_round_trips_as_binary_and_as_json() -> None:
    version = values_pb2.Version(run_id="run-1", epoch=2, step=30)
    assert values_pb2.Version.FromString(version.SerializeToString()) == version
    # protobuf's JSON mapping: camelCase names, 64-bit integers as strings.
    as_json = json_format.MessageToDict(version)
    assert as_json == {"runId": "run-1", "epoch": "2", "step": "30"}
    assert json_format.ParseDict(as_json, values_pb2.Version()) == version
