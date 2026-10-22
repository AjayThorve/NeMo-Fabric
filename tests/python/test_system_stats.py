# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native sampling, typed results, and explicit measurement availability."""

import json
import sys

import pytest

from nemo_fabric import Fabric
from nemo_fabric import FabricConfig
from nemo_fabric import FabricConfigError
from nemo_fabric import GaugeSummary
from nemo_fabric import Measurement
from nemo_fabric import RunRequest
from nemo_fabric import RunResult
from nemo_fabric import SampleSummary
from nemo_fabric import SystemStatsConfig


@pytest.mark.parametrize(
    "mapping",
    [
        {"sample_interval_ms": True},
        {"sample_interval_ms": 99},
        {"sample_interval_ms": 60001},
        {"max_samples": 1},
        {"max_samples": 100001},
        {"unknown": True},
    ],
)
def test_sampling_options_enforce_cadence_and_retention_bounds(mapping):
    with pytest.raises(ValueError):
        SystemStatsConfig.from_mapping(mapping)


@pytest.mark.parametrize(
    "mapping",
    [
        {"status": "available", "value": True},
        {"status": "available", "value": -1},
        {"status": "available", "value": float("nan")},
        {"status": "unavailable", "reason": ""},
        {"status": "unlimited", "value": 0},
    ],
)
def test_measurement_rejects_invalid_numeric_and_availability_states(mapping):
    with pytest.raises(FabricConfigError):
        Measurement.from_mapping(mapping)


@pytest.mark.skipif(
    sys.platform not in {"linux", "darwin"}, reason="requires a native process reader"
)
async def test_sdk_returns_typed_samples_and_artifacts_without_relay_or_recording(
    hermes_shim_agent_dir, tmp_path, monkeypatch
):
    monkeypatch.setenv("ADAPTER_PYTHON", sys.executable)
    config = FabricConfig.from_mapping(
        {
            "metadata": {"name": "native-stats-test"},
            "harness": {
                "adapter_id": "test.fabric.hermes_shim",
                "resolution": "preinstalled",
            },
            "discovery": {"local_paths": ["adapters"]},
            "runtime": {"artifacts": tmp_path / "artifacts"},
            "system_stats": {"sample_interval_ms": 100, "max_samples": 100},
        }
    )
    result = await Fabric().run(
        config,
        base_dir=hermes_shim_agent_dir,
        request=RunRequest(input="sample", context={"delay_seconds": 0.5}),
    )
    assert result.status == "succeeded"
    sampled = result.system_stats.sampled
    assert isinstance(sampled, SampleSummary)
    assert isinstance(sampled.resident_memory_bytes, GaugeSummary)
    assert sampled.sample_count >= 3
    assert (
        sampled.resident_memory_bytes.sampled_peak
        >= sampled.resident_memory_bytes.final_value
    )
    assert 0 < sampled.resident_memory_bytes.covered_fraction <= 1
    assert (
        RunResult.from_mapping(result.to_mapping()).system_stats == result.system_stats
    )
    refs = {reference.kind: reference for reference in result.artifacts.artifacts}
    assert sampled.samples.path == refs["system_stats_samples"].path
    rows = [json.loads(line) for line in sampled.samples.path.read_text().splitlines()]
    assert len(rows) == sampled.sample_count
    assert all(row["invocation_id"] == result.invocation_id for row in rows)
    assert rows[-1]["offset_seconds"] == result.system_stats.elapsed_seconds
    saved = json.loads(refs["system_stats_summary"].path.read_text())
    assert saved["invocation_id"] == result.invocation_id
    assert saved["system_stats"]["sampled"]["sample_count"] == sampled.sample_count
