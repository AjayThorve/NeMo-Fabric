# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native sampling, interval reductions, and explicit measurement availability."""

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
from nemo_fabric import summarize_system_stats


def _sample(offset, memory, cpu):
    observation = {"status": "unavailable", "reason": "process moved"}
    if memory is not None:
        observation = {
            "status": "available",
            "snapshot": {
                "runtime_id": "runtime-1",
                "started_at": "2026-10-21T00:00:00.000Z",
                "observed_at": "2026-10-21T00:01:00.000Z",
                "resident_memory_bytes": memory,
                "process_count": 1,
                "cpu_time_seconds": cpu,
            },
        }
    return {
        "schema_version": "fabric.system_stats/v1alpha1",
        "runtime_id": "runtime-1",
        "invocation_id": "invocation-1",
        "request_id": "request-1",
        "offset_seconds": offset,
        "adapter_tree": observation,
    }


@pytest.fixture(name="stats_file")
def stats_file_fixture(tmp_path):
    path = tmp_path / "system-stats.jsonl"
    rows = [_sample(0, 80, 0.1), _sample(1, 400, 0.6), _sample(2, 80, 1.1)]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def test_series_reductions_retain_a_spike_that_endpoint_memory_misses(stats_file):
    summary = summarize_system_stats(stats_file, target_interval_ms=1000)
    memory = summary["resident_memory_bytes"]
    assert memory["final_value"] == 80
    assert memory["sampled_peak"] == 400
    assert memory["range"] == [80, 400]
    assert memory["time_weighted_mean"] == 240
    assert memory["covered_fraction"] == 1
    assert summary["cpu_time_seconds_estimate"] == 1
    assert summary["cpu_percent_one_core"] == 50


def test_selected_window_discloses_interpolated_boundaries(stats_file):
    summary = summarize_system_stats(
        stats_file, target_interval_ms=1000, start_seconds=0.5, end_seconds=1.5
    )
    memory = summary["resident_memory_bytes"]
    assert not summary["start_observed"]
    assert not summary["end_observed"]
    assert memory["sampled_peak"] == 400
    assert memory["final_value"] is None
    assert memory["time_weighted_mean"] == 240
    assert memory["covered_fraction"] == 1
    assert summary["cpu_time_seconds_estimate"] is None


def test_missing_samples_and_counter_resets_stay_unknown(stats_file):
    stats_file.write_text(
        "".join(
            json.dumps(row) + "\n"
            for row in [
                _sample(0, 80, 2),
                _sample(1, 400, 1),
                _sample(2, 80, 3),
            ]
        )
    )
    assert (
        summarize_system_stats(stats_file, target_interval_ms=1000)[
            "cpu_time_seconds_estimate"
        ]
        is None
    )
    stats_file.write_text(
        "".join(
            json.dumps(row) + "\n"
            for row in [
                _sample(0, 80, 2),
                _sample(1, None, 0),
                _sample(2, 80, 3),
            ]
        )
    )
    summary = summarize_system_stats(stats_file, target_interval_ms=1000)
    assert summary["resident_memory_bytes"]["time_weighted_mean"] is None
    assert summary["resident_memory_bytes"]["covered_fraction"] == 0
    assert summary["cpu_time_seconds_estimate"] is None


def test_interrupted_tail_is_disclosed_and_mixed_invocations_are_rejected(stats_file):
    with stats_file.open("ab") as stream:
        stream.write(b'{"schema_')
    summary = summarize_system_stats(stats_file, target_interval_ms=1000)
    assert summary["diagnostics"] == ["partial_tail_ignored"]
    assert summary["resident_memory_bytes"]["sampled_peak"] == 400
    rows = [_sample(0, 80, 0), {**_sample(1, 80, 1), "invocation_id": "other"}]
    stats_file.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(FabricConfigError, match="one invocation"):
        summarize_system_stats(stats_file, target_interval_ms=1000)


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
    selected = summarize_system_stats(sampled.samples.path, target_interval_ms=100)
    assert selected["invocation_id"] == result.invocation_id
    assert (
        selected["resident_memory_bytes"]["sampled_peak"]
        == sampled.resident_memory_bytes.sampled_peak
    )
    assert selected["resident_memory_bytes"]["time_weighted_mean"] == pytest.approx(
        sampled.resident_memory_bytes.time_weighted_mean
    )
    saved = json.loads(refs["system_stats_summary"].path.read_text())
    assert saved["invocation_id"] == result.invocation_id
    assert saved["system_stats"]["sampled"]["sample_count"] == sampled.sample_count
