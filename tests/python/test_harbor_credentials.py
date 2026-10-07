# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Credential transport regressions using sentinel values, not real credentials."""

import json
import os
from unittest.mock import AsyncMock, MagicMock

import pytest

pytestmark = pytest.mark.usefixtures("requires_harbor")


@pytest.fixture(name="credential_agent")
def credential_agent_fixture(tmp_path):
    from nemo_fabric.integrations.harbor import FabricAgent

    return FabricAgent(
        logs_dir=tmp_path,
        fabric_adapter_id="example.adapter",
        model_name="nvidia/test-model",
        fabric_model_api_key_env="PILOT_API_KEY",
        fabric_environment_env={
            "PILOT_API_KEY": "pilot-sentinel-secret",
            "RUN_MODE": "test",
        },
    )


def test_values_use_harbor_redaction_and_name_only_transport(credential_agent):
    payload = credential_agent._build_spec("test")
    assert payload.environment_env_names == ("PILOT_API_KEY", "RUN_MODE")
    assert payload.config.environment.env == {}
    assert "pilot-sentinel-secret" not in payload.model_dump_json()
    assert credential_agent.extra_env["PILOT_API_KEY"] == "pilot-sentinel-secret"
    assert credential_agent._runner_env["RUN_MODE"] == "test"


def test_environment_conflicts_do_not_print_values(tmp_path):
    from nemo_fabric.integrations.harbor import FabricAgent

    with pytest.raises(ValueError, match="conflicting values") as caught:
        FabricAgent(
            logs_dir=tmp_path,
            fabric_adapter_id="example.adapter",
            extra_env={"PILOT_API_KEY": "first-secret"},
            fabric_environment_env={"PILOT_API_KEY": "second-secret"},
        )
    assert "first-secret" not in str(caught.value)
    assert "second-secret" not in str(caught.value)


@pytest.mark.parametrize("name", ["", "KEY=VALUE", "BAD NAME", "BAD\x00NAME"])
def test_invalid_environment_names_do_not_print_values(tmp_path, name):
    from nemo_fabric.integrations.harbor import FabricAgent

    with pytest.raises(ValueError, match="environment variable names") as caught:
        FabricAgent(
            logs_dir=tmp_path,
            fabric_adapter_id="example.adapter",
            fabric_environment_env={name: "pilot-sentinel-secret"},
        )
    assert "pilot-sentinel-secret" not in str(caught.value)


@pytest.mark.parametrize("name", ["", "KEY=VALUE", "BAD NAME", "BAD\x00NAME"])
def test_transport_rejects_invalid_environment_references(credential_agent, name):
    from nemo_fabric.integrations.harbor.models import FabricRunPayload

    document = credential_agent._build_spec("test").model_dump(mode="python")
    document["environment_env_names"] = [name]
    with pytest.raises(ValueError, match="environment variable names"):
        FabricRunPayload.model_validate(document)


async def test_runner_resolves_names_without_mutating_retained_payload(
    credential_agent, monkeypatch
):
    from nemo_fabric import Fabric, RunResult
    from nemo_fabric.integrations.harbor import runner

    os.environ["PILOT_API_KEY"] = "pilot-sentinel-secret"
    os.environ["RUN_MODE"] = "test"
    document = {
        "agent_name": "test",
        "harness": "external",
        "adapter_kind": "python",
        "adapter_id": "example.adapter",
        "runtime_id": "runtime",
        "invocation_id": "invocation",
        "request_id": "request",
        "status": "succeeded",
        "output": {"response": "done"},
        "error": None,
        "artifacts": {"artifacts": []},
        "telemetry": [],
        "events": [],
        "metadata": {},
    }
    mock_fabric = MagicMock(spec=Fabric)
    mock_fabric.run = AsyncMock(return_value=RunResult.from_mapping(document))
    monkeypatch.setattr(runner, "Fabric", MagicMock(return_value=mock_fabric))
    payload = credential_agent._build_spec("test")
    result = await runner.run(payload)
    assert mock_fabric.run.call_args.args[0].environment.env == {
        "PILOT_API_KEY": "pilot-sentinel-secret",
        "RUN_MODE": "test",
    }
    assert payload.config.environment.env == {}
    assert "pilot-sentinel-secret" not in json.dumps(result.to_mapping())


async def test_missing_reference_fails_before_execution(credential_agent, monkeypatch):
    from nemo_fabric import Fabric
    from nemo_fabric.integrations.harbor import runner

    os.environ.pop("PILOT_API_KEY", None)
    mock_fabric = MagicMock(spec=Fabric)
    monkeypatch.setattr(runner, "Fabric", mock_fabric)
    with pytest.raises(ValueError, match="PILOT_API_KEY is not set") as caught:
        await runner.run(credential_agent._build_spec("test"))
    mock_fabric.assert_not_called()
    assert "pilot-sentinel-secret" not in str(caught.value)


def test_harbor_scrubs_retained_specs_results_and_diagnostics(
    credential_agent, tmp_path
):
    from harbor.trial.trial import Trial

    mock_trial = MagicMock(spec=Trial)
    mock_trial.agent = credential_agent
    mock_trial.user_agent = None
    mock_trial.task = MagicMock()
    mock_trial.task.config.verifier.env = {}
    mock_trial.config = MagicMock()
    mock_trial.config.verifier.env = {}
    mock_trial.paths = MagicMock()
    mock_trial.paths.trial_dir = tmp_path
    for name in ("spec.json", "result.json", "runner.stderr"):
        (tmp_path / name).write_text(
            "diagnostic: pilot-sentinel-secret; non-sensitive: test", encoding="utf-8"
        )
    Trial._scrub_jobs_dir(mock_trial)
    for name in ("spec.json", "result.json", "runner.stderr"):
        text = (tmp_path / name).read_text(encoding="utf-8")
        assert "pilot-sentinel-secret" not in text
        assert "[REDACTED]" in text
        assert "non-sensitive: test" in text
