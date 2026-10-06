# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Credential-free accounting, transport, and capability pilot regressions."""

import json
import os
from unittest.mock import AsyncMock, MagicMock

import pytest

pytestmark = pytest.mark.usefixtures("requires_harbor")


@pytest.fixture(name="pilot_agent")
def pilot_agent_fixture(tmp_path):
    from nemo_fabric.integrations.harbor import FabricAgent

    return FabricAgent(
        logs_dir=tmp_path,
        fabric_adapter_id="nvidia.fabric.codex",
        model_name="nvidia/test-model",
        fabric_model_api_key_env="PILOT_API_KEY",
        fabric_environment_env={
            "PILOT_API_KEY": "pilot-sentinel-secret",
            "RUN_MODE": "test",
        },
    )


def test_environment_values_use_harbor_redaction_and_name_only_transport(pilot_agent):
    payload = pilot_agent._build_spec("test")
    assert payload.environment_env_names == ("PILOT_API_KEY", "RUN_MODE")
    assert payload.config.environment.env == {}
    assert "pilot-sentinel-secret" not in payload.model_dump_json()
    assert pilot_agent.extra_env["PILOT_API_KEY"] == "pilot-sentinel-secret"
    assert pilot_agent._runner_env["RUN_MODE"] == "test"


def test_environment_conflicts_fail_without_printing_values(tmp_path):
    from nemo_fabric.integrations.harbor import FabricAgent

    with pytest.raises(ValueError, match="conflicting values") as caught:
        FabricAgent(
            logs_dir=tmp_path,
            fabric_adapter_id="nvidia.fabric.codex",
            extra_env={"PILOT_API_KEY": "first-secret"},
            fabric_environment_env={"PILOT_API_KEY": "second-secret"},
        )
    assert "first-secret" not in str(caught.value)
    assert "second-secret" not in str(caught.value)


@pytest.mark.parametrize("name", ["", "KEY=VALUE", "BAD NAME", "BAD\x00NAME"])
def test_transport_rejects_invalid_environment_names(pilot_agent, name):
    from nemo_fabric.integrations.harbor.models import FabricRunPayload

    document = pilot_agent._build_spec("test").model_dump(mode="python")
    document["environment_env_names"] = [name]
    with pytest.raises(ValueError, match="environment variable names"):
        FabricRunPayload.model_validate(document)


@pytest.fixture(name="result_document")
def result_document_fixture():
    return {
        "agent_name": "harbor-test",
        "harness": "codex",
        "adapter_kind": "python",
        "adapter_id": "nvidia.fabric.codex",
        "runtime_id": "runtime-test",
        "invocation_id": "invocation-test",
        "request_id": "request-test",
        "status": "succeeded",
        "output": {"response": "done"},
        "error": None,
        "artifacts": {"artifacts": []},
        "telemetry": [],
        "events": [],
        "metadata": {},
    }


async def test_runner_reconstructs_environment_without_retaining_values(
    pilot_agent, result_document, monkeypatch
):
    from nemo_fabric import Fabric, RunResult
    from nemo_fabric.integrations.harbor import runner

    os.environ["PILOT_API_KEY"] = "pilot-sentinel-secret"
    os.environ["RUN_MODE"] = "test"
    mock_fabric = MagicMock(spec=Fabric)
    mock_fabric.run = AsyncMock(return_value=RunResult.from_mapping(result_document))
    monkeypatch.setattr(runner, "Fabric", MagicMock(return_value=mock_fabric))
    result = await runner.run(pilot_agent._build_spec("test"))
    config = mock_fabric.run.call_args.args[0]
    assert config.environment.env == {
        "PILOT_API_KEY": "pilot-sentinel-secret",
        "RUN_MODE": "test",
    }
    assert "pilot-sentinel-secret" not in json.dumps(result.to_mapping())
    assert result.metadata["harbor_provenance"]["adapter_id"] == "nvidia.fabric.codex"
    assert result.metadata["harbor_provenance"]["harness_version"] is None


async def test_runner_rejects_missing_environment_reference(pilot_agent):
    from nemo_fabric.integrations.harbor import runner

    os.environ.pop("PILOT_API_KEY", None)
    with pytest.raises(ValueError, match="PILOT_API_KEY is not set"):
        await runner.run(pilot_agent._build_spec("test"))


@pytest.mark.parametrize("status", ["succeeded", "failed", "cancelled"])
@pytest.mark.parametrize("cost", [None, 0.0, 1.25])
def test_adapter_usage_reaches_harbor_without_atif(
    result_document, tmp_path, status, cost
):
    from harbor.models.agent.context import AgentContext
    from nemo_fabric import RunResult
    from nemo_fabric_adapters.codex import adapter
    from nemo_fabric.integrations.harbor.fabric_agent import (
        populate_context_from_result,
    )

    typed = adapter._agent_run_result(
        {
            "usage": {
                "total": {
                    "inputTokens": 100,
                    "cachedInputTokens": 40,
                    "outputTokens": 25,
                    "totalTokens": 125,
                }
            }
        }
    )
    assert typed.usage.input_tokens == 100
    result_document["status"] = status
    usage = typed.usage.to_mapping()
    usage["metadata"] = usage.pop("extensions")
    usage["cost_usd"] = cost
    result_document["usage"] = usage
    result_document["metadata"]["harbor_provenance"] = {
        "harness_version": "test-version"
    }
    result = RunResult.from_mapping(result_document)
    path = tmp_path / "result.json"
    path.write_text(json.dumps(result.to_mapping()), encoding="utf-8")
    context = AgentContext()
    populate_context_from_result(context, path)
    assert (
        context.n_input_tokens,
        context.n_cache_tokens,
        context.n_output_tokens,
    ) == (100, 40, 25)
    assert context.cost_usd == cost
    assert context.metadata["fabric"]["provenance"]["harness_version"] == "test-version"


@pytest.mark.parametrize(
    "adapter_id,mcp", [("nvidia.fabric.pi", False), ("nvidia.fabric.codex", True)]
)
@pytest.mark.parametrize("telemetry", ["none", "relay"])
def test_pilot_capabilities_match_selected_execution_path(
    tmp_path, adapter_id, mcp, telemetry
):
    from nemo_fabric.integrations.harbor import FabricAgent

    agent = FabricAgent(
        logs_dir=tmp_path, fabric_adapter_id=adapter_id, fabric_telemetry=telemetry
    )
    assert FabricAgent.capabilities.atif is False
    assert agent.capabilities.atif is (telemetry == "relay")
    for name, expected in {"skills": True, "mcp_servers": mcp}.items():
        if name in type(agent.capabilities).model_fields:
            assert getattr(agent.capabilities, name) is expected


def test_pi_rejects_mcp_before_any_harness_execution(tmp_path):
    from harbor.models.task.config import MCPServerConfig
    from nemo_fabric.integrations.harbor import FabricAgent

    with pytest.raises(ValueError, match="Pi adapter does not support MCP"):
        FabricAgent(
            logs_dir=tmp_path,
            fabric_adapter_id="nvidia.fabric.pi",
            mcp_servers=[
                MCPServerConfig(name="test", transport="stdio", command="test-mcp"),
            ],
        )


def test_supported_codex_mcp_passes_fabric_planning(tmp_path):
    from harbor.models.task.config import MCPServerConfig
    from nemo_fabric import Fabric
    from nemo_fabric.integrations.harbor import FabricAgent

    agent = FabricAgent(
        logs_dir=tmp_path,
        fabric_adapter_id="nvidia.fabric.codex",
        model_name="openai/test-model",
        mcp_servers=[
            MCPServerConfig(name="test", transport="stdio", command="test-mcp"),
        ],
    )
    plan = Fabric().plan(agent._build_config())
    assert plan.agent_config["mcp"]["servers"]["test"]["url"] == "test-mcp"


def test_runner_failure_provenance_is_explicit_when_versions_unknown(
    pilot_agent, monkeypatch
):
    from nemo_fabric.integrations.harbor import runner
    import importlib.metadata

    monkeypatch.setattr(
        runner.importlib.metadata,
        "version",
        MagicMock(side_effect=importlib.metadata.PackageNotFoundError),
    )
    assert runner.execution_provenance(pilot_agent._build_spec("test")) == {
        "harness": "codex",
        "harness_version": None,
        "adapter_id": "nvidia.fabric.codex",
        "adapter_version": None,
        "harness_sdk_version": None,
        "fabric_runtime_version": None,
    }


@pytest.fixture(name="pi_provenance")
def pi_provenance_fixture(tmp_path, monkeypatch):
    from nemo_fabric import Fabric, RunPlan
    from nemo_fabric.integrations.harbor import FabricAgent, runner

    payload = FabricAgent(
        logs_dir=tmp_path, fabric_adapter_id="nvidia.fabric.pi"
    )._build_spec("test")
    mock_plan = MagicMock(spec=RunPlan)
    mock_plan.get.return_value = {
        "provenance": [{"root": str(tmp_path)}, {"root": "/unselected/adapter"}]
    }
    mock_fabric = MagicMock(spec=Fabric)
    mock_fabric.plan.return_value = mock_plan
    monkeypatch.setattr(runner, "Fabric", MagicMock(return_value=mock_fabric))
    return payload, tmp_path / "package.json"


def test_runner_pi_lifecycle_failure_retains_selected_adapter_version(
    pi_provenance, tmp_path, monkeypatch
):
    import sys
    from nemo_fabric import FabricRuntimeError
    from nemo_fabric.integrations.harbor import runner

    payload, package_path = pi_provenance
    package_path.write_text(
        json.dumps({"name": "nemo-fabric-adapters-pi", "version": "0.5.0"}),
        encoding="utf-8",
    )
    spec_path = tmp_path / "spec.json"
    result_path = tmp_path / "result.json"
    spec_path.write_text(payload.model_dump_json(), encoding="utf-8")
    monkeypatch.setattr(
        sys, "argv", ["runner", "--spec", str(spec_path), "--result", str(result_path)]
    )
    monkeypatch.setattr(
        runner, "run", AsyncMock(side_effect=FabricRuntimeError("startup failed"))
    )
    with pytest.raises(FabricRuntimeError, match="startup failed"):
        runner.main()
    document = json.loads(result_path.read_text())
    assert document["provenance"]["adapter_version"] == "0.5.0"
    assert document["provenance"]["harness_version"] is None
    assert document["runner_error"]["message"] == "startup failed"


@pytest.mark.parametrize(
    "contents",
    [
        None,
        "not json",
        "[]",
        '{"name": "different-adapter", "version": "0.5.0"}',
        '{"name": "nemo-fabric-adapters-pi"}',
        '{"name": "nemo-fabric-adapters-pi", "version": 5}',
        '{"name": "nemo-fabric-adapters-pi", "version": " "}',
    ],
)
def test_runner_pi_unavailable_version_does_not_mask_failure(pi_provenance, contents):
    from nemo_fabric.integrations.harbor import runner

    payload, package_path = pi_provenance
    if contents is not None:
        package_path.write_text(contents, encoding="utf-8")
    assert runner.execution_provenance(payload)["adapter_version"] is None


def test_runner_pi_unresolved_descriptor_keeps_version_unknown(pi_provenance):
    from nemo_fabric import FabricConfigError
    from nemo_fabric.integrations.harbor import runner

    payload, _ = pi_provenance
    runner.Fabric.return_value.plan.side_effect = FabricConfigError("cannot resolve Pi")
    assert runner.execution_provenance(payload)["adapter_version"] is None


@pytest.mark.parametrize(
    "usage",
    [
        {"input_tokens": 10},
        {"output_tokens": 3},
        {"metadata": {"cached_input_tokens": 4}},
    ],
)
def test_omitted_optional_usage_fields_remain_unknown(result_document, tmp_path, usage):
    from harbor.models.agent.context import AgentContext
    from nemo_fabric.integrations.harbor.fabric_agent import (
        populate_context_from_result,
    )

    result_document["usage"] = usage
    path = tmp_path / "result.json"
    path.write_text(json.dumps(result_document), encoding="utf-8")
    context = AgentContext()
    populate_context_from_result(context, path)
    assert context.n_input_tokens == usage.get("input_tokens")
    assert context.n_output_tokens == usage.get("output_tokens")
    assert context.cost_usd is None
