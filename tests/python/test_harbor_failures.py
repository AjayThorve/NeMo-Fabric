# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Credential-free regression tests for the Harbor runner boundary."""

import asyncio
import json
import sys
from unittest.mock import AsyncMock

import pytest

pytestmark = pytest.mark.usefixtures("requires_harbor")


@pytest.fixture(name="run_result")
def run_result_fixture():
    from nemo_fabric import RunResult

    return RunResult.from_mapping(
        {
            "agent_name": "harbor-test",
            "harness": "test",
            "adapter_kind": "process",
            "adapter_id": "test.adapter",
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
    )


@pytest.fixture(name="bridge")
def bridge_fixture(tmp_path, run_result):
    from harbor.environments.base import BaseEnvironment, ExecResult
    from nemo_fabric.integrations.harbor import FabricAgent

    document = run_result.to_mapping()
    environment = AsyncMock(spec=BaseEnvironment)
    environment.exec.return_value = ExecResult(return_code=0, stdout="", stderr="")

    async def download(source, target):
        target.write_text(json.dumps(document), encoding="utf-8")

    environment.download_file.side_effect = download
    agent = FabricAgent(logs_dir=tmp_path, fabric_adapter_id="test.adapter")
    return agent, environment, document


@pytest.mark.parametrize("status", ["failed", "cancelled"])
@pytest.mark.parametrize("return_code", [0, 1])
async def test_bridge_rejects_unsuccessful_result_and_preserves_evidence(
    bridge, status, return_code
):
    from harbor.models.agent.context import AgentContext

    agent, environment, document = bridge
    document["status"] = status
    environment.exec.return_value.return_code = return_code
    context = AgentContext()

    exception = asyncio.CancelledError if status == "cancelled" else RuntimeError
    with pytest.raises(exception, match=status):
        await agent.run("test", environment, context)

    assert agent._result_path.is_file()
    assert context.is_empty()
    agent.populate_context_post_run(context)
    assert context.metadata["fabric"]["status"] == status
    assert context.metadata["fabric"]["runtime_id"] == "runtime-test"


@pytest.mark.parametrize("code", ["host_timeout", "timeout", "codex_timed_out"])
async def test_bridge_maps_normalized_timeout_to_timeout_error(bridge, code):
    from harbor.models.agent.context import AgentContext

    agent, environment, document = bridge
    document["status"] = "failed"
    document["error"] = {
        "stage": "invoke",
        "code": code,
        "message": "deadline exceeded",
        "retryable": False,
    }
    context = AgentContext()
    with pytest.raises(TimeoutError, match="deadline exceeded"):
        await agent.run("test", environment, context)
    agent.populate_context_post_run(context)
    assert context.metadata["fabric"]["error"]["code"] == code


async def test_cancellation_takes_precedence_over_error_classification(bridge):
    from harbor.models.agent.context import AgentContext

    agent, environment, document = bridge
    document["status"] = "cancelled"
    document["error"] = {
        "stage": "invoke",
        "code": "host_timeout",
        "message": "cancelled",
        "retryable": False,
    }
    with pytest.raises(asyncio.CancelledError):
        await agent.run("test", environment, AgentContext())


async def test_bridge_preserves_runner_exception_and_timeout_classification(bridge):
    from harbor.models.agent.context import AgentContext

    agent, environment, document = bridge
    document.clear()
    document["runner_error"] = {
        "stage": "run",
        "code": "host_timeout",
        "message": "invoke deadline exceeded",
        "retryable": False,
    }
    environment.exec.return_value.return_code = 1
    context = AgentContext()
    with pytest.raises(TimeoutError, match="invoke deadline exceeded"):
        await agent.run("test", environment, context)
    agent.populate_context_post_run(context)
    assert context.metadata["fabric"]["status"] == "failed"
    assert context.metadata["fabric"]["error"]["code"] == "host_timeout"
    assert list(agent.logs_dir.glob("fabric-result-*.json"))
    agent.populate_context_post_run(context)
    assert context.metadata["fabric"]["error"]["code"] == "host_timeout"


async def test_bridge_does_not_mask_process_failure_when_result_is_missing(bridge):
    from harbor.models.agent.context import AgentContext

    agent, environment, _ = bridge
    environment.exec.return_value.return_code = 2
    environment.exec.return_value.stderr = "runner import failed"
    environment.download_file.side_effect = FileNotFoundError("no result")
    with pytest.raises(RuntimeError, match="runner import failed"):
        await agent.run("test", environment, AgentContext())


async def test_bridge_does_not_accept_success_result_from_failed_process(bridge):
    from harbor.models.agent.context import AgentContext

    agent, environment, _ = bridge
    environment.exec.return_value.return_code = 2
    environment.exec.return_value.stderr = "post-run failure"
    with pytest.raises(RuntimeError, match="post-run failure"):
        await agent.run("test", environment, AgentContext())
    assert agent._result_path.is_file()


async def test_bridge_does_not_mask_process_failure_with_invalid_result(bridge):
    from harbor.models.agent.context import AgentContext

    agent, environment, document = bridge
    document.clear()
    environment.exec.return_value.return_code = 2
    environment.exec.return_value.stderr = "runner failed before completing result"
    with pytest.raises(RuntimeError, match="runner failed before completing result"):
        await agent.run("test", environment, AgentContext())


@pytest.fixture(name="runner_cli")
def runner_cli_fixture(tmp_path, monkeypatch):
    from nemo_fabric.integrations.harbor.fabric_agent import build_harbor_config
    from nemo_fabric.integrations.harbor.models import FabricRunPayload

    spec = tmp_path / "spec.json"
    result = tmp_path / "result.json"
    payload = FabricRunPayload(
        config=build_harbor_config(adapter_id="test.adapter", workspace="/app"),
        config_base_dir="/app",
        request={"input": "test"},
    )
    spec.write_text(payload.model_dump_json(), encoding="utf-8")
    monkeypatch.setattr(
        sys, "argv", ["runner", "--spec", str(spec), "--result", str(result)]
    )
    return result


@pytest.mark.parametrize("status", ["succeeded", "failed", "cancelled"])
def test_runner_writes_normalized_evidence_before_exit(
    runner_cli, monkeypatch, run_result, status
):
    from nemo_fabric import RunResult
    from nemo_fabric.integrations.harbor import runner

    document = run_result.to_mapping()
    document["status"] = status
    mock_run = AsyncMock(return_value=RunResult.from_mapping(document))
    monkeypatch.setattr(runner, "run", mock_run)
    if status == "succeeded":
        runner.main()
    else:
        with pytest.raises(SystemExit) as caught:
            runner.main()
        assert caught.value.code == 1
    assert json.loads(runner_cli.read_text())["status"] == status


@pytest.mark.parametrize("code", ["host_timeout", "adapter_error"])
def test_runner_serializes_structured_exception(runner_cli, monkeypatch, code):
    from nemo_fabric import FabricRuntimeError
    from nemo_fabric.integrations.harbor import runner

    monkeypatch.setattr(
        runner,
        "run",
        AsyncMock(
            side_effect=FabricRuntimeError(
                "test failure",
                stage="invoke",
                code=code,
            )
        ),
    )
    with pytest.raises(FabricRuntimeError):
        runner.main()
    error = json.loads(runner_cli.read_text())["runner_error"]
    assert error["code"] == code
    assert error["stage"] == "invoke"


def test_runner_recognizes_native_host_timeout_without_scraping_stderr(
    runner_cli, monkeypatch
):
    from nemo_fabric import FabricRuntimeError
    from nemo_fabric.integrations.harbor import runner

    error = FabricRuntimeError(
        "adapter lifecycle invoke failed for runtime `test` (host_timeout): deadline exceeded",
        stage="run",
    )
    monkeypatch.setattr(runner, "run", AsyncMock(side_effect=error))
    with pytest.raises(FabricRuntimeError):
        runner.main()
    assert json.loads(runner_cli.read_text())["runner_error"]["code"] == "host_timeout"


def test_runner_does_not_swallow_cancellation(runner_cli, monkeypatch):
    from nemo_fabric.integrations.harbor import runner

    monkeypatch.setattr(runner, "run", AsyncMock(side_effect=asyncio.CancelledError()))
    with pytest.raises(asyncio.CancelledError):
        runner.main()
    assert not runner_cli.exists()


def test_runner_does_not_classify_arbitrary_message_as_native_timeout(
    runner_cli, monkeypatch
):
    from nemo_fabric.integrations.harbor import runner

    monkeypatch.setattr(
        runner,
        "run",
        AsyncMock(
            side_effect=RuntimeError(
                "tool output mentions (host_timeout): not a native deadline",
            )
        ),
    )
    with pytest.raises(RuntimeError):
        runner.main()
    assert json.loads(runner_cli.read_text())["runner_error"]["code"] == "RuntimeError"


async def test_bridge_rejects_error_even_with_succeeded_status(bridge):
    from harbor.models.agent.context import AgentContext

    agent, environment, document = bridge
    document["error"] = {
        "stage": "stop",
        "code": "runtime_stop_failed",
        "message": "stop failed",
        "retryable": False,
    }
    with pytest.raises(RuntimeError, match="stop failed"):
        await agent.run("test", environment, AgentContext())
