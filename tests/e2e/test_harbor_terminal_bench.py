# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real Pi and TB2.1, with scripted model responses rather than live inference.

Opt-in because this needs Harbor 0.24, the pinned TB2.1 task, and a qualified image.
The original instruction and verifier are unchanged. The model server supplies
the reference solution, so reward 1 is integration evidence, not a benchmark score.
"""

import asyncio
import hashlib
import json
import os
import platform
import shutil
import subprocess
import tomllib
from pathlib import Path, PurePosixPath

import pytest
import pytest_asyncio
from aiohttp import web

pytestmark = pytest.mark.usefixtures("requires_harbor")

_TASK_HASHES = {
    "instruction.md": "4f7ac05e70cf9220ea0f1e5a052c5f908cd0fa884e847d80b0bd51bae2e96f9c",
    "tests/test.sh": "4770437ea96c3cc84684b4f99d55fb148fcac09f9ea1e8ef49de487716e6c334",
    "tests/test_outputs.py": "345c3bd09ab6f6fe8c8361a58c0a47bf0a13b3fcb38a5ac7824e44ff855e8f72",
}


@pytest.fixture(name="fabric_tb21_task")
def fabric_tb21_task_fixture(tmp_path):
    from harbor.trial.trial import Trial

    if not hasattr(Trial, "create"):
        pytest.skip("requires a qualified Harbor 0.24 host")
    source = os.environ.get("HARBOR_FABRIC_TB21_TASK")
    image = os.environ.get("HARBOR_FABRIC_TB21_IMAGE")
    if not source or not image:
        pytest.skip("set HARBOR_FABRIC_TB21_TASK and HARBOR_FABRIC_TB21_IMAGE")
    pytest.importorskip("nemo_fabric.integrations.harbor")
    source_path = Path(source)
    for relative, expected in _TASK_HASHES.items():
        assert (
            hashlib.sha256((source_path / relative).read_bytes()).hexdigest()
            == expected
        )
    subprocess.run(
        ["docker", "image", "inspect", image], check=True, capture_output=True
    )
    task = tmp_path / "regex-log"
    shutil.copytree(source_path, task)
    config_path = task / "task.toml"
    config_text = config_path.read_text()
    original_image = tomllib.loads(config_text)["environment"]["docker_image"]
    image_line = f"docker_image = {json.dumps(original_image)}"
    assert config_text.count(image_line) == 1
    config_path.write_text(
        config_text.replace(image_line, f"docker_image = {json.dumps(image)}")
    )
    return task


@pytest_asyncio.fixture(name="fabric_model_server", loop_scope="function")
async def fabric_model_server_fixture(fabric_tb21_task, unused_tcp_port):
    """Serve Pi-compatible tool calls; no real provider or credentials are used."""
    requests = []
    release = asyncio.Event()
    outcome = {"mode": "success"}
    solution = (fabric_tb21_task / "solution/solve.sh").read_text()

    async def completion(request):
        payload = await request.json()
        requests.append(payload)
        if outcome["mode"] == "timeout":
            await release.wait()
        if outcome["mode"] == "failed":
            return web.json_response(
                {
                    "error": {
                        "message": "fixture rejected pilot-sentinel-secret",
                        "type": "authentication_error",
                    }
                },
                status=401,
            )
        first_turn = len(requests) == 1
        delta = {"role": "assistant", "content": "Reference solution executed."}
        reason = "stop"
        if first_turn:
            assert "bash" in {tool["function"]["name"] for tool in payload["tools"]}
            delta = {
                "role": "assistant",
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call-reference",
                        "type": "function",
                        "function": {
                            "name": "bash",
                            "arguments": json.dumps(
                                {
                                    "command": "printf '(?!)\\n' > /app/regex.txt"
                                    if outcome["mode"] == "incorrect"
                                    else solution,
                                }
                            ),
                        },
                    }
                ],
            }
            reason = "tool_calls"
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        for content, finish in [(delta, None), ({}, reason)]:
            chunk = {
                "id": "chatcmpl-fixture",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": payload["model"],
                "choices": [{"index": 0, "delta": content, "finish_reason": finish}],
            }
            await response.write(f"data: {json.dumps(chunk)}\n\n".encode())
        usage_chunk = {
            "id": "chatcmpl-fixture",
            "object": "chat.completion.chunk",
            "model": payload["model"],
            "created": 0,
            "choices": [],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 25,
                "total_tokens": 125,
                "prompt_tokens_details": {"cached_tokens": 40},
            },
        }
        await response.write(f"data: {json.dumps(usage_chunk)}\n\n".encode())
        await response.write(b"data: [DONE]\n\n")
        await response.write_eof()
        return response

    app = web.Application()
    app.router.add_post("/v1/chat/completions", completion)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "0.0.0.0", unused_tcp_port).start()
    if platform.system() in {"Darwin", "Windows"}:
        host = "host.docker.internal"
    else:
        host = subprocess.check_output(["hostname", "-I"], text=True).split()[0]
    try:
        yield f"http://{host}:{unused_tcp_port}/v1", requests, outcome
    finally:
        release.set()
        await runner.cleanup()


@pytest.mark.parametrize("mode", ["success", "incorrect", "failed", "timeout"])
async def test_fabric_pi_terminal_bench_trial(
    fabric_tb21_task, fabric_model_server, tmp_path, mode
):
    from harbor.environments.docker.docker import (
        _sanitize_docker_compose_project_name,
    )
    from harbor.models.environment_type import EnvironmentType
    from harbor.models.trial.config import (
        AgentConfig,
        EnvironmentConfig,
        TaskConfig,
        TrialConfig,
    )
    from harbor.trial.trial import Trial
    from nemo_fabric.integrations.harbor import FabricAgent

    base_url, requests, outcome = fabric_model_server
    outcome["mode"] = mode
    config = TrialConfig(
        task=TaskConfig(path=fabric_tb21_task),
        agent=AgentConfig(
            import_path="nemo_fabric.integrations.harbor:FabricAgent",
            model_name="nvidia/nemotron-3-ultra-550b-a55b",
            override_timeout_sec=90,
            kwargs={
                "fabric_adapter_id": "nvidia.fabric.pi",
                "fabric_workspace": "/app",
                "fabric_python": "/opt/fabric-venv/bin/python",
                "fabric_discovery_paths": [
                    "/opt/nemo-fabric/adapters/typescript/pi/pi.fabric-adapter.json"
                ],
                "fabric_model_base_url": base_url,
                "fabric_model_api_key_env": "FABRIC_TEST_API_KEY",
                "fabric_runtime_timeout_seconds": 5 if mode == "timeout" else 60,
                "fabric_environment_env": {
                    "FABRIC_TEST_API_KEY": "pilot-sentinel-secret"
                },
            },
        ),
        environment=EnvironmentConfig(type=EnvironmentType.DOCKER, delete=True),
        trials_dir=tmp_path / "trials",
    )
    trial = await Trial.create(config=config)
    assert isinstance(trial.agent, FabricAgent)
    result = await trial.run()
    assert requests, "Pi must actually call the fixture server"
    evidence = list(trial.paths.agent_dir.glob("fabric-result-*.json"))
    assert len(evidence) == 1
    document = json.loads(evidence[0].read_text())
    if mode in {"success", "incorrect"}:
        assert result.exception_info is None
        assert result.verifier_result is not None
        assert result.verifier_result.rewards is not None
        assert result.verifier_result.rewards["reward"] == (
            1.0 if mode == "success" else 0.0
        )
        assert document["status"] == "succeeded"
        assert result.agent_result.n_input_tokens == 200
        assert result.agent_result.n_cache_tokens == 80
        assert result.agent_result.n_output_tokens == 50
        assert result.agent_result.cost_usd is None
        provenance = result.agent_result.metadata["fabric"]["provenance"]
        assert provenance["harness"] == "pi"
        assert provenance["harness_version"] == "0.86.0"
        assert provenance["adapter_version"] is not None
        # This Pi adapter uses an in-memory session; assert the evidence it
        # actually publishes rather than claiming native trajectory support.
        artifact = document["artifacts"]["artifacts"][0]
        relative = PurePosixPath(artifact["path"]).relative_to("/logs/agent")
        assert (trial.paths.agent_dir / str(relative)).is_file()
    else:
        assert result.exception_info is not None
        expected = "AgentTimeoutError" if mode == "timeout" else "RuntimeError"
        assert result.exception_info.exception_type == expected
        assert result.agent_result is not None
        assert result.agent_result.metadata is not None
        assert result.agent_result.metadata["fabric"]["status"] == "failed"
        assert "runner_error" in document or document["status"] == "failed"
    for path in trial.paths.trial_dir.rglob("*"):
        if path.is_file():
            assert b"pilot-sentinel-secret" not in path.read_bytes(), path
    containers = subprocess.check_output(
        [
            "docker",
            "ps",
            "-aq",
            "--filter",
            "label=com.docker.compose.project="
            + _sanitize_docker_compose_project_name(trial.agent_environment.session_id),
        ],
        text=True,
    )
    assert not containers.strip(), "the trial must remove its own containers"
