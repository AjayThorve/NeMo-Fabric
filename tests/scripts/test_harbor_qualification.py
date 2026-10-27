# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[2] / ".agents/skills/qualify-harbor/scripts"),
)
import qualify  # noqa: E402
from prepare import adapter_dockerfile  # noqa: E402

pytestmark = pytest.mark.skipif(
    os.name != "posix",
    reason="Qualification uses POSIX process groups and Linux task images",
)


@pytest.fixture(name="qualification_plan")
def qualification_plan_fixture() -> dict:
    return qualify.make_plan(
        argparse.Namespace(
            task=qualify.ROOT / "examples/harbor/calculator/task",
            mode="source",
            version=qualify.load(qualify.ROOT / qualify.CATALOG)["catalog_version"],
            harbor_version="0.24.0",
            bridge="native",
            model="nvidia/local-model",
            base_url="http://127.0.0.1:8000/v1",
            api_key_env="MODEL_API_KEY",
            endpoint_kind="openai-and-anthropic",
            trusted_http_host=[],
            estimated_cost_usd=0,
            cost_basis="local",
            timeout=60,
        )
    )


@pytest.fixture(name="clean_qualification_checkout")
def clean_qualification_checkout_fixture():
    real_run = subprocess.run

    def run_with_clean_checkout(command, *args, **kwargs):
        if command[:2] == ["git", "diff"] or command[1:] == [
            str(qualify.ROOT / "scripts/ci/generate_adapter_catalog.py"),
            "--check",
        ]:
            return subprocess.CompletedProcess(command, 0)
        return real_run(command, *args, **kwargs)

    with patch.object(qualify.subprocess, "run", side_effect=run_with_clean_checkout):
        yield


@pytest.mark.usefixtures("clean_qualification_checkout")
def test_plan_uses_complete_catalog_and_distinguishes_prerequisites(qualification_plan):
    qualify.validate_plan(qualification_plan)
    catalog = qualify.load(qualify.ROOT / qualify.CATALOG)
    rows = {row["adapter_id"]: row for row in qualification_plan["adapters"]}
    assert set(rows) == set(catalog["adapters"])
    assert rows["nvidia.fabric.claude"]["action"] == "run"
    assert rows["nvidia.fabric.kiro"]["action"] == "blocked"
    assert rows["nvidia.fabric.remote-agent"]["reason"]
    assert rows["nvidia.fabric.pi"]["capabilities"]["mcp"] == "untested"


@pytest.mark.usefixtures("clean_qualification_checkout")
def test_registry_prerelease_keeps_python_and_npm_versions_distinct(qualification_plan):
    plan = copy.deepcopy(qualification_plan)
    base = plan["version"]
    plan["mode"] = "registry"
    plan["version"] = base + "rc1"
    plan["catalog_snapshot"]["catalog_version"] = plan["version"]
    for row in plan["adapters"]:
        row["catalog"]["package"]["version"] = base + (
            "rc1" if row["catalog"]["ecosystem"] == "python" else "-rc.1"
        )
    qualify.validate_plan(plan)
    row = next(
        row for row in plan["adapters"] if row["adapter_id"] == "nvidia.fabric.pi"
    )
    assert f"nemo-fabric-adapters-pi@{base}-rc.1" in adapter_dockerfile(
        plan, row, "base:exact"
    )
    plan["mode"] = "source"
    with pytest.raises(ValueError, match="unmodified catalog"):
        qualify.validate_plan(plan)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "duplicate",
        "descriptor",
        "source",
        "version",
        "credential_url",
        "untrusted_http",
        "inline_secret",
        "checkpoint",
        "catalog_extra",
    ],
)
@pytest.mark.usefixtures("clean_qualification_checkout")
def test_plan_rejects_incomplete_or_unattributable_coverage(
    qualification_plan, mutation
):
    plan = copy.deepcopy(qualification_plan)
    row = plan["adapters"][0]
    if mutation == "missing":
        plan["adapters"].pop()
    elif mutation == "duplicate":
        plan["adapters"].append(row)
    elif mutation == "descriptor":
        row["catalog"]["sha256"] = "0" * 64
    elif mutation == "source":
        plan["source_sha"] = "0" * 40
    elif mutation == "version":
        plan["version"] = "999.0.0"
    elif mutation == "credential_url":
        row["base_url"] = "https://user:secret@example.com/v1"
    elif mutation == "untrusted_http":
        row["base_url"] = "http://10.0.0.1:8000/v1"
    elif mutation == "inline_secret":
        row["harness_settings"] = {"config_overrides": {"api_key": "literal-secret"}}
    elif mutation == "checkpoint":
        row["model"] = "/mnt/checkpoint"
    else:
        plan["mode"] = "registry"
        plan["catalog_snapshot"]["adapters"]["extra.adapter"] = copy.deepcopy(
            row["catalog"]
        )
    with pytest.raises(ValueError):
        qualify.validate_plan(plan)


@pytest.mark.parametrize("outcome", ["blocked", "failed", "unsupported", "excluded"])
def test_nonpasses_never_turn_full_catalog_green(tmp_path, qualification_plan, outcome):
    report = {
        "scope": "full_catalog",
        "plan": qualification_plan,
        "adapters": [
            {"adapter_id": "one", "outcome": "passed"},
            {
                "adapter_id": "two",
                "outcome": outcome,
                "reason": "coverage gap",
            },
        ],
    }
    qualify.write_report(tmp_path, report)
    assert qualify.load(tmp_path / "report.json")["full_catalog_passed"] is False
    assert "denominator: 2" in (tmp_path / "summary.md").read_text()


def test_subset_is_never_a_full_catalog_pass(tmp_path, qualification_plan):
    report = {
        "scope": "developer_subset",
        "plan": qualification_plan,
        "adapters": [{"adapter_id": "one", "outcome": "passed"}],
    }
    qualify.write_report(tmp_path, report)
    assert report["all_harnesses_passed"] is True
    assert report["full_catalog_passed"] is False


@pytest.mark.parametrize("state", ["completed", "running", "cancelled"])
def test_full_catalog_pass_requires_completed_run(tmp_path, qualification_plan, state):
    report = {
        "scope": "full_catalog",
        "state": state,
        "plan": qualification_plan,
        "adapters": [{"adapter_id": "one", "outcome": "passed"}],
    }
    qualify.write_report(tmp_path, report)
    assert report["full_catalog_passed"] is (state == "completed")
    assert "release_qualified" not in report


def test_credential_is_transported_as_reference_and_logs_are_redacted(
    tmp_path, qualification_plan
):
    os.environ["MODEL_API_KEY"] = "qualification-secret-value"
    row = qualification_plan["adapters"][0]
    command = qualify.harbor_command(
        qualification_plan, row, "python", tmp_path, tmp_path, "trial"
    )
    assert "MODEL_API_KEY=${MODEL_API_KEY}" in command
    assert "qualification-secret-value" not in " ".join(command)
    log = tmp_path / "run.log"
    code = qualify.execute(
        [sys.executable, "-c", "import os; print(os.environ['MODEL_API_KEY'])"],
        log,
        10,
        key_env="MODEL_API_KEY",
    )
    assert code == 0
    assert "qualification-secret-value" not in log.read_text()
    assert "[REDACTED]" in log.read_text()
    assert "MODEL_API_KEY" not in qualify.process_env()


def test_timeout_terminates_process_and_retains_partial_log(tmp_path):
    log = tmp_path / "timeout.log"
    code = qualify.execute(
        [sys.executable, "-u", "-c", "import time; print('started'); time.sleep(60)"],
        log,
        1,
    )
    assert code == 124
    assert "started" in log.read_text()


def test_interrupt_terminates_only_its_process_group(tmp_path):
    with (
        patch("qualify.subprocess.Popen", autospec=True) as mock_popen,
        patch("qualify.os.killpg") as kill,
    ):
        process = mock_popen.return_value
        process.pid = 12345
        process.stdout = iter([])
        process.wait.side_effect = [KeyboardInterrupt, 0]
        assert qualify.execute(["test-process"], tmp_path / "cancel.log", 10) == 130
    kill.assert_called_once_with(process.pid, qualify.signal.SIGINT)


def test_cleanup_does_not_remove_another_jobs_resources(tmp_path):
    with (
        patch("qualify.subprocess.run") as run,
        patch("qualify.subprocess.check_output") as output,
        patch("qualify.execute") as execute,
    ):
        run.side_effect = [
            subprocess.CompletedProcess([], 0, stdout="probe"),
            subprocess.CompletedProcess([], 0, stdout="owned unrelated"),
            subprocess.CompletedProcess([], 0, stdout="owned unrelated"),
        ]
        output.side_effect = [
            json.dumps({"com.docker.compose.project": "fq-unique-0__trial"}),
            json.dumps({"com.docker.compose.project": "another-user"}),
            json.dumps({"com.docker.compose.project": "fq-unique-0__trial"}),
            json.dumps({"com.docker.compose.project": "another-user"}),
        ]
        execute.return_value = 0
        assert qualify.cleanup("fq-unique", tmp_path / "cleanup.log") is True
        assert [call.args[0] for call in execute.call_args_list] == [
            ["docker", "rm", "-f", "probe"],
            ["docker", "rm", "-f", "owned"],
            ["docker", "network", "rm", "owned"],
        ]


@pytest.mark.parametrize(
    "message",
    [
        "Provider List: nvidia; ModuleNotFoundError: No module named 'mcp'",
        "APIConnectionError: Nvidia_nimException - No module named 'mcp'",
    ],
)
def test_missing_dependency_is_not_classified_as_a_provider_failure(message):
    assert qualify.classify(message) == "install"


@pytest.fixture(name="qualification_job")
def qualification_job_fixture(tmp_path, qualification_plan):
    from nemo_fabric.integrations.harbor.fabric_agent import build_harbor_config

    row = next(
        row
        for row in qualification_plan["adapters"]
        if row["adapter_id"] == "nvidia.fabric.codex"
    )
    trial = tmp_path / "trial"
    (trial / "agent").mkdir(parents=True)
    (trial / "verifier").mkdir()
    qualify.save(
        tmp_path / "result.json",
        {"stats": {"n_completed_trials": 1, "n_errored_trials": 0}},
    )
    result = {
        "status": "succeeded",
        "adapter_id": row["adapter_id"],
        "harness": "codex",
        "agent_name": "qualification",
        "adapter_kind": "python",
        "runtime_id": "runtime",
        "invocation_id": "invocation",
        "request_id": "request",
        "output": {"response": "fixed"},
        "events": [],
        "metadata": {},
        "artifacts": {"artifacts": []},
        "telemetry": [],
        "usage": {
            "input_tokens": 12,
            "input_tokens_include_cache": True,
            "output_tokens": 3,
        },
        "error": None,
    }
    qualify.save(trial / "agent/fabric-result-one.json", result)
    spec = {
        "config": build_harbor_config(
            adapter_id=row["adapter_id"],
            workspace="/app",
            model_name=row["model"],
            model_base_url=row["base_url"],
            model_api_key_env=row["api_key_env"],
        ).to_mapping()
    }
    spec["adapter_descriptor_sha256"] = qualify.admission_fingerprint(
        spec, row["catalog"]["descriptor"]
    )
    qualify.save(trial / "agent/fabric-run-one.json", spec)
    (trial / "verifier/reward.txt").write_text("1\n")
    identity = {"editable": [], "packages": {"nemo-fabric-runtime": "0.6.0"}}
    qualify.save(trial / "verifier/qualification-identity.json", identity)
    selection = {
        "adapter_id": row["adapter_id"],
        "python_prefix": "/usr/local",
        "provenance": [
            {
                "source": "installed_package",
                "path": "/usr/local/share/nemo-fabric/adapters/codex/codex.fabric-adapter.json",
            }
        ],
    }
    qualify.save(trial / "verifier/qualification-selection.json", selection)
    qualify.save(
        trial / "result.json",
        {
            "agent_result": {
                "n_input_tokens": 12,
                "n_output_tokens": 3,
                "metadata": {
                    "fabric": {
                        "adapter_id": row["adapter_id"],
                        "invocation_id": "invocation",
                    }
                },
            }
        },
    )
    return tmp_path, row, identity, result, selection


@pytest.mark.parametrize(
    "mutation,category",
    [
        ("none", None),
        ("reward", "harbor_verifier"),
        ("identity", "adapter_admission"),
        ("usage", "usage"),
        ("counter", "usage"),
        ("source_origin", "adapter_admission"),
        ("fingerprint", "adapter_admission"),
        ("missing_result", "transport_runner"),
        ("runner_error", "adapter_admission"),
    ],
)
@pytest.mark.usefixtures("requires_harbor")
def test_result_checks_remain_independent(qualification_job, mutation, category):
    job, row, identity, result, selection = qualification_job
    if mutation == "reward":
        (job / "trial/verifier/reward.txt").write_text("0")
    elif mutation == "identity":
        identity = {"editable": ["runtime"]}
    elif mutation == "usage":
        result["usage"] = None
        qualify.save(job / "trial/agent/fabric-result-one.json", result)
    elif mutation == "counter":
        trial = qualify.load(job / "trial/result.json")
        trial["agent_result"]["n_input_tokens"] = 999
        qualify.save(job / "trial/result.json", trial)
    elif mutation == "source_origin":
        observed = copy.deepcopy(selection)
        observed["provenance"][0] = {
            "source": "bundled",
            "path": "/src/adapters/python/codex/codex.fabric-adapter.json",
        }
        qualify.save(job / "trial/verifier/qualification-selection.json", observed)
    elif mutation == "fingerprint":
        spec = qualify.load(job / "trial/agent/fabric-run-one.json")
        spec["adapter_descriptor_sha256"] = "wrong"
        qualify.save(job / "trial/agent/fabric-run-one.json", spec)
    elif mutation == "missing_result":
        (job / "trial/agent/fabric-result-one.json").unlink()
    elif mutation == "runner_error":
        qualify.save(
            job / "trial/agent/fabric-result-one.json",
            {"runner_error": {"message": "descriptor fingerprint mismatch"}},
        )
    summary = qualify.inspect_results(job, row, identity, selection)
    assert {failure["category"] for failure in summary["failures"]} == (
        {category} if category else set()
    )


@pytest.mark.usefixtures("requires_harbor")
def test_admission_compares_normalized_descriptor_not_raw_catalog_hash(
    qualification_job,
):
    job, row, identity, _, selection = qualification_job
    row["catalog"]["sha256"] = "0" * 64
    summary = qualify.inspect_results(job, row, identity, selection)
    assert summary["checks"]["descriptor_fingerprint"] is True
    assert summary["failures"] == []


def test_source_task_uses_packed_installs_and_exact_runtime(qualification_plan):
    row = next(
        row
        for row in qualification_plan["adapters"]
        if row["adapter_id"] == "nvidia.fabric.pi"
    )
    dockerfile = adapter_dockerfile(qualification_plan, row, "qualification-base:exact")
    assert "npm pack" in dockerfile
    assert "npm install --prefix /opt/harness /packs/*.tgz" in dockerfile
    assert "@earendil-works/pi-ai@1.0.3" in dockerfile
    assert "-e /src" not in dockerfile


@pytest.mark.usefixtures("clean_qualification_checkout")
def test_failed_host_environment_preserves_complete_report(
    tmp_path, qualification_plan
):
    output = tmp_path / "report"
    assert (
        qualify.run(
            qualification_plan, {"plan": qualification_plan, "adapters": {}}, output, []
        )
        == 1
    )
    report = qualify.load(output / "report.json")
    assert len(report["adapters"]) == len(qualification_plan["adapters"])
    assert all(row["outcome"] == "blocked" for row in report["adapters"])
    assert report["full_catalog_passed"] is False


@pytest.mark.usefixtures("clean_qualification_checkout")
def test_single_task_is_pinned_and_dataset_roots_are_rejected(
    tmp_path, qualification_plan
):
    with pytest.raises(ValueError, match="single Harbor task missing"):
        qualify.task_identity(tmp_path)
    task = tmp_path / "task"
    qualify.stage_task(qualification_plan, task, "prepared:exact")
    wrapper = (task / "tests/test.sh").read_text()
    assert wrapper.index("qualification-identity.json") < wrapper.index(
        "bash /tests/upstream-test.sh"
    )
    assert not (task / "solution").exists()
    assert "docker_image" not in (task / "task.toml").read_text()
    assert (task / "environment/Dockerfile").read_text() == "FROM prepared:exact\n"
    assert (task / "tests/upstream-test.sh").read_bytes() == (
        Path(qualification_plan["task"]["path"]) / "tests/test.sh"
    ).read_bytes()
    changed = copy.deepcopy(qualification_plan)
    changed["task"]["sha256"] = "wrong"
    with pytest.raises(ValueError, match="task changed"):
        qualify.validate_plan(changed)


def test_task_package_installation_is_allowed_but_dependency_drift_is_rejected():
    expected = {
        "editable": [],
        "packages": {"nemo-fabric-runtime": "0.6.0", "protobuf": "6.33.0"},
    }
    observed = copy.deepcopy(expected)
    observed["packages"]["grpcio-tools"] = "1.73.0"
    assert qualify.unchanged_install(expected, observed)
    observed["packages"]["protobuf"] = "5.0.0"
    assert not qualify.unchanged_install(expected, observed)


def test_dirty_checkout_is_rejected(tmp_path, qualification_plan):
    repository = tmp_path / "repo"
    repository.mkdir()
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    (repository / "crates").mkdir()
    probe = repository / "crates/probe"
    probe.write_text("committed\n")
    subprocess.run(["git", "add", "crates/probe"], cwd=repository, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Qualification test",
            "-c",
            "user.email=test@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "-c",
            "core.hooksPath=/dev/null",
            "commit",
            "-q",
            "-m",
            "fixture",
        ],
        cwd=repository,
        check=True,
    )
    plan = copy.deepcopy(qualification_plan)
    plan["source_sha"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repository, text=True
    ).strip()
    probe.write_text("dirty\n")
    catalog = qualify.ROOT / qualify.CATALOG
    with (
        patch.object(qualify, "ROOT", repository),
        patch.object(qualify, "CATALOG", catalog),
    ):
        with pytest.raises(subprocess.CalledProcessError) as error:
            qualify.validate_plan(plan)
    assert error.value.cmd[:2] == ["git", "diff"]


def test_report_redaction_preserves_types_and_custom_credentials(
    tmp_path, qualification_plan
):
    os.environ["MODEL_API_KEY"] = "true"
    os.environ["AWS_ACCESS_KEY_ID"] = "fake-access-key-id"
    os.environ["AZURE_STORAGE_CONNECTION_STRING"] = "fake-connection-string"
    os.environ["BROKER_CREDENTIAL_HANDLE"] = "opaque-custom-secret"
    os.environ["DOCKER_AUTH_CONFIG"] = (
        '{"auths":{"registry":{"auth":"encoded-secret"}}}'
    )
    report = {
        "scope": "full_catalog",
        "state": "completed",
        "plan": qualification_plan,
        "adapters": [
            {
                "adapter_id": "one",
                "outcome": "passed",
                "selected": {"api_key_env": "BROKER_CREDENTIAL_HANDLE"},
                "reason": "opaque-custom-secret fake-access-key-id fake-connection-string "
                + os.environ["DOCKER_AUTH_CONFIG"],
                "nested": [True, False, None, 1, "true"],
            }
        ],
    }
    qualify.write_report(tmp_path, report)
    text = (tmp_path / "report.json").read_text()
    redacted = json.loads(text)
    assert redacted["full_catalog_passed"] is True
    assert redacted["adapters"][0]["nested"] == [True, False, None, 1, "[REDACTED]"]
    assert "fake-access-key-id" not in text
    assert "fake-connection-string" not in text
    assert "opaque-custom-secret" not in text
    assert os.environ["DOCKER_AUTH_CONFIG"] not in text
    assert "opaque-custom-secret" not in (tmp_path / "summary.md").read_text()


def test_custom_credential_is_redacted_before_log_write(tmp_path):
    os.environ["BROKER_CREDENTIAL_HANDLE"] = "opaque-custom-secret"
    log = tmp_path / "process.log"
    assert (
        qualify.execute(
            [
                sys.executable,
                "-c",
                "import os; print(os.environ['BROKER_CREDENTIAL_HANDLE'])",
            ],
            log,
            10,
            key_env="BROKER_CREDENTIAL_HANDLE",
        )
        == 0
    )
    assert "opaque-custom-secret" not in log.read_text()
    assert "[REDACTED]" in log.read_text()


@pytest.mark.parametrize(
    "text,category",
    [
        ("status=401", "auth"),
        ("HTTP 403", "auth"),
        ("HTTP 404", "provider_model"),
        ("HTTP 429", "infrastructure"),
        ("HTTP 502", "infrastructure"),
        ("HTTP 503", "infrastructure"),
        ("run-1401-1403-1404-1429-1502-1503", "transport_runner"),
        ("id_401_403_404_429_502_503", "transport_runner"),
    ],
)
def test_http_codes_do_not_match_inside_identifiers(text, category):
    assert qualify.classify(text) == category


@pytest.mark.parametrize("suffix", [".json", ".jsonl"])
def test_artifact_redaction_preserves_machine_readable_evidence(suffix):
    os.environ["MODEL_API_KEY"] = "true"
    original = {"flag": True, "value": None, "credential": "true"}
    text = json.dumps(original) + "\n"
    if suffix == ".jsonl":
        text += json.dumps({"flag": False, "count": 1}) + "\n"
    result = qualify.sanitize_artifact(text, suffix, ("MODEL_API_KEY",))
    objects = [json.loads(line) for line in result.splitlines()]
    assert objects[0] == {"flag": True, "value": None, "credential": "[REDACTED]"}
    if suffix == ".jsonl":
        assert objects[1] == {"flag": False, "count": 1}
    os.environ["MODEL_API_KEY"] = "different-secret"
    assert qualify.sanitize_artifact(text, suffix, ("MODEL_API_KEY",)) == text


@pytest.mark.parametrize("suffix", [".jsonl", ".ndjson"])
def test_jsonl_redaction_keeps_unicode_separators_inside_records(suffix):
    secret = "prefix\u2028suffix"
    os.environ["MODEL_API_KEY"] = secret
    records = [{"credential": secret, "flag": True}, {"text": "other\u2028text"}]
    original = "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n"
    cleaned = qualify.sanitize_artifact(original, suffix, ("MODEL_API_KEY",))
    assert secret not in cleaned
    assert cleaned.endswith("\n")
    parsed = [json.loads(line) for line in cleaned.split("\n") if line]
    assert parsed == [{"credential": "[REDACTED]", "flag": True}, records[1]]
