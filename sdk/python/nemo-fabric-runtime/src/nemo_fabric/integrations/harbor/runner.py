# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the Fabric SDK inside a Harbor task environment."""

from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import os
from pathlib import Path

from nemo_fabric import Fabric
from nemo_fabric import FabricError
from nemo_fabric import FabricRuntimeError
from nemo_fabric import RunResult
from nemo_fabric.integrations.harbor.models import FabricRunPayload
from nemo_fabric.integrations.harbor.telemetry import publish_telemetry_evidence


async def run(payload: FabricRunPayload) -> RunResult:
    config = payload.config.model_copy(deep=True)
    for name in payload.environment_env_names:
        if name not in os.environ:
            raise ValueError(f"Harbor runner environment variable {name} is not set")
        config.environment.env[name] = os.environ[name]
    result = await Fabric().run(
        config,
        base_dir=payload.config_base_dir,
        request=payload.request,
    )
    mapping = result.to_mapping()
    provenance = execution_provenance(payload)
    provenance.update(mapping["metadata"].get("adapter", {}).get("provenance", {}))
    mapping["metadata"]["harbor_provenance"] = provenance
    result = RunResult.from_mapping(mapping)
    publish_telemetry_evidence(
        result,
        Path(payload.logs_dir),
        harbor_session_id=payload.request.context.get("harbor_session_id"),
        harbor_context_id=payload.request.context.get("harbor_context_id"),
    )
    return result


def execution_provenance(payload: FabricRunPayload) -> dict[str, str | None]:
    """Record task-runtime versions; unknown is not the runtime's version."""

    adapter_id = payload.config.harness.adapter_id
    harness = {
        "nvidia.fabric.pi": "pi",
        "nvidia.fabric.codex": "codex",
    }.get(adapter_id)

    def version(distribution: str) -> str | None:
        try:
            return importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            return None

    adapter_version = (
        version("nemo-fabric-adapters-codex") if harness == "codex" else None
    )
    if harness == "pi":
        adapter_version = _pi_adapter_version(payload)
    return {
        "harness": harness,
        "harness_version": None,
        "adapter_id": adapter_id,
        "adapter_version": adapter_version,
        "harness_sdk_version": version("openai-codex") if harness == "codex" else None,
        "fabric_runtime_version": version("nemo-fabric-runtime"),
    }


def _pi_adapter_version(payload: FabricRunPayload) -> str | None:
    """Read metadata beside the selected descriptor without starting Pi."""

    try:
        plan = Fabric().plan(payload.config, base_dir=payload.config_base_dir)
    except FabricError:
        return None
    resolved = plan.get("adapter_descriptor", {})
    provenance = resolved.get("provenance", [])
    if not provenance:
        return None
    # Core resolves descriptor-local runner paths against the primary root.
    package_path = Path(provenance[0]["root"]) / "package.json"
    try:
        package = json.loads(package_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if (
        not isinstance(package, dict)
        or package.get("name") != "nemo-fabric-adapters-pi"
    ):
        return None
    version = package.get("version")
    return version if isinstance(version, str) and version.strip() else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()

    payload = FabricRunPayload.model_validate_json(
        args.spec.read_text(encoding="utf-8")
    )
    args.result.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = asyncio.run(run(payload))
    except Exception as error:
        # A lifecycle exception has no RunResult. Preserve it without inventing
        # runtime IDs or pretending that a normalized invocation completed.
        code = error.code if isinstance(error, FabricError) else None
        if isinstance(error, TimeoutError):
            code = "timeout"
        elif (
            isinstance(error, FabricRuntimeError)
            and code is None
            and str(error).startswith("adapter lifecycle ")
            and "(host_timeout):" in str(error)
        ):
            # Older native bindings expose this lifecycle code only in the
            # exception message. Never infer timeout from runner stderr.
            code = "host_timeout"
        diagnostic = {
            "stage": (error.stage or "run")
            if isinstance(error, FabricError)
            else "run",
            "code": code or type(error).__name__,
            "message": str(error),
            "retryable": error.retryable if isinstance(error, FabricError) else False,
        }
        args.result.write_text(
            json.dumps(
                {
                    "runner_error": diagnostic,
                    "provenance": execution_provenance(payload),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        raise
    args.result.write_text(json.dumps(result.to_mapping(), indent=2), encoding="utf-8")
    if result.status != "succeeded" or result.error is not None:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
