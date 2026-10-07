# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the Fabric SDK inside a Harbor task environment."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from nemo_fabric import Fabric
from nemo_fabric import FabricError
from nemo_fabric import RunResult
from nemo_fabric.integrations.harbor.models import FabricRunPayload
from nemo_fabric.integrations.harbor.models import FabricRunnerError
from nemo_fabric.integrations.harbor.models import FabricRunnerFailure
from nemo_fabric.integrations.harbor.telemetry import publish_telemetry_evidence


async def run(payload: FabricRunPayload) -> RunResult:
    config = payload.config.model_copy(deep=True)
    result = await Fabric().run(
        config,
        base_dir=payload.config_base_dir,
        request=payload.request,
    )
    publish_telemetry_evidence(
        result,
        Path(payload.logs_dir),
        harbor_session_id=payload.request.context.get("harbor_session_id"),
        harbor_context_id=payload.request.context.get("harbor_context_id"),
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()

    args.result.parent.mkdir(parents=True, exist_ok=True)
    try:
        payload = FabricRunPayload.model_validate_json(
            args.spec.read_text(encoding="utf-8")
        )
        result = asyncio.run(run(payload))
    except Exception as error:
        diagnostic = FabricRunnerError(
            stage=error.stage if isinstance(error, FabricError) else None,
            code=error.code if isinstance(error, FabricError) else None,
            message=(
                str(error)
                if isinstance(error, FabricError)
                else f"Fabric runner failed ({type(error).__name__})"
            ),
            retryable=error.retryable if isinstance(error, FabricError) else False,
        )
        args.result.write_text(
            FabricRunnerFailure(runner_error=diagnostic).model_dump_json(indent=2),
            encoding="utf-8",
        )
        raise SystemExit(1) from None
    args.result.write_text(json.dumps(result.to_mapping(), indent=2), encoding="utf-8")
    if result.status != "succeeded" or result.error is not None:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
