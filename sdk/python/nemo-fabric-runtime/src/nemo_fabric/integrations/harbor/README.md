<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Harbor Integration

Use `nemo_fabric.integrations.harbor:FabricAgent` to run NVIDIA NeMo Fabric
adapters in Harbor tasks. Harbor options select the model, harness, skills, MCP
servers, tool policy, and telemetry; `FabricAgent` translates them into one
typed `FabricConfig` for the task run.

Refer to the [Harbor example](../../../../../../../examples/harbor/README.md)
for runnable SWE-Bench commands, configuration variations, reward checks, and
Relay artifacts.

## Credential Transport

Use `fabric_model_api_key_env` to name the model credential variable. Supply its value through Harbor's `extra_env` or `fabric_environment_env`. The bridge merges the latter into Harbor's managed environment and redaction inputs, rejects conflicting values, and transports only variable names in the retained run specification. The task runner resolves those names from its execution environment; a missing variable fails with a name-only diagnostic. Non-sensitive environment settings follow the same transport.

Deploy matching bridge and runner versions: a runner predating name-only transport cannot read the new payload. Do not put credentials in harness settings or native configuration files.
