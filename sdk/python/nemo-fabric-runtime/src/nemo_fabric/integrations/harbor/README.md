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

## Execution Outcomes

The task runner writes normalized evidence before exiting nonzero for failed or cancelled runs. The bridge downloads that evidence before reporting a Harbor execution failure or cancellation, including when the runner exits nonzero. A completed answer can still receive verifier reward zero; verifier scoring is not an execution failure. A missing or malformed result does not hide an available process failure.

Lifecycle failures before a normalized result is available produce a separate
`runner_error` record. Fabric normalizes deadline errors to code `timeout`;
the bridge raises `TimeoutError`, which Harbor translates into
`AgentTimeoutError`. Cancellation takes precedence. The bridge does not
inspect harness IDs, native error-code lists, or diagnostic text.
