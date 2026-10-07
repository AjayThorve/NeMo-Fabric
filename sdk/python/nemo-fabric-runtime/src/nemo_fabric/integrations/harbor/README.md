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

Use `fabric_model_api_key_env` to name the model credential variable. The name
must match Harbor's redaction policy, for example `MODEL_API_KEY`. Names such as
`MODEL_DSN` are rejected during options validation, including Harbor preflight.
Supply literal credentials through Harbor's managed `extra_env` (`--ae`).
In `fabric_environment_env`, sensitive entries must use host-variable references
such as `{"MODEL_API_KEY": "${MODEL_API_KEY}"}` without inline defaults.
The bridge resolves these references with Harbor's utilities and merges them into
Harbor's managed environment and redaction
inputs, rejects conflicting values, and transports only variable names in the
retained run specification. The task runner resolves those names from its
execution environment; a missing variable fails with a name-only diagnostic.
This keeps raw credentials out of Harbor's top-level job configuration, which
is outside its trial-directory scrubber.

Non-sensitive settings such as `RUN_MODE` can use arbitrary valid environment
variable names. Harbor does not scrub their values. Use names recognized by
Harbor's `is_sensitive_env_key` policy for all credentials, including credentials
other than the configured model key; the bridge cannot infer secrets from values.

Deploy matching bridge and runner versions: a runner predating name-only transport cannot read the new payload. Do not put credentials in harness settings or native configuration files.

## Execution Outcomes

The task runner writes normalized evidence before exiting nonzero for failed or cancelled runs. The bridge downloads that evidence before reporting a Harbor execution failure or cancellation, including when the runner exits nonzero. A completed answer can still receive verifier reward zero; verifier scoring is not an execution failure. A missing or malformed result does not hide an available process failure.

Lifecycle failures before a normalized result is available produce a separate
`runner_error` record. NeMo Fabric normalizes deadline errors to code `timeout`;
the bridge raises `TimeoutError`, which Harbor translates into
`AgentTimeoutError`. Cancellation takes precedence. The bridge does not
inspect harness IDs, native error-code lists, or diagnostic text.
