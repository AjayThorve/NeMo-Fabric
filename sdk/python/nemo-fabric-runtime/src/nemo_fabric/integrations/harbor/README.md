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

## Execution Failures

The bridge rejects unsuccessful normalized results even when the task runner
exits successfully. Failed invocations raise an execution error; cancelled
invocations enter Harbor's cancellation path. The runner saves normalized
results before exiting on failure and saves a separate `runner_error` diagnostic
when a lifecycle exception prevents a normalized result. Harbor retains this
evidence in the trial's agent logs. Native host deadlines and structured timeout
errors enter Harbor's timeout handling. A completed invocation with verifier
reward `0` remains a task-quality result, not an execution error.

## Credentials And Environment Variables

Pass model credentials through Harbor's `extra_env` (`--ae`) and select them by
name with `fabric_model_api_key_env`. Do not put credentials in harness settings,
instructions, or configuration bundles.

`fabric_environment_env` remains available for explicit task-local environment
configuration. Its values join Harbor's managed `extra_env`; the retained
transport specification contains only variable names. The runner reconstructs
the values from its execution environment. Conflicting values in the two
options are rejected without printing either value.

Use credential variable names recognized by Harbor's redaction policy, such as
`NVIDIA_API_KEY`. The pilot validates job-output redaction with Harbor 0.24.0;
older Harbor releases do not provide the same redaction guarantees. Redaction
does not make a task container a trusted place for secrets.

The name-only transport requires this bridge and runner on both the host and
the task image. Do not mix it with the previously released runner.

## Usage And Provenance

The bridge copies normalized input, cached-input, output, and reported cost
into Harbor results independently of Relay. Input counts include cached tokens.
Codex thread totals are converted into invocation deltas. Pi counts are summed
across assistant messages in the invocation, including cache reads and writes.
Pi's catalog cost estimate stays in usage metadata rather than being presented
as a provider-reported charge. Unknown cost remains unavailable. ATIF only
backfills missing accounting fields.

Retained result metadata includes the logical harness, harness version,
adapter ID/version, harness SDK version when applicable, and task-runtime
version. Unavailable versions are explicitly `null`; the runtime version is
not substituted for a harness version. Lifecycle failure diagnostics retain
the versions available before startup. Run Pi and Codex in separate pilot jobs
until Harbor supports separate harness identities in aggregation.

## Pilot Capabilities

Class-level discovery is conservative because one Fabric agent can select
different adapters. Pi/Codex instances expose supported skills; Codex also
exposes MCP configuration. Pi rejects MCP configuration before harness
execution. Other adapters do not receive these pilot capability claims until
qualified. Fabric planning validates configured adapter support before startup.

Baseline runs with `fabric_telemetry=none` do not promise ATIF. Pi/Codex instances
declare ATIF only when Relay telemetry is selected, which also requires the
adapter's documented Relay setup in the task environment. Resume, trajectory
loading, and handoff remain outside this single-run pilot.

## Regression Tests

The credential-free bridge tests run in the Python suite:

```bash
uv run --no-sync pytest tests/python/test_harbor_failures.py tests/python/test_harbor_pilot.py
```

`tests/e2e/test_harbor_terminal_bench.py` additionally runs real Pi in Docker
against the unchanged TB2.1 `regex-log` instruction and verifier at commit
`7131e4375048a0e408a8fb404b5f499d726b695b`. Its scripted model endpoint tests
integration behavior, not model quality. It covers success, verifier reward
`0`, harness failure, timeout, usage, provenance, credential redaction, and
container cleanup.

Run it from a qualified Harbor 0.24 environment with this bridge on the host
and this runner/Pi adapter in the task image:

```bash
export HARBOR_FABRIC_TB21_TASK="<tb2.1-checkout>/tasks/regex-log"
export HARBOR_FABRIC_TB21_IMAGE="<qualified-fabric-task-image>"
python -m pytest <fabric-checkout>/tests/e2e/test_harbor_terminal_bench.py -q
```

The test checks the task instruction and verifier hashes and skips when the
qualified environment is not supplied. Keep live model compatibility checks
separate from this deterministic regression test.
