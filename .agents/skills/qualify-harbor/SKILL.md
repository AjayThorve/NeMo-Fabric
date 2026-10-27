---
name: qualify-harbor
description: Qualify a NeMo Fabric release or developer subset through Harbor, real harnesses, inference, and a deterministic verifier. Use for Harbor integration release gates and reproducible all-harness failure triage; not for publishing releases or benchmarking model quality.
license: Apache-2.0
---

# Qualify NVIDIA NeMo Fabric With Harbor

Use the runnable gate in this directory to test Harbor → NeMo Fabric → harness → inference → result → verifier. Harbor owns containers, task lifecycle, and scoring. The gate runs exactly one shared Terminal-Bench task across the catalog, builds isolated package installs on that task environment, and retains a complete adapter denominator. It does not install or manage inference servers.

Read [the runbook](references/runbook.md) for commands, environment requirements, and failure triage. Release policy lives in [RELEASING.md](../../../RELEASING.md#harbor-integration-qualification).

1. Identify the exact release candidate or source commit and compatible Harbor package version. Use a disposable checkout of that version with a fresh catalog (`just check-adapter-catalog`). Do not use a developer's ambient editable environment as release evidence.
2. Establish inference prerequisites before running. A local checkpoint must already be served through a compatible HTTP API. Verify the served model ID, container reachability, protocol, authentication, and available resources. Use the same served model for compatible harnesses; record provider restrictions rather than assuming universal compatibility. Do not stop existing services, take allocated GPUs, mount credential directories, or publish new network listeners.
3. Select one task directory, not a benchmark dataset. The runbook pins TB2 `kv-store-grpc` as a comprehensive baseline. Generate and inspect the plan; its task hash covers the instruction, environment, and unchanged upstream verifier. It includes every catalog adapter, exact package/descriptor metadata, settings, estimated provider cost, and capability coverage. Resolve prerequisites by configuring individual rows and dedicated task images. Keep blocked, unsupported, and excluded rows in the plan with reasons. Record waivers only when a named human actually accepted the gap, with a decision reference.
4. Prepare isolated host and task installs, then run the full matrix. Use `--only` for a developer subset; it can never qualify a release. The default is one trial at a time, one attempt, bounded runtime/build deadlines, and no automatic retries. Preserve each failure before reproducing it in a new output directory.
5. Inspect `report.json`, `summary.md`, installation logs, and trial artifacts. Separate task reward from transport, selection/fingerprint, package identity, usage, optional telemetry, and cleanup. Reproduce findings and classify them before attributing a regression. Fix gate defects narrowly; report unrelated product defects separately.
6. Hand the release PIC the sanitized report and exact evidence locations. A full pass requires every catalog entry to pass. Missing credentials, incompatible providers, install failures, absent usage, timeout, cancellation, and skipped coverage remain nonpasses. A waiver records a release decision; it does not rewrite the automated gate to green.

One shared TB2 task is a scored integration check, not a full benchmark or proof of all-feature equivalence. MCP, skills, ATIF/export, and subagent paths require their own evidence where supported. The report labels untested paths. Use the existing Harbor integration suites for deterministic lifecycle, error, credential, MCP, skills, and telemetry coverage; record any live capability tests separately. Never infer a capability pass from one successful task.
