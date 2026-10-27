<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Harbor Qualification Runbook

## Prerequisites And Scope

Run from the repository root on Linux x86_64 with Git, Python 3.12 or newer, `uv`, Docker, and Docker Compose. The image builder preserves the selected task Docker environment and adds Node.js 24.16.0, Rust 1.94.0, and packaged NVIDIA NeMo Fabric and harness dependencies. The baseline task uses Python 3.13 on Debian Bookworm; custom tasks must provide a compatible Python 3.12+ Debian environment with pip and apt-get. It can download an isolated host Python interpreter through `uv`. Use a dedicated output directory on storage visible to the Docker daemon; Snap Docker needs a `TMPDIR` visible outside its private `/tmp`, as described in the [Harbor examples](../../../../examples/harbor/README.md#docker-installed-with-snap). Remote Docker daemons need the same absolute paths as the client. No host credential directories are mounted.

Choose the exact NVIDIA NeMo Fabric package version and matching checkout/catalog, plus an exact Harbor version compatible with that NeMo Fabric line. The native Harbor `fabric` agent is the default; `--bridge sdk` explicitly tests `nemo_fabric.integrations.harbor:FabricAgent` instead. Record the distinction when comparing runs. Native Harbor and the SDK bridge are independently versioned implementations.

`--mode registry` installs the exact published version. `--mode source` builds noneditable wheels from the recorded committed source, installs the same runtime wheel on host and task, and builds TypeScript tarballs from the checked-in lockfiles. Source mode qualifies source integration, not registry publication. Its local tarballs can satisfy package dependencies that are not published yet. Run registry mode against the candidate before treating registry installability as proven. Build logs and identities retain actual resolved harness versions; unpinned upstream ranges can resolve differently on another date.

For a published beta or RC, pass `--catalog-json` with the catalog resource extracted from that exact catalog wheel. Release workflows stamp prerelease metadata in disposable checkouts, so the committed catalog can retain the stable base version. Keep the reference checkout at the intended release source: the gate still checks its complete adapter denominator and descriptor bytes against the candidate, then verifies the installed catalog's full hash. It preserves Python and npm versions independently (for example, `0.6.0rc1` and `0.6.0-rc.1`). To obtain the resource without importing any harness, run:

```bash
uv tool run --from pip pip download --no-deps --only-binary=:all: \
  --dest "$WORK/catalog" "nemo-fabric-adapter-catalog==$CANDIDATE_VERSION"
python3 - "$WORK/catalog" <<'PY'
from pathlib import Path
from zipfile import ZipFile
import sys
directory = Path(sys.argv[1])
with ZipFile(next(directory.glob('*.whl'))) as wheel:
    (directory / 'catalog.json').write_bytes(
        wheel.read('nemo_fabric_adapter_catalog/catalog.json'))
PY
```

Use `--mode registry --version "$CANDIDATE_VERSION" --catalog-json "$WORK/catalog/catalog.json"` with the other plan arguments below. Use a fresh catalog directory per candidate.

The denominator is every descriptor in the checked-in metadata catalog, including both NOOA variants and remote-agent. Generic NOOA requires registered workflow selection, which the standalone Harbor bridge does not transport; its catalog target IDs remain listed as unsupported coverage. The separate BenchAgent entry exercises standalone NOOA. Future catalog entries appear automatically and need a reviewed installation recipe/provider profile. They are not silently removed when no harness or credential is available. The builder derives Python extras and TypeScript optional peers from package manifests; a few harnesses deliberately installed outside adapter dependencies have explicit tested recipes in `prepare.py`. Review those recipes when the supported harness pins change.

## Serve A Local Checkpoint Or Select A Provider

A filesystem checkpoint path is not an inference API. Start with an existing operator-managed NIM/vLLM or other compatible server for that checkpoint, using an allocated GPU and the server's documented serving command. This gate deliberately does not load checkpoints, install serving infrastructure, allocate GPUs, or modify existing model services. Validate these prerequisites:

- The endpoint advertises the exact served model ID and supports tool calling. Responses, Chat Completions, and Anthropic Messages are different protocols. Check the API description and exercise the selected harness; an advertised route alone is not a passing integration test.
- The task container can reach the endpoint. Its `127.0.0.1` is the container, not the host. Keep a local server bound to loopback by default and arrange an operator-approved container route/proxy if needed. Do not expose a model service publicly to make the gate work.
- HTTPS is the default for remote endpoints. For an explicitly trusted private HTTP endpoint, name the exact host with `--trusted-http-host`; it is also forwarded to harnesses that advertise `insecure_http_model_hosts`. This transmits prompts and any credential without TLS to that host.
- Export the selected credential variable through your usual secret source. Never put the value in the plan, command arguments, Docker build arguments, settings, URLs, or committed files. For a confirmed unauthenticated endpoint, a nonsecret `MODEL_API_KEY=unused` satisfies clients that require a key. It is not an authentication test.

Claude needs Anthropic Messages, Codex normally uses Responses, and most other local coding harnesses use Chat Completions. `--endpoint-kind openai-and-anthropic` permits the same endpoint to be attempted through both families. Droid's descriptor requires Factory and Kiro's requires Kiro; neither declares arbitrary model endpoints. Remote-agent requires a remote **agent** with access to the task workspace, not just an inference server. A single local checkpoint cannot override these adapter contracts.

Hermes requires a supported preprovisioned task image. The source revision maintained in `just install-hermes-agent` needs Python 3.14 and explicitly rejects wheel/sdist builds. The development recipe uses editable installs and is not release-installation evidence. Do not set an unrelated build-system escape hatch to bypass that guard. Until a suitable supported image is available, retain the entry as blocked; an attempted unsupported installation is a recorded install failure, not a pass. A custom source image can include `/opt/qualification-harness-sources.json` mapping harness names to exact revisions, which the identity collector retains.

The Codex profile uses `danger-full-access` and `deny_all` **inside the gate-created Harbor task container**. Harbor supplies the isolation boundary; nested `bwrap` sandboxing requires user namespaces that many Docker hosts prohibit. Do not reuse this profile on a host workspace or attach host credentials, Docker sockets, or unrelated files to the task. The gate never enables privileged Docker mode or changes host namespace policy.

Estimate paid-provider cost before running, using the selected model's current prices, expected prompt/output tokens per trial, and the number of runnable entries. Supply the estimate and assumptions explicitly. The estimate is recorded evidence, not an enforced spending cap. The deadline bounds runtime, not token billing. Never retry silently or infer a zero cost from unavailable accounting.

## Plan, Prepare, And Run

Run **one shared task across the harnesses**, never the full TB2 dataset. The baseline is [`kv-store-grpc`](https://github.com/harbor-framework/terminal-bench-2/tree/2fd12b88aafdd04a52c298e3940bcb189f9766d6/kv-store-grpc): dependency installation, protocol definition, generated code, server implementation, and a background service checked with real RPCs. Fetch the pinned task outside the NeMo Fabric checkout:

```bash
WORK=/path/to/private/qualification
mkdir -m 700 "$WORK"
git clone --no-checkout https://github.com/harbor-framework/terminal-bench-2.git "$WORK/tb2"
git -C "$WORK/tb2" sparse-checkout set kv-store-grpc
git -C "$WORK/tb2" checkout 2fd12b88aafdd04a52c298e3940bcb189f9766d6
```

The gate requires `--task` pointing to that single directory. It records a hash of the task inputs and verifier, rejects a dataset root, and rejects changed files after planning. Preparation builds the task's original Dockerfile before adding NeMo Fabric and each harness. Harbor builds from the prepared image rather than pulling the task-only `docker_image`. The instruction, task assets, resources, and upstream verifier stay intact; the oracle solution is excluded from staging. A wrapper collects the agent's final package and selection evidence before running the upstream verifier, whose own dependency installation is outside the agent's execution. Task-installed packages may be added; changing an existing harness/runtime dependency fails identity verification. The original base and final image IDs are retained. This integration overlay is not an unmodified leaderboard submission.

Set the version, compatible Harbor pin, served model ID, and container-reachable endpoint for your environment. The following example uses an already running local NIM endpoint and a nonsecret placeholder key:

```bash
QUALIFY=.agents/skills/qualify-harbor/scripts
export MODEL_API_KEY=unused

python3 "$QUALIFY/qualify.py" plan \
  --output "$WORK/plan.json" \
  --task "$WORK/tb2/kv-store-grpc" \
  --mode source --version 0.6.0 \
  --harbor-version 0.24.1.dev202610250408 \
  --model nvidia/nemotron-3-ultra-550b-a55b \
  --base-url http://model-gateway:8000/v1 \
  --trusted-http-host model-gateway \
  --endpoint-kind openai-and-anthropic \
  --backend 'Operator-managed NIM; record the exact image/version' \
  --estimated-cost-usd 0 \
  --cost-basis 'Existing local inference; no metered provider charge; hardware excluded' \
  --timeout 900

# Inspect and configure rows before preparing; retain the full denominator.
python3 "$QUALIFY/prepare.py" --plan "$WORK/plan.json" --work "$WORK/environment"

"$WORK/environment/host/bin/python" "$QUALIFY/qualify.py" run \
  --plan "$WORK/plan.json" --environment "$WORK/environment/environment.json" \
  --output "$WORK/full-run"
```

Preparation and execution each require new directories and never overwrite previous evidence. Pass `--host-python /absolute/path/to/python` to `prepare.py` to use an existing Python 3.12+ interpreter rather than downloading Python 3.12. A base installation failure writes `environment.json` and logs before exiting; run the gate with `python3` in that case to produce a fully blocked report. `prepare.py` continues past individual harness installation failures, which become failed matrix rows. Retain the environment directory while inspecting or reproducing a run. Images are tagged with the source revision; record their immutable IDs in the report. No image is pushed.

For a keyed HTTPS provider, export `MODEL_API_KEY` securely, use its model slug and `--base-url https://...`, supply a nonzero cost estimate with token/pricing assumptions, and set the appropriate endpoint kind. Each plan row can override `model`, `base_url`, `api_key_env`, `harness_settings`, `telemetry`, and `image`. A custom task image must contain the matching runtime/catalog/adapter, its harness, `/qualification/identity.py`, and `/opt/qualification-identity.json` produced by that script. It must use noneditable package installs. Set a prerequisite row's `action` to `run` only after providing those dependencies. Do not pass a generic local model to a proprietary provider under a fabricated model alias.

A developer subset retains every other row as excluded and sets `scope=developer_subset`:

```bash
"$WORK/environment/host/bin/python" "$QUALIFY/qualify.py" run \
  --plan "$WORK/plan.json" --environment "$WORK/environment/environment.json" \
  --output "$WORK/pi-reproduction" --only nvidia.fabric.pi
```

Exit zero means the requested subset passed, or every entry passed in a full run. A subset never sets `release_qualified=true`. Nonzero means failure, blocked coverage, cancellation, or invalid prerequisites. A release waiver has the shape `{"accepted_by":"<actual human>","reason":"<coverage gap>","reference":"<decision URL or record>"}` on the relevant row. No default waiver is supplied and waived rows remain nonpasses.

## Evidence And Triage

`report.json` is the sanitized machine summary; `summary.md` is the human matrix. Both are rewritten atomically after each row, with pending rows retained if interrupted. `plan.json` retains the full descriptor snapshot. Each attempted row includes a reproducible command, timestamps, image identity, host/task package and harness versions, runtime wheel hash, observed runtime/invocation/adapter identity, reward, usage, capability limitations, and cleanup status. Catalog hashes cover raw metadata; Harbor admission hashes cover the core-normalized descriptor. The gate records both and compares the uploaded admission hash against SDK metadata inspection, never directly against the raw catalog hash. Raw task artifacts stay in the private run directory. The gate redacts known credential values from text output, but binary artifacts and unknown secrets are not shareable by assumption; inspect before exporting them.

Source-built task images remove the build tree after packaging. The native core can discover its compile-time source directory when that directory still exists; identical descriptor bytes do not guarantee the intended execution origin. A metadata-only task-container preflight verifies installed Python package data or the explicitly selected npm package before any model call. The verifier records the actual task's descriptor provenance again. A bundled/source-tree origin or drift between those observations fails admission even when versions and fingerprints match.

Use the failure categories to choose the next investigation:

| Category | Start With |
| --- | --- |
| Prerequisite/install | `environment.json`, `task-build.log`, `base-build.log`, `host-install.log`, per-image `install.log`; package availability, Python/Node compatibility, harness pins |
| Auth/provider/model | Endpoint protocol, exact served ID, credential environment reference, harness error code, and task's model config |
| Adapter admission/fingerprint | Selected catalog SHA, uploaded `fabric-run-*.json`, installed descriptor, package versions, task identity |
| Transport/runner | Harbor trial exception, `agent/fabric-result-*.json` or runner-error envelope, runner stdout/stderr |
| Harbor lifecycle/verifier | Trial `result.json`, `exception_info`, verifier output, reward file; reward zero and missing reward are distinct failures |
| Usage | Invocation usage and Harbor accounting; unknown is not zero, reward success does not prove usage transport |
| Telemetry/export | Relay validation, canonical ATIF schema, promoted `trajectory.json`, artifacts; baseline runs do not test telemetry |
| Timeout/cancellation/cleanup | Recorded deadline/exit code, partial logs, owned Compose resources and `cleanup.log` |
| Infrastructure | Connectivity, image fetch/build, storage exhaustion, server/rate-limit errors; reproduce before declaring a product regression |

Harbor normally deletes task containers. After completion or interruption, the gate removes only Compose containers/networks whose project label starts with this run's randomly allocated task prefix, plus metadata-probe containers carrying its exact qualification label. It reads resource labels without collecting container environments. It never prunes Docker globally or stops a model service. SIGINT is forwarded to its own process group, followed by bounded TERM/KILL escalation. Abrupt host loss or SIGKILL cannot guarantee cleanup; retain the run's task prefix and inspect its exact labels before manual cleanup. Images and build evidence are deliberately retained for reproduction. The gate does not retry trials; harness-native retries can still occur within the recorded deadline and remain visible in the logs.

For deterministic gate and integration checks, use the repository environment:

```bash
uv run --no-sync pytest tests/scripts/test_harbor_qualification.py \
  tests/integrations/test_harbor_runner.py tests/integrations/test_harbor_mcp.py \
  tests/integrations/test_harbor_skills.py tests/integrations/test_harbor_telemetry.py \
  tests/python/test_harbor_accounting.py tests/python/test_harbor_credentials.py \
  tests/python/test_adapter_capabilities.py
```

The baseline leaves MCP, skills, subagents, and telemetry untested where declared, or unsupported/not declared where metadata does not advertise them. For optional Relay qualification, configure a supported row with `telemetry=relay` and a compatible image containing Relay and any required harness extension; the gate validates export and canonical ATIF. Use the [SWE-Bench examples](../../../../examples/harbor/swebench/README.md) for live MCP/skills/telemetry scenarios and attach their exact commands and artifacts separately. Subagent functionality needs harness-specific evidence; neither one task nor ATIF schema validity establishes it. Historical flaky CI tests are context, not present-run exemptions.
