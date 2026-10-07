<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NVIDIA NeMo Fabric Adapter Catalog

`nemo-fabric-adapter-catalog` provides release snapshots of the bundled Python and TypeScript adapter descriptors and their registered targets. It has no runtime dependencies and imports no adapter code or harness SDKs.

## Install And Inspect

Install the matching catalog release:

```bash
pip install nemo-fabric-adapter-catalog
```

Look up a descriptor by its exact ID:

```python
from nemo_fabric_adapter_catalog import get_adapter_descriptor, get_target_descriptor

adapter = get_adapter_descriptor("nvidia.fabric.codex")
print(adapter["config"]["accepts"])
target = get_target_descriptor("nvidia.nooa.arc-solver")
```

Each call returns a fresh dictionary containing the canonical descriptor object. Unknown IDs raise `KeyError`; lookup does not guess an adapter or fall back to another version. Catalog-format errors and unreadable or malformed resources propagate as errors.

## Metadata Is Not Execution

The bundle is a package resource, not an installed descriptor under `share/nemo-fabric`. Installing it does not register executable adapters or change Fabric runner selection, including when it is co-installed with adapter packages. Task execution continues to discover and validate task-owned descriptors, runners, and harness dependencies through the existing runtime APIs.

A snapshot claim is not proof of the task environment's capabilities. Pin compatible releases and compare the snapshot with the actual task descriptor before relying on it. A provider declaring ATIF support does not establish that it is enabled or that an artifact was produced. Capability normalization, host/task compatibility enforcement, and runtime-observed provenance are separate concerns, not implemented by this package.

## Maintain The Bundle

`scripts/ci/generate_adapter_catalog.py` collects canonical descriptors from leaf packages under `adapters/python/` and `adapters/typescript/`. It also includes targets under each leaf's `targets/` directory. Common support packages, presets, fixtures, and source-only external integrations are excluded. There is no adapter-ID allowlist.

The generated `catalog.json` keeps source package names, declared versions, paths, ecosystems, and fingerprints outside the descriptor objects. SHA-256 uses UTF-8 JSON with sorted keys, compact separators, and unescaped Unicode. These fields describe the release snapshot, not software observed during execution. The resource format is versioned separately from the adapter contract; use the lookup functions rather than depending on its layout.

```bash
just adapter-catalog
just check-adapter-catalog
uv build --wheel --sdist --out-dir dist sdk/python/nemo-fabric-adapter-catalog
```

`just set-version` refreshes the bundle after stamping Python and TypeScript versions. Python tests and wheel builds reject stale bundles. Wheels and source distributions contain the bundle and build independently of the adapter source tree.
