---
title: "System Stats"
slug: "/reference/api/python-library-reference/stats"
description: "Reduce native resource samples over invocation or trajectory windows."
---
<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0 -->

# <kbd>module</kbd> `nemo_fabric.stats`

Read native resource series and select an invocation or trajectory window.


---


## <kbd>function</kbd> `summarize_system_stats`

```python
def summarize_system_stats(
    path: str | Path,
    *,
    target_interval_ms: int,
    start_seconds: float = 0.0,
    end_seconds: float | None = None,
) -> dict[str, Any]
```

Reduce a retained native JSONL series over a selected monotonic window.

This synchronous, local-file helper returns adapter-tree gauge reductions, counter estimates, identity and coverage. Select ``sampled_peak``, ``min``, ``range``, ``final_value`` or ``time_weighted_mean`` from each gauge.

Extrema cover actual samples inside the selected window. Means hold the previous value across adjacent valid readings at most twice cadence apart, clipping the integral at window boundaries. A mean can include a reading just before the window; it is an interpolation, not a new observation. Final values and CPU deltas require observed boundaries. Different IDs, counter resets and long/missing intervals never become valid CPU deltas.

Reads are bounded to 100,000 records, 512 MiB total, and 1 MiB per line. An interrupted final JSON line is ignored and disclosed; malformed complete lines fail. The source file and missing readings are never modified.




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
