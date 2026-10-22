# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read native resource series and select an invocation or trajectory window."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from nemo_fabric.errors import FabricConfigError
from nemo_fabric.types import ResourceObservation
from nemo_fabric.types import _stats_number


def summarize_system_stats(
    path: str | Path,
    *,
    target_interval_ms: int,
    start_seconds: float = 0.0,
    end_seconds: float | None = None,
) -> dict[str, Any]:
    """Reduce a retained native JSONL series over a selected monotonic window.

    This synchronous, local-file helper returns adapter-tree gauge reductions,
    counter estimates, identity and coverage. Select ``sampled_peak``, ``min``,
    ``range``, ``final_value`` or ``time_weighted_mean`` from each gauge.

    Extrema cover actual samples inside the selected window. Means hold the
    previous value across adjacent valid readings at most twice cadence apart,
    clipping the integral at window boundaries. A mean can include a reading
    just before the window; it is an interpolation, not a new observation.
    Final values and CPU deltas require observed boundaries. Different IDs,
    counter resets and long/missing intervals never become valid CPU deltas.

    Reads are bounded to 100,000 records, 512 MiB total, and 1 MiB per line. An
    interrupted final JSON line is ignored and disclosed; malformed complete
    lines fail. The source file and missing readings are never modified.
    """
    if (
        isinstance(target_interval_ms, bool)
        or not isinstance(target_interval_ms, int)
        or not 100 <= target_interval_ms <= 60000
    ):
        raise FabricConfigError("target interval must be between 100 and 60000 ms")
    for bound in (start_seconds, end_seconds):
        if bound is not None:
            _stats_number(bound, "window bound")
    rows: list[tuple[float, ResourceObservation]] = []
    identity: tuple[str, str, str] | None = None
    diagnostics = []
    total = 0
    try:
        with Path(path).open("rb") as stream:
            while raw := stream.readline(1_048_577):
                total += len(raw)
                if len(raw) > 1_048_576 or total > 536_870_912 or len(rows) >= 100_000:
                    raise FabricConfigError("resource series exceeds read bounds")
                try:
                    row = json.loads(raw)
                except (ValueError, UnicodeDecodeError) as error:
                    if not raw.endswith(b"\n"):
                        diagnostics.append("partial_tail_ignored")
                        break
                    raise FabricConfigError(
                        "invalid complete resource sample"
                    ) from error
                if (
                    not isinstance(row, dict)
                    or row.get("schema_version") != "fabric.system_stats/v1alpha1"
                ):
                    raise FabricConfigError("unsupported resource sample schema")
                ids = tuple(
                    row.get(key)
                    for key in ("runtime_id", "invocation_id", "request_id")
                )
                if any(
                    not isinstance(value, str) or not value.strip() for value in ids
                ) or (identity is not None and identity != ids):
                    raise FabricConfigError(
                        "resource samples must identify one invocation"
                    )
                identity = ids
                offset = _stats_number(row.get("offset_seconds"), "resource offset")
                if offset > (1 << 64) - 1 or (rows and offset < rows[-1][0]):
                    raise FabricConfigError(
                        "resource offsets must be bounded and monotonic"
                    )
                observation = ResourceObservation.from_mapping(row.get("adapter_tree"))
                if (
                    observation.snapshot is not None
                    and observation.snapshot.runtime_id != ids[0]
                ):
                    raise FabricConfigError(
                        "resource snapshot belongs to a different runtime"
                    )
                rows.append((float(offset), observation))
    except OSError as error:
        raise FabricConfigError("resource series could not be read") from error
    if not rows or identity is None:
        raise FabricConfigError("resource series has no complete samples")
    end = rows[-1][0] if end_seconds is None else float(end_seconds)
    start = float(start_seconds)
    if start < rows[0][0] or end > rows[-1][0] or end <= start:
        raise FabricConfigError(
            "window must be nonempty and inside the retained series"
        )
    duration = end - start
    maximum_gap = target_interval_ms / 500
    selected = [(t, observation) for t, observation in rows if start <= t <= end]
    result: dict[str, Any] = {
        "runtime_id": identity[0],
        "invocation_id": identity[1],
        "request_id": identity[2],
        "scope": "adapter_host_tree",
        "start_seconds": start,
        "end_seconds": end,
        "start_observed": any(t == start for t, _ in selected),
        "end_observed": any(t == end for t, _ in selected),
        "diagnostics": diagnostics,
    }
    for field, unit in (
        ("resident_memory_bytes", "byte"),
        ("process_count", "process"),
    ):
        values = [
            getattr(obs.snapshot, field)
            for _, obs in selected
            if obs.snapshot is not None
        ]
        area = covered = 0.0
        for (left, before), (right, after) in zip(rows, rows[1:]):
            dt = max(0.0, min(right, end) - max(left, start))
            if (
                dt
                and right - left <= maximum_gap
                and before.snapshot is not None
                and after.snapshot is not None
            ):
                area += getattr(before.snapshot, field) * dt
                covered += dt
        final = next(
            (
                getattr(obs.snapshot, field)
                for t, obs in selected
                if t == end and obs.snapshot is not None
            ),
            None,
        )
        result[field] = {
            "unit": unit,
            "final_value": final,
            "min": min(values) if values else None,
            "sampled_peak": max(values) if values else None,
            "range": [min(values), max(values)] if values else None,
            "time_weighted_mean": area / covered if covered else None,
            "covered_seconds": covered,
            "covered_fraction": min(covered / duration, 1.0),
            "valid_samples": len(values),
        }
    snapshots = [obs.snapshot for _, obs in selected]
    cpu_valid = (
        bool(selected)
        and result["start_observed"]
        and result["end_observed"]
        and all(s is not None for s in snapshots)
    )
    cpu_valid = cpu_valid and all(
        right - left <= maximum_gap
        and before.snapshot.started_at == after.snapshot.started_at
        and before.snapshot.cpu_time_seconds <= after.snapshot.cpu_time_seconds
        for (left, before), (right, after) in zip(selected, selected[1:])
    )
    delta = (
        snapshots[-1].cpu_time_seconds - snapshots[0].cpu_time_seconds
        if cpu_valid
        else None
    )
    if delta is not None and not math.isfinite(delta):
        delta = None
    percent = 100 * delta / duration if delta is not None else None
    result["cpu_time_seconds_estimate"] = delta
    result["cpu_percent_one_core"] = (
        percent if percent is not None and math.isfinite(percent) else None
    )
    return result
