// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Invocation measurements collected by Fabric in the executing runtime.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::error::{FabricError, Result};
use crate::resource_snapshot::RuntimeResourceSnapshot;

/// One attempt to observe the adapter host process tree. Missing readings are
/// explicit; an unavailable observation never stands for zero resource usage.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum ResourceObservation {
    /// The process tree was read successfully.
    Available {
        /// Timestamped cumulative counters and instantaneous gauges.
        snapshot: RuntimeResourceSnapshot,
    },
    /// The reader supports this runtime, but could not read its process tree.
    Unavailable {
        /// Why this observation could not be collected.
        reason: String,
    },
    /// The runtime's provider or operating system has no reader.
    Unsupported {
        /// Which collection capability is missing.
        reason: String,
    },
}

impl ResourceObservation {
    pub(crate) fn from_result(result: Result<RuntimeResourceSnapshot>) -> Self {
        match result {
            Ok(snapshot) => Self::Available { snapshot },
            Err(FabricError::ResourceObservationUnsupported { reason, .. }) => {
                Self::Unsupported { reason }
            }
            Err(FabricError::ResourceObservationUnavailable { reason, .. }) => {
                Self::Unavailable { reason }
            }
            Err(error) => Self::Unavailable {
                reason: error.to_string(),
            },
        }
    }

    pub(crate) fn snapshot(&self) -> Option<&RuntimeResourceSnapshot> {
        match self {
            Self::Available { snapshot } => Some(snapshot),
            Self::Unavailable { .. } | Self::Unsupported { .. } => None,
        }
    }
}

/// System measurements around one invocation, collected inside the executing
/// runner. Resource observations cover its adapter host process tree, excluding
/// the orchestrator, other workloads, and remote model/service processes.
///
/// Endpoints are instantaneous readings, not peaks or time averages. CPU time
/// is cumulative since runtime startup; its difference is only an estimate
/// because process reaping and reparenting can change which work is visible.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct InvocationSystemStats {
    /// Monotonic elapsed seconds for the adapter exchange. Excludes runtime
    /// startup, resource collection, artifact preparation, and teardown.
    #[schemars(range(min = 0.0))]
    pub elapsed_seconds: f64,
    /// Observation immediately before the adapter exchange.
    pub before: ResourceObservation,
    /// Observation immediately after the adapter exchange, before another
    /// invocation can acquire this runtime's host lock.
    pub after: ResourceObservation,
    /// Nonnegative CPU counter difference, in seconds. Absent if an endpoint
    /// is missing, the runtime identity differs, or its CPU counter decreased.
    /// It can undercount work whose processes leave the tree between reads.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(min = 0.0))]
    pub cpu_time_seconds_estimate: Option<f64>,
}

impl InvocationSystemStats {
    pub(crate) fn new(
        before: ResourceObservation,
        after: ResourceObservation,
        elapsed_seconds: f64,
    ) -> Self {
        let cpu_time_seconds_estimate =
            before
                .snapshot()
                .zip(after.snapshot())
                .and_then(|(before, after)| {
                    let delta = after.cpu_time_seconds - before.cpu_time_seconds;
                    (before.runtime_id == after.runtime_id
                        && before.started_at == after.started_at
                        && delta.is_finite()
                        && delta >= 0.0)
                        .then_some(delta)
                });
        Self {
            elapsed_seconds,
            before,
            after,
            cpu_time_seconds_estimate,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn observation(cpu: f64) -> ResourceObservation {
        ResourceObservation::Available {
            snapshot: RuntimeResourceSnapshot {
                runtime_id: "runtime-1".into(),
                started_at: "2026-10-21T00:00:00.000Z".into(),
                observed_at: "2026-10-21T00:01:00.000Z".into(),
                resident_memory_bytes: 100,
                process_count: 1,
                cpu_time_seconds: cpu,
            },
        }
    }

    #[test]
    fn cpu_estimate_excludes_runtime_history_and_does_not_clamp_counter_resets() {
        let stats = InvocationSystemStats::new(observation(12.0), observation(14.5), 3.0);
        assert_eq!(stats.cpu_time_seconds_estimate, Some(2.5));
        let unchanged = InvocationSystemStats::new(observation(12.0), observation(12.0), 3.0);
        assert_eq!(unchanged.cpu_time_seconds_estimate, Some(0.0));
        let reset = InvocationSystemStats::new(observation(12.0), observation(2.0), 3.0);
        assert_eq!(reset.cpu_time_seconds_estimate, None);
    }

    #[test]
    fn missing_or_different_runtime_observations_have_no_cpu_estimate() {
        let mut replaced = observation(14.0);
        if let ResourceObservation::Available { snapshot } = &mut replaced {
            snapshot.started_at = "2026-10-21T00:00:30.000Z".into();
        }
        for after in [
            replaced,
            ResourceObservation::Unavailable {
                reason: "host exited".into(),
            },
            ResourceObservation::Unsupported {
                reason: "no OS reader".into(),
            },
        ] {
            let stats = InvocationSystemStats::new(observation(12.0), after, 3.0);
            assert_eq!(stats.cpu_time_seconds_estimate, None);
            let json = serde_json::to_value(&stats).unwrap();
            assert!(json.get("cpu_time_seconds_estimate").is_none());
            assert_eq!(
                serde_json::from_value::<InvocationSystemStats>(json).unwrap(),
                stats
            );
        }
    }
}
