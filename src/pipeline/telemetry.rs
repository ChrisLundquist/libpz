//! Unified scheduler telemetry: timing/counter stats for profiling parallel compression.

use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    Mutex, OnceLock,
};
use std::time::{Duration, Instant};

use std::sync::Arc;

/// Aggregated timing/counter telemetry for the unified scheduler.
///
/// Collection is disabled by default and can be enabled via
/// [`set_unified_scheduler_stats_enabled()`].
#[derive(Debug, Clone, Copy, Default)]
pub struct UnifiedSchedulerStats {
    pub runs: u64,
    pub total_ns: u64,
    pub stage_compute_ns: u64,
    pub queue_wait_ns: u64,
    pub queue_admin_ns: u64,
    pub gpu_handoff_ns: u64,
    pub gpu_try_send_full_count: u64,
    pub gpu_try_send_disconnected_count: u64,
}

impl UnifiedSchedulerStats {
    /// Sum of tracked scheduler thread-time across workers/coordinator.
    pub fn scheduler_overhead_ns(&self) -> u64 {
        self.queue_wait_ns
            .saturating_add(self.queue_admin_ns)
            .saturating_add(self.gpu_handoff_ns)
    }

    /// Sum of tracked thread-time for scheduler + stage execution.
    pub fn tracked_thread_time_ns(&self) -> u64 {
        self.stage_compute_ns
            .saturating_add(self.scheduler_overhead_ns())
    }

    /// Fraction of tracked thread-time spent in scheduler overhead (0.0..=1.0).
    pub fn scheduler_overhead_pct(&self) -> f64 {
        let denom = self.tracked_thread_time_ns();
        if denom == 0 {
            0.0
        } else {
            self.scheduler_overhead_ns() as f64 / denom as f64
        }
    }
}

#[derive(Default)]
pub(super) struct LocalSchedulerStats {
    stage_compute_ns: AtomicU64,
    queue_wait_ns: AtomicU64,
    queue_admin_ns: AtomicU64,
}

impl LocalSchedulerStats {
    pub(super) fn add_stage_compute(&self, d: Duration) {
        self.stage_compute_ns
            .fetch_add(duration_to_ns(d), Ordering::Relaxed);
    }

    pub(super) fn add_queue_wait(&self, d: Duration) {
        self.queue_wait_ns
            .fetch_add(duration_to_ns(d), Ordering::Relaxed);
    }

    pub(super) fn add_queue_admin(&self, d: Duration) {
        self.queue_admin_ns
            .fetch_add(duration_to_ns(d), Ordering::Relaxed);
    }
}

pub(super) fn duration_to_ns(d: Duration) -> u64 {
    d.as_nanos().min(u64::MAX as u128) as u64
}

pub(super) static UNIFIED_SCHEDULER_STATS_ENABLED: AtomicBool = AtomicBool::new(false);
static UNIFIED_SCHEDULER_STATS: OnceLock<Mutex<UnifiedSchedulerStats>> = OnceLock::new();

pub(super) struct SchedulerRunRecorder {
    start: Instant,
    local: Option<Arc<LocalSchedulerStats>>,
}

impl SchedulerRunRecorder {
    pub(super) fn new(local: Option<Arc<LocalSchedulerStats>>) -> Self {
        Self {
            start: Instant::now(),
            local,
        }
    }
}

impl Drop for SchedulerRunRecorder {
    fn drop(&mut self) {
        let Some(local) = self.local.as_ref() else {
            return;
        };
        let mut guard = UNIFIED_SCHEDULER_STATS
            .get_or_init(|| Mutex::new(UnifiedSchedulerStats::default()))
            .lock()
            .expect("unified scheduler stats lock poisoned");
        guard.runs = guard.runs.saturating_add(1);
        guard.total_ns = guard
            .total_ns
            .saturating_add(duration_to_ns(self.start.elapsed()));
        guard.stage_compute_ns = guard
            .stage_compute_ns
            .saturating_add(local.stage_compute_ns.load(Ordering::Relaxed));
        guard.queue_wait_ns = guard
            .queue_wait_ns
            .saturating_add(local.queue_wait_ns.load(Ordering::Relaxed));
        guard.queue_admin_ns = guard
            .queue_admin_ns
            .saturating_add(local.queue_admin_ns.load(Ordering::Relaxed));
        // GPU telemetry fields (gpu_handoff_ns, gpu_try_send_*) are retained
        // in UnifiedSchedulerStats for API stability but always 0: the parallel
        // scheduler is CPU-only; GPU work uses the streaming path.
    }
}

pub(crate) fn set_unified_scheduler_stats_enabled(enabled: bool) {
    UNIFIED_SCHEDULER_STATS_ENABLED.store(enabled, Ordering::Relaxed);
}

pub(crate) fn reset_unified_scheduler_stats() {
    let mut guard = UNIFIED_SCHEDULER_STATS
        .get_or_init(|| Mutex::new(UnifiedSchedulerStats::default()))
        .lock()
        .expect("unified scheduler stats lock poisoned");
    *guard = UnifiedSchedulerStats::default();
}

pub(crate) fn unified_scheduler_stats() -> UnifiedSchedulerStats {
    *UNIFIED_SCHEDULER_STATS
        .get_or_init(|| Mutex::new(UnifiedSchedulerStats::default()))
        .lock()
        .expect("unified scheduler stats lock poisoned")
}
