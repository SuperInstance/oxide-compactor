//! # oxide-compactor
//!
//! Background compaction for GPU data structures with ternary compaction state.
//!
//! Compaction state is represented as a ternary value:
//! - `+1` → **Compacted** (healthy segment, skip)
//! - `0`  → **In Progress** (consider for compaction)
//! - `-1` → **Needs Compaction** (fragmented, compact now)
//!
//! ## Architecture
//!
//! - [`CompactionState`] — ternary trigger for segment health
//! - [`CompactionJob`] — tracks a single compaction operation
//! - [`CompactionScheduler`] — picks segments based on fragmentation ratio
//! - [`Segment`] — a sorted run of data in GPU memory
//! - [`CompactionStats`] — cumulative statistics
//! - [`MergeCompactor`] — merge-sort compaction of sorted runs

use std::fmt;
use std::ops::{Add, AddAssign};

// ---------------------------------------------------------------------------
// Ternary compaction state
// ---------------------------------------------------------------------------

/// Ternary compaction trigger for a segment.
///
/// | Value | Variant          | Meaning                        |
/// |-------|------------------|--------------------------------|
/// | +1    | Compacted        | Healthy, skip compaction       |
/// |  0    | InProgress       | Borderline, consider           |
/// | -1    | NeedsCompaction  | Fragmented, compact now        |
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CompactionState {
    NeedsCompaction = -1,
    InProgress = 0,
    Compacted = 1,
}

impl CompactionState {
    /// Classify a segment from its fragmentation ratio (0.0 – 1.0).
    ///
    /// - ratio < 0.3 → Compacted (plenty of free space)
    /// - ratio < 0.6 → InProgress (worth watching)
    /// - ratio ≥ 0.6 → NeedsCompaction (reclaim now)
    pub fn from_fragmentation(ratio: f64) -> Self {
        if ratio >= 0.6 {
            Self::NeedsCompaction
        } else if ratio >= 0.3 {
            Self::InProgress
        } else {
            Self::Compacted
        }
    }

    /// Numeric value (+1, 0, -1).
    pub fn value(&self) -> i8 {
        match self {
            Self::Compacted => 1,
            Self::InProgress => 0,
            Self::NeedsCompaction => -1,
        }
    }
}

impl fmt::Display for CompactionState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Compacted => write!(f, "Compacted(+1)"),
            Self::InProgress => write!(f, "InProgress(0)"),
            Self::NeedsCompaction => write!(f, "NeedsCompaction(-1)"),
        }
    }
}

// ---------------------------------------------------------------------------
// Segment — a sorted run in GPU memory
// ---------------------------------------------------------------------------

/// A segment of sorted data residing in GPU memory.
#[derive(Debug, Clone)]
pub struct Segment {
    /// Unique segment identifier.
    pub id: u64,
    /// Total capacity in bytes.
    pub capacity: u64,
    /// Bytes actually in use (live data).
    pub used_bytes: u64,
    /// Current compaction state.
    pub state: CompactionState,
}

impl Segment {
    /// Create a new segment.
    pub fn new(id: u64, capacity: u64, used_bytes: u64) -> Self {
        let ratio = Self::frag_ratio(capacity, used_bytes);
        Self {
            id,
            capacity,
            used_bytes,
            state: CompactionState::from_fragmentation(ratio),
        }
    }

    /// Fragmentation ratio: dead space / capacity.
    ///
    /// Returns 0.0 when the segment is full (no fragmentation) and 1.0 when
    /// empty (fully fragmented in the sense of wasted capacity).
    /// Fragmentation ratio: dead space / capacity.
    ///
    /// Returns 0.0 when the segment is full (no fragmentation) and 1.0 when
    /// empty (fully fragmented in the sense of wasted capacity).
    pub fn fragmentation_ratio(&self) -> f64 {
        Self::frag_ratio(self.capacity, self.used_bytes)
    }

    fn frag_ratio(capacity: u64, used_bytes: u64) -> f64 {
        if capacity == 0 {
            return 0.0;
        }
        1.0 - (used_bytes as f64 / capacity as f64)
    }

    /// Dead (reclaimable) bytes.
    pub fn dead_bytes(&self) -> u64 {
        self.capacity.saturating_sub(self.used_bytes)
    }

    /// Refresh the ternary state from the current fragmentation ratio.
    pub fn refresh_state(&mut self) {
        self.state = CompactionState::from_fragmentation(self.fragmentation_ratio());
    }
}

// ---------------------------------------------------------------------------
// Compaction job
// ---------------------------------------------------------------------------

/// Tracks a single compaction operation from source segments into a target.
#[derive(Debug, Clone)]
pub struct CompactionJob {
    /// Unique job identifier.
    pub id: u64,
    /// Source segment ids that will be merged.
    pub source_ids: Vec<u64>,
    /// Target segment id for the compacted output.
    pub target_id: u64,
    /// Bytes already compacted.
    pub bytes_compacted: u64,
    /// Total bytes to compact.
    pub total_bytes: u64,
    /// Whether the job is finished.
    pub done: bool,
}

impl CompactionJob {
    /// Create a new job.
    pub fn new(id: u64, source_ids: Vec<u64>, target_id: u64, total_bytes: u64) -> Self {
        Self {
            id,
            source_ids,
            target_id,
            bytes_compacted: 0,
            total_bytes,
            done: false,
        }
    }

    /// Advance compaction by `n` bytes.
    pub fn advance(&mut self, n: u64) {
        self.bytes_compacted = (self.bytes_compacted + n).min(self.total_bytes);
        if self.bytes_compacted >= self.total_bytes {
            self.done = true;
        }
    }

    /// Progress as a fraction 0.0 – 1.0.
    pub fn progress(&self) -> f64 {
        if self.total_bytes == 0 {
            1.0
        } else {
            self.bytes_compacted as f64 / self.total_bytes as f64
        }
    }
}

// ---------------------------------------------------------------------------
// Compaction statistics
// ---------------------------------------------------------------------------

/// Cumulative statistics for compaction operations.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CompactionStats {
    /// Total bytes that have been compacted.
    pub total_bytes_compacted: u64,
    /// Bytes of dead space reclaimed.
    pub space_reclaimed: u64,
    /// Number of compaction jobs completed.
    pub jobs_completed: u64,
    /// Number of source segments merged.
    pub segments_merged: u64,
}

impl CompactionStats {
    /// Overall compaction ratio: reclaimed / compacted.
    ///
    /// Higher is better — it means more waste was removed per byte processed.
    pub fn compaction_ratio(&self) -> f64 {
        if self.total_bytes_compacted == 0 {
            0.0
        } else {
            self.space_reclaimed as f64 / self.total_bytes_compacted as f64
        }
    }
}

impl AddAssign for CompactionStats {
    fn add_assign(&mut self, other: Self) {
        self.total_bytes_compacted += other.total_bytes_compacted;
        self.space_reclaimed += other.space_reclaimed;
        self.jobs_completed += other.jobs_completed;
        self.segments_merged += other.segments_merged;
    }
}

impl Add for CompactionStats {
    type Output = Self;
    fn add(mut self, other: Self) -> Self {
        self += other;
        self
    }
}

// ---------------------------------------------------------------------------
// Compaction scheduler
// ---------------------------------------------------------------------------

/// Picks segments to compact based on fragmentation ratio and ternary state.
#[derive(Debug, Clone)]
pub struct CompactionScheduler {
    /// Segments under management.
    segments: Vec<Segment>,
    /// Monotonically increasing job counter.
    next_job_id: u64,
    /// Monotonically increasing segment counter (for targets).
    next_segment_id: u64,
    /// Cumulative statistics.
    stats: CompactionStats,
}

impl CompactionScheduler {
    /// Create a new scheduler with the given segments.
    pub fn new(segments: Vec<Segment>) -> Self {
        let max_id = segments.iter().map(|s| s.id).max().unwrap_or(0);
        Self {
            segments,
            next_job_id: 0,
            next_segment_id: max_id + 1,
            stats: CompactionStats::default(),
        }
    }

    /// Refresh the ternary state of every managed segment.
    pub fn refresh_all(&mut self) {
        for seg in &mut self.segments {
            seg.refresh_state();
        }
    }

    /// Return segments that need compaction (state = NeedsCompaction).
    pub fn segments_needing_compaction(&self) -> Vec<&Segment> {
        self.segments
            .iter()
            .filter(|s| s.state == CompactionState::NeedsCompaction)
            .collect()
    }

    /// Return segments that are in progress (state = InProgress).
    pub fn segments_in_progress(&self) -> Vec<&Segment> {
        self.segments
            .iter()
            .filter(|s| s.state == CompactionState::InProgress)
            .collect()
    }

    /// Return segments that are healthy (state = Compacted).
    pub fn healthy_segments(&self) -> Vec<&Segment> {
        self.segments
            .iter()
            .filter(|s| s.state == CompactionState::Compacted)
            .collect()
    }

    /// Pick the most fragmented segments and create a compaction job.
    ///
    /// Returns `None` if no segments need compaction. Picks up to `max_sources`
    /// segments, sorted by fragmentation ratio descending.
    pub fn schedule_compaction(&mut self, max_sources: usize) -> Option<CompactionJob> {
        // Refresh states first.
        self.refresh_all();

        // Collect indices of segments needing compaction.
        let mut candidates: Vec<usize> = self
            .segments
            .iter()
            .enumerate()
            .filter(|(_, s)| s.state == CompactionState::NeedsCompaction)
            .map(|(i, _)| i)
            .collect();

        if candidates.is_empty() {
            return None;
        }

        // Sort by fragmentation ratio descending.
        candidates.sort_by(|a, b| {
            self.segments[*b]
                .fragmentation_ratio()
                .partial_cmp(&self.segments[*a].fragmentation_ratio())
                .unwrap()
        });

        candidates.truncate(max_sources);

        let total_bytes: u64 = candidates.iter().map(|i| self.segments[*i].used_bytes).sum();
        let source_ids: Vec<u64> = candidates.iter().map(|i| self.segments[*i].id).collect();

        let job_id = self.next_job_id;
        self.next_job_id += 1;

        let target_id = self.next_segment_id;
        self.next_segment_id += 1;

        let job = CompactionJob::new(job_id, source_ids.clone(), target_id, total_bytes);

        Some(job)
    }

    /// Finalize a completed job: remove source segments, add target, update stats.
    pub fn finalize_job(&mut self, job: &CompactionJob) {
        if !job.done {
            return;
        }

        let source_set: std::collections::HashSet<u64> =
            job.source_ids.iter().copied().collect();

        let source_capacity: u64 = self
            .segments
            .iter()
            .filter(|s| source_set.contains(&s.id))
            .map(|s| s.capacity)
            .sum();

        let reclaimed = source_capacity.saturating_sub(job.total_bytes);

        // Remove source segments.
        self.segments
            .retain(|s| !source_set.contains(&s.id));

        // Add compacted target segment.
        let mut target = Segment::new(job.target_id, job.total_bytes, job.total_bytes);
        target.state = CompactionState::Compacted;
        self.segments.push(target);

        // Update stats.
        self.stats.total_bytes_compacted += job.total_bytes;
        self.stats.space_reclaimed += reclaimed;
        self.stats.jobs_completed += 1;
        self.stats.segments_merged += job.source_ids.len() as u64;
    }

    /// Access cumulative statistics.
    pub fn stats(&self) -> &CompactionStats {
        &self.stats
    }

    /// Access managed segments.
    pub fn segments(&self) -> &[Segment] {
        &self.segments
    }

    /// Add a segment to the scheduler.
    pub fn add_segment(&mut self, seg: Segment) {
        if seg.id >= self.next_segment_id {
            self.next_segment_id = seg.id + 1;
        }
        self.segments.push(seg);
    }
}

// ---------------------------------------------------------------------------
// Merge compactor — merge-sort compaction of sorted runs
// ---------------------------------------------------------------------------

/// Performs merge-sort compaction of multiple sorted runs.
///
/// Each "sorted run" is a `Vec<T: Ord>`. The compactor merges them into a
/// single sorted output, simulating the kind of merge that would happen on
/// GPU data structures.
pub struct MergeCompactor;

impl MergeCompactor {
    /// Merge-sort multiple sorted runs into a single sorted output.
    ///
    /// All input runs **must** be sorted in ascending order. The output is
    /// guaranteed to be sorted in ascending order.
    pub fn compact<T: Ord + Clone>(runs: &[Vec<T>]) -> Vec<T> {
        if runs.is_empty() {
            return Vec::new();
        }
        if runs.len() == 1 {
            return runs[0].clone();
        }

        // Classic k-way merge using a binary heap (min-heap via Reverse).
        use std::cmp::Reverse;
        use std::collections::BinaryHeap;

        // (value, run_index, element_index)
        let mut heap: BinaryHeap<Reverse<(T, usize, usize)>> = BinaryHeap::new();

        for (ri, run) in runs.iter().enumerate() {
            if !run.is_empty() {
                heap.push(Reverse((run[0].clone(), ri, 0)));
            }
        }

        let total_len: usize = runs.iter().map(|r| r.len()).sum();
        let mut result = Vec::with_capacity(total_len);

        while let Some(Reverse((val, ri, ei))) = heap.pop() {
            result.push(val);
            let next_ei = ei + 1;
            if next_ei < runs[ri].len() {
                heap.push(Reverse((runs[ri][next_ei].clone(), ri, next_ei)));
            }
        }

        result
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // 1. Ternary state classification from fragmentation ratios.
    #[test]
    fn test_ternary_state_classification() {
        assert_eq!(CompactionState::from_fragmentation(0.1), CompactionState::Compacted);
        assert_eq!(CompactionState::from_fragmentation(0.29), CompactionState::Compacted);
        assert_eq!(CompactionState::from_fragmentation(0.3), CompactionState::InProgress);
        assert_eq!(CompactionState::from_fragmentation(0.5), CompactionState::InProgress);
        assert_eq!(CompactionState::from_fragmentation(0.6), CompactionState::NeedsCompaction);
        assert_eq!(CompactionState::from_fragmentation(0.9), CompactionState::NeedsCompaction);
    }

    // 2. Ternary numeric values.
    #[test]
    fn test_ternary_values() {
        assert_eq!(CompactionState::Compacted.value(), 1);
        assert_eq!(CompactionState::InProgress.value(), 0);
        assert_eq!(CompactionState::NeedsCompaction.value(), -1);
    }

    // 3. Segment fragmentation ratio and dead bytes.
    #[test]
    fn test_segment_fragmentation() {
        let seg = Segment::new(1, 1000, 400);
        // fragmentation = 1 - 400/1000 = 0.6 → NeedsCompaction
        assert!((seg.fragmentation_ratio() - 0.6).abs() < 1e-9);
        assert_eq!(seg.dead_bytes(), 600);
        assert_eq!(seg.state, CompactionState::NeedsCompaction);

        let healthy = Segment::new(2, 1000, 900);
        assert!((healthy.fragmentation_ratio() - 0.1).abs() < 1e-9);
        assert_eq!(healthy.state, CompactionState::Compacted);
    }

    // 4. CompactionJob progress tracking.
    #[test]
    fn test_job_progress() {
        let mut job = CompactionJob::new(0, vec![1, 2], 99, 1000);
        assert!((job.progress() - 0.0).abs() < 1e-9);
        assert!(!job.done);

        job.advance(500);
        assert!((job.progress() - 0.5).abs() < 1e-9);
        assert!(!job.done);

        job.advance(500);
        assert!((job.progress() - 1.0).abs() < 1e-9);
        assert!(job.done);
    }

    // 5. CompactionJob advance saturates at total_bytes.
    #[test]
    fn test_job_advance_saturates() {
        let mut job = CompactionJob::new(0, vec![1], 50, 100);
        job.advance(999);
        assert!(job.done);
        assert_eq!(job.bytes_compacted, 100);
    }

    // 6. Scheduler picks most fragmented segments.
    #[test]
    fn test_scheduler_picks_fragmented() {
        let segs = vec![
            Segment::new(0, 1000, 900), // 10% frag → Compacted
            Segment::new(1, 1000, 200), // 80% frag → NeedsCompaction
            Segment::new(2, 1000, 500), // 50% frag → InProgress
        ];
        let mut sched = CompactionScheduler::new(segs);
        let job = sched.schedule_compaction(2).unwrap();

        assert_eq!(job.source_ids, vec![1]); // only NeedsCompaction
        assert_eq!(job.total_bytes, 200);
    }

    // 7. Scheduler returns None when nothing needs compaction.
    #[test]
    fn test_scheduler_nothing_to_compact() {
        let segs = vec![
            Segment::new(0, 1000, 950),
            Segment::new(1, 1000, 900),
        ];
        let mut sched = CompactionScheduler::new(segs);
        assert!(sched.schedule_compaction(2).is_none());
    }

    // 8. Full scheduler lifecycle: schedule → finalize → stats.
    #[test]
    fn test_scheduler_lifecycle() {
        let segs = vec![
            Segment::new(0, 1000, 200), // 80% frag → NeedsCompaction
            Segment::new(1, 1000, 300), // 70% frag → NeedsCompaction
            Segment::new(2, 1000, 900), // 10% frag → Compacted
        ];
        let mut sched = CompactionScheduler::new(segs);

        let mut job = sched.schedule_compaction(4).unwrap();
        assert_eq!(job.source_ids.len(), 2);
        job.advance(job.total_bytes);
        assert!(job.done);

        sched.finalize_job(&job);

        let stats = sched.stats();
        assert_eq!(stats.total_bytes_compacted, 500);
        assert_eq!(stats.space_reclaimed, 1500); // 800 + 700 dead = 1500
        assert_eq!(stats.jobs_completed, 1);
        assert_eq!(stats.segments_merged, 2);

        // Segments should now be: original healthy + new target.
        assert_eq!(sched.segments().len(), 2);
        assert!(sched
            .segments()
            .iter()
            .any(|s| s.id == 2 && s.state == CompactionState::Compacted));
    }

    // 9. Merge compactor: k-way merge of sorted runs.
    #[test]
    fn test_merge_compactor_sorted() {
        let runs: Vec<Vec<i32>> = vec![
            vec![1, 4, 7],
            vec![2, 5, 8],
            vec![3, 6, 9],
        ];
        let merged = MergeCompactor::compact(&runs);
        assert_eq!(merged, vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }

    // 10. Merge compactor with empty runs.
    #[test]
    fn test_merge_compactor_edge_cases() {
        // No runs.
        let merged: Vec<i32> = MergeCompactor::compact(&[]);
        assert!(merged.is_empty());

        // Single run.
        let merged = MergeCompactor::compact(&[vec![1, 2, 3]]);
        assert_eq!(merged, vec![1, 2, 3]);

        // Runs with empties.
        let merged = MergeCompactor::compact(&[vec![], vec![1, 3], vec![2]]);
        assert_eq!(merged, vec![1, 2, 3]);
    }

    // 11. CompactionStats arithmetic.
    #[test]
    fn test_stats_addition() {
        let a = CompactionStats {
            total_bytes_compacted: 100,
            space_reclaimed: 40,
            jobs_completed: 1,
            segments_merged: 2,
        };
        let b = CompactionStats {
            total_bytes_compacted: 200,
            space_reclaimed: 60,
            jobs_completed: 2,
            segments_merged: 3,
        };
        let sum = a + b;
        assert_eq!(sum.total_bytes_compacted, 300);
        assert_eq!(sum.space_reclaimed, 100);
        assert_eq!(sum.jobs_completed, 3);
        assert_eq!(sum.segments_merged, 5);
        assert!((sum.compaction_ratio() - (100.0 / 300.0)).abs() < 1e-9);
    }

    // 12. Segment refresh state.
    #[test]
    fn test_segment_refresh_state() {
        let mut seg = Segment::new(0, 1000, 900);
        assert_eq!(seg.state, CompactionState::Compacted);

        // Simulate data deletion.
        seg.used_bytes = 200;
        seg.refresh_state();
        assert_eq!(seg.state, CompactionState::NeedsCompaction);
    }

    // 13. Scheduler compaction ratio stat.
    #[test]
    fn test_scheduler_compaction_ratio() {
        let segs = vec![Segment::new(0, 100, 10)];
        let mut sched = CompactionScheduler::new(segs);
        let mut job = sched.schedule_compaction(4).unwrap();
        job.advance(job.total_bytes);
        sched.finalize_job(&job);

        let stats = sched.stats();
        // Reclaimed = 90 dead bytes, compacted = 10 used bytes.
        assert!((stats.compaction_ratio() - 9.0).abs() < 1e-9);
    }
}
