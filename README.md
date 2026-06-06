# oxide-compactor

**Background compaction for GPU data structures with ternary compaction state.**

A Rust library that implements merge-sort compaction of sorted runs in GPU memory, with a scheduler that uses a three-valued (ternary) trigger to decide which segments need attention — and which are healthy enough to skip.

---

## Why This Matters

GPU workloads that maintain sorted data structures (columnar stores, LSM-tree-style indexes, spatial indexes) accumulate **fragmentation** over time as records are inserted and deleted. Dead space inside GPU segments wastes VRAM, degrades memory locality, and increases the cost of scans.

Compaction reclaims that space by merging partially-full sorted runs into dense, contiguous segments. But compaction itself costs GPU bandwidth and latency — you don't want to compact *too* aggressively, nor ignore segments until they're nearly empty.

**oxide-compactor** solves this with a **ternary compaction trigger** that classifies every segment into one of three states:

| State | Value | Meaning |
|-------|-------|---------|
| **Compacted** | +1 | Healthy — plenty of live data, skip |
| **In Progress** | 0 | Borderline — watch, maybe compact soon |
| **Needs Compaction** | −1 | Fragmented — compact now |

This gives you fine-grained control: compact aggressively only when fragmentation is high, keep an eye on borderline segments, and leave healthy ones alone.

---

## Architecture

```
┌──────────────┐     ┌──────────────────────┐     ┌──────────────────┐
│  Segment(s)  │────▶│  CompactionScheduler │────▶│  CompactionJob   │
│  (GPU mem)   │     │  (picks by frag %)   │     │  (tracks work)   │
└──────────────┘     └──────────────────────┘     └──────┬───────────┘
                              │                          │
                     ternary trigger              ┌──────▼───────────┐
                    (+1 / 0 / −1)                 │  MergeCompactor  │
                                                   │  (k-way merge)   │
                                                   └──────┬───────────┘
                                                          │
                                                   ┌──────▼───────────┐
                                                   │  CompactionStats │
                                                   │  (cumulative)    │
                                                   └──────────────────┘
```

### Key Types

#### `CompactionState`

The ternary enum. Classifies a segment from its **fragmentation ratio** (dead space ÷ capacity):

- `< 0.3` → `Compacted` (+1)
- `0.3 – 0.6` → `InProgress` (0)
- `≥ 0.6` → `NeedsCompaction` (−1)

Thresholds are chosen so that a segment that's more than 60% dead space gets immediate attention, while one under 30% is left alone.

#### `Segment`

A sorted run in GPU memory. Tracks:

- **id** — unique identifier
- **capacity** — total bytes allocated
- **used_bytes** — live data bytes
- **state** — current `CompactionState`

Automatically computes fragmentation ratio and dead (reclaimable) bytes.

#### `CompactionJob`

Tracks a single compaction operation:

- Source segments being merged
- Target segment for the output
- Progress (bytes compacted / total bytes)
- Completion flag

#### `CompactionScheduler`

The brain. Manages a collection of segments and:

1. **Refreshes** ternary state on every segment
2. **Selects** segments with `NeedsCompaction` state, sorted by fragmentation ratio (worst first)
3. **Creates** a `CompactionJob` with a fresh target segment
4. **Finalizes** completed jobs — removes source segments, adds the compacted target, updates stats

#### `MergeCompactor`

Performs a **k-way merge sort** of multiple sorted runs into a single sorted output. Uses a min-heap for O(n log k) performance.

#### `CompactionStats`

Cumulative accounting:

- **total_bytes_compacted** — bytes processed through compaction
- **space_reclaimed** — dead bytes recovered
- **jobs_completed** — number of compaction rounds
- **segments_merged** — source segments consumed
- **compaction_ratio** — `reclaimed / compacted` (higher = more efficient)

---

## Usage

Add to `Cargo.toml`:

```toml
[dependencies]
oxide-compactor = "0.1"
```

### Create segments and schedule compaction

```rust
use oxide_compactor::{CompactionScheduler, Segment};

fn main() {
    // Simulate GPU segments with varying fragmentation.
    let segments = vec![
        Segment::new(0, 1024 * 1024, 900_000),  // ~14% frag → Compacted
        Segment::new(1, 1024 * 1024, 200_000),  // ~81% frag → NeedsCompaction
        Segment::new(2, 1024 * 1024, 400_000),  // ~62% frag → NeedsCompaction
        Segment::new(3, 1024 * 1024, 600_000),  // ~43% frag → InProgress
    ];

    let mut scheduler = CompactionScheduler::new(segments);

    // Pick up to 4 segments to compact.
    if let Some(mut job) = scheduler.schedule_compaction(4) {
        println!("Compacting {} segments ({} bytes)",
                 job.source_ids.len(), job.total_bytes);

        // Simulate compaction work.
        job.advance(job.total_bytes);

        // Finalize: remove sources, add target, update stats.
        scheduler.finalize_job(&job);
    }

    let stats = scheduler.stats();
    println!("Bytes compacted: {}", stats.total_bytes_compacted);
    println!("Space reclaimed: {}", stats.space_reclaimed);
    println!("Compaction ratio: {:.2}", stats.compaction_ratio());
}
```

### Merge-sort compaction of sorted runs

```rust
use oxide_compactor::MergeCompactor;

let runs: Vec<Vec<i32>> = vec![
    vec![1, 4, 7, 10],
    vec![2, 5, 8],
    vec![3, 6, 9, 11, 12],
];

let merged = MergeCompactor::compact(&runs);
assert_eq!(merged, vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
```

### Ternary state directly

```rust
use oxide_compactor::CompactionState;

let state = CompactionState::from_fragmentation(0.75);
assert_eq!(state, CompactionState::NeedsCompaction);
assert_eq!(state.value(), -1);
```

---

## How It Fits the Oxide Stack

oxide-compactor is designed as a **leaf dependency** for GPU-accelerated data systems in the oxide ecosystem:

- **No GPU dependency at the library level** — the scheduling logic, ternary triggers, and statistics are pure Rust. Actual GPU memory management is handled by the caller.
- **Plug-in ready** — wrap `CompactionScheduler` in your GPU buffer manager and call `schedule_compaction()` from a background thread.
- **Testable** — all logic is deterministic and unit-testable without a GPU.
- **Composable** — `MergeCompactor` works on any `T: Ord + Clone`, so you can test merge logic with simple integers before wiring it to GPU kernels.

In a full GPU data pipeline, oxide-compactor sits between the **mutation layer** (inserts/deletes that create fragmentation) and the **storage layer** (dense sorted runs that feeds scans and queries). The scheduler runs in a background task, periodically checking fragmentation ratios and triggering compaction only when the ternary state warrants it.

---

## License

MIT
