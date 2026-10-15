# File and Application Search and Low-Power Indexing Design

As of October 15, 2026. Spotlight name/text search and application launching form the default search path. Recently explored media, PDFs, and code receive priority for embedding. Text, image, and audio inference now use Core ML, with USearch providing ANN search. The earlier text-only experiments and validation of the current multimodal implementation are distinguished below. Measuring energy per operation remains future work.

## Search Implemented in This Update

- File, folder, and application results are integrated into the macOS app's search panel and search window. Enter opens a file in its default app or launches an application. The existing Show in Finder action remains available.
- File names and system-extracted text are queried through Spotlight's existing index within the selected search folders. Formats without a content extractor, such as ZIP archives and spreadsheets, can also be found by name. Folder and extension exclusions apply. Hidden files, key files, files inside application bundles, some dependency directories, and cloud files that have not been downloaded are excluded.
- The application catalog is collected from `/Applications`, `/System/Applications`, and `~/Applications`. Collection does not descend into app bundles, and applications remain searchable without selected file folders or an embedding model. The catalog refreshes on search at most once every 60 seconds; deleted applications are immediately excluded from results.
- Names are scored by exact match, prefix, substring, and inclusion of all query terms, in that order. Single-character application names and Korean are supported. When a content result has the same path, the two are merged into one result while preserving PDF pages, video timestamps, and excerpts.
- `code:` retains the existing separate code search. This integration applies to the native macOS app; it is not connected to the CLI or Tauri search UI.

File discovery depends on Spotlight, so name search may omit disks with indexing disabled, locations excluded by Spotlight privacy settings, and files the system has not yet collected. Files in the existing content index remain searchable. The number of metadata candidates retrieved per query is bounded, so very broad searches do not display every file. Name/text search does not support matching entire path strings or typo correction.

The implementation is in `Sources/DigUpKit/LocalSearch.swift`, `Platform/Apple/LocalSearch+Apple.swift`, `Sources/DigUpApp/Engine.swift`, and the result UI. Metadata results do not receive vectors, and the existing SQLite index does not need to be rebuilt.

## Indexing Architecture Found in the Existing Code

| Area | Current behavior | Reason for the next improvement |
|---|---|---|
| Discovery | `Crawler` walks selected folders and selects supported formats | Name-search candidates must be separated from content-analysis candidates to find ordinary files as well |
| Change detection | Receives FSEvents and waits for changes to settle: 0.75 seconds for ordinary files, 60 seconds by default for code | Event paths are not currently passed to the helper; all selected folders are walked again |
| Change classification | Compares size and modification time; moves with the same device/inode reuse vectors | Saving or copying identical content can still trigger re-embedding |
| Initial processing | Unfinished files used within the last 30 days, then the remaining files by kind and modification order | Recent files receive priority in both the main and code indexes |
| Long files | Processes the first section of a PDF/document before handling the remainder later | Audio and video also need resumable checkpoints that bound work and memory usage |
| Power policy | The native app applies existing backfill power, thermal, and time-window policies to recent and new files too | Work/energy budgets and idle detection remain future work |
| Search | SQLite FTS5 + USearch HNSW, with exact search for small collections | Continue measuring latency, accuracy, and energy at different media-segment counts |
| Query inference | Shows name/text results first and schedules inference after typing pauses for 120 ms | Further evaluate skipping inference when name results are sufficient |

Previously, the 120 ms delay only postponed result display. This change delays the `embedAhead` call itself. New input cancels a previous search that is still waiting. A model call already in progress cannot be forcibly interrupted. Once text assets are available, the query encoder prepares in the background at app startup. Opening the panel after an idle shutdown prepares it again; name and Spotlight results do not wait for this preparation.

## Core ML and ANE — Full Multimodal Implementation

`CoreMLEmbedder` in `CoreMLRuntime` provides text, image, and audio embeddings through the same `Embedder` interface and vector source. `CoreMLTextEmbedder` runs the shared text transformer, and `CoreMLMediaEmbedder` passes vision/audio encoder outputs to that transformer. Video retains the existing frame and audio extraction. Native support requires Apple silicon and macOS 15 or later. The llama.cpp/GGUF runtime and its download/build dependencies have been removed.

[FluidInference's models](https://huggingface.co/FluidInference/embeddinggemma-2-coreml) are pinned to revision `3f978ddf92cac9aa07eb2dcf6eb36cf450f3dba4`. The 12 assets total **1,481,900,463 bytes**.

| Assets | Size |
|---|---:|
| Text model, tokenizer, and token table | 584,745,430 bytes |
| Vision model | 309,277,919 bytes |
| Audio model | 587,877,114 bytes |

Downloads resume after interruption and are published to their final paths after size and SHA-256 verification. BF16 is the token table's on-disk format; actual Core ML inputs are FP16. Compilation and device caches consume storage beyond the download size.

- All models use `MLComputeUnits.all`. ANE, GPU, and CPU are allowed, but Core ML decides operator placement and execution order. The public API **cannot enforce an ANE → GPU → CPU priority order**. [Apple compute units](https://developer.apple.com/documentation/coreml/mlcomputeunits/all)
- Existing `Prompts` are retained for ordinary search, code, and documents. The query worker uses the 128-token function for ordinary queries and the 512-token function for longer queries. Queries are capped at 512 tokens including BOS/EOS. Indexing chooses the appropriate function from 32/48/64/128/256/512 tokens.
- Long document/code segments are split into multiple windows without dropping body tokens, repeating the title prompt. Window vectors are averaged with weights based on body-token counts, then normalized. This is different from a single long attention context.
- Image budgets are 70/140/280 tokens. The bundle does not include 560/1120 graphs, so the CLI and runtime return explicit errors for those budgets. Existing aspect-ratio handling and bicubic resizing are retained.
- Audio search retains the existing 30-second windows at 25-second intervals. Core ML processes every sample in each window in pieces of up to 10 seconds, then averages by sample count and normalizes. Existing segment locations, including PDF pages, code lines, and video timestamps, are preserved.
- At startup, the app runs the common 128-token query function in the background using available text assets. During the initial download, `ModelDownload.onTextReady` calls `Engine.prepareQueryEncoder()` as soon as text assets arrive, preparing the query model while image/audio assets continue downloading. When the full download completes, `modelArrived()` keeps the prepared encoder and starts indexing. The query worker reports ready only after model-file verification and the first prediction complete. After a long idle period, the encoder exits and is prepared again when the search panel/window next opens. First use of the 512-token query function, media models, or other document functions may require separate device preparation.
- Full SHA verification is not repeated in every process when the verification receipt matches the expected asset hash/revision and device/inode/size/mtime/ctime. Changed file metadata or a missing receipt triggers verification again. This does not permit arbitrary model changes.
- Package compilation results are stored at stable paths, with publication serialized by interprocess file locks. The operating system manages Core ML's device-specific specialization cache. Actual preparation remains necessary on first use or after cache invalidation, so zero first-search latency is not guaranteed. The indexing worker retains up to six text functions it has used; the query worker retains up to two. Worker exit or `close()` releases them.

Both `auto` and `coreml` select the same native Core ML runtime. Missing assets or execution failures do not trigger fallback to a different inference engine. Queries can prepare with text assets alone, but full indexing requires text, vision, and audio assets. `--backend llama` is unsupported. Name, Spotlight, and keyword search operate independently of the model.

The new unified fingerprint includes `coreml-all-text128-windows512-vision280-audio10-v2`, the model revision, dtype, and indexing options. Vectors from an older runtime are not mixed with new query vectors. After the model has downloaded and power policy permits work, the native helper clears derived vectors and reindexes **only after successfully loading and verifying the new runtime's assets**. On failure, previous vectors remain on disk. Names and activity records are preserved. Semantic-search coverage initially decreases during reindexing and grows again as files are processed. This does not retain a separate old semantic index for an atomic replacement.

The CLI does not automatically reuse an index from a different source. Switching requires `index --reindex`; even then, new-runtime verification precedes resetting the existing index.

```sh
digup download-model --backend coreml --to .dev/models
DIGUP_MODELS="$PWD/.dev/models" digup index <folder> --index .dev/coreml-index --backend coreml
DIGUP_MODELS="$PWD/.dev/models" digup search <query> --index .dev/coreml-index --backend coreml
```

`DIGUP_COREML_MODELS` points directly to the `CoreML/` subdirectory. The default is `CoreML/` under the app's Models folder. Linux/Windows do not bundle a native model and retain only the explicitly configured `DIGUP_WORKER_DIR` protocol. The repository's `worker/` implementation targets MLX and should not itself be considered a portable model implementation. Python environments are managed with `uv`.

The implementation uses FP16 precision. [Apple's ANE Transformer description](https://machinelearning.apple.com/research/neural-engine-transformers) explicitly identifies FP16. Reading BF16 weights alone is not evidence of native BF16 arithmetic on the ANE.

### Current `.all` Compute Plans and Actual Media Validation — October 15, 2026

On an Apple M5 running macOS 27.0.1 (26A434), `MLComputePlan` was queried for the current revision's prepared `.mlmodelc` assets using the same `.all` configuration as the app. The inspection process recognized the ANE, Apple M5 GPU, and CPU. The table shows the device Core ML **reported as preferred** for each executable operation.

| Function | ANE operations | GPU operations | CPU operations | Constants without device assignments |
|---|---:|---:|---:|---:|
| Text `embed_128` | 3,862 | 0 | 0 | 3,151 |
| Image `vision_70` | 2,389 | 0 | 0 | 2,200 |
| Image `vision_140` | 2,389 | 0 | 0 | 2,200 |
| Image `vision_280` | 2,389 | 0 | 0 | 2,200 |
| Audio `main` | 2,702 | 197 | 4 | 3,912 |

Summing the audio model's estimated per-operation workload weights by device gives **97.3273% ANE**, **2.2410% GPU**, and **0.4317% CPU**. The estimated workloads for text128 and all three image functions are assigned entirely to the ANE. All unassigned entries are `const`, with no reported compute cost. GPU use is reflected not just in the allowed configuration: it is the preferred device for 197 audio-model operations, including `matmul`, `softmax`, and `tanh`.

These are **estimated placement and workloads before execution**, not measurements of runtime utilization, execution-time shares, power, or energy. CPU work for tokenization, image/audio preprocessing, vector normalization, and search-database operations is not included in these model-graph figures. This device's plan cannot be assumed to apply to other devices or operating systems. Apple defines `preferred` as the device the framework prefers for an operation, and `weight` as the operation's estimated workload relative to the full model. [Apple MLComputePlan](https://developer.apple.com/documentation/coreml/mlcomputeplan-1w21n)

On the same day, `CoreMLMediaInferenceTests.imagesAudioAndVideoFramesRemainSearchable` was actually run in a debug build with `DIGUP_TEST_COREML=1`. Finite 768-dimensional vectors produced through the image/audio encoders and shared text transformer were compared with fixed references saved using the previous runtime.

| Fixed input | Cosine similarity to reference vector |
|---|---:|
| Image budget 70, used by the video-frame path | 0.99858505 |
| Photo budget 140 | 0.99939567 |
| Screenshot budget 280 | 0.9998209 |
| Audio | 0.999867 |

That single test passed in **130.705 seconds**. This was the first execution of the new `.all` media path and included compilation, device-function preparation, and inference; debug tokenizer initialization alone took 5.024 seconds. It was not a run with all OS caches cleared, and it should not be interpreted as warm per-file inference latency or release-app startup time. Individual failure for a nonexistent image was also verified. The budget-70 case processed a fixed photo with the same budget as the video-frame path; it did not validate end-to-end video decoding.

This verifies agreement with four fixed reference inputs, not overall search quality or energy efficiency.

### Current Build Integration Validation and Startup Latency — October 15, 2026

- `DIGUP_TEST_COREML=1 DIGUP_TEST_SPOTLIGHT=1 swift test --no-parallel`: 143 tests passed, comprising 119 DigUpKit tests and 24 CoreMLRuntime tests. Coverage includes actual model references, multilingual query128 padding, 128/129/512-token boundaries, noncontiguous media-tensor strides, long documents, and the distinction between model-preparation failures and file-reading failures.
- The actual Core ML-only CLI processed temporary PDF, image, screenshot, and audio inputs; all reached done, with text/image/audio vectors stored. Validation also covered exclusion of ordinary txt files, matching query/index sources, keyword and semantic search, repeated-query caching, reload, keyword search with missing model assets, and rejection of the removed llama backend. A separate actual 12-second MPEG-TS video produced three frame vectors with timestamps.
- A release query worker was launched twice as a new process. The first ready time, including model preparation, was 19,660.8 ms; the second, using the same caches, was 782.8 ms. The first query after preparation took 8.9 ms/6.7 ms respectively, and the medians for five distinct subsequent queries were 7.2 ms/6.6 ms. The worker itself has no query-vector cache, so all were actual inference calls. This was a small sample on a device also indexing in the background, not an end-to-end UI measurement or a run with all OS caches cleared.
- `./build.sh` built and signed the release app and helper, and the actual app was relaunched. Its query encoder became ready 19.6 seconds after startup, in parallel with file/Spotlight search. The new helper was checked for the absence of llama/ggml symbols. After backing up the existing user databases, automatic reindexing with the new fingerprint was in progress, and search-vector counts increased as files were processed.

The initial cost has not been eliminated. Preparation during download and app startup pays that cost before the user types; subsequent runs reuse verification receipts and package/OS compilation caches. Searching before preparation completes may delay semantic results, but name, Spotlight, and keyword results appear first.

### Fix for Repeated Model Loading During Indexing — October 15, 2026

Actual helper logs identified the policy of caching only two of six sequence-length functions as a bottleneck. In one snapshot of 402 model loads, the 396 loads after the initial six consumed 216.2 seconds, with a median repeated-load time of 482 ms. Recreating `MLModel` instances still costs time even with the OS device-compilation cache. The indexing worker was changed to retain all six functions as they are used. Input shapes, preprocessing, model, and fingerprint were unchanged, so reindexing is unnecessary.

A release worker cycled through the same six shapes three times. Excluding the initial six preparations, the total for 12 calls fell from 8,856.1 ms to 477.5 ms, and the median fell from 682.5 ms to 22.7 ms. All 18 vectors were byte-identical in Float32. These were synthetic inputs repeatedly alternating lengths, not evidence that total file-indexing throughput improves by the same factor. Separate device preparation and the existing app's background indexing were also running. Actual-model reference and long-document regression tests passed as well.

In a separate worker-memory comparison, measured two seconds after processing the same six inputs once, RSS rose from 395.30 to 1,106.12 MiB and physical footprint from 181 to 198 MiB. Resident file mappings accounted for much of the increase. Neural Engine memory was accounted for under different categories in the two processes, so the footprint difference alone cannot establish the total cost. There is an additional memory cost, released when the worker exits. This measurement covers the text path, not peak usage with every media model loaded simultaneously.

ANN's automatic 50,000 threshold counts segment vectors per index. At the 02:37 check, the main index had 384 vectors and the code index 524; both used exact search and had no HNSW graph. The observed latency was not caused by ANN construction or training. USearch HNSW is a search data structure that adds connections between fixed embedding vectors; it does not train model weights.

### Earlier Text-Runtime Experiments — Separate from the Current Full Core ML Results

The following results were recorded when media still used the previous runtime and text ran with `.cpuAndNeuralEngine`. They do not substitute for measurements of the current `.all` placement, 128-token query preparation, or Core ML vision/audio execution.

- On the M5 at that stage, compute plans for all six text functions assigned 3,862 costed operations to the ANE and zero to the CPU/GPU. Another 3,151 unassigned operations had zero cost. This is placement information from [MLComputePlan](https://developer.apple.com/documentation/coreml/mlcomputeplan-85vdw), not a power measurement.
- Text-asset SHA-256 checks, 19 saved tokenizer fixtures, and 124 additional Unicode/special-token cases matched. Six actual Core ML tests passed at that stage.
- Across 23 inputs compared with the previous Metal text implementation, minimum cosine similarity was 0.99970439 and the mean was 0.99991528. Rankings were preserved in a small search comparison against four saved media references. This was not a full search-quality evaluation.
- The median of 20 warm Korean-query calls was 5.82 ms. Initialization in the same run took about five seconds, and the first prediction about 525 ms. These figures do not guarantee latency or energy use under the current query-preparation policy.

Tests were added for preprocessing, multi-window handling, integrity caching, and safe fingerprint transitions in the current full multimodal path. Progress integration checks using temporary PDFs/code and a fake worker also passed with the new Core ML-only CLI. Actual-model and full integration validation are recorded in the current-path sections above; energy efficiency has not yet been measured.

## Policy Centered on Explored Files — Implemented

The default scope is the user's selected ordinary folders and separately enabled code folders. Opening a file elsewhere in Finder does not automatically expand the search scope.

| Signal | Collection method | Meaning and limitations |
|---|---|---|
| Opening in the app | Recorded when an open request is made | Represents intent, not confirmed success; a failed open may still remain as a priority hint |
| Explicit Quick Look | Space/Quick Look and navigation within an open preview panel | Automatic thumbnails and merely displaying search results do not record activity |
| Finder/other apps | Spotlight query at startup, followed by NSMetadataQuery updates | Uses system-reported `kMDItemLastUsedDate`; does not guarantee detection of every in-app open |

Apple describes automatic last-used-date updates for opens through LaunchServices. Opening files inside an editor may not take that route. The app does not broaden permissions to monitor global keystrokes/UI or read private recent-document databases. [Apple metadata definitions](https://developer.apple.com/library/archive/documentation/CoreServices/Reference/MetadataAttributesRef/Reference/CommonAttrs.html)

`activity(path, last_used, source)` stores only each file's latest use within the last 30 days. Activity discovered before the initial crawl is preserved too. Equal or older timestamps are ignored, and records are reconciled with deletions, renames, selected-folder changes, and exclusions. A completed file is not re-embedded merely because it was reopened. If its content changes and it returns to pending, preserved recent-use information gives it priority.

Embedding targets are images, screenshots, video, audio, PDFs, and code. Ordinary txt/md/Office documents use Spotlight name/body search. Repository documents explicitly included in code folders remain part of the existing code context. Documents are not all read upfront just to determine their length. Extending this to long ordinary documents later will require explicit selection or metadata-based criteria.

Processing order is recently used unfinished files, including remaining pages of long PDFs, then other pending files, then other long files. Recent work in both the main and code indexes is checked before bulk work. Bulk processing checks the queue again after at most eight files and yields between processing units when new activity arrives. This does not mean only opened files are ever indexed: the remaining eligible files subsequently follow the existing background schedule.

Progress is calculated from total and pending file counts, not completion of these small batches. The helper sends current ready, pending, failed, and skipped counts after each file/batch completes, and clears batch progress when the batch ends. This fixes the simultaneous display of `100%` and a stale `0 files searchable` after only four recent files had been processed while 597 remained pending. Ready counts in Settings describe semantic-search coverage, distinct from Spotlight name search. Three progress unit tests and the `uv run --no-project scripts/test-index-progress.py .build/debug/digup` integration check verified recent batches, main/code transitions, failures/skips, later processing of long PDFs, and restarts.

Recent files do not bypass low-power, thermal, charger, or allowed-hours policies. Application names and Spotlight results remain available without a model or while processing is waiting. Activity notifications currently also request a full folder rescan; incremental updates are part of the future design below. Resuming within an audio/video file is not yet implemented either.

The native app passes `--rich-content-only --recent-first` to the helper. The CLI retains its default policy and can opt in with those flags. The schema 5→6 migration adds the activity table without recalculating existing vectors. The next crawl removes only the app's own ordinary-document index entries; those files remain searchable through Spotlight.

## Vector Database — USearch Implemented

SQLite remains the source of truth for files, segments, FTS, and FP16 vectors. The separately pinned USearch 2.24.0 C API is integrated. A newer release's NumKong package reproduced an Xcode linking problem, so 2.24.0 was selected because it provides the required C API. No external server is started. [USearch](https://github.com/unum-cloud/USearch/tree/v2.24.0)

`Searcher` builds per-modality HNSW graphs for large collections and uses exact scans for small collections or narrow filters. Vectors are 768-dimensional FP16 with cosine distance. ANN candidates are rescored using the original vectors and grouped by file. If many segments from one file consume the candidate list, more candidates are requested; if sufficient file diversity cannot be obtained, that modality falls back to exact search. FTS candidates and candidates matching the query's file-kind hints are also included and rescored.

The default crossover setting is 50,000 segments. HNSW connectivity is 32, insertion expansion is 256, and search expansion is 2,048. A kind filter selecting fewer than 2,048 segments uses an exhaustive scan. `Searcher.VectorSearchMode` offers `exact` and `approximate` to validate both paths on the same index. These settings are not claimed to be optimal for every collection.

The distinction from the current z-score method matters. Top ANN results are not treated as the full score distribution. Means and variances are estimated from query-independent samples per modality and weighted by actual per-modality segment counts. The rule that later pages of long PDFs do not change the existing search scale is preserved. This sample estimate is not identical to the previous full distribution, so it is validated against an exact baseline.

Caches are stored in `index.sqlite.ann/` and `code.sqlite.ann/`. Segment IDs, vector hashes, modalities, and source are read from a consistent SQLite snapshot to reconcile additions, changes, and deletions in the graph. ID reuse and model changes are distinguished too. Once graph files are complete, a manifest containing checksums is atomically replaced. Interprocess locks prevent loading, publication, and old-cache cleanup from conflicting. Corruption, missing files, and version mismatches trigger reconstruction or recovery through exact search. Caches are saved at most once every 30 seconds; stale caches are reconciled with the database on the next load. SQL and ANN are not independently written as two sources of truth.

After graph construction, the duplicate FP16 matrix is released. Initial construction and reload still temporarily read and hash all vectors, so transient memory and I/O costs remain. `releaseVectors` also releases the graph. Disabling code search deletes its ANN cache, and displayed storage usage includes cache sizes.

The native app's local search runs on a separate queue so that name/Spotlight results do not wait for a slow initial graph build. Semantic results are merged once ready. Cache construction currently runs in the search process; having the background helper prebuild it under power policy remains future work.

Raw 768-dimensional FP16 vectors require 1,536 bytes per segment: 153.6 MB for 100,000 segments and 1.536 GB for one million. Graphs, metadata, stored source data, and temporary buffers are additional. Capacity is measured in segments, not files. Vectors are currently stored in both SQLite and the graph, so no storage reduction is claimed.

Future improvements include a durable change log to reduce full hash-rescan costs, background generation builds, actual search evaluations across narrow scopes/languages/modalities, and energy-per-operation measurements. LanceDB has not been adopted. The initial design compared [SQLite's suitability for local storage](https://www.sqlite.org/whentouse.html) and [LanceDB's alternative indexes](https://docs.lancedb.com/indexing/vector-index); USearch is the implementation actually selected.

### ANN Measurements and Limitations

A release build on the M5 evaluated 50,000 synthetic 768-dimensional FP16 vectors. Queries and vectors were fixed-seed random unit vectors, not representative of actual file embeddings. Latencies below use prepared query vectors and exclude model inference. The p95 values come from a small sample of ten queries.

| Final settings: M32 / build 256 / search 2,048 | Result |
|---|---:|
| Initial graph construction | 227.044 s |
| First ANN query | 14.83 ms |
| Warm ANN search p95 | 14.76 ms |
| Warm exact search p95 | 13.31 ms |
| Top-10 recall | 100/100 |

At this scale and with this data, **ANN was not faster**. An earlier search-expansion-1,024 experiment measured ANN at 10.80 ms and exact search at 12.87 ms, but its 94/100 recall missed the 98% criterion and was not adopted. With the final expansion of 2,048, all five ANN tests passed without lowering that criterion. The small default test with 2,500 segments also achieved 100/100.

The automatic threshold of 50,000 is an initial setting for introducing USearch, not a validated performance crossover. It needs tuning with actual media/code embeddings and larger collections. Initial construction occupies the search queue for a long time, hence the separate name-search queue. Even when caches can be reused, full hash reconciliation still has a cost. These results alone do not justify claims of faster search or lower energy use.

## Future Indexing Design

Separate search readiness into names/system text, locally extracted text, and semantic vectors. Spotlight is already independent of the model, but fully separating local FTS extraction from vector generation remains future work.

**Update only changed paths.** Pass FSEvents paths, flags, and event IDs to the helper, merging duplicates within a small debounce window. File creation/modification/deletion should process only that file; directory changes should be scoped to the affected subtree. `FolderWatcher` currently discards flags and IDs, so that handoff must change first. Dropped events, `MustScanSubDirs`, missing event history, and mount recovery should trigger full reconciliation, and only completed event IDs should be persisted. Record crawl completeness so that a read failure is not mistaken for deletion.

**Reduce what must be recalculated.** Keep size/mtime as a fast way to identify possible changes. Hash only candidate files, and use extracted-content/segment hashes, model source, preprocessing version, and prompts together as cache keys. If a document title or path is part of its embedding prompt, include it in the key too. Reuse unchanged segments, but mark old results as stale or hide them according to policy while a changed file is being analyzed.

**Separate power budgets for search and indexing.** Show name results immediately and start semantic inference after a short typing pause. New input cancels a previous pending request, and active searches receive priority over backfill. Prefer AC, idle, and acceptable thermal conditions for backfill, while also limiting new-file throughput per second or energy per minute. Tune delays against measured p95 latency and evaluate skipping inference when names alone are sufficient.

**Handle restarts and large files.** Store file version, stage, priority, retry count, and next segment in a durable job queue. A single writer should replace results transactionally and reject stale work if a file changes again during analysis. Process long audio/video in bounded, resumable sections and cap the extraction queue's disk and memory usage.

**Tune the search index for real collections.** Evaluate USearch candidate counts, expansion, and exact-search thresholds on actual media and code collections. Record HNSW construction/reload times separately from query latency. Reuse system metadata for name search, adding a small local name index only if coverage of locations unsupported by Spotlight is required.

## Follow-Up Priorities

1. Measure missed results and lookup latency for actual file/application search and recent-use detection. Locations unsupported by Spotlight and in-app file opens remain separate problems.
2. Implement FSEvents-based incremental updates and event-loss recovery. Recent-use notifications currently trigger full walks too, so reduce that cost first.
3. Measure USearch accuracy, memory, and update costs per actual collection, and add background cache construction and a change log.
4. Extend current full Core ML placement-plan and fixed-reference validation to other devices and actual collections. Measure runtime device usage, initial preparation, and energy per file/query. Do not assume `.all` guarantees use of a particular device or lower power consumption.
5. Add segment caching and a resumable media work queue. Preserve activity-based priority and power policies across all backends.

## Earlier Implementation Validation Records

The following figures and model-execution checks describe the implementation before the full Core ML migration. They should not be confused with validation of the current build.

At that stage, running `DIGUP_TEST_COREML=1 swift test` with actual models configured passed all 121 tests: 109 DigUpKit, six LlamaRuntime, and six CoreMLRuntime tests, without skipping model-reference tests. Search/activity tests cover name normalization, excluded paths, application search without a model, page preservation, schema migration, recent-use priority, completed-file reuse, moves/deletions, policy scope, and prioritized completion of long PDFs. ANN tests cover corrupt caches, concurrent publication, segment deletion and ID reuse, model changes, kind filters, file grouping, and agreement with exact search. `./build.sh` built the release app and helper and completed local signing.

Integration checks used the actual `digup index-service`, temporary databases/PDF/code files, and a fake embedding worker. They verified recent PDF → recent code → remaining PDF order; no embedding of recent files when power permission was off; exclusion of ordinary txt files; and no re-embedding of completed files after restart. These checks ran outside the sandbox to access system services needed for file-type lookup. They validate the scheduler, not actual model inference.

Within the actual system's `/System/Applications` scope, both Spotlight and the separate application catalog found `Calculator.app`. The built app was also launched with separate settings and an empty index to verify the Calculator result, icon, preview, and open request in a dry run. Name results appeared in about 95 ms, while initial semantic search completed in about 11 seconds; the two appeared sequentially. Calculator itself was not launched. Seven files were imported into a separate temporary model folder, verifying the size and SHA-256 of the seven assets used at that stage and readiness afterward. Recent-use metadata updates from every external app and power consumption were not validated.

Later real-use logs showed embeddings taking 0–14 ms while the search stage took 10–26 seconds. Ordinary searches were repeating the same Spotlight request and waiting for stale requests from prior keystrokes and synchronous system queries. The panel and search window were changed to independently display and merge local keyword/semantic results and system results. Spotlight requests now start after a 120 ms typing pause and cancel previous requests. System collection stops after at most 300 ms and uses results gathered so far; this does not bound the duration of an individual OS metadata call itself. The application catalog compares names first and fetches file metadata only for matching applications.

Validating the release app against a copy of the actual index and separate settings measured the search stage at 2–25 ms and system results at 313–331 ms. The first query after starting a new model had an 11.9-second embedding wait; first queries in subsequent processes were around 1.1 seconds. Model initialization remains a separate cost from waiting on the search queue. For the same short queries repeatedly inferred directly in the worker, five warm calls had a median of 4.6 ms on CPU+ANE and 4.9 ms on llama.cpp Metal, without a query-vector cache. This is a small sample on that device, not representative of long inputs, concurrent indexing, or power consumption. Ten tests from `DIGUP_TEST_SPOTLIGHT=1 swift test --filter LocalSearchTests` verified cancellation, collection deadlines, preservation of application results, and actual system Spotlight responses.

The earlier hybrid model processed one PDF, two images, and one audio file into a temporary index, with ordinary txt files excluded. Query and indexing fingerprints matched, and forcing llama rejected the source mismatch. On mismatch, the service provided a warning and keyword results; a cold keyword search did not start the encoder. Repeated-query embedding caching, result preservation after reload, and keyword search without Core ML assets were also verified.

The search service was also updated to treat `/var` and `/private/var` path aliases as the same location. Search folders, excluded folders, and settings updates using both spellings were regression-tested with the latest CLI.
