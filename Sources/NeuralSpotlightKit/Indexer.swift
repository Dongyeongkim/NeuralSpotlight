import Foundation

public enum IndexerError: Error, CustomStringConvertible {
    case fingerprintMismatch(index: String, now: String)

    public var description: String {
        switch self {
        case .fingerprintMismatch(let index, let now):
            "this index was built with different settings or model:\n  index: \(index)\n  now:   \(now)\n"
                + "re-run with --reindex (vectors from different settings can't be compared)"
        }
    }
}

/// Keeps the index in step with the chosen folders, then extracts and embeds whatever is pending.
///
/// Model work goes to an `Embedder` (made on first use, freed by `closeWorker`). Extraction (ImageIO, Vision, PDFKit,
/// AVFoundation) runs on a second thread one batch ahead, so reading files and embedding them overlap (see `run`).
public final class Indexer {
    public struct SyncSummary: Sendable {
        public var added = 0, changed = 0, moved = 0, removed = 0, unchanged = 0
        public var crawl: CrawlReport
        /// Files this sync queued for embedding (new or changed), so a watcher can index exactly those first.
        public var queued: [Int64] = []
    }

    public struct Event: Sendable {
        public let done: Int
        public let total: Int
        public let path: String
        public let kind: FileKind
        public let state: String
        public let seconds: Double
        public let note: String?
    }

    let store: IndexStore
    let options: IndexOptions
    private let makeEmbedder: () throws -> any Embedder
    private var worker: (any Embedder)?
    private let scratch: URL

    /// When set, an index built with another model or other settings is emptied and embedded again (the app: a new
    /// version may bring a new model), and this hears about it. When nil, indexing stops with `fingerprintMismatch`.
    public var resetOnModelChange: ((_ indexed: String, _ now: String) -> Void)?

    /// The vector source of the embedders `makeEmbedder` makes, when known without loading one. `checkModel()` notices
    /// changes before any file needs embedding, but validates the replacement runtime before discarding vectors.
    public var embedderSource: String?
    /// The index folder, where what indexing costs on this Mac is kept (`MeasuredCosts`) for estimates. Unset: nothing
    /// is kept.
    public var costsFolder: URL?
    private var measured = MeasuredCosts()
    /// Cached FP16 text segment vectors `(prompt -> halfBlob)` from files that were modified and re-queued in `sync`,
    /// so unchanged pages, document chunks, or code stretches do not need to be re-embedded by the model.
    private var reusableTextVectors: [Int64: [String: Data]] = [:]

    public init(store: IndexStore, options: IndexOptions, makeEmbedder: @escaping () throws -> any Embedder) throws {
        self.store = store
        self.options = options
        self.makeEmbedder = makeEmbedder
        // One folder per process ($TMPDIR/NeuralSpotlight-<pid>, which the helper clears up after a process that died),
        // and one in it per indexer: the helper runs two (the main index and the code index), and each clears its own away.
        scratch = FileManager.default.temporaryDirectory
            .appendingPathComponent("NeuralSpotlight-\(ProcessInfo.processInfo.processIdentifier)")
            .appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: scratch, withIntermediateDirectories: true)
    }

    deinit {
        worker?.close()
        try? FileManager.default.removeItem(at: scratch)
    }

    /// Frees the embedding model (about 1 GB) until the next file needs it.
    public func closeWorker() {
        worker?.close()
        worker = nil
    }

    /// Stops the worker and removes the scratch folder, for a process that's about to exit (deinit won't run then).
    public func finish() {
        closeWorker()
        try? FileManager.default.removeItem(at: scratch)
    }

    public var workerIsRunning: Bool { worker != nil }

    // MARK: Sync

    /// Crawls `roots` and reconciles the index: new files are queued, changed ones re-queued, moved ones re-pathed
    /// (no re-embedding), and files that are gone or newly excluded are removed.
    public func sync(_ roots: [URL]) throws -> SyncSummary {
        let crawl = Crawler.crawl(roots, options: options)
        let existing = try store.files(under: roots.map { $0.standardizedFileURL.path })
        return try reconcile(crawl: crawl, existing: existing, roots: roots)
    }

    /// Reconciles only the specified changed file or directory `paths` within `roots`, avoiding a full crawl of
    /// untouched folders when FSEvents or recent activity reports localized changes.
    public func sync(paths: [String], roots: [URL]) throws -> SyncSummary {
        guard !paths.isEmpty else { return SyncSummary(crawl: CrawlReport()) }
        let crawl = Crawler.crawl(paths: paths, roots: roots, options: options)
        let existing = try store.files(matching: paths)
        var byPath: [String: FileRecord] = [:]
        var byInode: [String: FileRecord] = [:]
        for file in existing {
            byPath[file.path] = file
            byInode["\(file.device):\(file.inode)"] = file
        }
        // Check inode matches across the whole store in case a file was moved from/to another folder.
        for candidate in crawl.candidates where byPath[candidate.path] == nil {
            let key = "\(candidate.device):\(candidate.inode)"
            if byInode[key] == nil {
                try store.db.query(
                    "SELECT \(IndexStore.fileColumns) FROM files WHERE device = ? AND inode = ? LIMIT 1",
                    [.int(Int64(bitPattern: candidate.device)), .int(Int64(bitPattern: candidate.inode))]
                ) { row in
                    let record = IndexStore.record(row)
                    if !FileManager.default.fileExists(atPath: record.path) {
                        byInode[key] = record
                    }
                }
            }
        }
        return try reconcile(crawl: crawl, existing: existing, roots: roots, extraByInode: byInode)
    }

    private static func contentSHA256(of path: String, maxBytes: Int64 = 32 << 20) -> String? {
        guard let attrs = try? FileManager.default.attributesOfItem(atPath: path),
              let size = attrs[.size] as? Int64, size <= maxBytes,
              let data = try? Data(contentsOf: URL(fileURLWithPath: path), options: .mappedIfSafe) else { return nil }
        // Fast FNV-1a 64-bit + length digest for content identity check
        var hash: UInt64 = 0xcbf29ce484222325
        for byte in data {
            hash ^= UInt64(byte)
            hash &*= 0x100000001b3
        }
        return String(format: "%016llx:%lld", hash, size)
    }

    private func reconcile(crawl: CrawlReport, existing: [FileRecord], roots: [URL],
                           extraByInode: [String: FileRecord] = [:]) throws -> SyncSummary {
        var byPath: [String: FileRecord] = [:]
        var byInode = extraByInode
        for file in existing {
            byPath[file.path] = file
            byInode["\(file.device):\(file.inode)"] = file
        }
        let seen = Set(crawl.candidates.map(\.path))
        var movedFrom = Set<String>()
        var summary = SyncSummary(crawl: crawl)
        try store.db.transaction {
            for file in crawl.candidates {
                if let row = byPath[file.path] {
                    if row.size != file.size || abs(row.mtime - file.mtime) > 0.001 || row.kind != file.kind {
                        // If only mtime changed on a completed file and its stored content digest matches, update metadata in-place without re-embedding.
                        if row.size == file.size, row.kind == file.kind, row.state == "done",
                           let oldInfo = try? store.info(of: row.id), let oldHash = oldInfo["sha256"],
                           let newHash = Self.contentSHA256(of: file.path), oldHash == newHash {
                            try store.db.run(
                                "UPDATE files SET mtime = ?, inode = ?, device = ? WHERE id = ?",
                                [.double(file.mtime), .int(Int64(bitPattern: file.inode)),
                                 .int(Int64(bitPattern: file.device)), .int(row.id)]
                            )
                            summary.unchanged += 1
                            continue
                        }
                        if [.doc, .pdf, .code].contains(row.kind),
                           let cached = try? store.existingTextVectors(of: row.id), !cached.isEmpty {
                            reusableTextVectors[row.id] = cached
                        }
                        try store.requeue(row.id, as: file)
                        summary.changed += 1
                        summary.queued.append(row.id)
                    } else {
                        summary.unchanged += 1
                    }
                } else if let row = byInode["\(file.device):\(file.inode)"], !seen.contains(row.path),
                          !movedFrom.contains(row.path), row.size == file.size, abs(row.mtime - file.mtime) < 0.001 {
                    try store.move(row.id, to: file.path)
                    movedFrom.insert(row.path)
                    summary.moved += 1
                } else {
                    summary.queued.append(try store.insert(file))
                    summary.added += 1
                }
            }
            // A root that went away while it was being walked (a drive ejected mid-crawl) looked emptier than it is:
            // keep its rows rather than take that for "deleted".
            let vanished = roots.map { $0.standardizedFileURL.path }
                .filter { !FileManager.default.fileExists(atPath: $0) }.map { $0.hasSuffix("/") ? $0 : $0 + "/" }
            for row in existing where !seen.contains(row.path) && !movedFrom.contains(row.path)
                && !vanished.contains(where: { row.path.hasPrefix($0) })
                && !crawl.unreadablePaths.contains(where: {
                    row.path == $0 || row.path.hasPrefix($0.hasSuffix("/") ? $0 : $0 + "/")
                }) {
                reusableTextVectors.removeValue(forKey: row.id)
                try store.delete(row.id)
                summary.removed += 1
            }
        }
        return summary
    }

    // MARK: Run

    /// Processes pending files, screenshots first, newest first (`only`: just these files). Long PDFs and documents get
    /// their first step (`IndexOptions.pdfStep` pages, `docStep` chunks) and stay `partial` for `finishLongFiles`.
    /// Per-file problems are recorded on the file; a dead worker or a settings mismatch stops the run (pending files stay
    /// pending, so the next run resumes). `shouldStop` is asked between batches: pausing, or new files that should go
    /// first.
    @discardableResult
    public func run(limit: Int? = nil, only ids: Set<Int64>? = nil, shouldStop: () -> Bool = { false },
                    onEvent: (Event) -> Void = { _ in }) throws -> [String: Int] {
        let queue = try store.pending(limit: limit, only: ids)
        var states: [String: Int] = [:]
        var done = 0
        try process(Self.batches(queue).map { Job(files: $0) }, shouldStop: shouldStop) { _, job, results, seconds in
            for (offset, file) in job.files.enumerated() {
                let (state, note) = results[offset]
                states[state, default: 0] += 1
                onEvent(Event(done: done + offset + 1, total: queue.count, path: file.path, kind: file.kind,
                              state: state, seconds: seconds, note: note))
            }
            done += job.files.count
        }
        return states
    }

    /// One step of the later pass: how far it has got (`done` of the `total` pages and chunks it set out to read, in
    /// `files` files), and the file it just read more of.
    public struct Step: Sendable {
        public let done: Int
        public let total: Int
        public let files: Int        // long files not finished yet, this one included until its last step
        public let path: String
        public let kind: FileKind
        public let state: String     // "partial" until the file's last step, then "done"
        public let seconds: Double
        public let note: String?
    }

    /// The later pass: reads the rest of long PDFs, documents, audio and video (`partial` after their first step), in
    /// indexing order, a step at a time. Each step is saved as one transaction, so stopping between steps (pausing,
    /// new files going first) loses nothing, and the next pass goes on from there. It runs once nothing is pending: a
    /// book's last pages matter less than everything else's first ones, and they take a while (0.04–0.2 s a page, 2026-10-09).
    @discardableResult
    public func finishLongFiles(only ids: Set<Int64>? = nil, shouldStop: () -> Bool = { false }, onStep: (Step) -> Void = { _ in }) throws
        -> [String: Int] {
        var jobs: [Job] = []
        for file in try store.unfinished() {
            if let ids, !ids.contains(file.id) { continue }
            let info = try store.info(of: file.id)
            let size: Int = switch file.kind {
            case .pdf: options.pdfStep
            case .audio: options.audioStep
            case .video: options.videoStep
            default: options.docStep
            }
            // Where it stopped (after the last page, chunk, or second stored, should its info not say; nothing stored: the start).
            var start = try info["read_to"].flatMap(Int.init) ?? store.lastPlace(of: file.id) ?? 0
            // At least one step, which settles a file whose length was only guessed (documents cut before schema 5).
            let end = max(info["read_end"].flatMap(Int.init) ?? 0, start + 1)
            while start < end {
                jobs.append(Job(files: [file], step: start..<min(start + size, end)))
                start += size
            }
        }
        // Progress counts pages and chunks: what's read once each job is through.
        var through: [Int] = []
        for job in jobs { through.append((through.last ?? 0) + (job.step?.count ?? 0)) }
        var files = Set(jobs.map { $0.files[0].id }).count
        var states: [String: Int] = [:]
        var finished = Set<Int64>()
        try process(jobs, shouldStop: shouldStop) { index, job, results, seconds in
            let file = job.files[0]
            let (state, note) = results[0]
            if state != "partial", finished.insert(file.id).inserted {
                states[state, default: 0] += 1
                files -= 1
            }
            onStep(Step(done: through[index], total: through.last ?? 0, files: files, path: file.path, kind: file.kind,
                        state: state, seconds: seconds, note: note))
        } skip: { job in
            // Finished sooner than its steps said: a document whose length was guessed, or a file that went unreadable.
            finished.contains(job.files[0].id)
        }
        return states
    }

    /// Runs `jobs` in order until `shouldStop` (asked between jobs), passing each one's results to `committed` (with
    /// its place in `jobs`). Jobs that `skip` turns down when their turn comes are left out.
    ///
    /// Two stages overlap: while the model embeds one job, the next one is read and decoded on another thread
    /// (images scaled, screenshots OCR'd, PDF pages rendered, audio and video decoded). OCR alone is ~0.24 s per Retina
    /// screenshot, about as long as embedding it.
    private func process(_ jobs: [Job], shouldStop: () -> Bool,
                         committed: (Int, Job, [(String, String?)], Double) -> Void,
                         skip: (Job) -> Bool = { _ in false }) throws {
        var ahead: PendingBatch?
        defer {
            // A job prepared for a run that stopped early: throw its files away.
            if let ahead { try? FileManager.default.removeItem(at: ahead.wait().folder) }
            if let costsFolder, !measured.tallies.isEmpty {
                MeasuredCosts.merge(measured, into: costsFolder)
                measured = MeasuredCosts()
            }
        }
        for (index, job) in jobs.enumerated() {
            guard !shouldStop() else { break }
            let started = Date()
            let prepared = (ahead ?? prepare(job)).wait()
            ahead = index + 1 < jobs.count ? prepare(jobs[index + 1]) : nil
            defer { try? FileManager.default.removeItem(at: prepared.folder) }
            guard !skip(job) else { continue }
            // The helper's and the CLI's main thread has no run loop to drain what Foundation autoreleases while a job
            // commits: without a pool per job it all stayed until the process exited (~30 KB a text chunk).
            let modelWasRunning = worker != nil
            let results = try autoreleasepool { try commit(prepared) }
            let seconds = Date().timeIntervalSince(started)
            if modelWasRunning { measure(prepared, seconds: seconds) }   // else it's seconds of the model starting
            committed(index, job, results, seconds / Double(job.files.count))
        }
    }

    /// What a job read and how long it took (its reading and its model's work, which overlap), for `MeasuredCosts`.
    private func measure(_ prepared: Prepared, seconds: Double) {
        switch prepared.payload {
        case .images(let screenshot, let ready):
            measured.add(screenshot ? "screenshot" : "image", units: Double(ready.count), seconds: seconds)
        case .pdf(let pages, _, _, _): measured.add("pdfPage", units: Double(pages.count), seconds: seconds)
        case .doc(let chunks, _, _, _):
            // Chunks with a vector: a data dump's go to keyword search only and take next to nothing, which isn't
            // what a chunk of prose costs (the estimate can't tell them apart from a file's size).
            let worded = chunks.filter(SearchText.hasWords).count
            if worded > 0 { measured.add("docChunk", units: Double(worded), seconds: seconds) }
        case .audio(_, let read, _): measured.add("audioSecond", units: Double(read.count), seconds: seconds)
        case .video(_, _, let read, _): measured.add("videoSecond", units: Double(read.count), seconds: seconds)
        case .code(let file): measured.add("codeStretch", units: Double(file.pieces.count), seconds: seconds)
        case .nothing: break
        }
    }

    /// Files read and embedded together (images 8 at a time, other files one by one), or one step of a long file
    /// (`step`: the pages of a PDF, the chunks of a document, or the seconds of audio/video to read, from 0). The first
    /// step comes with the file.
    struct Job: Sendable {
        let files: [FileRecord]
        var step: Range<Int>?
    }

    /// Images and screenshots go to the model 8 at a time; other files one by one.
    static func batches(_ queue: [FileRecord]) -> [[FileRecord]] {
        var batches: [[FileRecord]] = []
        var index = 0
        while index < queue.count {
            var batch = [queue[index]]
            if batch[0].kind == .image || batch[0].kind == .screenshot {
                while index + batch.count < queue.count, batch.count < 8,
                      queue[index + batch.count].kind == batch[0].kind {
                    batch.append(queue[index + batch.count])
                }
            }
            batches.append(batch)
            index += batch.count
        }
        return batches
    }

    /// Compares the index with the current model and settings (`embedderSource`). An automatic rebuild first loads
    /// and validates the replacement runtime; missing/corrupt replacement assets leave the existing index intact.
    public func checkModel() throws {
        guard let embedderSource else { return }
        let fingerprint = "\(embedderSource) | \(options.fingerprint)"
        guard let indexed = try store.meta("fingerprint"), indexed != fingerprint else { return }
        guard resetOnModelChange != nil else {
            throw IndexerError.fingerprintMismatch(index: indexed, now: fingerprint)
        }
        _ = try fullWorker() // load/checksum verification succeeds before adopt resets the old index
    }

    private func adopt(_ fingerprint: String) throws {
        guard let indexed = try store.meta("fingerprint"), indexed != fingerprint else { return }
        guard let resetOnModelChange else { throw IndexerError.fingerprintMismatch(index: indexed, now: fingerprint) }
        // Everything is queued again: a run that's under way goes on with its batch, and later runs take the rest.
        reusableTextVectors.removeAll()
        try store.resetAll()
        resetOnModelChange(indexed, fingerprint)
    }

    private func fullWorker() throws -> any Embedder {
        if let worker { return worker }
        let started = try makeEmbedder()
        let fingerprint = "\(started.info.vectorSource) | \(options.fingerprint)"
        do {
            try adopt(fingerprint)
        } catch {
            started.close()
            throw error
        }
        try store.setMeta("fingerprint", fingerprint)
        try store.setMeta("vector_source", started.info.vectorSource)
        worker = started
        return started
    }

    // MARK: Prepare (any thread: files only, no database, no model)

    /// A job read from disk and decoded, ready for the model. Its scratch files live in `folder`.
    struct Prepared: Sendable {
        let batch: [FileRecord]
        let step: Range<Int>?                // a long file's later step (`Job.step`)
        let folder: URL
        var outcomes: [Int: Outcome] = [:]   // files already settled (skipped or failed while reading)
        var payload = Payload.nothing
    }

    struct Outcome: Sendable {
        let state: String
        let detail: String?
    }

    enum Payload: Sendable {
        case nothing
        case images(screenshot: Bool, ready: [ReadyImage])
        /// `read`: the pages read (from 0) of the PDF's `total`.
        case pdf(pages: [PDFExtractor.Page], read: Range<Int>, total: Int, title: String?)
        /// `chunks`: the ones in `read` (from 0) of the document's `total` (counted up to `IndexOptions.maxChunks` + 1).
        case doc(chunks: [String], read: Range<Int>, total: Int, characters: Int)
        case audio(pieces: [MediaPiece], read: Range<Int>, duration: Double)
        case video(frames: [MediaPiece], sound: [MediaPiece], read: Range<Int>, duration: Double)
        case code(CodeExtractor.File)
    }

    struct ReadyImage: Sendable {
        let slot: Int
        let image: ImageExtractor.Prepared
        let ocr: String
    }

    /// A batch being prepared on the preparation queue.
    final class PendingBatch: @unchecked Sendable {
        private let ready = DispatchSemaphore(value: 0)
        private var value: Prepared?

        fileprivate func fulfill(_ prepared: Prepared) {
            value = prepared
            ready.signal()
        }

        /// Blocks until it's ready. Call once.
        func wait() -> Prepared {
            ready.wait()
            return value!
        }
    }

    private let preparing = DispatchQueue(label: "NeuralSpotlight.prepare", qos: .utility)
    private var jobsPrepared = 0   // names the scratch folders

    /// The file being read (extracted) right now, and since when: for a watchdog. Apple's frameworks have run away on
    /// some files (PDF text extraction growing by gigabytes a minute, never returning), and nothing stops them from
    /// outside but ending the process, so it has to know which file that was. One value for the whole process, since
    /// files are read one at a time.
    public static var nowReading: (path: String, since: Date)? { readingLock.withLock { reading } }
    nonisolated(unsafe) private static var reading: (path: String, since: Date)?
    private static let readingLock = NSLock()

    private static func startReading(_ path: String) {
        readingLock.withLock { reading = (path, Date()) }
    }

    private func prepare(_ job: Job) -> PendingBatch {
        let pending = PendingBatch()
        jobsPrepared += 1
        let (options, folder) = (options, scratch.appendingPathComponent("batch-\(jobsPrepared)"))
        preparing.async {
            pending.fulfill(Self.prepare(job, options: options, into: folder))
        }
        return pending
    }

    static func prepare(_ job: Job, options: IndexOptions, into folder: URL) -> Prepared {
        let batch = job.files
        var prepared = Prepared(batch: batch, step: job.step, folder: folder)
        let url = URL(fileURLWithPath: batch[0].path)
        startReading(batch[0].path)
        defer { readingLock.withLock { reading = nil } }
        do {
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
            switch batch[0].kind {
            case .file, .folder, .application:
                for slot in batch.indices { prepared.outcomes[slot] = Outcome(state: "done", detail: nil) }
            case .image, .screenshot:
                let screenshot = batch[0].kind == .screenshot
                var ready: [ReadyImage] = []
                for (slot, file) in batch.enumerated() {
                    let url = URL(fileURLWithPath: file.path)
                    startReading(file.path)
                    do {
                        let image = try ImageExtractor.prepare(url, maxPixel: screenshot ? 1536 : 1024,
                                                               minSide: options.minImageSide, to: folder,
                                                               png: screenshot)
                        let ocr = screenshot ? OCR.text(ofImageAt: url) : ""
                        ready.append(ReadyImage(slot: slot, image: image, ocr: ocr))
                    } catch let error as ExtractError {
                        if case .tooSmall = error {
                            prepared.outcomes[slot] = Outcome(state: "skipped", detail: "\(error)")
                        } else {
                            prepared.outcomes[slot] = Outcome(state: "failed", detail: "\(error)")
                        }
                    }
                }
                prepared.payload = .images(screenshot: screenshot, ready: ready)
            case .pdf:
                // Long PDFs (books) get their first pages now and the rest after everything else.
                var wanted = 0..<0
                let read = try PDFExtractor.pages(of: url, range: { total in
                    wanted = (job.step ?? (total > options.bigPDFPages ? 0..<options.pdfStep : 0..<options.maxPages))
                        .clamped(to: 0..<min(total, options.maxPages))
                    return wanted
                }, minText: options.minPageText, renderPixels: 1400, to: folder)
                prepared.payload = .pdf(pages: read.pages, read: wanted, total: read.total, title: read.title)
            case .doc:
                // The chunks are cut the same way every time, so a later step goes on where the last one stopped.
                let text = try DocExtractor.text(of: url)
                let chunks = DocExtractor.chunks(text, limit: options.maxChunks + 1)
                let read = (job.step ?? 0..<options.docStep).clamped(to: 0..<min(chunks.count, options.maxChunks))
                prepared.payload = .doc(chunks: Array(chunks[read]), read: read, total: chunks.count,
                                        characters: text.count)
            case .audio:
                let (window, hop) = (options.audioWindow, options.audioHop)
                let step = job.step
                let (bigAudio, audioStep, maxMedia) = (options.bigAudioSeconds, options.audioStep, options.maxMediaSeconds)
                let (pieces, duration) = try runBlocking {
                    try await AudioExtractor.windows(of: url, window: window, hop: hop, range: { total in
                        let totalSec = max(0, Int(ceil(total)))
                        let endCap = min(totalSec, maxMedia)
                        let r = (step ?? (totalSec > bigAudio ? 0..<audioStep : 0..<maxMedia)).clamped(to: 0..<endCap)
                        return Double(r.lowerBound)..<min(total, Double(r.upperBound))
                    }, to: folder)
                }
                let totalSec = max(0, Int(ceil(duration)))
                let endCap = min(totalSec, maxMedia)
                let wanted = (step ?? (totalSec > bigAudio ? 0..<audioStep : 0..<maxMedia)).clamped(to: 0..<endCap)
                prepared.payload = .audio(pieces: pieces, read: wanted, duration: duration)
            case .video:
                let (interval, window, hop) = (options.frameInterval, options.audioWindow, options.audioHop)
                let step = job.step
                let (bigVideo, videoStep, maxMedia) = (options.bigVideoSeconds, options.videoStep, options.maxMediaSeconds)
                let (frames, duration, hasAudio) = try runBlocking {
                    try await VideoExtractor.keyframes(of: url, every: interval, maxPixel: 1024, range: { total in
                        let totalSec = max(0, Int(ceil(total)))
                        let endCap = min(totalSec, maxMedia)
                        let r = (step ?? (totalSec > bigVideo ? 0..<videoStep : 0..<maxMedia)).clamped(to: 0..<endCap)
                        return Double(r.lowerBound)..<min(total, Double(r.upperBound))
                    }, to: folder)
                }
                let totalSec = max(0, Int(ceil(duration)))
                let endCap = min(totalSec, maxMedia)
                let wanted = (step ?? (totalSec > bigVideo ? 0..<videoStep : 0..<maxMedia)).clamped(to: 0..<endCap)
                let sound = hasAudio
                    ? try runBlocking {
                        try await AudioExtractor.windows(of: url, window: window, hop: hop, range: { total in
                            Double(wanted.lowerBound)..<min(total, Double(wanted.upperBound))
                        }, to: folder)
                    }.pieces
                    : []
                prepared.payload = .video(frames: frames, sound: sound, read: wanted, duration: duration)
            case .code:
                prepared.payload = .code(try CodeExtractor.read(url, options: options))
            }
        } catch let error as ExtractError {
            if case .unsupported = error {
                prepared.outcomes[0] = Outcome(state: "unsupported", detail: "\(error)")
            } else if case .leftOut = error {
                prepared.outcomes[0] = Outcome(state: "skipped", detail: "\(error)")
            } else {
                for slot in batch.indices where prepared.outcomes[slot] == nil {
                    prepared.outcomes[slot] = Outcome(state: "failed", detail: "\(error)")
                }
            }
            prepared.payload = .nothing
        } catch {
            for slot in batch.indices where prepared.outcomes[slot] == nil {
                prepared.outcomes[slot] = Outcome(state: "failed", detail: "\(error)")
            }
            prepared.payload = .nothing
        }
        return prepared
    }

    // MARK: Commit (the indexing thread: the model and the database)

    private func commit(_ prepared: Prepared) throws -> [(String, String?)] {
        let batch = prepared.batch
        let later = prepared.step != nil
        if let step = prepared.step, let outcome = prepared.outcomes[0] {
            return [try stopReading(batch[0], at: step.lowerBound, because: outcome.detail ?? outcome.state)]
        }
        var results = [(String, String?)](repeating: ("failed", nil), count: batch.count)
        for (slot, outcome) in prepared.outcomes {
            try store.finish(batch[slot].id, state: outcome.state, detail: outcome.detail)
            results[slot] = (outcome.state, outcome.detail)
        }
        do {
            switch prepared.payload {
            case .nothing:
                break
            case .images(let screenshot, let ready):
                try images(batch, screenshot: screenshot, ready: ready, results: &results)
            case .pdf(let pages, let read, let total, let title):
                results[0] = try pdf(batch[0], pages: pages, read: read, total: total, title: title, later: later)
            case .doc(let chunks, let read, let total, let characters):
                results[0] = try doc(batch[0], chunks: chunks, read: read, total: total, characters: characters,
                                     later: later)
            case .audio(let pieces, let read, let duration):
                results[0] = try audio(batch[0], pieces: pieces, read: read, duration: duration, later: later)
            case .video(let frames, let sound, let read, let duration):
                results[0] = try video(batch[0], frames: frames, sound: sound, read: read, duration: duration, later: later)
            case .code(let code):
                results[0] = try self.code(batch[0], code)
            }
        } catch let error as WorkerError {
            throw error
        } catch let error as IndexerError {
            throw error
        } catch {
            if let step = prepared.step { return [try stopReading(batch[0], at: step.lowerBound, because: "\(error)")] }
            for (slot, file) in batch.enumerated() where prepared.outcomes[slot] == nil {
                try store.finish(file.id, state: "failed", detail: "\(error)")
                results[slot] = ("failed", "\(error)")
            }
        }
        return results
    }

    /// A long file that can't be read any further: it keeps the pages, chunks or seconds read so far, which stay
    /// searchable, and counts as done rather than failed or waiting to be tried again and again.
    private func stopReading(_ file: FileRecord, at place: Int, because problem: String) throws -> (String, String?) {
        var info = try store.info(of: file.id)
        info["read_to"] = nil
        info["read_end"] = nil
        let unit: String = switch file.kind {
        case .pdf: "page"
        case .audio, .video: "second"
        default: "chunk"
        }
        try store.finish(file.id, state: "done", detail: "read to \(unit) \(place), not after: \(problem)", info: info)
        return ("done", "stopped before \(unit) \(place + 1): \(problem)")
    }

    private func images(_ batch: [FileRecord], screenshot: Bool, ready: [ReadyImage],
                        results: inout [(String, String?)]) throws {
        guard !ready.isEmpty else { return }
        let worker = try fullWorker()
        let vectors = try worker.embedImages(ready.map(\.image.path),
                                             budget: screenshot ? options.screenshotBudget : options.photoBudget)
        try store.db.transaction {
            for (position, item) in ready.enumerated() {
                let file = batch[item.slot]
                guard let vector = vectors[position].vector else {
                    try store.finish(file.id, state: "failed", detail: vectors[position].error)
                    results[item.slot] = ("failed", vectors[position].error)
                    continue
                }
                try store.addSegment(file: file.id, kind: .image, modality: .image, vector: vector)
                // The words on screen are for keyword search: codes, names and errors typed exactly. Meaning comes from
                // the picture alone; a vector of the words made screenshots outrank better matches (2026-10-08).
                if !item.ocr.isEmpty {
                    try store.addSegment(file: file.id, kind: .ocr, modality: .text, excerpt: excerpt(item.ocr),
                                         text: item.ocr)
                }
                var info = ["width": "\(item.image.width)", "height": "\(item.image.height)"]
                if let taken = item.image.taken { info["taken"] = taken }
                if screenshot { info["ocr_chars"] = "\(item.ocr.count)" }
                if let sha = Self.contentSHA256(of: file.path) { info["sha256"] = sha }
                try store.finish(file.id, state: "done", info: info)
                results[item.slot] = ("done", screenshot ? "ocr \(item.ocr.count) chars" : nil)
            }
        }
    }

    /// Pages `read` of a PDF (all of a short one, a step of a long one). A long one stays `partial` with `read_to` and
    /// `read_end` (pages, from 0) in its info until its last step.
    private func pdf(_ file: FileRecord, pages: [PDFExtractor.Page], read: Range<Int>, total: Int, title: String?,
                     later: Bool) throws -> (String, String?) {
        let end = min(total, options.maxPages)
        let unread = read.upperBound < end
        guard !pages.isEmpty || later || unread else {
            try store.finish(file.id, state: "skipped", detail: "no pages")
            return ("skipped", "no pages")
        }
        let title = title ?? URL(fileURLWithPath: file.path).deletingPathExtension().lastPathComponent
        let textPages = pages.filter { $0.imagePath == nil }
        // A page that's mostly digits (a table, a data listing) gets keyword search only.
        let wordPages = textPages.filter { SearchText.hasWords($0.text) }
        let imagePages = pages.filter { $0.imagePath != nil }
        let garbledPages = pages.filter { $0.garbled != nil }   // rendered too: among `imagePages`
        let cached = reusableTextVectors[file.id] ?? [:]
        if !unread { reusableTextVectors.removeValue(forKey: file.id) }
        let uncachedWordPages = wordPages.filter { cached[$0.text] == nil }
        var textVectors: [Int: [Float]] = [:]   // by page number
        var imageVectors: [Embedding] = []
        var garbledVectors: [[Float]] = []
        if !uncachedWordPages.isEmpty || !imagePages.isEmpty || !garbledPages.isEmpty {
            let worker = try fullWorker()
            if !uncachedWordPages.isEmpty {
                let vectors = try worker.embedTexts(uncachedWordPages.map {
                    Prompts.document(title: title, text: String($0.text.prefix(4000)))
                })
                textVectors = Dictionary(uniqueKeysWithValues: zip(uncachedWordPages.map(\.number), vectors))
            }
            if !imagePages.isEmpty {
                imageVectors = try worker.embedImages(imagePages.compactMap(\.imagePath), budget: options.pageBudget)
            }
            if !garbledPages.isEmpty {
                garbledVectors = try worker.embedTexts(garbledPages.map {
                    Prompts.document(title: title, text: String($0.garbled!.prefix(4000)))
                })
            }
        }
        try store.db.transaction {
            // Meaning only: the garbled words are never shown or matched against what you type.
            for (page, vector) in zip(garbledPages, garbledVectors) {
                try store.addSegment(file: file.id, kind: .page, modality: .text, loc: Double(page.number), vector: vector)
            }
            for page in textPages {
                let half = SearchText.hasWords(page.text) ? cached[page.text] : nil
                try store.addSegment(file: file.id, kind: .page, modality: .text, loc: Double(page.number),
                                     excerpt: excerpt(page.text), text: page.text, vector: textVectors[page.number],
                                     halfVector: half)
            }
            for (page, embedding) in zip(imagePages, imageVectors) {
                try store.addSegment(file: file.id, kind: .pageImage, modality: .image, loc: Double(page.number),
                                     excerpt: page.text.isEmpty ? nil : excerpt(page.text), text: page.text,
                                     vector: embedding.vector)
            }
            var info = later ? try store.info(of: file.id) : [:]
            info["pages"] = "\(total)"
            info["scanned_pages"] = "\((info["scanned_pages"].flatMap(Int.init) ?? 0) + imagePages.count)"
            info["read_to"] = unread ? "\(read.upperBound)" : nil
            info["read_end"] = unread ? "\(end)" : nil
            if !unread, let sha = Self.contentSHA256(of: file.path) { info["sha256"] = sha }
            let detail = unread ? "read to page \(read.upperBound) of \(total)"
                : end < total ? "the first \(end) of \(total) pages" : nil
            try store.finish(file.id, state: unread ? "partial" : "done", detail: detail, info: info)
        }
        let ocr = pages.filter(\.ocr).count
        let span = later ? "pages \(read.lowerBound + 1)–\(read.upperBound) of \(total)" : "\(pages.count)/\(total) pages"
        return (unread ? "partial" : "done", "\(span), \(imagePages.count) as images" + (ocr > 0 ? ", \(ocr) read by OCR" : ""))
    }

    /// Chunks `read` of a document (all of a short one, a step of a long one), the way `pdf` reads pages.
    private func doc(_ file: FileRecord, chunks: [String], read: Range<Int>, total: Int, characters: Int, later: Bool)
        throws -> (String, String?) {
        guard total > 0 else {
            reusableTextVectors.removeValue(forKey: file.id)
            try store.finish(file.id, state: "skipped", detail: "no text")
            return ("skipped", "no text")
        }
        let end = min(total, options.maxChunks)
        let unread = read.upperBound < end
        let cached = reusableTextVectors[file.id] ?? [:]
        if !unread { reusableTextVectors.removeValue(forKey: file.id) }
        let title = URL(fileURLWithPath: file.path).deletingPathExtension().lastPathComponent
        // Text that's mostly digits (a data dump saved as .txt) gets keyword search only.
        let worded = chunks.indices.filter { SearchText.hasWords(chunks[$0]) }
        let uncachedWorded = worded.filter { cached[chunks[$0]] == nil }
        let vectors = uncachedWorded.isEmpty ? [] : try fullWorker().embedTexts(uncachedWorded.map {
            Prompts.document(title: title, text: chunks[$0])
        })
        let vector = Dictionary(uniqueKeysWithValues: zip(uncachedWorded, vectors))
        try store.db.transaction {
            for (offset, chunk) in chunks.enumerated() {
                let half = SearchText.hasWords(chunk) ? cached[chunk] : nil
                try store.addSegment(file: file.id, kind: .chunk, modality: .text,
                                     loc: Double(read.lowerBound + offset + 1), excerpt: excerpt(chunk), text: chunk,
                                     vector: vector[offset], halfVector: half)
            }
            var info = ["chunks": "\(end)", "chars": "\(characters)"]
            info["read_to"] = unread ? "\(read.upperBound)" : nil
            info["read_end"] = unread ? "\(end)" : nil
            if !unread, let sha = Self.contentSHA256(of: file.path) { info["sha256"] = sha }
            let detail = unread ? "read to chunk \(read.upperBound) of \(end)" : end < total ? "the first \(end) chunks" : nil
            try store.finish(file.id, state: unread ? "partial" : "done", detail: detail, info: info)
        }
        let span = later ? "chunks \(read.lowerBound + 1)–\(read.upperBound) of \(end)"
            : unread ? "\(chunks.count) of \(end) chunks" : "\(chunks.count) chunks"
        return (unread ? "partial" : "done",
                span + (worded.count < chunks.count ? ", \(chunks.count - worded.count) for keywords only" : ""))
    }

    /// A code file's stretches of lines (a notebook's of cells), each embedded with the file's path in its repo as its
    /// title, and kept with its lines for keyword search and for opening at the place.
    private func code(_ file: FileRecord, _ code: CodeExtractor.File) throws -> (String, String?) {
        let cached = reusableTextVectors.removeValue(forKey: file.id) ?? [:]
        guard !code.pieces.isEmpty else {
            try store.finish(file.id, state: "skipped", detail: "no code")
            return ("skipped", "no code")
        }
        // Lines that are mostly numbers and symbols (a table of constants) get keyword search only.
        let worded = code.pieces.indices.filter { SearchText.hasWords(code.pieces[$0].text) }
        let uncachedWorded = worded.filter { cached[code.pieces[$0].text] == nil }
        let vectors = uncachedWorded.isEmpty ? [] : try fullWorker().embedTexts(uncachedWorded.map {
            Prompts.document(title: code.title, text: code.pieces[$0].text)
        })
        let vector = Dictionary(uniqueKeysWithValues: zip(uncachedWorded, vectors))
        try store.db.transaction {
            for (offset, piece) in code.pieces.enumerated() {
                let half = SearchText.hasWords(piece.text) ? cached[piece.text] : nil
                try store.addSegment(file: file.id, kind: code.cells == nil ? .lines : .cells, modality: .text,
                                     loc: Double(piece.first), locEnd: Double(piece.last),
                                     excerpt: CodeExtractor.summary(piece.text), text: piece.text, vector: vector[offset],
                                     halfVector: half)
            }
            var info = ["lines": "\(code.lines)"]
            info["cells"] = code.cells.map(String.init)
            info["repo"] = code.repo
            if let sha = Self.contentSHA256(of: file.path) { info["sha256"] = sha }
            try store.finish(file.id, state: "done", info: info)
        }
        return ("done", "\(code.pieces.count) stretches, \(code.cells.map { "\($0) cells" } ?? "\(code.lines) lines")"
                + (worded.count < code.pieces.count ? ", \(code.pieces.count - worded.count) for keywords only" : ""))
    }

    private func audio(_ file: FileRecord, pieces: [MediaPiece], read: Range<Int>, duration: Double,
                       later: Bool = false) throws -> (String, String?) {
        let totalSec = max(0, Int(ceil(duration)))
        let end = min(totalSec, options.maxMediaSeconds)
        let unread = read.upperBound < end && end > 0
        guard !pieces.isEmpty || later || unread else {
            try store.finish(file.id, state: "skipped", detail: "silent or shorter than 2 s")
            return ("skipped", "silent or too short")
        }
        let vectors = pieces.isEmpty ? [] : try fullWorker().embedAudio(pieces.map(\.path))
        try store.db.transaction {
            for (piece, embedding) in zip(pieces, vectors) {
                try store.addSegment(file: file.id, kind: .audio, modality: .audio, loc: piece.start, locEnd: piece.end,
                                     vector: embedding.vector)
            }
            var info = later ? try store.info(of: file.id) : [:]
            info["duration"] = String(format: "%.1f", duration)
            info["read_to"] = unread ? "\(read.upperBound)" : nil
            info["read_end"] = unread ? "\(end)" : nil
            let detail = unread ? "read to \(read.upperBound) s of \(Int(duration)) s"
                : end < totalSec ? "the first \(end) s of \(Int(duration)) s" : nil
            try store.finish(file.id, state: unread ? "partial" : "done", detail: detail, info: info)
        }
        return (unread ? "partial" : "done", "\(pieces.count) windows, \(Int(duration)) s")
    }

    private func video(_ file: FileRecord, frames: [MediaPiece], sound: [MediaPiece], read: Range<Int>, duration: Double,
                       later: Bool = false) throws -> (String, String?) {
        let totalSec = max(0, Int(ceil(duration)))
        let end = min(totalSec, options.maxMediaSeconds)
        let unread = read.upperBound < end && end > 0
        guard !frames.isEmpty || !sound.isEmpty || later || unread else {
            try store.finish(file.id, state: "skipped", detail: "no frames or sound")
            return ("skipped", "no frames or sound")
        }
        var frameVectors: [Embedding] = []
        var soundVectors: [Embedding] = []
        if !frames.isEmpty || !sound.isEmpty {
            let worker = try fullWorker()
            if !frames.isEmpty { frameVectors = try worker.embedImages(frames.map(\.path), budget: options.frameBudget) }
            if !sound.isEmpty { soundVectors = try worker.embedAudio(sound.map(\.path)) }
        }
        try store.db.transaction {
            for (frame, embedding) in zip(frames, frameVectors) {
                try store.addSegment(file: file.id, kind: .frame, modality: .image, loc: frame.start,
                                     locEnd: frame.end, vector: embedding.vector)
            }
            for (piece, embedding) in zip(sound, soundVectors) {
                try store.addSegment(file: file.id, kind: .audio, modality: .audio, loc: piece.start,
                                     locEnd: piece.end, vector: embedding.vector)
            }
            var info = later ? try store.info(of: file.id) : [:]
            info["duration"] = String(format: "%.1f", duration)
            info["frames"] = "\((info["frames"].flatMap(Int.init) ?? 0) + frames.count)"
            info["sound_windows"] = "\((info["sound_windows"].flatMap(Int.init) ?? 0) + sound.count)"
            info["read_to"] = unread ? "\(read.upperBound)" : nil
            info["read_end"] = unread ? "\(end)" : nil
            let detail = unread ? "read to \(read.upperBound) s of \(Int(duration)) s"
                : end < totalSec ? "the first \(end) s of \(Int(duration)) s" : nil
            try store.finish(file.id, state: unread ? "partial" : "done", detail: detail, info: info)
        }
        return (unread ? "partial" : "done", "\(frames.count) frames, \(sound.count) sound windows, \(Int(duration)) s")
    }
}
