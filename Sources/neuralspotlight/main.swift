import NeuralSpotlightKit
import Foundation

// NeuralSpotlight's developer CLI. Every command works on an explicit index folder, so tests never touch the real one.

#if !os(Windows)
signal(SIGPIPE, SIG_IGN)   // a dead worker must surface as an error, not kill us (Windows has no SIGPIPE)
#endif

let usage = """
    neuralspotlight: search your files by meaning (NeuralSpotlight developer CLI)

      neuralspotlight estimate <folder>... [--json]     what would be indexed, and how long the first pass takes
      neuralspotlight index <folder>... [--limit N]     sync the index with these folders, embed what's new, then the rest
                                              of long PDFs and documents
      neuralspotlight search <words>... [--kind K,K] [-n 10] [--json]
                                              "code: <words>" searches the code index
      neuralspotlight status                            what's in the index
      neuralspotlight speed [--model [--if-needed] [--when-quiet S] [--tries N]] [--json]
                                              this machine's speed next to the M4 Pro behind the estimates, and
                                              what indexing has measured here (costs.json in the index folder);
                                              --model times the active runtime and keeps its measurements
      neuralspotlight eval <queries.json> [-n 10]       recall of known answers (see evals/; "code: …" queries search code)
      neuralspotlight index-service <folder>... [--code-root DIR]... [--code-exclude DIR]... [--backfill] [--prune]
                          [--memory-limit MB] [--time-limit S]
                          [--rich-content-only] [--recent-first] [--sync-only]
                                              the app's indexing helper (JSON lines; see IndexService); it stops
                                              on a file whose reading passes the limits (see Watchdog)
      neuralspotlight worker [--text-only]              the model as a process (JSON lines; the app's query encoder)
      neuralspotlight search-service <folder>... [--code-root DIR]... [--encoder-idle S]
                                              the searcher as a process, index open and encoder warm (JSON lines;
                                              see SearchService): for shells that can't link NeuralSpotlightKit
      neuralspotlight download-model [--to DIR]         the model files, from the Hugging Face cache or downloaded with
                                              resume, checked (JSON lines: see DownloadModel)

    Options:
      --index DIR          index folder (default: NeuralSpotlight's Application Support folder; see `neuralspotlight status`)
      --backend MODE       auto (default) or coreml; both use Core ML (ANE, GPU and CPU allowed)
                           Core ML needs Apple silicon/macOS 15+; download-model installs its pinned assets
      --code               estimate, index or show the code index of these folders (code.sqlite in the index
                           folder): source files, their repos' docs and notebooks, minus what git ignores
      --kind K,K           screenshot, image, pdf, doc, audio, video
      --include-code       also index loose source code and data files as documents (outside code projects)
      --include-datasets   don't skip folders that look like datasets
      --exclude DIR        don't index this folder (repeatable)
      --exclude-ext EXT    don't index files with this extension (repeatable, or comma-separated: "heic,txt")
      --reindex            drop vectors and embed everything again (after changing settings or model)
      --zgate Z            how far a meaning match must stand out (default 2.0)
      --word-weight W      bonus for containing all of the query's words (default 2.0, and for code see Searcher)
      --shot-tokens N      vision tokens per screenshot: 70, 140, or 280 (default)
      --code-stretch N     characters a stretch of code holds at most (default 1800; 1200 and 3600 lost on the evals)

    The native model: EmbeddingGemma 2 on Core ML, from $NEURALSPOTLIGHT_MODELS/CoreML, else
    ~/Library/Application Support/NeuralSpotlight/Models/CoreML. NEURALSPOTLIGHT_COREML_MODELS overrides that folder.
    Other platforms require an explicitly provisioned NEURALSPOTLIGHT_WORKER_DIR embedding worker.
    """

struct CLIError: Error, CustomStringConvertible {
    let description: String
    init(_ description: String) { self.description = description }
}

struct Arguments {
    var positional: [String] = []
    var values: [String: String] = [:]
    var lists: [String: [String]] = [:]
    var flags: Set<String> = []
    static let valued: Set<String> = ["--index", "--limit", "--kind", "-n", "--zgate", "--word-weight", "--new-limit",
                                      "--linger", "--shot-tokens", "--code-stretch", "--encoder-idle", "--to",
                                      "--memory-limit", "--time-limit", "--tries", "--when-quiet", "--backend"]
    static let repeated: Set<String> = ["--exclude", "--exclude-ext", "--code-root", "--code-exclude"]

    init(_ raw: ArraySlice<String>) throws {
        var index = raw.startIndex
        while index < raw.endIndex {
            let argument = raw[index]
            if Self.valued.contains(argument) || Self.repeated.contains(argument) {
                guard index + 1 < raw.endIndex else { throw CLIError("\(argument) needs a value") }
                if Self.repeated.contains(argument) {
                    lists[argument, default: []].append(raw[index + 1])
                } else {
                    values[argument] = raw[index + 1]
                }
                index += 2
            } else if argument.hasPrefix("--") || argument == "-h" {
                flags.insert(argument)
                index += 1
            } else {
                positional.append(argument)
                index += 1
            }
        }
    }

    var indexFolder: URL {
        guard let chosen = values["--index"] else { return AppPaths.defaultIndex.standardizedFileURL }
        return URL(fileURLWithPath: (chosen as NSString).expandingTildeInPath).standardizedFileURL
    }

    /// `--code`: the code index, not the main one.
    var code: Bool { flags.contains("--code") }

    func store() throws -> IndexStore {
        try IndexStore(directory: indexFolder, file: code ? IndexStore.codeFile : IndexStore.mainFile)
    }

    var options: IndexOptions {
        var options = IndexOptions()
        options.code = code
        options.richContentOnly = flags.contains("--rich-content-only")
        options.includeCode = flags.contains("--include-code")
        options.includeDatasets = flags.contains("--include-datasets")
        options.excludedFolders = (lists["--exclude"] ?? []).map { Self.folderURL($0).path }
        options.excludedExtensions = IndexOptions.extensions((lists["--exclude-ext"] ?? []).joined(separator: ","))
        if let tokens = values["--shot-tokens"].flatMap(Int.init) { options.screenshotBudget = tokens }
        if let characters = values["--code-stretch"].flatMap(Int.init) { options.codeStretch = characters }
        return options
    }

    func folders() throws -> [URL] {
        guard !positional.isEmpty else { throw CLIError("which folders? e.g. neuralspotlight index ~/NeuralSpotlightTestbed") }
        return try positional.map {
            let url = Self.folderURL($0)
            guard Self.isFolder(url) else { throw CLIError("not a folder: \($0)") }
            guard Self.isReadable(url) else { throw CLIError("can't read \($0) (macOS privacy settings?)") }
            return url
        }
    }

    static func folderURL(_ path: String) -> URL {
        URL(fileURLWithPath: (path as NSString).expandingTildeInPath).standardizedFileURL
    }

    static func isFolder(_ url: URL) -> Bool {
        var isFolder: ObjCBool = false
        return FileManager.default.fileExists(atPath: url.path, isDirectory: &isFolder) && isFolder.boolValue
    }

    /// The folder can be listed. One that macOS won't let us read (Files and Folders privacy) looks empty to a crawl,
    /// which must never be mistaken for "everything in it was deleted". (The first look may bring up macOS's prompt.)
    static func isReadable(_ url: URL) -> Bool {
        (try? FileManager.default.contentsOfDirectory(atPath: url.path)) != nil
    }

    /// Dev knobs for A/B runs in evals/.
    var zGate: Double { values["--zgate"].flatMap(Double.init) ?? 2.0 }
    var wordWeight: Double? { values["--word-weight"].flatMap(Double.init) }   // nil: the index's own (`Searcher`)

    func kinds() throws -> Set<FileKind>? {
        guard let list = values["--kind"] else { return nil }
        return Set(try list.split(separator: ",").map {
            guard let kind = FileKind(rawValue: $0.trimmingCharacters(in: .whitespaces)) else {
                throw CLIError("unknown kind \($0); use screenshot, image, pdf, doc, audio, video")
            }
            return kind
        })
    }
}

// MARK: Formatting

let home = FileManager.default.homeDirectoryForCurrentUser.path
func tilde(_ path: String) -> String { path.hasPrefix(home) ? "~" + path.dropFirst(home.count) : path }
func pad(_ text: String, _ width: Int) -> String { text.count >= width ? text : text + String(repeating: " ", count: width - text.count) }
func lpad(_ text: String, _ width: Int) -> String { text.count >= width ? text : String(repeating: " ", count: width - text.count) + text }
func number(_ value: Int) -> String { value.formatted(.number) }

func duration(_ seconds: Double) -> String {
    switch seconds {
    case ..<90: "\(Int(seconds.rounded())) s"
    case ..<5400: "\(Int((seconds / 60).rounded())) min"
    default: String(format: "%.1f h", seconds / 3600)
    }
}

func clock(_ seconds: Double) -> String {
    let total = Int(seconds)
    return total >= 3600 ? String(format: "%d:%02d:%02d", total / 3600, total / 60 % 60, total % 60)
                         : String(format: "%d:%02d", total / 60, total % 60)
}

func location(_ hit: SearchHit) -> String {
    switch hit.segment {
    case .page, .pageImage: hit.loc.map { "p. \(Int($0))" } ?? ""
    case .audio, .frame: hit.loc.map { "at \(clock($0))" + (hit.segment == .audio ? " (sound)" : "") } ?? ""
    case .chunk: hit.loc.map { "part \(Int($0))" } ?? ""
    case .lines: hit.loc.map { "lines \(Int($0))–\(Int(hit.locEnd ?? $0))" + (hit.focus.map { ", opens at \(Int($0))" } ?? "") } ?? ""
    case .cells: hit.loc.map { $0 == hit.locEnd ? "cell \(Int($0))" : "cells \(Int($0))–\(Int(hit.locEnd ?? $0))" } ?? ""
    case .ocr: "text in image"
    default: ""
    }
}

func skippedLine(_ skipped: [String: Int]) -> String {
    skipped.sorted { $0.value > $1.value }.map { "\(number($0.value)) \($0.key)" }.joined(separator: " · ")
}

// MARK: Commands

/// Core ML in this process, or an explicitly configured external embedding worker.
func makeEmbedder(textOnly: Bool, log: URL, qos: QualityOfService = .default) throws -> any Embedder {
    if var config = try WorkerConfig.fromEnvironment() {
        config.log = log
        config.qos = qos
        return try EmbeddingWorker(config, textOnly: textOnly)
    }
    return try EmbeddingRuntime.make(textOnly: textOnly, log: log)
}

/// The native encoder's source is known without loading models; an external worker reports its own identity.
func embedderSource() -> String? {
    (ProcessInfo.processInfo.environment["NEURALSPOTLIGHT_WORKER_DIR"] ?? ProcessInfo.processInfo.environment["NEURALSPOTLIGHT_WORKER_DIR"]) == nil ? EmbeddingRuntime.modelInfo?.vectorSource : nil
}

func estimate(_ args: Arguments) throws {
    guard !args.positional.isEmpty else { throw CLIError("which folders? e.g. neuralspotlight estimate ~/Documents") }
    let options = args.options
    // What indexing costs on this machine: as measured by indexing here, kind by kind, once there's enough of it;
    // until then the M4 Pro's fixed costs, scaled by a second's speed check (done once and kept beside the index).
    var measured = MeasuredCosts.load(from: args.indexFolder)
    if measured.speed == nil, let speed = MeasuredCosts.Speed.measure() {
        measured.speed = speed
        MeasuredCosts.setSpeed(speed, in: args.indexFolder)
    }
    let costs = measured.costs
    // The times are this machine's (a speed check or indexing measured it); off the Mac, before `neuralspotlight speed --model`
    // has run, they're the M4 Pro's, far too short for most PCs.
    let calibrated = measured.speed != nil || !measured.tallies.isEmpty
    var total = 0.0
    for folder in args.positional.map(Arguments.folderURL) {
        // One folder that isn't there or can't be read (a denied privacy prompt) mustn't sink the others.
        guard Arguments.isFolder(folder), Arguments.isReadable(folder) else {
            if args.flags.contains("--json") {
                emit(["folder": folder.path, "readable": false, "seconds": 0, "files": [String: Int](),
                      "pdf_pages": 0, "audio_seconds": 0, "video_seconds": 0])
            } else {
                print("\(tilde(folder.path))    can't read it (not there, or not allowed in macOS privacy settings)")
            }
            continue
        }
        let started = Date()
        let crawl = Crawler.crawl([folder], options: options)
        let parts = Estimator.estimates(bySubfolder: crawl.candidates, of: folder.path, options: options, costs: costs)
        let estimate = parts.values.reduce(Estimate(), +)
        // Biggest first; the app lists them under the folder, so that a long first pass can be cut down.
        let subfolders = parts.filter { $0.key != folder.path }.sorted {
            $0.value.seconds != $1.value.seconds ? $0.value.seconds > $1.value.seconds : $0.key < $1.key
        }
        total += estimate.seconds
        if args.flags.contains("--json") {
            func files(_ estimate: Estimate) -> [String: Int] {
                Dictionary(uniqueKeysWithValues: estimate.files.map { ($0.key.rawValue, $0.value) })
            }
            // One line per folder, as soon as it's done (the app's onboarding shows them as they come).
            emit(["folder": folder.path, "seconds": estimate.seconds, "readable": true, "files": files(estimate),
                  "kind_seconds": Dictionary(uniqueKeysWithValues: estimate.kindSeconds.map { ($0.key.rawValue, $0.value) }),
                  "calibrated": calibrated,
                  "pdf_pages": estimate.pdfPages, "pdf_pages_total": estimate.pdfPagesTotal,
                  "audio_seconds": estimate.audioSeconds, "video_seconds": estimate.videoSeconds,
                  "long_files": estimate.longFiles, "later_seconds": estimate.laterSeconds,
                  "code_stretches": estimate.codeStretches, "bytes": estimate.bytes,
                  "subfolders": subfolders.prefix(500).map {
                      ["folder": $0.key, "seconds": $0.value.seconds, "files": files($0.value),
                       "kind_seconds": Dictionary(uniqueKeysWithValues: $0.value.kindSeconds.map { ($0.key.rawValue, $0.value) }),
                       "long_files": $0.value.longFiles, "later_seconds": $0.value.laterSeconds,
                       "bytes": $0.value.bytes] as [String: Any]
                  },
                  "skipped": crawl.skipped.values.reduce(0, +), "datasets": crawl.datasets.count,
                  "ms": Int(Date().timeIntervalSince(started) * 1000)])
            continue
        }
        var kinds: [String] = []
        for kind in FileKind.allCases {
            guard let count = estimate.files[kind], count > 0 else { continue }
            switch kind {
            case .pdf: kinds.append("\(number(count)) PDFs (\(number(estimate.pdfPagesTotal)) pages; \(number(estimate.pdfPages)) in the first pass)")
            case .audio: kinds.append("\(number(count)) audio (\(duration(estimate.audioSeconds)))")
            case .video: kinds.append("\(number(count)) video (\(duration(estimate.videoSeconds)))")
            case .code: kinds.append("\(number(count)) code \(count == 1 ? "file" : "files") "
                                     + "(\(number(estimate.codeStretches)) stretches)")
            default: kinds.append("\(number(count)) \(kind.rawValue)\(count == 1 ? "" : "s")")
            }
        }
        print("\(tilde(folder.path))    first index ≈ \(duration(estimate.seconds))"
              + (args.code ? ", index ≈ \(ByteCountFormatter.string(fromByteCount: Int64(estimate.bytes), countStyle: .file))" : ""))
        print("  " + (kinds.isEmpty ? "nothing to index" : kinds.joined(separator: " · ")))
        if estimate.longFiles > 0 {
            print("  then the rest of \(number(estimate.longFiles)) long \(estimate.longFiles == 1 ? "file" : "files") "
                  + "≈ \(duration(estimate.laterSeconds))")
        }
        if !subfolders.isEmpty {
            print("  biggest subfolders: " + subfolders.prefix(5).map {
                "\(($0.key as NSString).lastPathComponent) ≈ \(duration($0.value.seconds))"
            }.joined(separator: " · ") + (subfolders.count > 5 ? " · \(subfolders.count - 5) more" : ""))
        }
        if !crawl.skipped.isEmpty { print("  skipped: " + skippedLine(crawl.skipped)) }
        if !crawl.skippedFolders.isEmpty { print("  folders not walked: " + skippedLine(crawl.skippedFolders)) }
        for (folder, count) in crawl.datasets.sorted(by: { $0.value > $1.value }) {
            print("  looks like a dataset (skipped): \(tilde(folder)) (\(number(count)) files)")
        }
        print(String(format: "  (looked at it in %.2f s)", Date().timeIntervalSince(started)))
    }
    if args.positional.count > 1, !args.flags.contains("--json") { print("all together ≈ \(duration(total))") }
}

func index(_ args: Arguments) throws {
    let store = try args.store()
    let model = SharedEmbedder {
        try makeEmbedder(textOnly: false, log: store.directory.appendingPathComponent("worker.log"))
    }
    defer { model.close() }
    if args.flags.contains("--reindex") {
        _ = try model.embedder() // replacement assets must validate before existing semantic data is removed
        try store.resetAll()
    }
    let indexer = try Indexer(store: store, options: args.options, makeEmbedder: {
        try model.embedder()
    })
    indexer.embedderSource = embedderSource()
    indexer.costsFolder = store.directory
    try store.readAgainOnce()   // what older indexes couldn't read: scanned pages' words, SVGs
    try indexer.checkModel()
    let folders = try args.folders()
    let sync = try indexer.sync(folders)
    print("index: \(tilde(store.file.path))")
    print("sync: \(sync.added) new · \(sync.changed) changed · \(sync.moved) moved · \(sync.removed) removed · \(sync.unchanged) unchanged")
    if !sync.crawl.skipped.isEmpty { print("skipped: " + skippedLine(sync.crawl.skipped)) }
    for (folder, count) in sync.crawl.datasets { print("looks like a dataset (skipped): \(tilde(folder)) (\(number(count)) files)") }

    let limit = args.values["--limit"].flatMap(Int.init)
    var started = Date()
    let states = try indexer.run(limit: limit, onEvent: { event in
        let name = (event.path as NSString).lastPathComponent
        let note = event.note.map { "  (\($0))" } ?? ""
        print("[\(lpad("\(event.done)", 4))/\(event.total)] \(pad(event.kind.rawValue, 10)) \(pad(event.state, 7)) "
              + String(format: "%5.2fs  ", event.seconds) + name + note)
    })
    if states.isEmpty {
        print("nothing pending")
    } else {
        print("done in \(duration(Date().timeIntervalSince(started))): " + statesLine(states))
    }
    // The rest of long files, as the app does once nothing is pending (not after a --limit run: things still are).
    guard limit == nil, try store.unfinishedCount().files > 0 else { return }
    started = Date()
    let finished = try indexer.finishLongFiles(onStep: { step in
        let name = (step.path as NSString).lastPathComponent
        let note = step.note.map { "  (\($0))" } ?? ""
        print("[rest \(lpad("\(step.done)", 5))/\(step.total)] \(pad(step.kind.rawValue, 4)) \(pad(step.state, 7)) "
              + String(format: "%5.2fs  ", step.seconds) + name + note)
    })
    print("the rest of long files in \(duration(Date().timeIntervalSince(started))): "
          + (finished.isEmpty ? "none finished" : statesLine(finished)))
}

func statesLine(_ states: [String: Int]) -> String {
    states.sorted { $0.key < $1.key }.map { "\($0.value) \($0.key)" }.joined(separator: " · ")
}

/// Embeds queries with the text model alone (0.3B params, ~0.25 GB): the query encoder. `prompt`: the search's
/// (`Prompts.query`, or `Prompts.codeQuery` for the code index).
func queryVectors(_ queries: [String], store: IndexStore, searcher: Searcher,
                  prompt: (String) -> String = Prompts.query) throws -> [[Float]]? {
    guard searcher.vectorCount > 0, !queries.isEmpty else { return nil }
    let worker = try makeEmbedder(textOnly: true, log: store.directory.appendingPathComponent("worker-query.log"))
    defer { worker.close() }
    if let source = searcher.vectorSource, source != worker.info.vectorSource {
        throw CLIError("index vectors come from \(source) but the query encoder is \(worker.info.vectorSource)")
    }
    return try worker.embedTexts(queries.map(prompt))
}

func search(_ args: Arguments) throws {
    var query = args.positional.joined(separator: " ")
    guard !query.isEmpty else { throw CLIError("search for what?") }
    let code = CodeQuery.text(of: query)
    if let code { query = code }
    guard !query.isEmpty else { throw CLIError("search the code for what?") }
    let store = try IndexStore(directory: args.indexFolder, file: code == nil ? IndexStore.mainFile : IndexStore.codeFile)
    let searcher = try Searcher(store: store)
    let started = Date()
    let vector = try queryVectors([query], store: store, searcher: searcher,
                                  prompt: code == nil ? Prompts.query : Prompts.codeQuery)?.first
    let encoded = Date()
    let hits = try searcher.search(query, queryVector: vector, kinds: try args.kinds(),
                                   limit: args.values["-n"].flatMap(Int.init) ?? 10, zGate: args.zGate,
                                   wordWeight: args.wordWeight)
    let searched = Date()
    if args.flags.contains("--json") {
        let rows = hits.map(hitFields)
        let data = try JSONSerialization.data(withJSONObject: rows, options: [.prettyPrinted, .sortedKeys])
        print(String(data: data, encoding: .utf8) ?? "[]")
        return
    }
    if hits.isEmpty { print("no good matches") }
    for (rank, hit) in hits.enumerated() {
        let why = [hit.z.map { String(format: "z %.1f", $0) }, hit.keywordRank.map { "words #\($0)" }]
            .compactMap { $0 }.joined(separator: ", ")
        let place = location(hit)
        print("\(lpad("\(rank + 1)", 2)). \(pad(hit.kind.rawValue, 10)) \((hit.path as NSString).lastPathComponent)"
              + (place.isEmpty ? "" : "  [\(place)]") + "   (\(why))")
        print("      \(tilde((hit.path as NSString).deletingLastPathComponent))")
        if let excerpt = hit.excerpt, !excerpt.isEmpty { print("      “\(excerpt.prefix(160))”") }
    }
    print(String(format: "(%d vectors · query encoder %.2f s incl. start-up · search %.0f ms)", searcher.vectorCount,
                 encoded.timeIntervalSince(started), searched.timeIntervalSince(encoded) * 1000))
}

func status(_ args: Arguments) throws {
    let store = try args.store()
    let counts = try store.counts()
    print("index: \(tilde(store.file.path))")
    print("vectors: \(number(counts.vectors)) in \(number(counts.segments)) segments")
    for kind in FileKind.allCases {
        guard let states = counts.byKindAndState[kind.rawValue] else { continue }
        let total = states.values.reduce(0, +)
        print("  \(pad(kind.rawValue, 11)) \(lpad(number(total), 6))   " + statesLine(states))
    }
    if counts.unfinished > 0 {
        print("long files still being read: \(number(counts.unfinished)) (\(number(counts.left)) pages and chunks left)")
    }
    if let fingerprint = try store.meta("fingerprint") { print("built with: \(fingerprint)") }
    if let source = embedderSource() { print("native vector source: \(source)") }
    let runtime = (ProcessInfo.processInfo.environment["NEURALSPOTLIGHT_WORKER_DIR"] ?? ProcessInfo.processInfo.environment["NEURALSPOTLIGHT_WORKER_DIR"]) != nil ? "external worker"
        : EmbeddingBackend.supportsCoreML ? "coreml" : "unavailable (configure NEURALSPOTLIGHT_WORKER_DIR)"
    print("embedding backend: \(runtime)")
    print("size: \(ByteCountFormatter.string(fromByteCount: IndexStore.size(of: store.file), countStyle: .file))")
}

/// The app's indexing helper: sync + index on request, JSON events out (IndexService.swift).
func indexService(_ args: Arguments) throws -> Int32 {
    if !args.flags.contains("--sync-only") { try EmbeddingRuntime.requireSupport() }
    // One model for both indexes, started when the first file needs it (`SharedEmbedder`).
    let model = SharedEmbedder {
        try makeEmbedder(textOnly: false, log: args.indexFolder.appendingPathComponent("worker.log"), qos: .utility)
    }
    func part(_ file: String, options: IndexOptions, roots: [URL]) throws -> IndexService.Part {
        let store = try IndexStore(directory: args.indexFolder, file: file)
        store.prioritizeRecent = args.flags.contains("--recent-first")
        let indexer = try Indexer(store: store, options: options, makeEmbedder: { try model.embedder() })
        indexer.costsFolder = args.indexFolder   // what indexing costs here, for later estimates
        // A new app version may bring a new model: the index starts over by itself (keyword search on names goes on).
        indexer.embedderSource = embedderSource()
        indexer.resetOnModelChange = { indexed, now in
            emit(["event": "reindex", "from": indexed, "to": now, "code": options.code])
        }
        // A chosen folder that isn't there now (an unplugged drive) is skipped, not an error, and its files stay
        // indexed. With --prune, files under none of the given folders leave the index (the app passes all its folders).
        return IndexService.Part(store: store, indexer: indexer, roots: roots,
                                 keepOnly: args.flags.contains("--prune") ? roots : nil)
    }
    var codeOptions = args.options
    codeOptions.code = true
    codeOptions.excludedFolders = (args.lists["--code-exclude"] ?? []).map { Arguments.folderURL($0).path }
    let codeRoots = (args.lists["--code-root"] ?? []).map(Arguments.folderURL)
    // A file whose reading runs away ends this helper, noted for the next one (`Watchdog`).
    Watchdog(folder: args.indexFolder,
             memoryLimit: args.values["--memory-limit"].flatMap(UInt64.init).map { $0 << 20 } ?? Watchdog.defaultMemoryLimit,
             timeLimit: args.values["--time-limit"].flatMap(Double.init) ?? Watchdog.defaultTimeLimit).start()
    let service = IndexService(main: try part(IndexStore.mainFile, options: args.options,
                                              roots: args.positional.map(Arguments.folderURL)),
                               code: codeRoots.isEmpty ? nil : try part(IndexStore.codeFile, options: codeOptions,
                                                                        roots: codeRoots),
                               folder: args.indexFolder,
                               model: model,
                               modelReady: { EmbeddingRuntime.ready },
                               backfill: args.flags.contains("--backfill"),
                               recentFirst: args.flags.contains("--recent-first"),
                               syncOnly: args.flags.contains("--sync-only"),
                               newFilesLimit: args.values["--new-limit"].flatMap(Int.init) ?? 50,
                               linger: args.values["--linger"].flatMap(Double.init) ?? 60)
    return service.serve()
}

/// `neuralspotlight speed`: the speed check estimates start from, and what indexing has measured on this machine. `--model`
/// times the model itself instead (`ModelSpeed`: Linux and Windows) and keeps the result beside the index.
func speed(_ args: Arguments) throws {
    if args.flags.contains("--model") { return try modelSpeed(args) }
    if let seconds = MeasuredCosts.Speed.measuredSeconds() {
        let speed = MeasuredCosts.Speed.compared(seconds)
        print(String(format: "20 float16 matrix multiplies (2048²) %.3f s · Vision on a made-up screenshot %.3f s",
                     seconds.multiplies, seconds.ocr))
        print(String(format: "next to the M4 Pro behind the fixed costs: the model's work ×%.2f, OCR ×%.2f (1 = as fast)",
                     speed.embedding, speed.ocr))
    } else {
        print("no speed check here (no Metal): the fixed costs stand in until indexing has measured this machine's")
    }
    let measured = MeasuredCosts.load(from: args.indexFolder)
    print("measured by indexing (\(tilde(MeasuredCosts.file(in: args.indexFolder).path))):"
          + (measured.tallies.isEmpty ? " nothing yet" : ""))
    for (name, tally) in measured.tallies.sorted(by: { $0.key < $1.key }) {
        print(String(format: "  %@ %.4f s each, over %.0f", name, tally.seconds / tally.units, tally.units))
    }
}

/// Times the active runtime, recording per-kind costs under its vector identity. Core ML chooses operator
/// placement across ANE, GPU and CPU; timing must not silently switch the encoder or pretend ANE is a GPU.
func modelSpeed(_ args: Arguments) throws {
    let folder = args.indexFolder
    let json = args.flags.contains("--json")
    let tries = args.values["--tries"].flatMap(Int.init) ?? 3
    let log = folder.appendingPathComponent("worker.log")
    if args.flags.contains("--measure-only") {
        let started = Date()
        let model = try makeEmbedder(textOnly: false, log: log)
        let load = Date().timeIntervalSince(started)
        let measured = try ModelSpeed.measure(model, tries: tries)
        model.close()
        emit(["event": "speed", "seconds": measured.times.fields, "load": load, "vectors": measured.vectors,
              "tries": measured.tries, "steady": measured.steady, "others": measured.others])
        return
    }
    let saved = MeasuredCosts.load(from: folder)
    let source = embedderSource()
    let changedBackend = saved.speed?.vectorSource != source
    var speed = changedBackend ? MeasuredCosts.Speed(embedding: 1, ocr: 1)
        : saved.speed ?? MeasuredCosts.Speed(embedding: 1, ocr: 1)
    speed.vectorSource = source
    let checks = speed.kinds == nil ? 0 : speed.checks ?? 1
    let again = checks < 3 && (speed.doubtful == true || saved.pace.map { $0 < 0.25 } == true)
    if args.flags.contains("--if-needed"), let kinds = speed.kinds, !again {
        if json { emit(["event": "speed", "kinds": kinds, "measured": false]) }
        return
    }
    let quiet = args.values["--when-quiet"].flatMap(Double.init).map { SystemLoad.waitForQuiet(limit: $0) }
        ?? (waited: 0, settled: true, calmest: nil)
    let started = Date()
    let loading = Date()
    let model = try makeEmbedder(textOnly: false, log: log)
    let load = Date().timeIntervalSince(loading)
    let kept = try ModelSpeed.measure(model, tries: tries)
    model.close()
    speed.device = source == nil ? "external" : "coreml"
    speed.kinds = ModelSpeed.factors(kept.times)
    speed.checks = checks + 1
    speed.doubtful = !kept.steady ? true : nil
    // What indexing measured on the other device doesn't hold on this one.
    MeasuredCosts.setSpeed(speed, in: folder,
                          forgetTallies: changedBackend || (saved.speed?.device.map { $0 != speed.device } ?? false))
    let total = Date().timeIntervalSince(started)
    if json {
        var fields: [String: Any] = ["event": "speed", "kinds": speed.kinds ?? [:], "seconds": kept.times.fields,
                                     "load": load, "total": total, "measured": true, "tries": kept.tries,
                                     "others": kept.others, "waited": quiet.waited, "settled": quiet.settled,
                                     "doubtful": speed.doubtful ?? false, "checks": checks + 1]
        if let calmest = quiet.calmest { fields["calmest"] = calmest }
        if let device = speed.device { fields["device"] = device }
        emit(fields)
        return
    }
    if quiet.waited > 0 {
        print(String(format: "waited %.0f s for other programs to settle%@", quiet.waited, quiet.settled ? "" : " (they didn't)"))
    }
    print("runtime: \(source == nil ? "external worker" : "Core ML · ANE, GPU and CPU allowed")")
    print(String(format: "the model loaded in %.2f s; the check took %.1f s; tries %@", load, total,
                 kept.tries.map { String(format: "%.2f s", $0) }.joined(separator: ", ")))
    for (name, seconds) in kept.times.fields.sorted(by: { $0.key < $1.key }) {
        let base = ModelSpeed.baseline.fields[name] ?? seconds
        print(String(format: "  %@ %.3f s (M4 Pro %.3f s, ×%.1f)", name, seconds, base, seconds / base))
    }
    if speed.doubtful == true { print("in doubt (a busy moment): timed again at the next --if-needed") }
    print("kept in \(tilde(MeasuredCosts.file(in: folder).path)): " + (speed.kinds ?? [:]).sorted { $0.key < $1.key }
        .map { String(format: "%@ ×%.1f", $0.key, $0.value) }.joined(separator: " · "))
}

// MARK: Eval

struct EvalFile: Decodable {
    struct Query: Decodable {
        let q: String
        let expect: [String]   // paths relative to `base`; "path@6-12" also requires the hit to start in [6, 12) s
        let tag: String?
        let kind: String?
    }
    let base: String
    let queries: [Query]
}

func evaluate(_ args: Arguments) throws {
    guard let file = args.positional.first else { throw CLIError("which eval file?") }
    let eval = try JSONDecoder().decode(EvalFile.self, from: Data(contentsOf: URL(fileURLWithPath: (file as NSString).expandingTildeInPath)))
    var base = (eval.base as NSString).expandingTildeInPath
    // The testbed kept elsewhere (another drive on Windows, wherever CI puts it): NEURALSPOTLIGHT_TESTBED stands for
    // ~/NeuralSpotlightTestbed in the eval's base.
    if let testbed = ProcessInfo.processInfo.environment["NEURALSPOTLIGHT_TESTBED"] ?? ProcessInfo.processInfo.environment["NEURALSPOTLIGHT_TESTBED"] {
        for prefix in ["~/NeuralSpotlightTestbed", "~/NeuralSpotlightTestbed"] where eval.base.hasPrefix(prefix) {
            base = testbed + eval.base.dropFirst(prefix.count)
            break
        }
    }
    // "code: …" queries search the code index (with its prompt), the others the main one.
    let isCode = eval.queries.map { CodeQuery.text(of: $0.q) != nil }
    let texts = eval.queries.map { CodeQuery.text(of: $0.q) ?? $0.q }
    var searchers: [Bool: Searcher] = [:]
    var vectors = [[Float]?](repeating: nil, count: texts.count)
    for code in Set(isCode) {
        let store = try IndexStore(directory: args.indexFolder, file: code ? IndexStore.codeFile : IndexStore.mainFile)
        let searcher = try Searcher(store: store)
        searchers[code] = searcher
        let indices = texts.indices.filter { isCode[$0] == code }
        let embedded = try queryVectors(indices.map { texts[$0] }, store: store, searcher: searcher,
                                        prompt: code ? Prompts.codeQuery : Prompts.query)
        for (index, vector) in zip(indices, embedded ?? []) { vectors[index] = vector }
    }
    let depth = args.values["-n"].flatMap(Int.init) ?? 10

    var byTag: [String: (n: Int, at1: Int, at5: Int, at10: Int, mrr: Double)] = [:]
    var shown = 0, listed = 0, folded = 0   // the app's weaker-matches fold (Searcher.strongCount), over all queries
    for (index, query) in eval.queries.enumerated() {
        let kinds = query.kind.flatMap { FileKind(rawValue: $0) }.map { Set([$0]) }
        let hits = try searchers[isCode[index]]!.search(texts[index], queryVector: vectors[index], kinds: kinds,
                                                       limit: depth, zGate: args.zGate, wordWeight: args.wordWeight)
        let strong = Searcher.strongCount(hits)
        shown += strong
        listed += hits.count
        let targets: [(path: String, range: ClosedRange<Double>?)] = query.expect.map { entry in
            let parts = entry.split(separator: "@", maxSplits: 1)
            let range = parts.count == 2 ? parts[1].split(separator: "-").compactMap { Double($0) } : []
            return (base + "/" + parts[0], range.count == 2 ? range[0]...range[1] : nil)
        }
        // A place to be at: a page or moment starts in the range; a stretch of code overlaps it (its lines or cells).
        func at(_ hit: SearchHit, _ range: ClosedRange<Double>) -> Bool {
            guard let loc = hit.loc else { return false }
            if hit.segment == .lines || hit.segment == .cells { return loc <= range.upperBound && (hit.locEnd ?? loc) >= range.lowerBound }
            return range.contains(loc)
        }
        let rank = hits.firstIndex { hit in
            targets.contains { target in
                hit.path == target.path && (target.range.map { at(hit, $0) } ?? true)
            }
        }.map { $0 + 1 }
        if let rank, rank > strong { folded += 1 }
        let tag = query.tag ?? "all"
        for key in Set([tag, "ALL"]) {
            var row = byTag[key] ?? (0, 0, 0, 0, 0)
            row.n += 1
            if let rank {
                if rank <= 1 { row.at1 += 1 }
                if rank <= 5 { row.at5 += 1 }
                if rank <= 10 { row.at10 += 1 }
                row.mrr += 1 / Double(rank)
            }
            byTag[key] = row
        }
        let top = hits.first.map { "\(($0.path as NSString).lastPathComponent)" + (location($0).isEmpty ? "" : " [\(location($0))]") } ?? "-"
        print("\(rank.map { lpad("#\($0)", 4) } ?? "miss")  \(pad(tag, 11)) \(pad("“\(query.q)”", 42)) top: \(top)")
    }
    print("")
    print("\(pad("tag", 12)) \(lpad("n", 3))  \(lpad("R@1", 5))  \(lpad("R@5", 5))  \(lpad("R@10", 5))  \(lpad("MRR", 5))")
    for (tag, row) in byTag.sorted(by: { ($0.key == "ALL" ? "~" : $0.key) < ($1.key == "ALL" ? "~" : $1.key) }) {
        let n = Double(row.n)
        print("\(pad(tag, 12)) \(lpad("\(row.n)", 3))  " + String(format: "%5.2f  %5.2f  %5.2f  %5.2f",
              Double(row.at1) / n, Double(row.at5) / n, Double(row.at10) / n, row.mrr / n))
    }
    let count = Double(max(1, eval.queries.count))
    print(String(format: "weaker-matches fold (below 0.5× the top score): %.1f of %.1f results shown per query; ",
                 Double(shown) / count, Double(listed) / count)
          + "\(folded) expected \(folded == 1 ? "answer" : "answers") folded")
}

// MARK: Main

do {
    let raw = CommandLine.arguments.dropFirst()
    guard let command = raw.first, command != "-h", command != "--help", command != "help" else {
        print(usage)
        exit(0)
    }
    let args = try Arguments(raw.dropFirst())
    if let value = args.values["--backend"], EmbeddingBackend(rawValue: value) == nil {
        throw CLIError("unknown backend \(value); choose auto or coreml")
    }
    if (ProcessInfo.processInfo.environment["NEURALSPOTLIGHT_WORKER_DIR"] ?? ProcessInfo.processInfo.environment["NEURALSPOTLIGHT_WORKER_DIR"]) == nil,
       let value = args.values["--shot-tokens"], ![70, 140, 280].contains(Int(value) ?? -1) {
        throw CLIError("Core ML supports --shot-tokens 70, 140, or 280; no larger vision graph is bundled")
    }
    if ["index-service", "worker", "search-service", "download-model"].contains(command)
        || command == "estimate" && args.flags.contains("--json") {
        ParentWatch.start()   // an app's helper goes when the app does
    }
    switch command {
    case "estimate": try estimate(args)
    case "index": try index(args)
    case "search": try search(args)
    case "status": try status(args)
    case "speed": try speed(args)
    case "eval": try evaluate(args)
    case "index-service": exit(try indexService(args))
    case "worker": exit(workerService(args))
    case "search-service": exit(try searchService(args))
    case "download-model": exit(try downloadModel(args))
    default:
        print(usage)
        exit(2)
    }
} catch {
    FileHandle.standardError.write(Data("error: \(error)\n".utf8))
    exit(1)
}
