import AppKit
import DigUpKit
import Foundation
import Observation

/// Gets the pinned Core ML text, image and audio models onto this Mac.
///
/// - Resumable: a file downloads into `<name>.partial`, and an interrupted download carries on from where it stopped
///   (an HTTP range), after a relaunch too.
/// - Checked: a finished file must have the exact size and SHA-256 in `ModelFiles` before it gets its real name, so a
///   file under its real name is always whole.
/// - Network problems retry by themselves (2 s, 5 s, 15 s, then every 30 s). A full disk, or a file that fails its
///   check twice, waits for the user (`start()` again).
/// - The folder is excluded from Time Machine: the files can always be downloaded again.
///
/// This is the app's only network use. It starts when onboarding's Get Started is clicked, or at launch when a set-up
/// app finds its model missing (an update that brings a new model). A file that's already in the Hugging Face cache
/// (`hf download`) is copied from there instead and checked the same way.
/// So are files chosen by hand (`use`), for networks that block Hugging Face.
@Observable
final class ModelDownload {
    enum Phase: Equatable {
        case needed            // files missing, waiting for the go-ahead
        case downloading
        case verifying
        case retrying(String)  // a network problem: tries again by itself
        case failed(String)    // needs the user
        case ready
    }

    private(set) var phase: Phase
    /// Bytes on disk over every model asset, including files left by earlier runs.
    private(set) var received: Int64 = 0
    private(set) var bytesPerSecond: Double = 0
    let files: [ModelFiles.File]
    let total: Int64
    let folder: URL

    @ObservationIgnored var onReady: () -> Void = {}
    /// Text assets arrive first, so query preparation can overlap the remaining image/audio download.
    @ObservationIgnored var onTextReady: () -> Void = {}
    @ObservationIgnored private var textReadyReported = false
    @ObservationIgnored private var fetcher: FileFetcher?
    @ObservationIgnored private var retryWork: DispatchWorkItem?
    @ObservationIgnored private var failuresInARow = 0
    @ObservationIgnored private var failedChecks = 0
    @ObservationIgnored private var samples: [(time: Date, bytes: Int64)] = []
    @ObservationIgnored private var fileStarted = Date()
    @ObservationIgnored private var lastLogged = Date.distantPast
    @ObservationIgnored private var attempt = 0
    /// Files whose `.partial` was chosen by hand (`use`): one that fails its check says so rather than downloading.
    @ObservationIgnored private var chosen = Set<String>()

    init(folder: URL, backend: EmbeddingBackend = .automatic) {
        self.folder = folder
        files = backend.modelFiles
        total = files.reduce(0) { $0 + $1.bytes }
        phase = files.allSatisfy { FileFetcher.size(of: folder.appendingPathComponent($0.name)) == $0.bytes }
            ? .ready : .needed
        received = bytesOnDisk()
    }

    private var missing: [ModelFiles.File] {
        files.filter { FileFetcher.size(of: folder.appendingPathComponent($0.name)) != $0.bytes }
    }

    var isReady: Bool { phase == .ready }
    var isActive: Bool {
        switch phase {
        case .downloading, .verifying, .retrying: true
        default: false
        }
    }
    var fraction: Double { min(1, Double(received) / Double(total)) }
    var secondsLeft: Double? {
        guard phase == .downloading, bytesPerSecond > 0 else { return nil }
        return Double(total - received) / bytesPerSecond
    }

    /// "42% · about 1 min left", "Checking the download…", "Waiting for the network…"
    var progressText: String {
        switch phase {
        case .needed: return "Not downloaded yet"
        case .downloading:
            let percent = "\(Int((fraction * 100).rounded(.down)))%"
            guard let left = secondsLeft else { return percent }
            return percent + " · " + (left < 50 ? "less than a minute left" : "about \(roughDuration(left)) left")
        case .verifying: return "Checking the download…"
        case .retrying: return "Waiting for the network…"
        case .failed(let message): return message
        case .ready: return "Ready"
        }
    }

    /// Starts, resumes or retries; does nothing while it runs (waiting for the network included) or once it's done.
    func start() {
        switch phase {
        case .ready, .downloading, .verifying: return
        default: break
        }
        guard fetcher == nil else { return }   // one download at a time: two would write the same .partial
        retryWork?.cancel()
        failedChecks = 0
        next()
    }

    private func next() {
        if !textReadyReported, CoreMLModelFiles.complete(in: folder.appendingPathComponent("CoreML"), textOnly: true) {
            textReadyReported = true
            onTextReady()
        }
        guard let file = missing.first else {
            phase = .ready
            received = total
            log("model: ready in \(tildePath(folder.path))")
            onReady()
            return
        }
        do {
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
            var values = URLResourceValues()
            values.isExcludedFromBackup = true
            var folder = self.folder
            try folder.setResourceValues(values)
        } catch {
            phase = .failed("Can't create \(tildePath(folder.path))")
            log("model: can't create \(folder.path): \(error)")
            return
        }
        let needed = missing.reduce(0) { $0 + $1.bytes } - partialBytes(of: file) + 800_000_000
        if let free = freeSpace(), free < needed {
            phase = .failed("Needs \(bytes(needed)) free on this disk")
            log("model: needs \(bytes(needed)) free, \(bytes(free)) is free")
            return
        }
        if let cached = ModelFiles.cachedCopy(of: file) {
            phase = .verifying
            let partial = folder.appendingPathComponent(file.name + ".partial")
            DispatchQueue.global(qos: .utility).async {
                try? FileManager.default.createDirectory(at: partial.deletingLastPathComponent(), withIntermediateDirectories: true)
                try? FileManager.default.removeItem(at: partial)
                let copied = (try? FileManager.default.copyItem(at: cached, to: partial)) != nil
                DispatchQueue.main.async { [weak self] in
                    guard let self else { return }
                    log("model: \(file.name) " + (copied ? "copied from the Hugging Face cache"
                                                          : "is in the Hugging Face cache but couldn't be copied"))
                    if copied {
                        fileStarted = Date()
                        finished(file, .downloaded(partial))
                    } else {
                        download(file)
                    }
                }
            }
            return
        }
        download(file)
    }

    private func download(_ file: ModelFiles.File) {
        let done = completedBytes()
        let resumeAt = partialBytes(of: file)
        log("model: downloading \(file.name)" + (resumeAt > 0 ? " from \(bytes(resumeAt)) (resumed)" : ""))
        phase = .downloading
        fileStarted = Date()
        samples = []
        attempt += 1
        let current = attempt   // callbacks from an earlier attempt are ignored
        let fetcher = FileFetcher(file: file, folder: folder, progress: { [weak self] written in
            DispatchQueue.main.async { [weak self] in
                guard let self, attempt == current else { return }
                progress(done + written)
            }
        }, waiting: { [weak self] in
            DispatchQueue.main.async { [weak self] in
                guard let self, attempt == current else { return }
                phase = .retrying("Waiting for a network connection")
                log("model: waiting for a network connection")
            }
        }, done: { [weak self] outcome in
            DispatchQueue.main.async { [weak self] in
                guard let self, attempt == current else { return }
                finished(file, outcome)
            }
        })
        self.fetcher = fetcher
        fetcher.start()
    }

    private func progress(_ onDisk: Int64) {
        if phase != .downloading { phase = .downloading }
        if onDisk > received { failuresInARow = 0 }   // bytes came: the link works again (a failure reports too)
        received = onDisk
        let now = Date()
        samples.append((now, onDisk))
        samples.removeAll { now.timeIntervalSince($0.time) > 6 }
        if let first = samples.first, now.timeIntervalSince(first.time) >= 1 {
            bytesPerSecond = Double(onDisk - first.bytes) / now.timeIntervalSince(first.time)
        }
        if now.timeIntervalSince(lastLogged) >= 10 {
            lastLogged = now
            log("model: \(progressText) (\(bytes(onDisk)) of \(bytes(total)), \(bytes(Int64(bytesPerSecond)))/s)")
        }
    }

    private func finished(_ file: ModelFiles.File, _ outcome: FileFetcher.Outcome) {
        fetcher = nil
        switch outcome {
        case .downloaded(let partial):
            let seconds = Date().timeIntervalSince(fileStarted)
            phase = .verifying
            let started = Date()
            DispatchQueue.global(qos: .utility).async { [folder] in
                let size = FileFetcher.size(of: partial)
                let hash = size == file.bytes ? try? ModelFiles.sha256(of: partial) : nil
                let good = hash == file.sha256
                var saveError: String?
                if good {
                    let destination = folder.appendingPathComponent(file.name)
                    do {
                        if FileManager.default.fileExists(atPath: destination.path) {
                            try FileManager.default.removeItem(at: destination)
                        }
                        try FileManager.default.moveItem(at: partial, to: destination)
                    } catch { saveError = error.localizedDescription }
                } else {
                    try? FileManager.default.removeItem(at: partial)
                }
                let failure = saveError
                DispatchQueue.main.async { [weak self] in
                    guard let self else { return }
                    if let failure {
                        phase = .failed("Can't save the model: \(failure)")
                        return
                    }
                    log("model: \(file.name) \(good ? "checked" : "FAILED its check (\(size) of \(file.bytes) bytes)") in "
                        + "\(ms(since: started)), downloaded in \(String(format: "%.1f s", seconds))")
                    if good {
                        chosen.remove(file.name)
                        next()
                    } else if chosen.remove(file.name) != nil {
                        // Chosen by hand: say so, rather than download what the network may block.
                        phase = .failed("The chosen \(file.name) isn't the right file")
                    } else {
                        failedChecks += 1
                        received = bytesOnDisk()
                        if failedChecks >= 2 {
                            phase = .failed("The download was damaged twice. Try again later.")
                        } else {
                            next()
                        }
                    }
                }
            }
        case .failed(let message, let retry):
            received = bytesOnDisk()
            bytesPerSecond = 0
            guard retry else {
                phase = .failed(message)
                log("model: stopped: \(message)")
                return
            }
            let delays: [Double] = [2, 5, 15, 30]
            let delay = delays[min(failuresInARow, delays.count - 1)]
            failuresInARow += 1
            phase = .retrying(message)
            log("model: \(message.trimmingCharacters(in: CharacterSet(charactersIn: "."))); trying again in \(Int(delay)) s")
            let work = DispatchWorkItem { [weak self] in
                guard let self, case .retrying = phase else { return }
                next()
            }
            retryWork = work
            DispatchQueue.main.asyncAfter(deadline: .now() + delay, execute: work)
        }
    }

    // MARK: Files chosen by hand

    /// Model files, a Models/CoreML folder, or a Core ML package obtained on another network or from a mirror.
    /// Each goes where a download would put it (`<name>.partial`; an APFS clone when it can
    /// be) and is checked the same way, size and SHA-256, before it's used. A file counts by its size, and so by its
    /// name only at the right size (a smaller one would be taken for a download to resume). What's still missing
    /// afterwards downloads as usual.
    func use(_ urls: [URL]) {
        var picked: [(file: ModelFiles.File, url: URL)] = []
        var problem = "Those aren't the model's files"
        let needed = missing
        func consider(_ original: URL, expected: ModelFiles.File? = nil) {
            let url = original.resolvingSymlinksInPath()
            guard (try? url.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) == true else { return }
            let size = FileFetcher.size(of: url)
            guard let file = expected ?? files.first(where: { $0.bytes == size }), file.bytes == size else {
                if let named = expected ?? files.first(where: { ($0.name as NSString).lastPathComponent == url.lastPathComponent }) {
                    problem = "The chosen \(named.name) is \(bytes(size)), not \(bytes(named.bytes))"
                }
                return
            }
            let wanted = needed.contains(file) && !picked.contains { $0.file == file }
            if wanted { picked.append((file, url)) }
        }
        for original in urls {
            let url = original.resolvingSymlinksInPath()
            guard (try? url.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true else {
                consider(url)
                continue
            }
            for file in needed {
                // Only probe manifest paths: no recursive scan of a selected folder or package.
                // Models/ uses the installation name; CoreML/ or an HF snapshot uses the remote path.
                var relativePaths = [file.name, file.remotePath]
                let components = file.remotePath.split(separator: "/")
                if url.pathExtension.lowercased() == "mlpackage",
                   components.first?.hasSuffix(".mlpackage") == true {
                    relativePaths.append(components.dropFirst().joined(separator: "/"))
                }
                var seen = Set<String>()
                for relative in relativePaths where seen.insert(relative).inserted {
                    consider(url.appendingPathComponent(relative), expected: file)
                }
            }
        }
        guard !picked.isEmpty else {
            log("model: none of the chosen files is a model file (\(urls.map(\.lastPathComponent))): \(problem)")
            if phase != .ready, !isActive { phase = .failed(problem) }
            return
        }
        // What was downloading stops: these take its place (a stopped fetcher's callbacks are ignored).
        fetcher?.cancel()
        fetcher = nil
        attempt += 1
        retryWork?.cancel()
        phase = .verifying
        DispatchQueue.global(qos: .utility).async { [folder, picked] in
            try? FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
            var copied: [String] = []
            for (file, url) in picked {
                let partial = folder.appendingPathComponent(file.name + ".partial")
                // A user can select an already-downloaded .partial file in this folder. Check it in place.
                if url.standardizedFileURL == partial.standardizedFileURL {
                    copied.append(file.name)
                    continue
                }
                try? FileManager.default.createDirectory(at: partial.deletingLastPathComponent(), withIntermediateDirectories: true)
                try? FileManager.default.removeItem(at: partial)
                if (try? FileManager.default.copyItem(at: url, to: partial)) != nil { copied.append(file.name) }
            }
            DispatchQueue.main.async { [weak self] in
                guard let self else { return }
                log("model: chosen by hand: \(copied.joined(separator: ", "))")
                chosen.formUnion(copied)
                phase = .needed
                start()   // checks them, and downloads what's still missing
            }
        }
    }

    /// An open panel for `use`, as a sheet on `window` when there is one.
    func chooseFiles(for window: NSWindow?) {
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.canChooseDirectories = true
        panel.treatsFilePackagesAsDirectories = false   // selecting an .mlpackage imports its known files together
        panel.allowsMultipleSelection = true
        panel.prompt = "Use"
        panel.message = "Choose your downloaded model files, their folder, or a Core ML model package. "
            + "Each file is checked before use; missing files will download automatically."
        let finish = { [weak self] (response: NSApplication.ModalResponse) in
            guard response == .OK else { return }
            self?.use(panel.urls)
        }
        if let window {
            panel.beginSheetModal(for: window, completionHandler: finish)
        } else {
            finish(panel.runModal())
        }
    }

    /// Where the files download from (the pinned revisions), one a line: to fetch them on another network.
    var links: String {
        files.map(\.url.absoluteString).joined(separator: "\n")
    }

    // MARK: Disk

    private func partialBytes(of file: ModelFiles.File) -> Int64 {
        size(of: folder.appendingPathComponent(file.name + ".partial"))
    }

    private func completedBytes() -> Int64 {
        files.filter { !missing.contains($0) }.reduce(0) { $0 + $1.bytes }
    }

    private func bytesOnDisk() -> Int64 {
        completedBytes() + missing.reduce(0) { $0 + min(partialBytes(of: $1), $1.bytes) }
    }

    private func size(of url: URL) -> Int64 {
        FileFetcher.size(of: url)
    }

    private func freeSpace() -> Int64? {
        var url = folder
        while !FileManager.default.fileExists(atPath: url.path), url.pathComponents.count > 1 {
            url.deleteLastPathComponent()
        }
        return try? url.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey])
            .volumeAvailableCapacityForImportantUsage
    }
}

/// "865 MB", "1.2 GB".
nonisolated func bytes(_ count: Int64) -> String {
    ByteCountFormatter.string(fromByteCount: count, countStyle: .file)
}
