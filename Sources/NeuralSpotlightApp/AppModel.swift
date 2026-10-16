import AppKit
import NeuralSpotlightKit
import Observation

/// State the menubar, the panel and the windows share. Engine updates arrive on the main thread.
@Observable
final class AppModel {
    var status = EngineStatus()
    var settings: AppSettings
    var hotkey: KeyCombo?
    let download: ModelDownload
    /// A newer NeuralSpotlight the daily check found ("0.6.1"), until you open it (`Updates`).
    var updateAvailable: String?

    init(settings: AppSettings, download: ModelDownload) {
        self.settings = settings
        self.download = download
    }

    var hasFolders: Bool { !settings.roots.isEmpty }
    /// Code search is on (there are code folders).
    var hasCode: Bool { !settings.codeRoots.isEmpty }

    /// One line for the menu and the windows: what the index is doing.
    var statusLine: String {
        let status = self.status
        guard hasFolders else { return "No folders chosen yet" }
        switch status.activity {
        case .starting: return "Starting…"
        case .failed(let problem): return "⚠︎ \(problem)"
        case .syncing: return "Checking folders…"
        case .indexing:
            return indexingLine(status.indexingProgress)
        case .finishing(let done, let total, let files):
            let percent = total > 0 ? Int((Double(done) / Double(total) * 100).rounded(.down)) : 0
            let which = files > 0 ? "\(files.formatted()) long \(files == 1 ? "file" : "files")" : "long files"
            return "Reading the rest of \(which) · \(percent)%"
        case .idle:
            if let line = downloadLine { return line }
            if status.code.indexing != nil {
                return indexingLine(status.code.indexingProgress, code: true)
            }
            let waiting = "\(status.pending.formatted()) \(status.pending == 1 ? "file" : "files") waiting"
            if status.pending > 0, status.paused { return "Paused · \(waiting)" }
            if let until = status.heldUntil, !status.paused {
                // Outside the set hours: what's known to wait, and when indexing starts again.
                let when = "indexing from \(until.formatted(date: .omitted, time: .shortened))"
                if status.pending > 0 { return "\(waiting) · \(when)" }
                return "\(status.searchable.formatted()) \(status.searchable == 1 ? "file" : "files") indexed · \(when)"
            }
            if status.pending > 0, let reason = status.waitingFor { return "\(waiting) · \(reason)" }
            if status.pending > 0 { return "\(waiting) to be indexed" }
            if status.unfinished > 0 {
                let rest = "\(status.unfinished.formatted()) long \(status.unfinished == 1 ? "file" : "files") to finish"
                if status.paused { return "Paused · \(rest)" }
                if let reason = status.waitingFor { return "\(rest) · \(reason)" }
            }
            return "\(status.searchable.formatted()) \(status.searchable == 1 ? "file" : "files") indexed"
                + issuesSummary(failed: status.failed, skipped: status.skipped, unsupported: status.unsupported)
                + (status.paused ? " · paused" : "")
        }
    }

    /// While the model isn't there, how its download goes.
    private var downloadLine: String? {
        switch download.phase {
        case .ready: nil
        case .downloading: "Downloading the model · \(Int(download.fraction * 100))%"
        case .verifying: "Checking the model download…"
        case .retrying: "Model download: waiting for the network"
        case .failed: "⚠︎ The model download stopped"
        case .needed: status.pending > 0 ? "The model isn't downloaded yet" : nil
        }
    }

    /// 0…1 for the menubar ring and progress bars: download, the whole collection, or the rest of long files.
    var progress: Double? {
        if download.isActive { return download.fraction }
        switch status.activity {
        case .indexing: return status.indexingProgress.fraction
        case .finishing(let done, let total, _) where total > 0: return Double(done) / Double(total)
        case .idle:
            return status.code.indexing != nil ? status.code.indexingProgress.fraction : nil
        default: return nil
        }
    }

    /// One line about the code index, for Settings → Code: "812 code files indexed", "Indexing code 42%…".
    var codeStatusLine: String {
        let code = status.code
        guard hasCode else { return "Code search is off" }
        if !download.isReady { return "Code is read once the model has downloaded" }
        if code.indexing != nil {
            return indexingLine(code.indexingProgress, code: true)
        }
        let waiting = code.pending > 0 ? " · \(code.pending.formatted()) waiting" : ""
        if code.pending > 0, status.paused { return "Paused · \(code.pending.formatted()) code files waiting" }
        if code.pending > 0, let reason = status.waitingFor { return "\(code.pending.formatted()) code files waiting · \(reason)" }
        if code.pending > 0, let until = status.heldUntil {
            return "\(code.pending.formatted()) code files waiting · indexing from "
                + until.formatted(date: .omitted, time: .shortened)
        }
        // Code is read once the other folders have nothing waiting (the helper's order).
        if code.pending > 0, status.pending > 0 {
            let files = "\(code.pending.formatted()) code \(code.pending == 1 ? "file" : "files")"
            return files + " waiting · read after your other files"
        }
        return "\(code.searchable.formatted()) code \(code.searchable == 1 ? "file" : "files") indexed" + waiting
            + issuesSummary(failed: code.failed, skipped: code.skipped, unsupported: code.unsupported)
    }

    /// A few words for the search bar while results may be incomplete: "Indexing 42%", "Downloading model · 42%".
    var activityBadge: String? {
        if download.isActive { return "Downloading model · \(Int(download.fraction * 100))%" }
        if case .indexing = status.activity {
            return status.indexingProgress.percent.map { "Indexing \($0)%" } ?? "Indexing…"
        }
        if status.code.indexing != nil {
            return status.code.indexingProgress.percent.map { "Indexing code \($0)%" } ?? "Indexing code…"
        }
        if status.paused, status.pending > 0 { return "Indexing paused" }
        return nil
    }

    /// Why meaning search can't run yet (words still match), or nil.
    var meaningUnavailable: String? {
        guard !download.isReady else { return nil }
        if download.isActive {
            return "Matching words only until the model has downloaded (\(Int(download.fraction * 100))%)"
        }
        return "Matching words only: the model isn't downloaded"
    }

    /// Why results may be incomplete right now, or nil.
    var resultsCaveat: String? {
        if let meaningUnavailable { return meaningUnavailable }
        if case .indexing = status.activity, let percent = status.indexingProgress.percent {
            return "Still indexing (\(percent)%): some files aren't in meaning search yet"
        }
        if status.pending > 0 {
            return "\(status.pending.formatted()) \(status.pending == 1 ? "file isn't" : "files aren't") indexed yet"
        }
        if status.failed > 0 {
            return "\(status.failed.formatted()) \(status.failed == 1 ? "file couldn't" : "files couldn't") be indexed"
        }
        return nil
    }

    private func indexingLine(_ progress: IndexingProgress, code: Bool = false) -> String {
        let label = code ? "Indexing code" : "Indexing"
        guard let percent = progress.percent else { return "\(label)…" }
        return "\(label) \(percent)% · \(progress.processed.formatted()) of \(progress.files.formatted()) processed"
    }

    private func issuesSummary(failed: Int, skipped: Int, unsupported: Int) -> String {
        (failed > 0 ? " · \(failed.formatted()) couldn't be indexed" : "")
            + (skipped > 0 ? " · \(skipped.formatted()) skipped" : "")
            + (unsupported > 0 ? " · \(unsupported.formatted()) unsupported" : "")
    }
}
