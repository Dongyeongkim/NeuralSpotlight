#if os(macOS)
import CoreServices
import Foundation
import UniformTypeIdentifiers

/// Runs on a dedicated serial queue. Reuses Spotlight's names and indexed text; never extracts file contents.
/// Applications have a small, separately cached catalog, so launching works even without a semantic index.
public final class LocalSearchProvider {
    private struct Application {
        let url: URL
        let displayName: String
    }
    private var applications: [Application] = []
    private var refreshed = Date.distantPast
    private let applicationRoots: [URL]
    private let spotlightTimeout: TimeInterval

    public init(applicationRoots: [URL] = [URL(fileURLWithPath: "/Applications"),
        URL(fileURLWithPath: "/System/Applications"),
        FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Applications")],
                spotlightTimeout: TimeInterval = 0.3) {
        self.applicationRoots = applicationRoots
        self.spotlightTimeout = spotlightTimeout.isFinite ? max(0, spotlightTimeout) : 0.3
    }

    public func searchApplications(_ text: String, excluded: [URL], excludedTypes: Set<String>, limit: Int = 10,
                                   isCancelled: @Sendable () -> Bool = { false }) -> [SearchHit] {
        guard !isCancelled(), limit > 0, !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return [] }
        if Date().timeIntervalSince(refreshed) > 60 { refreshApplications(isCancelled: isCancelled) }
        var hits: [SearchHit] = []
        for application in applications {
            guard !isCancelled() else { return [] }
            let url = application.url
            // Match cached names first: a keystroke must not fetch file-provider metadata for every installed app.
            guard LocalSearch.nameScore(url.lastPathComponent, query: text) != nil
                    || LocalSearch.nameScore(application.displayName, query: text) != nil,
                  Self.allowed(url.path, excludedTypes: excludedTypes),
                  !excluded.contains(where: { url.path == $0.path || FolderSelection.isInside(url.path, $0.path) }),
                  let result = hit(url, text: text, application: true) else { continue }
            hits.append(result)
        }
        guard !isCancelled() else { return [] }
        return LocalSearch.merge([], with: hits, limit: limit)
    }

    public func search(_ text: String, roots: [URL], excluded: [URL], excludedTypes: Set<String>,
                       kinds: Set<FileKind>? = nil, limit: Int = 40,
                       isCancelled: @Sendable () -> Bool = { false }) -> [SearchHit] {
        guard !isCancelled(), limit > 0, !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return [] }
        var hits: [SearchHit] = []
        if kinds == nil || kinds!.contains(.application) {
            hits.append(contentsOf: searchApplications(text, excluded: excluded, excludedTypes: excludedTypes,
                                                       limit: limit, isCancelled: isCancelled))
        }
        // No selected folders means applications only, never an implicit search of the whole home folder.
        let words = LocalSearch.terms(text)
        if !roots.isEmpty, !words.isEmpty, kinds != [.application], spotlightTimeout > 0, !isCancelled() {
            // Only alphanumeric tokens enter the Spotlight expression. Quotes, wildcards and backslashes in a
            // filename cannot become query syntax. Name matches receive a stronger score than text-only matches.
            let expression = words.map {
                "(kMDItemFSName == \"*\($0)*\"cd || kMDItemTextContent == \"*\($0)*\"cd)"
            }.joined(separator: " && ")
            if let query = MDQueryCreate(kCFAllocatorDefault, expression as CFString, nil, nil) {
                let folders = Array(Set(roots.flatMap { [$0.path, FolderSelection.realPath($0.path)] }))
                let skips = Array(Set(excluded.flatMap { [$0.path, FolderSelection.realPath($0.path)] }))
                MDQuerySetSearchScope(query, folders as CFArray, 0)
                MDQuerySetMaxCount(query, max(200, limit * 5))
                // MDQuery.h: flags=0 starts gathering asynchronously on this thread's run loop. A synchronous
                // query can stall for seconds, holding up every newer keystroke behind it. Drive short run-loop
                // slices instead, then stop before reading: MDQueryStop guarantees a static result list, including
                // any partial results gathered before the deadline. Query access stays on this one thread.
                let deadline = ContinuousClock.now + .seconds(spotlightTimeout)
                if MDQueryExecute(query, 0) {
                    while !MDQueryIsGatheringComplete(query), !isCancelled(), ContinuousClock.now < deadline {
                        let remaining = ContinuousClock.now.duration(to: deadline).components
                        let seconds = Double(remaining.seconds) + Double(remaining.attoseconds) / 1e18
                        CFRunLoopRunInMode(CFRunLoopMode.defaultMode, min(0.01, max(0, seconds)), true)
                    }
                    MDQueryStop(query)
                    guard !isCancelled() else { return [] }
                    for index in 0..<MDQueryGetResultCount(query) {
                        guard !isCancelled() else { return [] }
                        guard let pointer = MDQueryGetResultAtIndex(query, index) else { continue }
                        let item = Unmanaged<MDItem>.fromOpaque(pointer).takeUnretainedValue()
                        guard let path = MDItemCopyAttribute(item, kMDItemPath) as? String,
                              FolderSelection.isSearched(path, folders: folders, skipped: skips),
                              Self.allowed(path, excludedTypes: excludedTypes),
                              let result = hit(URL(fileURLWithPath: path), text: text, contentMatch: true) else { continue }
                        hits.append(result)
                    }
                } else {
                    MDQueryStop(query)
                }
            }
        }
        guard !isCancelled() else { return [] }
        return LocalSearch.merge([], with: hits.filter { kinds == nil || kinds!.contains($0.kind) }, limit: limit)
    }

    static func allowed(_ path: String, excludedTypes: Set<String>) -> Bool {
        let url = URL(fileURLWithPath: path)
        guard !excludedTypes.contains(url.pathExtension.lowercased()), !Rules.isSecret(url.lastPathComponent)
        else { return false }
        // Avoid hidden/private files, dependency trees, and the contents of app bundles.
        return !url.pathComponents.dropLast().contains {
            $0.hasPrefix(".") || $0.lowercased().hasSuffix(".app") || Rules.skippedFolderNames.contains($0)
        } && !url.lastPathComponent.hasPrefix(".")
    }

    private func refreshApplications(isCancelled: @Sendable () -> Bool) {
        var found = Set<URL>()
        for root in applicationRoots {
            guard !isCancelled() else { return }
            guard let walker = FileManager.default.enumerator(at: root, includingPropertiesForKeys: [.isPackageKey],
                    options: [.skipsHiddenFiles, .skipsPackageDescendants]) else { continue }
            for case let url as URL in walker where url.pathExtension.lowercased() == "app" {
                guard !isCancelled() else { return }
                found.insert(url.standardizedFileURL)
            }
        }
        var catalog: [Application] = []
        for url in found.sorted(by: { $0.path < $1.path }) {
            guard !isCancelled() else { return }
            let name = (try? url.resourceValues(forKeys: [.localizedNameKey]))?.localizedName ?? url.lastPathComponent
            catalog.append(Application(url: url, displayName: name))
        }
        applications = catalog
        refreshed = Date()
    }

    func hit(_ url: URL, text: String, application: Bool = false, contentMatch: Bool = false) -> SearchHit? {
        guard FileManager.default.fileExists(atPath: url.path),
              let values = try? url.resourceValues(forKeys: [.localizedNameKey, .isDirectoryKey, .isPackageKey,
                  .contentModificationDateKey, .fileSizeKey, .isSymbolicLinkKey, .contentTypeKey,
                  .isUbiquitousItemKey, .ubiquitousItemDownloadingStatusKey]) else { return nil }
        guard values.isSymbolicLink != true, !FileSystem.isInCloud(values),
              FileSystem.info(of: url.path)?.dataless != true else { return nil }
        let app = application || url.pathExtension.lowercased() == "app"
        let display = values.localizedName ?? url.lastPathComponent
        guard let score = [LocalSearch.nameScore(url.lastPathComponent, query: text),
                           LocalSearch.nameScore(display, query: text)].compactMap({ $0 }).max()
            ?? (contentMatch ? 2 : nil) else { return nil }
        let kind: FileKind
        if app { kind = .application }
        else if values.isDirectory == true && values.isPackage != true { kind = .folder }
        else if let type = values.contentType, type.conforms(to: .pdf) { kind = .pdf }
        else if let type = values.contentType, type.conforms(to: .image) {
            kind = Rules.isScreenshot(url) ? .screenshot : .image
        } else if let type = values.contentType, type.conforms(to: .audio) { kind = .audio }
        else if let type = values.contentType, type.conforms(to: .movie),
                !["ts", "mts"].contains(url.pathExtension.lowercased()) { kind = .video }
        else if Rules.docExtensions.contains(url.pathExtension.lowercased()) { kind = .doc }
        else { kind = .file }
        return SearchHit(path: url.path, kind: kind, score: score, z: nil, cosine: nil, keywordRank: 1,
            segment: .name, segmentID: -1, loc: nil, locEnd: nil, excerpt: nil,
            modified: values.contentModificationDate?.timeIntervalSince1970 ?? 0,
            size: Int64(values.fileSize ?? 0), info: ["display_name": display, "source": "spotlight"])
    }
}
#endif
