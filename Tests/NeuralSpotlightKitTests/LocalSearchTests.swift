import Foundation
import Testing
@testable import NeuralSpotlightKit

@Suite struct LocalSearchTests {
    @Test func namesMatchWithoutContentStopwords() {
        #expect(LocalSearch.nameScore("Files.app", query: "files") == 8)
        #expect(LocalSearch.nameScore("R.app", query: "r") == 8)
        #expect(LocalSearch.nameScore("Report.zip", query: "rep") == 6)
        #expect(LocalSearch.nameScore("2026 Report.zip", query: "report") == 4)
        #expect(LocalSearch.nameScore("Research Notes.zip", query: "notes research") == 3)
        #expect(LocalSearch.nameScore("Report.zip", query: "   ") == nil)
        #expect(LocalSearch.nameScore("Report.zip", query: "tax") == nil)
    }

    @Test func unicodeAndQuerySyntax() {
        #expect(LocalSearch.nameScore("연구계획서.pdf", query: "연구") == 6)
        #expect(LocalSearch.nameScore("연구계획서.pdf".decomposedStringWithCanonicalMapping, query: "연구") == 6)
        #expect(LocalSearch.nameScore("Résumé.pdf", query: "resume") == 8)
        #expect(LocalSearch.terms("\" || kMDItemFSName == '*' ").allSatisfy {
            $0.unicodeScalars.allSatisfy(CharacterSet.alphanumerics.contains)
        })
        #expect(LocalSearch.nameScore("notes.txt", query: "*") == nil)
    }

    @Test func mergingRetainsPageAndSnippet() {
        let content = SearchHit(path: "/D/report.pdf", kind: .pdf, score: 3, z: 3, cosine: 0.8,
            keywordRank: nil, segment: .page, segmentID: 19, loc: 7, locEnd: nil, excerpt: "the matched clause",
            modified: 1, size: 10, info: [:])
        let name = SearchHit(path: content.path, kind: .file, score: 8, z: nil, cosine: nil,
            keywordRank: 1, segment: .name, segmentID: -1, loc: nil, locEnd: nil, excerpt: nil,
            modified: 1, size: 10, info: [:])
        let merged = LocalSearch.merge([content], with: [name, name], limit: 40)
        #expect(merged.count == 1)
        #expect(merged.first?.score == 8)
        #expect(merged.first?.kind == .pdf)
        #expect(merged.first?.loc == 7)
        #expect(merged.first?.segmentID == 19)
        #expect(merged.first?.excerpt == "the matched clause")
        #expect(LocalSearch.merge([content], with: [name], limit: 0).isEmpty)
    }

    #if os(macOS)
    @Test func metadataFiltersRespectExcludedAndPrivatePaths() {
        #expect(LocalSearchProvider.allowed("/D/report.zip", excludedTypes: []))
        #expect(!LocalSearchProvider.allowed("/D/report.zip", excludedTypes: ["zip"]))
        #expect(!LocalSearchProvider.allowed("/D/.ssh/config", excludedTypes: []))
        #expect(!LocalSearchProvider.allowed("/D/node_modules/package/file.txt", excludedTypes: []))
        #expect(!LocalSearchProvider.allowed("/D/Test.app/Contents/Info.plist", excludedTypes: []))
        #expect(!LocalSearchProvider.allowed("/D/password.key", excludedTypes: []))
        #expect(!LocalSearchProvider.allowed("/D/.env", excludedTypes: []))
    }

    @Test func applicationsWorkWithoutSelectedFoldersOrModels() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let app = root.appendingPathComponent("Utilities/Example Editor.app")
        try FileManager.default.createDirectory(at: app, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let provider = LocalSearchProvider(applicationRoots: [root])
        func search(_ text: String, types: Set<String> = [], skips: [URL] = []) -> [SearchHit] {
            provider.search(text, roots: [], excluded: skips, excludedTypes: types)
        }
        let results = search("example")
        #expect(results.count == 1)
        #expect(results.first?.path == app.path)
        #expect(results.first?.kind == .application)
        #expect(search("example", types: ["app"]).isEmpty)
        #expect(search("example", skips: [root]).isEmpty)
        #expect(search("").isEmpty)
        try FileManager.default.removeItem(at: app)
        #expect(search("example").isEmpty) // stale cached application is not offered
    }

    @Test func cancelledCatalogDoesNotPoisonTheNextSearch() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        for name in ["Example One.app", "Example Two.app", "Example Three.app"] {
            try FileManager.default.createDirectory(at: root.appendingPathComponent(name), withIntermediateDirectories: true)
        }
        let provider = LocalSearchProvider(applicationRoots: [root])
        // Cancel while discovering the catalog, rather than before the search starts. Its incomplete catalog must
        // not be cached for the next minute, and a cancelled request must not offer stale application matches.
        let cancellation = CancelAfterChecks(4)
        #expect(provider.search("example", roots: [], excluded: [], excludedTypes: [],
                                isCancelled: { cancellation.isCancelled() }).isEmpty)
        #expect(provider.search("example", roots: [], excluded: [], excludedTypes: []).count == 3)
        #expect(provider.search("example", roots: [], excluded: [], excludedTypes: [],
                                isCancelled: { true }).isEmpty)
    }

    @Test func exhaustedSpotlightBudgetKeepsApplications() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let app = root.appendingPathComponent("Example.app")
        try FileManager.default.createDirectory(at: app, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let provider = LocalSearchProvider(applicationRoots: [root], spotlightTimeout: 0)
        let hits = provider.search("example", roots: [root], excluded: [], excludedTypes: [])
        #expect(hits.map(\.path) == [app.path])
        #expect(hits.first?.kind == .application)
    }

    @Test func nativeSpotlightQueryReturnsWithinItsBudget() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let provider = LocalSearchProvider(applicationRoots: [], spotlightTimeout: 0.02)
        let started = ContinuousClock.now
        let hits = provider.search("nothing matches this empty directory", roots: [root], excluded: [], excludedTypes: [])
        #expect(hits.isEmpty)
        // Allow scheduling and native query setup/stop overhead; the old synchronous query had no upper bound.
        #expect(started.duration(to: .now) < .seconds(1))
    }

    @Test(.enabled(if: (ProcessInfo.processInfo.environment["NEURALSPOTLIGHT_TEST_SPOTLIGHT"] ?? ProcessInfo.processInfo.environment["DIGUP_TEST_SPOTLIGHT"]) == "1"))
    func nativeSpotlightReturnsSystemApplication() {
        // Opt-in integration check: sandboxed/headless builders may not have access to the system metadata service.
        // The catalog is deliberately empty, so only the async Spotlight query can produce this result.
        let provider = LocalSearchProvider(applicationRoots: [], spotlightTimeout: 0.3)
        let hits = provider.search("Calculator", roots: [URL(fileURLWithPath: "/System/Applications")],
                                   excluded: [], excludedTypes: [])
        #expect(hits.contains { $0.path == "/System/Applications/Calculator.app" })
    }

    private final class CancelAfterChecks: @unchecked Sendable {
        private let lock = NSLock()
        private var remaining: Int

        init(_ checks: Int) { remaining = checks }

        func isCancelled() -> Bool {
            lock.withLock {
                remaining -= 1
                return remaining <= 0
            }
        }
    }

    @Test func arbitraryFilesAndFoldersNeedNoExtractor() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("Research-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let file = root.appendingPathComponent("Research.zip")
        try Data().write(to: file)
        let provider = LocalSearchProvider(applicationRoots: [])
        #expect(provider.hit(file, text: "Research")?.kind == .file)
        #expect(provider.hit(file, text: "unrelated") == nil)
        #expect(provider.hit(file, text: "unrelated", contentMatch: true)?.score == 2)
        #expect(provider.hit(root, text: "Research")?.kind == .folder)
        let link = root.appendingPathComponent("Research-link.zip")
        try FileManager.default.createSymbolicLink(at: link, withDestinationURL: file)
        #expect(provider.hit(link, text: "Research") == nil)
    }
    #endif
}
