import NeuralSpotlightKit
import Foundation

/// What to index, where the index and the model live, and the hotkey.
///
/// Saved in the app's defaults domain (com.neuralspotlight.app). Launch arguments override any of it without being
/// saved, which is how tests point the app at the testbed and never at real folders:
///
///     NeuralSpotlight.app/Contents/MacOS/NeuralSpotlight -roots ~/NeuralSpotlightTestbed/real -indexDir <repo>/.dev/app-index \
///         -modelsDir <repo>/.dev/models -defaultsSuite com.neuralspotlight.dev
///
/// `-roots` takes one folder, several joined with ":", or a plist array. `-defaultsSuite` keeps what the app saves
/// (onboarding's folders, a new hotkey) out of the real settings. With no roots the app indexes nothing and opens
/// onboarding: a fresh install never starts reading someone's home folder on its own.
nonisolated struct AppSettings: Sendable {
    var roots: [URL]
    /// Folders inside the roots that are never indexed.
    var excludedFolders: [URL]
    /// File extensions that are never indexed ("heic", "txt"), on top of what NeuralSpotlight never reads anyway.
    var excludedTypes: [String]
    var indexDirectory: URL
    /// Where the model files are downloaded to and loaded from (the helpers get it as NEURALSPOTLIGHT_MODELS).
    var modelsDirectory: URL
    var embeddingBackend: EmbeddingBackend
    var hotkey: String
    /// On (the default), semantic indexing runs on battery too; off, it waits for a charger. Recent and new files
    /// obey the same gate; Low Power Mode or a hot Mac pause semantic indexing either way.
    var indexOnBattery: Bool
    /// "Index only at set hours": off by default. Outside the hours no helper runs (no crawl, no new files, no
    /// backfill); search works all day. The hours are kept while it's off.
    var indexHoursOn: Bool
    var indexHours: IndexHours
    /// The query encoder exits after this long without a query; the next debounced semantic query starts it again.
    var encoderIdleSeconds: Double
    /// Onboarding was finished (or closed after choosing folders); it doesn't open by itself again.
    var onboarded: Bool
    /// Code search: the folders whose code goes into the code index (none: code search is off), and the folders
    /// skipped in them (`FolderSelection`'s rule, as for `roots`). Searched with "code:" only.
    var codeRoots: [URL]
    var codeExcluded: [URL]
    /// The bundle id of the editor code results open in ("" for each file's default app); nil until one is chosen
    /// (`CodeEditor.chosen`).
    var codeEditor: String?
    /// The code index catches up once the code folders have been quiet this long (`-codeQuietSeconds`, default 60).
    var codeQuietSeconds: Double
    /// When on, ordinary documents (txt, md, rtf, docx, odt, html) are also embedded into the semantic index.
    var embedDocuments: Bool = false

    static let supportDirectory = AppPaths.applicationSupport
    static let defaultIndexDirectory = supportDirectory.appendingPathComponent("index.noindex")
    static let defaultHotkey = "cmd+shift+space"

    /// The battery switch, in onboarding and Settings.
    static let batteryTitle = "Index on battery power"
    static let batteryDetail = "When it's off, semantic indexing waits for a charger. Recently used files are processed first. Spotlight search stays available."

    /// Where settings are saved. Reads see launch arguments first either way. (UserDefaults is thread-safe.)
    nonisolated(unsafe) static let defaults: UserDefaults = {
        guard let suite = UserDefaults.standard.string(forKey: "defaultsSuite"), !suite.isEmpty,
              let defaults = UserDefaults(suiteName: suite) else { return .standard }
        return defaults
    }()

    static func load() -> AppSettings {
        importLegacyPreferences()
        let defaults = Self.defaults
        return AppSettings(
            roots: folders(defaults, "roots"),
            excludedFolders: folders(defaults, "excludedFolders"),
            excludedTypes: (defaults.stringArray(forKey: "excludedTypes")
                ?? defaults.string(forKey: "excludedTypes").map { [$0] } ?? [])
                .flatMap { IndexOptions.extensions($0) }.sorted(),
            indexDirectory: defaults.string(forKey: "indexDir").map(folderURL) ?? defaultIndexDirectory,
            modelsDirectory: defaults.string(forKey: "modelsDir").map(folderURL) ?? ModelFiles.installed,
            embeddingBackend: defaults.string(forKey: "embeddingBackend").flatMap(EmbeddingBackend.init(rawValue:))
                ?? EmbeddingBackend.environment,
            hotkey: defaults.string(forKey: "hotkey") ?? defaultHotkey,
            indexOnBattery: defaults.object(forKey: "indexOnBattery") == nil || defaults.bool(forKey: "indexOnBattery"),
            indexHoursOn: defaults.bool(forKey: "indexHoursOn"),
            indexHours: IndexHours(
                start: defaults.object(forKey: "indexHoursFrom") == nil ? IndexHours.night.start
                    : defaults.integer(forKey: "indexHoursFrom"),
                end: defaults.object(forKey: "indexHoursTo") == nil ? IndexHours.night.end
                    : defaults.integer(forKey: "indexHoursTo")),
            encoderIdleSeconds: defaults.object(forKey: "encoderIdleSeconds") == nil
                ? 600 : max(1, defaults.double(forKey: "encoderIdleSeconds")),
            onboarded: defaults.bool(forKey: "onboarded"),
            codeRoots: folders(defaults, "codeRoots"),
            codeExcluded: folders(defaults, "codeExcludedFolders"),
            codeEditor: defaults.string(forKey: "codeEditor"),
            codeQuietSeconds: defaults.object(forKey: "codeQuietSeconds") == nil
                ? 60 : max(1, defaults.double(forKey: "codeQuietSeconds")),
            embedDocuments: defaults.bool(forKey: "embedDocuments"))
    }

    /// A list of folders: a plist array, or paths joined with ":" (the form launch arguments use).
    /// Carry selected folders and shortcuts into the renamed app once. Never copy upstream updater state.
    private static func importLegacyPreferences() {
        guard UserDefaults.standard.string(forKey: "defaultsSuite") == nil,
              Bundle.main.bundleIdentifier == "com.neuralspotlight.app",
              !defaults.bool(forKey: "importedDigUpPreferences") else { return }
        let legacy = defaults.persistentDomain(forName: "com.abdurrahim.DigUp") ?? [:]
        let current = defaults.persistentDomain(forName: "com.neuralspotlight.app") ?? [:]
        let keys = ["roots", "excludedFolders", "excludedTypes", "indexDir", "modelsDir", "embeddingBackend",
                    "hotkey", "indexOnBattery", "indexHoursOn", "indexHoursFrom", "indexHoursTo",
                    "encoderIdleSeconds", "onboarded", "codeRoots", "codeExcludedFolders", "codeEditor",
                    "codeQuietSeconds", "embedDocuments"]
        for key in keys where current[key] == nil {
            if let value = legacy[key] { defaults.set(value, forKey: key) }
        }
        defaults.set(true, forKey: "importedDigUpPreferences")
    }

    private static func folders(_ defaults: UserDefaults, _ key: String) -> [URL] {
        let paths: [String]
        if let list = defaults.stringArray(forKey: key) {
            paths = list
        } else if let joined = defaults.string(forKey: key) {
            paths = joined.split(separator: ":").map(String.init)
        } else {
            paths = []
        }
        var seen = Set<String>()
        return paths.filter { !$0.isEmpty }.map(folderURL).filter { seen.insert($0.path).inserted }
    }

    static func save(roots: [URL]) {
        defaults.set(roots.map(\.path), forKey: "roots")
    }

    static func save(excludedFolders: [URL]) {
        defaults.set(excludedFolders.map(\.path), forKey: "excludedFolders")
    }

    static func save(excludedTypes: [String]) {
        defaults.set(excludedTypes, forKey: "excludedTypes")
    }

    static func save(hotkey: String) {
        defaults.set(hotkey, forKey: "hotkey")
    }

    static func save(indexOnBattery: Bool) {
        defaults.set(indexOnBattery, forKey: "indexOnBattery")
    }

    static func save(embedDocuments: Bool) {
        defaults.set(embedDocuments, forKey: "embedDocuments")
    }

    static func save(indexHoursOn: Bool, hours: IndexHours) {
        defaults.set(indexHoursOn, forKey: "indexHoursOn")
        defaults.set(hours.start, forKey: "indexHoursFrom")
        defaults.set(hours.end, forKey: "indexHoursTo")
    }

    static func saveOnboarded() {
        defaults.set(true, forKey: "onboarded")
    }

    static func save(codeRoots: [URL], excluded: [URL]) {
        defaults.set(codeRoots.map(\.path), forKey: "codeRoots")
        defaults.set(excluded.map(\.path), forKey: "codeExcludedFolders")
    }

    static func save(codeEditor: String) {
        defaults.set(codeEditor, forKey: "codeEditor")
    }

    /// Expands "~" and resolves symlinks, so crawled paths match the paths FSEvents reports.
    static func folderURL(_ path: String) -> URL {
        URL(fileURLWithPath: (path.trimmingCharacters(in: .whitespaces) as NSString).expandingTildeInPath)
            .resolvingSymlinksInPath().standardizedFileURL
    }

    static func isFolder(_ url: URL) -> Bool {
        var isFolder: ObjCBool = false
        return FileManager.default.fileExists(atPath: url.path, isDirectory: &isFolder) && isFolder.boolValue
    }
}

/// "~/NeuralSpotlightTestbed/real" instead of "/Users/…/NeuralSpotlightTestbed/real".
nonisolated func tildePath(_ path: String) -> String {
    (path as NSString).abbreviatingWithTildeInPath
}
