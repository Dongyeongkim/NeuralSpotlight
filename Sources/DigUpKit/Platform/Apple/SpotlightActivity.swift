#if os(macOS)
import CoreServices
import Foundation

public struct FileUse: Sendable {
    public let path: String
    public let date: Date
    public init(path: String, date: Date) { self.path = path; self.date = date }
}

/// System-reported use is a hint: applications do not all update kMDItemLastUsedDate.
public enum SpotlightActivity {
    public static func recent(roots: [URL]) -> [FileUse] {
        guard !roots.isEmpty, let query = MDQueryCreate(kCFAllocatorDefault,
            "kMDItemLastUsedDate >= $time.today(-30)" as CFString, nil, nil) else { return [] }
        MDQuerySetSearchScope(query, roots.map(\.path) as CFArray, 0)
        defer { MDQueryStop(query) }
        guard MDQueryExecute(query, CFOptionFlags(kMDQuerySynchronous.rawValue)) else { return [] }
        var uses: [FileUse] = []
        for i in 0..<MDQueryGetResultCount(query) {
            guard let pointer = MDQueryGetResultAtIndex(query, i) else { continue }
            let item = Unmanaged<MDItem>.fromOpaque(pointer).takeUnretainedValue()
            if let path = MDItemCopyAttribute(item, kMDItemPath) as? String,
               let date = MDItemCopyAttribute(item, kMDItemLastUsedDate) as? Date {
                uses.append(FileUse(path: path, date: date))
            }
        }
        return uses
    }

    /// Apply the same local/private/type rules before saving even an activity timestamp.
    public static func eligible(_ path: String, code: Bool, excludedTypes: Set<String>) -> Bool {
        guard LocalSearchProvider.allowed(path, excludedTypes: excludedTypes),
              let info = FileSystem.info(of: path), info.isRegular, !info.dataless else { return false }
        let url = URL(fileURLWithPath: path)
        guard let values = try? url.resourceValues(forKeys: Set(FileSystem.crawlKeys)),
              !FileSystem.isInCloud(values) else { return false }
        var options = IndexOptions()
        options.code = code
        options.excludedExtensions = excludedTypes
        let classification = code ? Rules.classifyCode(url, options: options)
            : Rules.classify(url, values: values, options: options)
        if case .index(let kind) = classification { return SemanticPolicy.embeds(kind) }
        return false
    }
}
#endif
