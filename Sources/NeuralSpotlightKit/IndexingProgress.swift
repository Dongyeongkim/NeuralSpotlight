import Foundation

/// Collection progress, independent of the small batches used to prioritize recently opened files.
/// Processed files include successes, failures and deliberate skips; searchable counts are shown separately.
public struct IndexingProgress: Sendable, Equatable {
    public let files: Int
    public let pending: Int

    public init(files: Int, pending: Int) {
        self.files = max(0, files)
        self.pending = min(max(0, pending), self.files)
    }

    public var processed: Int { files - pending }
    public var fraction: Double? { files > 0 ? Double(processed) / Double(files) : nil }
    public var percent: Int? { fraction.map { Int(($0 * 100).rounded(.down)) } }
}
