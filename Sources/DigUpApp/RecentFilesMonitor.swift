import Foundation
import DigUpKit

/// A live Spotlight query, not global UI monitoring. No polling or Accessibility permission.
/// Only the system's last-used date is observed; automatic row previews are not engagement.
final class RecentFilesMonitor {
    private var query: NSMetadataQuery?
    private var observers: [NSObjectProtocol] = []
    private var seen: [String: Date] = [:]
    private let onUse: @Sendable ([FileUse]) -> Void

    init(onUse: @escaping @Sendable ([FileUse]) -> Void) { self.onUse = onUse }

    func watch(_ roots: [URL]) {
        query?.stop()
        observers.forEach(NotificationCenter.default.removeObserver)
        observers = []
        query = nil
        seen = [:]
        guard !roots.isEmpty else { return }
        let query = NSMetadataQuery()
        query.searchScopes = roots
        query.predicate = NSPredicate(format: "%K >= %@", "kMDItemLastUsedDate",
                                      Date().addingTimeInterval(-30 * 86400) as NSDate)
        query.notificationBatchingInterval = 5
        for name in [Notification.Name.NSMetadataQueryDidFinishGathering, .NSMetadataQueryDidUpdate] {
            observers.append(NotificationCenter.default.addObserver(forName: name, object: query, queue: .main) {
                [weak self] _ in MainActor.assumeIsolated { self?.receive() }
            })
        }
        self.query = query
        if !query.start() { log("recent files: Spotlight monitoring unavailable") }
    }

    private func receive() {
        guard let query else { return }
        query.disableUpdates()
        defer { query.enableUpdates() }
        let cutoff = Date().addingTimeInterval(-30 * 86400)
        var uses: [FileUse] = []
        var latest: [String: Date] = [:]
        for index in 0..<query.resultCount {
            guard let item = query.result(at: index) as? NSMetadataItem,
                  let path = item.value(forAttribute: "kMDItemPath") as? String,
                  let date = item.value(forAttribute: "kMDItemLastUsedDate") as? Date, date >= cutoff else { continue }
            latest[path] = date
            if seen[path].map({ $0 < date }) ?? true { uses.append(FileUse(path: path, date: date)) }
        }
        seen = latest
        if !uses.isEmpty { onUse(uses) }
    }
}
