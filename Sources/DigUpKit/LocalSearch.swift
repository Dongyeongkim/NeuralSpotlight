import Foundation

/// Filename matching is independent of the embedding model and its content-word tokenizer.
public enum LocalSearch {
    static func normalized(_ text: String) -> String {
        text.folding(options: [.caseInsensitive, .diacriticInsensitive, .widthInsensitive],
                     locale: Locale(identifier: "en_US_POSIX")).precomposedStringWithCanonicalMapping
    }

    static func terms(_ text: String) -> [String] {
        normalized(text).components(separatedBy: CharacterSet.alphanumerics.inverted).filter { !$0.isEmpty }
    }

    static func nameScore(_ name: String, query: String) -> Double? {
        let name = normalized(name)
        let query = normalized(query.trimmingCharacters(in: .whitespacesAndNewlines))
        guard !query.isEmpty else { return nil }
        let stem = (name as NSString).deletingPathExtension
        if name == query || stem == query { return 8 }
        if name.hasPrefix(query) { return 6 }
        if name.contains(query) { return 4 }
        let words = terms(query)
        return !words.isEmpty && words.allSatisfy(name.contains) ? 3 : nil
    }

    /// Preserve a content hit's page, moment and snippet when its filename also matches.
    public static func merge(_ content: [SearchHit], with names: [SearchHit], limit: Int) -> [SearchHit] {
        var hits: [String: SearchHit] = [:]
        for hit in content { hits[hit.path] = hit }
        for name in names {
            if let hit = hits[name.path] {
                hits[name.path] = SearchHit(path: hit.path, kind: hit.kind, score: max(hit.score, name.score),
                    z: hit.z, cosine: hit.cosine, keywordRank: hit.keywordRank ?? name.keywordRank,
                    segment: hit.segment, segmentID: hit.segmentID, loc: hit.loc, locEnd: hit.locEnd,
                    excerpt: hit.excerpt, modified: hit.modified, size: hit.size, info: hit.info, focus: hit.focus)
            } else {
                hits[name.path] = name
            }
        }
        return Array(hits.values.sorted {
            if $0.score != $1.score { return $0.score > $1.score }
            if $0.kind == .application, $1.kind != .application { return true }
            if $1.kind == .application, $0.kind != .application { return false }
            return $0.path < $1.path
        }.prefix(max(0, limit)))
    }
}
