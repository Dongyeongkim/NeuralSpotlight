import Foundation

/// Query-independent, stratified reservoir: top ANN neighbors never set their own relevance threshold.
/// Population weighting keeps a small audio collection from outweighing a large image collection.
struct MeaningScale {
    let means: [Modality: Double]
    let center: Double
    let deviation: Double

    init(scores: [Float], modalities: [Modality], populations: [Modality: Int]) {
        var sums: [Modality: Double] = [:], counts: [Modality: Int] = [:]
        for (score, modality) in zip(scores, modalities) {
            sums[modality, default: 0] += Double(score)
            counts[modality, default: 0] += 1
        }
        let total = Double(max(populations.values.reduce(0, +), 1))
        let rawMeans = sums.map { modality, sum in (modality, sum / Double(counts[modality]!)) }
        let raw = Dictionary(uniqueKeysWithValues: rawMeans)
        let all = populations.reduce(0.0) { $0 + raw[$1.key, default: 0] * Double($1.value) } / total
        var means: [Modality: Double] = [:]
        for modality in Modality.allCases { means[modality] = populations[modality, default: 0] >= 20 ? raw[modality, default: all] : all }
        var sum = 0.0, squares = 0.0
        for (score, modality) in zip(scores, modalities) {
            let weight = Double(populations[modality, default: 0]) / Double(max(counts[modality, default: 0], 1))
            let centered = Double(score) - means[modality, default: 0]
            sum += weight * centered
            squares += weight * centered * centered
        }
        self.means = means
        center = sum / total
        deviation = max(sqrt(max(0, squares / total - center * center)), 0.02)
    }

    func score(_ cosine: Float, modality: Modality) -> Double {
        (Double(cosine) - means[modality, default: 0] - center) / deviation
    }

    static func sample(modalities: [Modality], setsScale: [Bool], perModality: Int = 1024)
        -> (rows: [Int], populations: [Modality: Int]) {
        var reservoirs: [Modality: [Int]] = [:], populations: [Modality: Int] = [:]
        // SplitMix64, seeded independently of query and process hash randomization.
        var state: UInt64 = 0x243F6A8885A308D3
        func next() -> UInt64 {
            state &+= 0x9E3779B97F4A7C15
            var value = state
            value = (value ^ (value >> 30)) &* 0xBF58476D1CE4E5B9
            value = (value ^ (value >> 27)) &* 0x94D049BB133111EB
            return value ^ (value >> 31)
        }
        for row in modalities.indices where setsScale[row] {
            let modality = modalities[row]
            populations[modality, default: 0] += 1
            let seen = populations[modality]!
            if seen <= perModality { reservoirs[modality, default: []].append(row) }
            else {
                let slot = Int(next() % UInt64(seen))
                if slot < perModality { reservoirs[modality]![slot] = row }
            }
        }
        return (Modality.allCases.flatMap { reservoirs[$0] ?? [] }, populations)
    }
}
