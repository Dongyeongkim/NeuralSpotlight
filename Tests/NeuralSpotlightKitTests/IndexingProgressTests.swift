import Testing
@testable import NeuralSpotlightKit

struct IndexingProgressTests {
    @Test func completedRecentBatchDoesNotFinishCollection() {
        // A four-file recent-first batch has finished, but the other 597 files still need indexing.
        let progress = IndexingProgress(files: 601, pending: 597)
        #expect(progress.processed == 4)
        #expect(progress.percent == 0)
        #expect(progress.fraction! < 0.01)

        // The next completed eight-file batch must advance the same collection counter.
        let next = IndexingProgress(files: 601, pending: 589)
        #expect(next.processed == 12)
        #expect(next.percent == 1)
        #expect(next.fraction! > progress.fraction!)
    }

    @Test func newlyDiscoveredFilesExtendWork() {
        let complete = IndexingProgress(files: 4, pending: 0)
        #expect(complete.percent == 100)
        let expanded = IndexingProgress(files: 773, pending: 769)
        #expect(expanded.processed == 4)
        #expect(expanded.percent == 0)
    }

    @Test func emptyOrTemporarilyInconsistentCountsDoNotProduceInvalidProgress() {
        #expect(IndexingProgress(files: 0, pending: 0).fraction == nil)
        #expect(IndexingProgress(files: 4, pending: 6).fraction == 0)
        #expect(IndexingProgress(files: 4, pending: -1).fraction == 1)
    }
}
