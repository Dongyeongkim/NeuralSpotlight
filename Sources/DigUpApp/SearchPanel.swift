import AppKit
import DigUpKit
import Observation
import Quartz
import SwiftUI

enum SearchPhase: Equatable {
    case empty       // nothing typed
    case words       // keyword results, meaning results on the way
    case meaning     // words + meaning: final
    case wordsOnly   // final, but there's no meaning search (the model is downloading, or the encoder failed)
    case codeHint    // "code:" and nothing yet: what to type
    case codeOff     // "code:" without code folders: how to turn code search on
}

/// The hotkey panel: type, see results grouped by kind with the selected one previewed big on the right, and open /
/// reveal / Quick Look / copy / drag them out.
///
/// Every keystroke shows keyword results at once (file names, OCR and document text, the last word as a prefix).
/// After a short pause in typing the query is embedded and the full hybrid results replace them. Results far weaker
/// than the best one are folded under "N weaker matches".
@Observable
final class SearchPanelController {
    static let width: CGFloat = 780
    static let listWidth: CGFloat = 330
    static let fieldHeight: CGFloat = 56
    static let expandedHeight: CGFloat = 480
    static let rowsPerGroup = 6
    static let typingPause: Duration = .milliseconds(120)
    /// It fades in and out, like Spotlight's panel.
    static let fadeIn: TimeInterval = 0.15
    static let fadeOut: TimeInterval = 0.12
    /// The selectable row that folds and unfolds the weaker matches.
    static let weakerToggle = "\u{0}weaker"

    var query = "" {
        didSet { if query != oldValue { queryChanged() } }
    }
    private(set) var results = ResultSet.empty
    private(set) var showWeaker = false
    private(set) var selection: String?
    private(set) var phase = SearchPhase.empty
    /// The query is a search of code ("code: …"): the code index, and nothing else.
    private(set) var isCode = false
    private(set) var note: String?
    var encoder: EngineStatus.Encoder { model.status.encoder }

    @ObservationIgnored let model: AppModel
    @ObservationIgnored private let engine: Engine
    @ObservationIgnored private unowned let app: AppController
    @ObservationIgnored private var window: SearchPanelWindow?
    @ObservationIgnored weak var field: NSTextField?
    @ObservationIgnored private var generation = 0
    /// The generation whose final (meaning) results are showing; late keyword results must not replace them.
    @ObservationIgnored private var finalGeneration = -1
    @ObservationIgnored private var meaningTask: Task<Void, Never>?
    @ObservationIgnored private var keywordTask: Task<Void, Never>?
    @ObservationIgnored private var localTask: Task<Void, Never>?
    @ObservationIgnored private var keyMonitor: Any?
    @ObservationIgnored private var mouseMonitor: Any?
    @ObservationIgnored private var resignObserver: NSObjectProtocol?
    @ObservationIgnored private var keyWindowObserver: NSObjectProtocol?
    @ObservationIgnored private var resignKeyObserver: NSObjectProtocol?
    /// Fading out: still on screen, but gone as far as everything else is concerned.
    @ObservationIgnored private var fading = false
    @ObservationIgnored private var activity: NSObjectProtocol?
    /// True once ↑/↓ moved the selection since the last edit: then Space previews instead of typing a space.
    @ObservationIgnored private var navigated = false
    @ObservationIgnored private var typedAt = Date()
    @ObservationIgnored private var noteTask: Task<Void, Never>?
    /// Called when the final results for the current query are on screen (debug hooks use it).
    @ObservationIgnored var onFinalResults: (() -> Void)?
    /// Where ⌘C puts the file (debug runs use a private pasteboard so the real clipboard stays untouched).
    @ObservationIgnored var pasteboard = NSPasteboard.general

    init(app: AppController) {
        self.app = app
        model = app.model
        engine = app.engine
    }

    var isVisible: Bool { window?.isVisible == true && !fading }
    var windowNumber: Int { window?.windowNumber ?? 0 }
    var isPreviewing: Bool { QLPreviewPanel.sharedPreviewPanelExists() && QLPreviewPanel.shared().isVisible }
    /// The selectable rows in order: strong results group by group, the fold, then the weaker ones if unfolded.
    var rowIDs: [String] {
        results.groups.flatMap { $0.rows.map(\.id) }
            + (results.weaker.isEmpty ? [] : [Self.weakerToggle])
            + (showWeaker ? results.weaker.map(\.id) : [])
    }
    var rows: [ResultRow] { results.groups.flatMap(\.rows) + (showWeaker ? results.weaker : []) }
    var selectedRow: ResultRow? { rows.first { $0.id == selection } }

    // MARK: Showing

    func toggle() {
        isVisible ? hide() : show()
    }

    func show() {
        let started = Date()
        let foreground = NSWorkspace.shared.frontmostApplication?.bundleIdentifier ?? "unknown"
        engine.prewarm()
        let window = self.window ?? makeWindow()
        placeWindow(window)
        refresh()
        if AppController.offscreen { return }
        // Take keyboard focus without activating the app or switching to its other windows' Space.
        // The activity below keeps interactive search out of App Nap while this panel is visible.
        fading = false   // shown again while it faded out: it comes back from where it got to
        window.ignoresMouseEvents = false
        window.canHide = true
        if !window.isVisible { window.alphaValue = 0 }
        window.makeKeyAndOrderFront(nil)
        NSAnimationContext.runAnimationGroup { context in
            context.duration = Self.fadeIn
            window.animator().alphaValue = 1
        }
        if let field {
            window.makeFirstResponder(field)
            field.currentEditor()?.selectAll(nil)   // like Spotlight: the last query, selected
        }
        installKeyMonitor()
        if activity == nil {
            activity = ProcessInfo.processInfo.beginActivity(options: .userInitiatedAllowingIdleSystemSleep,
                                                             reason: "Search panel open")
        }
        let shown = Date()
        DispatchQueue.main.async {
            // The next turn of the main loop, after the window's first layout and display.
            log("panel: shown in \(ms(since: started, until: shown)), drawn by \(ms(since: started))")
            log("panel: foreground \(foreground) → \(NSWorkspace.shared.frontmostApplication?.bundleIdentifier ?? "unknown"), "
                + "key=\(window.isKeyWindow), activeSpace=\(window.isOnActiveSpace), appActive=\(NSApp.isActive)")
        }
    }

    func hide() {
        if QLPreviewPanel.sharedPreviewPanelExists(), QLPreviewPanel.shared().isVisible {
            QLPreviewPanel.shared().orderOut(nil)
        }
        removeKeyMonitor()
        PreviewPlayback.shared.stop()
        if let activity { ProcessInfo.processInfo.endActivity(activity) }
        activity = nil
        Thumbnails.shared.clear()
        Previews.shared.clear()
        guard let window, window.isVisible, !fading else { return }
        fading = true
        window.ignoresMouseEvents = true
        NSAnimationContext.runAnimationGroup { context in
            context.duration = Self.fadeOut
            window.animator().alphaValue = 0
        } completionHandler: { [weak self] in
            MainActor.assumeIsolated {
                guard let self, self.fading else { return }   // shown again meanwhile
                self.fading = false
                window.orderOut(nil)
                window.canHide = true
            }
        }
    }

    // MARK: Searching

    /// Reopening the panel or changing files must rerun an unchanged query too.
    func refresh() {
        guard !query.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }
        results = .empty
        selection = nil
        queryChanged()
    }

    private func queryChanged() {
        generation += 1
        let current = generation
        typedAt = Date()
        navigated = false
        meaningTask?.cancel()
        keywordTask?.cancel()
        localTask?.cancel()
        let typed = query.trimmingCharacters(in: .whitespacesAndNewlines)
        let codeText = CodeQuery.text(of: typed)
        isCode = codeText != nil
        let code = isCode
        let text = codeText ?? typed
        guard !typed.isEmpty else {
            results = .empty
            selection = nil
            phase = .empty
            resizeWindow()
            return
        }
        // "code:" alone, or without code folders: what to do instead of results.
        if code, !model.hasCode || text.isEmpty {
            results = .empty
            selection = nil
            phase = model.hasCode ? .codeHint : .codeOff
            resizeWindow()
            return
        }
        // Each source updates independently. Late Spotlight results enrich the current meaning results instead
        // of making the index wait, or replacing those results with a keyword-only list.
        var localHits: [SearchHit] = []
        var indexedHits: [SearchHit] = []
        var indexedPhase = SearchPhase.words
        var localFinished = code
        var meaningFinished = false
        let present = {
            self.show(LocalSearch.merge(indexedHits, with: localHits, limit: 40), query: text, phase: indexedPhase)
        }
        let finish = {
            if localFinished && meaningFinished { self.onFinalResults?() }
        }
        let showAt = ContinuousClock.now + Self.typingPause
        if !code {
            localTask = Task {
                try? await Task.sleep(until: showAt, tolerance: .milliseconds(1), clock: .continuous)
                guard !Task.isCancelled, current == generation else { return }
                let started = Date()
                let names = await engine.searchLocal(text)
                guard !Task.isCancelled, current == generation else { return }
                localHits = names
                present()
                log("search: system results \(names.count) hits in \(ms(since: started))")
                localFinished = true
                finish()
            }
        }
        // Indexed words now. The last word is probably half-typed, so it matches as a prefix.
        keywordTask = Task {
            let started = Date()
            let hits = await engine.search(text, vector: nil, prefixLast: true, code: code)
            guard !Task.isCancelled, current == generation, finalGeneration != current else { return }
            indexedHits = hits
            present()
            log("search \"\(typed)\": words \(hits.count) hits in \(ms(since: started))")
        }
        // Start model work once typing pauses, independently of Spotlight gathering.
        meaningTask = Task {
            try? await Task.sleep(until: showAt, tolerance: .milliseconds(1), clock: .continuous)
            guard !Task.isCancelled, current == generation else { return }
            engine.embedAhead(text, code: code)
            let embedStarted = Date()
            let vector = await engine.embedQuery(text, code: code)
            guard !Task.isCancelled, current == generation else { return }
            let searchStarted = Date()
            let hits = await engine.search(text, vector: vector, prefixLast: false, code: code)
            let searched = Date()
            try? await Task.sleep(until: showAt, tolerance: .milliseconds(1), clock: .continuous)
            guard !Task.isCancelled, current == generation else { return }
            finalGeneration = current
            indexedHits = hits
            indexedPhase = vector == nil ? .wordsOnly : .meaning
            present()
            let shown = Date(), keystroke = typedAt
            log("search \"\(typed)\": \(vector == nil ? "words only" : "meaning") \(hits.count) hits "
                + "(\(results.strongRows.count) strong) · embed \(ms(since: embedStarted, until: searchStarted)) · "
                + "search \(ms(since: searchStarted, until: searched)) · shown \(ms(since: keystroke, until: shown)) after "
                + "the last keystroke")
            DispatchQueue.main.async {
                // The next turn of the main loop: the list and the preview pane have been laid out and drawn.
                log("search \"\(typed)\": drawn by \(ms(since: keystroke)) after the last keystroke")
            }
            meaningFinished = true
            finish()
        }
    }

    private func show(_ hits: [SearchHit], query: String, phase: SearchPhase) {
        // Code is one group: it gets the room of all of them.
        results = ResultSet(hits, query: query, rowsPerGroup: isCode ? Self.rowsPerGroup * 3 : Self.rowsPerGroup)
        if !navigated { showWeaker = false }
        // Follow the top hit until you pick something with the arrow keys.
        if !navigated || !rowIDs.contains(where: { $0 == selection }) {
            selection = rowIDs.first
        }
        self.phase = phase
        resizeWindow()
        if QLPreviewPanel.sharedPreviewPanelExists(), QLPreviewPanel.shared().isVisible { updateQuickLook() }
    }

    func toggleWeaker() {
        showWeaker.toggle()
        selection = Self.weakerToggle
    }

    // MARK: Keys

    private func installKeyMonitor() {
        guard keyMonitor == nil else { return }
        keyMonitor = NSEvent.addLocalMonitorForEvents(matching: .keyDown) { [weak self] event in
            guard let self, self.isVisible, self.owns(event) else { return event }
            return self.handle(event) ? nil : event
        }
        // A click back in the search field: Space types again (after a click on a result, it previewed).
        mouseMonitor = NSEvent.addLocalMonitorForEvents(matching: .leftMouseDown) { [weak self] event in
            guard let self, let field, event.window === field.window else { return event }
            if field.bounds.contains(field.convert(event.locationInWindow, from: nil)) { navigated = false }
            return event
        }
        resignObserver = NotificationCenter.default.addObserver(
            forName: NSApplication.didResignActiveNotification, object: nil, queue: .main
        ) { [weak self] _ in
            MainActor.assumeIsolated { self?.hide() }
        }
        // Another of the app's windows came forward (⌘, or ⌘N while the panel was up): the panel steps away, like
        // Spotlight, rather than float over it and take its keys.
        keyWindowObserver = NotificationCenter.default.addObserver(
            forName: NSWindow.didBecomeKeyNotification, object: nil, queue: .main
        ) { [weak self] note in
            let window = note.object as? NSWindow
            MainActor.assumeIsolated {
                guard let self, let window, window !== self.window, !(window is QLPreviewPanel) else { return }
                self.hide()
            }
        }
        // A nonactivating panel can lose focus while the app was never active, so app deactivation alone
        // cannot detect a click back into Safari. Allow one turn for Quick Look to take the key first.
        resignKeyObserver = NotificationCenter.default.addObserver(
            forName: NSWindow.didResignKeyNotification, object: nil, queue: .main
        ) { [weak self] note in
            let resigned = note.object as? NSWindow
            MainActor.assumeIsolated {
                guard let self, resigned === self.window || resigned is QLPreviewPanel else { return }
                DispatchQueue.main.async { [weak self] in
                    guard let self, self.isVisible, self.window?.isKeyWindow != true else { return }
                    if let preview = NSApp.keyWindow as? QLPreviewPanel, preview.dataSource === self.window { return }
                    self.hide()
                }
            }
        }
    }

    private func removeKeyMonitor() {
        for monitor in [keyMonitor, mouseMonitor].compactMap({ $0 }) { NSEvent.removeMonitor(monitor) }
        keyMonitor = nil
        mouseMonitor = nil
        for observer in [resignObserver, keyWindowObserver, resignKeyObserver].compactMap({ $0 }) {
            NotificationCenter.default.removeObserver(observer)
        }
        resignObserver = nil
        keyWindowObserver = nil
        resignKeyObserver = nil
    }

    /// Keys typed in the panel, or in Quick Look while the panel is the one showing it.
    private func owns(_ event: NSEvent) -> Bool {
        if event.window === window { return true }
        guard QLPreviewPanel.sharedPreviewPanelExists(), let preview = QLPreviewPanel.shared() else { return false }
        return event.window === preview && preview.dataSource === window
    }

    /// True when the key was handled here (it then doesn't reach the search field).
    private func handle(_ event: NSEvent) -> Bool {
        // An input method is composing (Japanese, Chinese…): ↩, the arrows and Esc are its keys until it's done.
        if (field?.currentEditor() as? NSTextView)?.hasMarkedText() == true { return false }
        let modifiers = event.modifierFlags.intersection(.deviceIndependentFlagsMask)
            .subtracting([.capsLock, .numericPad, .function])
        let previewing = QLPreviewPanel.sharedPreviewPanelExists() && QLPreviewPanel.shared().isVisible
        switch event.keyCode {
        case 53:  // Esc
            if previewing { QLPreviewPanel.shared().orderOut(nil) } else { dismiss() }
            return true
        case 125, 126:  // ↓ ↑
            guard modifiers.isEmpty else { return false }
            moveSelection(by: event.keyCode == 125 ? 1 : -1)
            navigated = true
            return true
        case 36, 76:  // ↩, enter
            if selection == Self.weakerToggle, modifiers.isEmpty {
                toggleWeaker()
            } else if modifiers == .command {
                revealSelected()
            } else if modifiers.isEmpty {
                openSelected()
            } else {
                return false
            }
            return true
        case 49 where modifiers.isEmpty && (navigated || previewing) && selection != nil:  // Space
            if selection == Self.weakerToggle { toggleWeaker() } else { toggleQuickLook() }
            return true
        case 16 where modifiers == .command:  // ⌘Y
            toggleQuickLook()
            return true
        case 31 where modifiers == .command:  // ⌘O: the same search in the window, for a bigger look
            openInWindow()
            return true
        case 8 where modifiers == .command:
            // ⌘C copies the selected file, except right after you selected some of the query text (the panel opens
            // with the last query selected, and arrowing to a result means you want the file).
            let textSelected = (field?.currentEditor()?.selectedRange.length ?? 0) > 0
            if textSelected, !navigated { return false }
            copySelected()
            return true
        default:
            return false
        }
    }

    func moveSelection(by step: Int) {
        let ids = rowIDs
        guard !ids.isEmpty else { return }
        let index = selection.flatMap { ids.firstIndex(of: $0) } ?? -1
        selection = ids[max(0, min(ids.count - 1, index + step))]
        if QLPreviewPanel.sharedPreviewPanelExists(), QLPreviewPanel.shared().isVisible { updateQuickLook() }
    }

    func select(_ row: ResultRow) {
        selection = row.id
        navigated = true
    }

    // MARK: Actions

    /// ↩: the file in its app, at the page or moment that matched when the app can be told where.
    func openSelected() {
        guard let row = selectedRow else { return }
        engine.open(row)
        hide()
    }

    func revealSelected() {
        guard let row = selectedRow else { return }
        NSWorkspace.shared.activateFileViewerSelecting([row.url])
        hide()
    }

    func copySelected() {
        guard let row = selectedRow else { return }
        pasteboard.clearContents()
        // As a file, like ⌘C in Finder: paste into a folder, a chat or a mail.
        pasteboard.writeObjects([row.url as NSURL])
        flash("Copied \(row.name)")
    }

    /// The search window, with this query (and only one kind, if given): for a bigger look.
    func openInWindow(_ kind: ResultGroupKind? = nil) {
        let text = query
        hide()
        app.showSearchWindow(query: text, kind: kind)
    }

    /// No folders yet: hide, and open where they're chosen.
    func chooseFolders() {
        hide()
        app.chooseFolders()
    }

    /// "code:" without code search: hide, and open where code folders are chosen.
    func setUpCodeSearch() {
        hide()
        app.chooseCodeFolders()
    }

    /// An example from the code hint, as if typed.
    func tryExample(_ text: String) {
        query = text
    }

    /// Esc: hide, and give the keyboard back to the app you came from (at once: the fade doesn't hold it).
    private func dismiss() {
        // Closing the nonactivating panel returns focus to the underlying app without hiding our other windows.
        hide()
    }

    private func flash(_ text: String) {
        note = text
        noteTask?.cancel()
        noteTask = Task {
            try? await Task.sleep(for: .seconds(1.6))
            guard !Task.isCancelled else { return }
            note = nil
        }
    }

    // MARK: Quick Look

    func toggleQuickLook() {
        let panel = QLPreviewPanel.shared()!
        if QLPreviewPanel.sharedPreviewPanelExists(), panel.isVisible {
            panel.orderOut(nil)
        } else if selectedRow != nil {
            updateQuickLook()
            panel.makeKeyAndOrderFront(nil)
        }
    }

    /// Quick Look shows the selected result at its page or moment.
    private func updateQuickLook() {
        if let row = selectedRow { engine.recordUses([FileUse(path: row.id, date: Date())], source: "preview") }
        window?.previewURL = selectedRow?.url
        QLPreviewPanel.shared().reloadData()
        QuickLookMoment.show(selectedRow, in: QLPreviewPanel.shared())
    }

    // MARK: Window

    private func makeWindow() -> SearchPanelWindow {
        let window = SearchPanelWindow()
        let hosting = NSHostingView(rootView: SearchPanelView(controller: self))
        hosting.sizingOptions = []
        window.contentView = Self.background(containing: hosting)
        self.window = window
        return window
    }

    private var desiredHeight: CGFloat {
        query.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? Self.fieldHeight : Self.expandedHeight
    }

    /// Centered on the screen with the pointer, a little above the middle, like Spotlight.
    private func placeWindow(_ window: NSWindow) {
        let screen = NSScreen.screens.first { NSMouseInRect(NSEvent.mouseLocation, $0.frame, false) } ?? NSScreen.main
        let area = screen?.visibleFrame ?? NSRect(x: 0, y: 0, width: 1440, height: 900)
        let top = area.maxY - area.height * 0.16
        let height = desiredHeight
        window.setFrame(NSRect(x: area.midX - Self.width / 2, y: top - height, width: Self.width, height: height),
                        display: true)
    }

    /// Grows or shrinks downward, keeping the search field where it is.
    private func resizeWindow() {
        guard let window else { return }
        let height = desiredHeight
        guard abs(window.frame.height - height) > 0.5 else { return }
        // Not in the middle of a SwiftUI update: resizing from inside one sizes the view wrong.
        DispatchQueue.main.async {
            var frame = window.frame
            frame.origin.y += frame.height - height
            frame.size.height = height
            window.setFrame(frame, display: true)
        }
    }

    /// Glass (or vibrancy before macOS 26) behind the SwiftUI content. The content sits beside the glass rather than
    /// inside it, since NSGlassEffectView resizes its contentView itself.
    private static func background(containing content: NSView) -> NSView {
        let radius: CGFloat = 18
        let backdrop: NSView
        if #available(macOS 26.0, *) {
            let glass = NSGlassEffectView()
            glass.cornerRadius = radius
            backdrop = glass
        } else {
            let effect = NSVisualEffectView()
            effect.material = .popover
            effect.state = .active
            effect.blendingMode = .behindWindow
            effect.wantsLayer = true
            effect.layer?.cornerRadius = radius
            effect.layer?.masksToBounds = true
            backdrop = effect
        }
        let container = NSView()
        for view in [backdrop, content] {
            view.frame = container.bounds
            view.autoresizingMask = [.width, .height]
            container.addSubview(view)
        }
        return container
    }

    /// Draws the panel into a PNG (see `Snapshot`); the glass doesn't draw that way, so it's on the window color.
    func snapshot(to url: URL) throws {
        try Snapshot.write(window, to: url, cornerRadius: 18)
    }
}

/// The floating panel. Unlike most floating panels it becomes key, because you type into it.
final class SearchPanelWindow: NSPanel, QLPreviewPanelDataSource {
    /// What Space / ⌘Y previews.
    var previewURL: URL?

    init() {
        super.init(contentRect: NSRect(x: 0, y: 0, width: SearchPanelController.width,
                                       height: SearchPanelController.fieldHeight),
                   styleMask: [.borderless, .fullSizeContentView, .nonactivatingPanel], backing: .buffered, defer: true)
        isFloatingPanel = true
        level = .floating
        hidesOnDeactivate = false   // the controller hides it, and also tidies up
        collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary, .canJoinAllApplications, .ignoresCycle]
        isOpaque = false
        backgroundColor = .clear
        hasShadow = true
        isReleasedWhenClosed = false
        animationBehavior = .none
    }

    override var canBecomeKey: Bool { true }
    override var canBecomeMain: Bool { false }

    // Quick Look asks the responder chain who controls it; the panel does. (These come from a nonisolated NSObject
    // category; Quick Look calls them on the main thread.)
    nonisolated override func acceptsPreviewPanelControl(_ panel: QLPreviewPanel!) -> Bool { true }

    nonisolated override func beginPreviewPanelControl(_ panel: QLPreviewPanel!) {
        MainActor.assumeIsolated {
            panel.dataSource = self
            // Above the floating search panel.
            panel.level = NSWindow.Level(rawValue: level.rawValue + 1)
        }
    }

    nonisolated override func endPreviewPanelControl(_ panel: QLPreviewPanel!) {
        MainActor.assumeIsolated {
            panel.dataSource = nil
        }
    }

    func numberOfPreviewItems(in panel: QLPreviewPanel!) -> Int {
        previewURL == nil ? 0 : 1
    }

    func previewPanel(_ panel: QLPreviewPanel!, previewItemAt index: Int) -> (any QLPreviewItem)! {
        previewURL as NSURL?
    }
}
