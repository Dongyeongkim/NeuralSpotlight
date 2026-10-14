// Regenerates NeuralSpotlight's original vector-drawn icon and tray assets. Run with `swift scripts/render-brand.swift`.
import AppKit

let root = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
let assets = root.appendingPathComponent("Resources/AppIcon")
func color(_ r: CGFloat, _ g: CGFloat, _ b: CGFloat, _ a: CGFloat = 1) -> NSColor {
    NSColor(srgbRed: r, green: g, blue: b, alpha: a)
}
let cyan = color(0.30, 0.88, 0.96), blue = color(0.36, 0.46, 0.98)
func render(_ size: Int, background: Bool, paused: Bool = false, dim: Bool = false, to url: URL) throws {
    let rep = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: size, pixelsHigh: size, bitsPerSample: 8,
                              samplesPerPixel: 4, hasAlpha: true, isPlanar: false,
                              colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0)!
    NSGraphicsContext.saveGraphicsState()
    NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: rep)
    let transform = AffineTransform(scale: CGFloat(size) / 1024)
    (transform as NSAffineTransform).concat()
    if background {
        let shape = NSBezierPath(roundedRect: NSRect(x: 40, y: 40, width: 944, height: 944), xRadius: 208, yRadius: 208)
        NSGradient(starting: color(0.12, 0.20, 0.37), ending: color(0.04, 0.07, 0.16))!.draw(in: shape, angle: -50)
    }
    NSGraphicsContext.current?.cgContext.setAlpha(dim ? 0.32 : 1)
    if paused {
        cyan.setFill()
        for x in [360, 560] { NSBezierPath(roundedRect: NSRect(x: x, y: 300, width: 100, height: 424), xRadius: 32, yRadius: 32).fill() }
    } else {
        // A lens surrounds three connected nodes, with the diagonal of an N inside it.
        let lens = NSBezierPath(ovalIn: NSRect(x: 216, y: 306, width: 500, height: 500))
        lens.lineWidth = 60; cyan.setStroke(); lens.stroke()
        let handle = NSBezierPath(); handle.move(to: NSPoint(x: 652, y: 370)); handle.line(to: NSPoint(x: 812, y: 210))
        handle.lineWidth = 80; handle.lineCapStyle = .round; blue.setStroke(); handle.stroke()
        let network = NSBezierPath(); network.move(to: NSPoint(x: 350, y: 430)); network.line(to: NSPoint(x: 350, y: 650))
        network.line(to: NSPoint(x: 580, y: 460)); network.line(to: NSPoint(x: 580, y: 680))
        network.lineWidth = 30; network.lineJoinStyle = .round; color(0.78, 0.84, 1).setStroke(); network.stroke()
        for (x,y) in [(350,430),(350,650),(580,460),(580,680)] {
            color(0.90, 0.96, 1).setFill(); NSBezierPath(ovalIn: NSRect(x: x-30, y: y-30, width: 60, height: 60)).fill()
        }
    }
    NSGraphicsContext.restoreGraphicsState()
    try rep.representation(using: .png, properties: [:])!.write(to: url)
}
let iconset = root.appendingPathComponent("build.noindex/NeuralSpotlight.iconset")
try FileManager.default.createDirectory(at: iconset, withIntermediateDirectories: true)
for size in [16,32,128,256,512] {
    try render(size, background: true, to: iconset.appendingPathComponent("icon_\(size)x\(size).png"))
    try render(size*2, background: true, to: iconset.appendingPathComponent("icon_\(size)x\(size)@2x.png"))
}
try render(1024, background: false, to: assets.appendingPathComponent("AppIcon.icon/Assets/neural-mark.png"))
try render(256, background: true, to: root.appendingPathComponent("docs/logo.png"))
for (name,paused,dim) in [("tray",false,false),("tray-dim",false,true),("tray-paused",true,false)] {
    try render(64, background: false, paused: paused, dim: dim, to: assets.appendingPathComponent("\(name).png"))
}
