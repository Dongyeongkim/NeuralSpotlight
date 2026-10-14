# NeuralSpotlight identity

An original lens and connected-node N, in cyan and periwinkle on midnight blue. The main app, onboarding,
search home, and menu bar share this mark and the palette in `Components.swift`.

`scripts/render-brand.swift` draws every raster asset directly with AppKit; no upstream artwork is used.
Run `swift scripts/render-brand.swift`, then
`iconutil -c icns build.noindex/NeuralSpotlight.iconset -o Resources/AppIcon.icns`.
The macOS 26 Icon Composer document uses the transparent neural mark over a navy fill.
Older macOS versions use `Resources/AppIcon.icns`. Tray variants communicate progress and pause.
