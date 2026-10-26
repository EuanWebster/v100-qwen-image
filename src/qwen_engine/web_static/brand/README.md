# Lightseed brand assets

The Lightseed wordmarks and app icons, served by the planning web UI
(`python -m qwen_engine.web.server`). Copied verbatim on 2026-10-26 from the
agreed brand drop at `/home/euan/Sandbox/v100-qwen-image/agreed-logos/`, which
stays outside this repository and is never git-tracked; these are artwork
inputs, so nothing here is hand-edited. The SHA-256 of every shipped copy is
pinned by `tests/test_web_static.py::test_brand_assets_are_the_agreed_files_and_stay_inert`.
Only these five files ship: the drop's symbol/banner/PNG exports and
`FONT-LICENSE.txt` are not needed — the wordmark lettering is outlined vector
path, so no font is embedded or fetched.

Each file has exactly one route and one content type; the type is sent with no
`charset` (an image that claims one is a lie the browser has to work around)
and never from a request-supplied path.

| file | what it is | route | `Content-Type` |
| --- | --- | --- | --- |
| `lightseed-logo-on-dark.svg` | wordmark, lettering `#F7F5EF` (brand ivory) | `/brand/lightseed-logo-on-dark.svg` | `image/svg+xml` |
| `lightseed-logo-on-light.svg` | wordmark, lettering `#0B102B` (brand ink) | `/brand/lightseed-logo-on-light.svg` | `image/svg+xml` |
| `favicon.svg` | square mark, 128 × 128 | `/brand/favicon.svg` | `image/svg+xml` |
| `favicon-180.png` | square mark, 180 × 180 | `/brand/favicon-180.png` | `image/png` |
| `favicon.ico` | square mark, frames 16/32/48/64 px | `/favicon.ico` (root only) | `image/x-icon` |

Both wordmark files are **503 × 148** px, so the aspect ratio is ≈ 3.40: at the
height the header uses (37 px in `app.css`) the lockup image is ≈ 126 px wide.
The two variants differ ONLY in the lettering colour: Original and Neon use
`on-dark`, Paper uses `on-light`. `app.css` has both `<img>` elements in the
markup and `display: none`s the one the current theme does not want, which
keeps exactly one visible logo — and therefore exactly one accessible name —
per theme. No CSS filter is used to recolour an SVG: it would tint the
cyan/periwinkle/violet gradient too.

The wordmarks are the only artwork in the header. The square mark below is the
**app icon**: tabs, bookmarks, home-screen shortcuts, the taskbar. It is NOT a
header element, and it must not be used as one — it carries no letters, so a
header that shows only the tile is the bug the rebrand fixed.

`favicon-180.png` is the Apple home-screen touch icon; the ICO's largest frame
is 64 × 64, which is what legacy `favicon.ico` consumers scale. The ICO is
deliberately ALSO published at the root path (`/favicon.ico`), because that is
where browsers look when no `<link rel="icon">` applies: it stays listed in
`STATIC_ROUTES` and `pyproject.toml` under exactly that name — dropping either
is a regression.

## Contrast invariants (WCAG 2.1 AA)

`--accent` is used as *text* (`.recipe-note`, `.transparency-note`,
`.back-link`) and as the *background* of filled buttons carrying `--on-accent`.
Both pairs clear 4.5:1 in every theme, which is what
`tests/test_web_static.py::test_accent_tokens_pass_wcag_contrast_in_every_theme`
recomputes from `app.css` rather than trusting this table:

| theme | `--bg` / `--bg-panel` | `--accent` | accent on those | `--on-accent` | on `--accent` |
| --- | --- | --- | --- | --- | --- |
| Original | `#16181b` / `#1d2024` | `#7C8CFF` | 5.97 / 5.49 | `#14161a` | 6.08 |
| Neon | `#07090f` / `#0d1220` | `#22d3ee` | 11.01 / 10.33 | `#062026` | 9.35 |
| Paper | `#f6f3ec` / `#fffdf7` | `#3F49C4` | 6.40 / 6.97 | `#F7F5EF` | 6.50 |

The retired amber/brown set also cleared 4.5:1 as text (Original `#d9a468`
8.01, Paper `#8a5a24` 5.31), so this was a palette change that had to *keep*
AA, not an AA repair. The binding constraint is that one bright accent cannot
serve both surfaces: `#7C8CFF` on the brand ivory reaches only 2.69:1, which is
why Paper carries a darkened periwinkle instead of the Original's. Neon's cyan
is pre-existing and untouched. `--accent-dim` is border-only (focus ring,
button edge), so it is held to the 3:1 non-text bar: 3.68 / 3.72 / 8.93.
