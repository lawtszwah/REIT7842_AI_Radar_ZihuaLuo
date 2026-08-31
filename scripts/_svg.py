"""Shared drawing primitives for the design diagrams."""

from __future__ import annotations

W, H = 1520, 900
ACCENT = {"a": "#1d4ed8", "b": "#6d28d9", "c": "#0f766e", "d": "#b45309", "e": "#be123c"}
INK, MUTED, LINE = "#0f172a", "#475569", "#cbd5e1"
FILL, WARN_FILL, WARN_LINE = "#f8fafc", "#fef3c7", "#d97706"
KEY_FILL, KEY_LINE = "#ecfdf5", "#0f766e"
STORE_FILL, STORE_LINE = "#eff6ff", "#1d4ed8"
MONO = "'SF Mono', ui-monospace, Menlo, monospace"
SANS = "Inter, 'Helvetica Neue', Helvetica, Arial, sans-serif"


class Canvas:
    def __init__(self, w: int = W, h: int = H):
        self.w, self.h = w, h
        self.out: list[str] = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
            f'width="{w}" height="{h}">',
            '<defs><marker id="arw" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
            'markerHeight="7" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" fill="#94a3b8"/>'
            '</marker></defs>',
        ]
        self.rect(0, 0, w, h, fill="#ffffff", stroke="#ffffff", rx=0)

    # -- primitives ---------------------------------------------------------
    @staticmethod
    def esc(s: str) -> str:
        return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    def rect(self, x, y, w, h, fill=FILL, stroke=LINE, rx=7, dash=None, sw=1.2):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
                        f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')

    def text(self, x, y, s, size=10, fill=INK, weight="normal", anchor="start",
             mono=False, style=None):
        st = f' font-style="{style}"' if style else ""
        self.out.append(f'<text x="{x}" y="{y}" font-family="{MONO if mono else SANS}" '
                        f'font-size="{size}" fill="{fill}" font-weight="{weight}" '
                        f'text-anchor="{anchor}"{st}>{self.esc(s)}</text>')

    def box(self, x, y, w, h, title, lines, accent, *, fill=FILL, stroke=LINE,
            dash=None, tag=None, size=9.6):
        self.rect(x, y, w, h, fill=fill, stroke=stroke, dash=dash)
        self.out.append(f'<rect x="{x}" y="{y}" width="4.5" height="{h}" rx="2" fill="{accent}"/>')
        self.text(x + 14, y + 19, title, size=11.5, weight="700")
        ty = y + 36
        for ln in lines:
            self.text(x + 14, ty, ln.strip("`"), size=size, fill=MUTED, mono=ln.startswith("`"))
            ty += 14.5
        if tag:
            tw = 7.0 * len(tag) + 12
            self.rect(x + w - tw - 10, y + 9, tw, 17, fill=accent, stroke=accent, rx=8)
            self.text(x + w - tw / 2 - 10, y + 21, tag, size=9.5, fill="#ffffff",
                      weight="700", anchor="middle")

    def arrow(self, x1, y1, x2, y2, color="#94a3b8", label=None, dash=None, above=7):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.out.append(f'<path d="M {x1} {y1} L {x2} {y2}" stroke="{color}" stroke-width="1.8" '
                        f'fill="none" marker-end="url(#arw)"{d}/>')
        if label:
            self.text((x1 + x2) / 2, (y1 + y2) / 2 - above, label, size=8.8,
                      fill=MUTED, anchor="middle")

    def elbow(self, x1, y1, x2, y2, color="#94a3b8", dash=None):
        """Right-angled connector: horizontal then vertical."""
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.out.append(f'<path d="M {x1} {y1} H {x2} V {y2}" stroke="{color}" stroke-width="1.8" '
                        f'fill="none" marker-end="url(#arw)"{d}/>')

    def header(self, title, subtitle, chips=None):
        self.text(40, 38, title, size=21, weight="700")
        self.text(40, 60, subtitle, size=11, fill=MUTED)
        if chips:
            for i, (label, colour) in enumerate(chips):
                x = 40 + i * 372
                self.rect(x, 72, 356, 22, fill="#ffffff", stroke=colour, rx=11)
                self.text(x + 12, 87, label, size=9.8, fill=colour, weight="600")

    def column_head(self, x, w, title, colour, y=122):
        self.text(x, y, title, size=13.5, weight="700", fill=colour)
        self.out.append(f'<path d="M {x} {y + 7} L {x + w} {y + 7}" stroke="{colour}" '
                        'stroke-width="2"/>')

    def save(self, path):
        import pathlib
        self.out.append("</svg>")
        p = pathlib.Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n".join(self.out))
        return p
