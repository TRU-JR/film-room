"""Visual layer: theme CSS and small HTML helpers shared by every page."""
from __future__ import annotations

import html

import streamlit as st

NAVY = "#16233a"
GOLD = "#f2b705"  # the broadcast first-down line

CSS = """
<style>
@font-face { font-family: 'Saira Condensed'; font-weight: 600; font-style: normal; font-display: swap;
  src: url('app/static/fonts/saira-condensed-latin-600-normal.woff2') format('woff2'); }
@font-face { font-family: 'Saira Condensed'; font-weight: 700; font-style: normal; font-display: swap;
  src: url('app/static/fonts/saira-condensed-latin-700-normal.woff2') format('woff2'); }
@font-face { font-family: 'Saira Condensed'; font-weight: 800; font-style: normal; font-display: swap;
  src: url('app/static/fonts/saira-condensed-latin-800-normal.woff2') format('woff2'); }
@font-face { font-family: 'IBM Plex Sans'; font-weight: 400; font-style: normal; font-display: swap;
  src: url('app/static/fonts/ibm-plex-sans-latin-400-normal.woff2') format('woff2'); }
@font-face { font-family: 'IBM Plex Sans'; font-weight: 500; font-style: normal; font-display: swap;
  src: url('app/static/fonts/ibm-plex-sans-latin-500-normal.woff2') format('woff2'); }
@font-face { font-family: 'IBM Plex Sans'; font-weight: 600; font-style: normal; font-display: swap;
  src: url('app/static/fonts/ibm-plex-sans-latin-600-normal.woff2') format('woff2'); }

:root {
  --navy: #16233a; --navy-2: #22324f; --ink: #16233a; --ink-2: #4a5568; --muted: #7a8597;
  --paper: #eef1f5; --card: #ffffff; --line: #dce2ea; --gold: #f2b705;
  --good: #0a7f3f; --good-bg: #e3f4ea; --bad: #b42318; --bad-bg: #fbe9e7;
}
html, body, .stApp, [data-testid="stMarkdownContainer"], button, input, textarea, select {
  font-family: 'IBM Plex Sans', system-ui, sans-serif;
}
.stApp { background: var(--paper); }
h1, h2, h3, h4 { font-family: 'Saira Condensed', 'IBM Plex Sans', sans-serif !important;
  color: var(--ink) !important; letter-spacing: 0.01em; }
h3 { font-weight: 700 !important; font-size: 1.55rem !important; margin-bottom: .15rem !important; }
.block-container { padding-top: 4.2rem; max-width: 1320px; }

/* top navigation */
header[data-testid="stHeader"] { background: var(--navy); }
header[data-testid="stHeader"] a, header[data-testid="stHeader"] span, header[data-testid="stHeader"] p { color: #dfe6f0 !important; }
header[data-testid="stHeader"] [aria-current="page"] span, header[data-testid="stHeader"] [aria-current="page"] p { color: #ffffff !important; }
header[data-testid="stHeader"] [aria-current="page"] { box-shadow: inset 0 -3px 0 var(--gold); border-radius: 0 !important; }
header[data-testid="stHeader"] button svg { fill: #dfe6f0; color: #dfe6f0; }

/* sidebar = sideline */
section[data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid var(--line); min-width: 320px !important; }
section[data-testid="stSidebar"] h2 { font-size: 1.35rem !important; margin: 0 !important; }

/* field-style hero: yard lines with a gold first-down line */
.fr-hero { position: relative; overflow: hidden; border-radius: 14px; margin: 0 0 1.4rem 0;
  padding: 1.6rem 1.8rem 1.5rem; color: #fff;
  background:
    linear-gradient(90deg, transparent 0 calc(100% - 172px), var(--gold) calc(100% - 172px) calc(100% - 166px), transparent calc(100% - 166px)),
    repeating-linear-gradient(90deg, transparent 0 94px, rgba(255,255,255,.13) 94px 96px),
    repeating-linear-gradient(90deg, transparent 0 17px, rgba(255,255,255,.18) 17px 19px) top / 100% 10px no-repeat,
    repeating-linear-gradient(90deg, transparent 0 17px, rgba(255,255,255,.18) 17px 19px) bottom / 100% 10px no-repeat,
    linear-gradient(135deg, #16233a 0%, #1d3050 100%);
}
.fr-hero h1 { color: #fff !important; font-size: 3rem !important; line-height: 1 !important;
  font-weight: 800 !important; margin: 0 0 .45rem 0 !important; padding: 0 !important; max-width: 760px; }
.fr-hero p { color: #c9d3e1; margin: 0; max-width: 64ch; font-size: 1rem; }
.fr-hero .fr-kicker { color: var(--gold); font-weight: 600; font-size: .9rem; margin-bottom: .35rem; }

/* cards */
[class*="st-key-card"] { background: var(--card) !important; border-color: var(--line) !important; border-radius: 12px !important;
  padding: 1.1rem 1.25rem 1rem !important; }

/* scoreboard strip of key figures */
.fr-board { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  background: var(--card); border: 1px solid var(--line); border-radius: 12px; margin: .2rem 0 1.1rem; }
.fr-board > div { padding: .8rem 1rem; border-left: 1px solid var(--line); }
.fr-board > div:first-child { border-left: none; }
.fr-board .v { font-family: 'Saira Condensed', sans-serif; font-weight: 700; font-size: 2rem; line-height: 1.05;
  color: var(--ink); font-variant-numeric: tabular-nums; }
.fr-board .l { color: var(--muted); font-size: .85rem; }
.fr-board .s { color: var(--ink-2); font-size: .8rem; margin-top: .1rem; }

.fr-chip { display: inline-block; padding: .18rem .6rem; margin: 0 .35rem .35rem 0; border-radius: 999px;
  background: #e6ebf2; color: var(--ink); font-size: .85rem; font-weight: 500; }
.fr-chip.gold { background: #fff4cf; color: #6b4e00; }
.fr-chip.good { background: var(--good-bg); color: var(--good); }
.fr-chip.bad { background: var(--bad-bg); color: var(--bad); }
.fr-note { color: var(--muted); font-size: .88rem; margin: .1rem 0 .6rem; }
.fr-notes li { margin-bottom: .45rem; color: var(--ink); }

.fr-player { display: flex; gap: 1rem; align-items: center; }
.fr-avatar { width: 64px; height: 64px; border-radius: 50%; display: grid; place-items: center; flex: none;
  font-family: 'Saira Condensed', sans-serif; font-weight: 800; font-size: 1.6rem; color: #fff;
  border: 3px solid rgba(255,255,255,.55); }

.fr-slot { display: grid; grid-template-columns: 64px 1fr auto; align-items: center; gap: .75rem;
  padding: .55rem .2rem; border-bottom: 1px solid var(--line); }
.fr-slot:last-child { border-bottom: none; }
.fr-slot .tag { font-family: 'Saira Condensed', sans-serif; font-weight: 700; text-align: center;
  border-radius: 6px; padding: .15rem 0; background: #e6ebf2; color: var(--ink); }
.fr-slot .who { font-weight: 600; color: var(--ink); }
.fr-slot .who small { font-weight: 400; color: var(--muted); }
.fr-slot .empty { color: var(--muted); font-style: italic; }
.fr-slot .pts { font-family: 'Saira Condensed', sans-serif; font-weight: 700; font-size: 1.3rem;
  font-variant-numeric: tabular-nums; color: var(--ink); }
.fr-slot.bench .tag { background: transparent; border: 1px dashed var(--line); color: var(--muted); }

.stButton button[kind="primary"] { background: var(--navy); border-color: var(--navy); }
.stButton button[kind="primary"]:hover { background: var(--navy-2); border-color: var(--navy-2); }
button:focus-visible, a:focus-visible { outline: 3px solid var(--gold) !important; outline-offset: 2px; }
[data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 10px; }
@media (max-width: 640px) {
  .fr-hero { padding: 1.2rem 1.1rem; background: linear-gradient(135deg, #16233a, #1d3050); }
  .fr-hero h1 { font-size: 2.2rem !important; }
}
</style>
"""

POSITION_TINT = {"QB": "#2a78d6", "RB": "#d9541f", "WR": "#13865d", "TE": "#6250d6", "DST": "#2f6b2f"}


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def hero(title: str, text: str, kicker: str | None = None) -> None:
    k = f'<div class="fr-kicker">{html.escape(kicker)}</div>' if kicker else ""
    st.markdown(f'<div class="fr-hero">{k}<h1>{html.escape(title)}</h1><p>{html.escape(text)}</p></div>',
                unsafe_allow_html=True)


def board(items: list[tuple[str, str, str | None]]) -> None:
    """A strip of key figures: (value, label, small note)."""
    cells = "".join(
        f'<div><div class="v">{html.escape(v)}</div><div class="l">{html.escape(l)}</div>'
        + (f'<div class="s">{html.escape(s)}</div>' if s else "") + "</div>" for v, l, s in items)
    st.markdown(f'<div class="fr-board">{cells}</div>', unsafe_allow_html=True)


def chips(items: list[str], kind: str = "") -> None:
    st.markdown("".join(f'<span class="fr-chip {kind}">{html.escape(i)}</span>' for i in items),
                unsafe_allow_html=True)


def note(text: str) -> None:
    st.markdown(f'<p class="fr-note">{html.escape(text)}</p>', unsafe_allow_html=True)


def player_header(name: str, position: str, team: str, facts: list[str], flags: list[tuple[str, str]]) -> None:
    initials = "".join(w[0] for w in name.replace(".", "").split()[:2]).upper()
    tint = POSITION_TINT.get(position, "#16233a")
    flag_html = "".join(f'<span class="fr-chip {k}">{html.escape(t)}</span>' for t, k in flags)
    st.markdown(
        f'<div class="fr-hero"><div class="fr-player"><div class="fr-avatar" style="background:{tint}">{initials}</div>'
        f'<div><div class="fr-kicker">{html.escape(position)} for {html.escape(team)}</div>'
        f'<h1>{html.escape(name)}</h1><p>{html.escape("   |   ".join(facts))}</p></div></div>'
        f'<div style="margin-top:.8rem">{flag_html}</div></div>', unsafe_allow_html=True)


def slot_row(slot: str, who: str | None, sub: str, pts: str, bench: bool = False) -> str:
    body = (f'<div class="who">{html.escape(who)} <small>{html.escape(sub)}</small></div>' if who
            else '<div class="empty">Open slot</div>')
    return (f'<div class="fr-slot{" bench" if bench else ""}"><div class="tag">{html.escape(slot)}</div>'
            f'{body}<div class="pts">{html.escape(pts)}</div></div>')
