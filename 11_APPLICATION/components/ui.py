"""Shared UI primitives: safety framing, flag rendering, provenance blocks.

Every visual element that could be mistaken for a prescribing instruction is
built here, once, so the framing cannot drift between tabs.
"""
from __future__ import annotations

import streamlit as st

SAFETY_DISCLAIMER = (
    "**This application is a research / clinical decision-support prototype. "
    "It does not replace qualified healthcare professionals, antimicrobial "
    "susceptibility testing, clinical guidelines, or professional medical "
    "judgement. AI-generated predictions and recommendations must be reviewed "
    "by an appropriately qualified healthcare professional.**"
)

SHORT_DISCLAIMER = (
    "Research prototype — decision support only. Not a prescribing system. "
    "All output requires clinician review."
)

SEVERITY_STYLE = {
    "critical": ("🛑", "#dc2626", "#fee2e2"),
    "caution": ("⚠️", "#b45309", "#fef3c7"),
    "info": ("ℹ️", "#1d4ed8", "#dbeafe"),
}


def page_config(title: str, icon: str = "🧫") -> None:
    st.set_page_config(page_title=title, page_icon=icon, layout="wide",
                       initial_sidebar_state="expanded")


def inject_css() -> None:
    st.markdown(
        """
        <style>
          :root {
              --cds-bg: #f4f6f9;
              --cds-card: #ffffff;
              --cds-border: #e1e6ed;
              --cds-navy: #0f2c4c;
              --cds-navy-dark: #0a1e33;
              --cds-text: #1f2937;
              --cds-muted: #64748b;
              --cds-green-fg: #166534; --cds-green-bg: #e7f6ec;
              --cds-amber-fg: #92400e; --cds-amber-bg: #fef3c7;
              --cds-red-fg: #991b1b;   --cds-red-bg: #fdecec;
              --cds-blue-fg: #1d4ed8;  --cds-blue-bg: #eaf1fd;
          }
          html, body, [class*="css"] { color: var(--cds-text); }
          .stApp { background: var(--cds-bg); }
          .block-container {padding-top: 2rem; max-width: 1400px;}
          h1, h2, h3 { color: var(--cds-navy-dark); font-weight: 650; letter-spacing: -0.01em; }
          hr { border-color: var(--cds-border) !important; }

          /* Sidebar */
          section[data-testid="stSidebar"] {
              background: var(--cds-navy-dark); border-right: 1px solid var(--cds-border);
          }
          section[data-testid="stSidebar"] * { color: #e7edf5 !important; }
          section[data-testid="stSidebar"] .cds-brand-title {
              font-size: 1.02rem; font-weight: 700; color: #ffffff !important;
              margin-bottom: 0.1rem;
          }
          section[data-testid="stSidebar"] .cds-brand-sub {
              font-size: 0.78rem; color: #a9b7c8 !important; margin-bottom: 0.6rem;
          }

          /* Safety / disclaimer */
          .safety-banner {
              border-left: 5px solid #b45309; background: #fffaf0;
              padding: 0.85rem 1.05rem; border-radius: 6px; margin-bottom: 1rem;
              color: var(--cds-text); font-size: 0.88rem; line-height: 1.5;
          }
          .flag {
              border-radius: 6px; padding: 0.55rem 0.75rem; margin: 0.3rem 0;
              font-size: 0.86rem; line-height: 1.45;
          }
          .prov {
              font-size: 0.76rem; color: var(--cds-muted);
              border-top: 1px dashed var(--cds-border);
              padding-top: 0.4rem; margin-top: 0.6rem;
          }
          .assumed {
              background:#fffbeb; border:1px solid #fcd34d; border-radius:6px;
              padding:0.5rem 0.75rem; font-size:0.82rem; color:#92400e;
          }

          /* Cards */
          .optcard {
              border:1px solid var(--cds-border); border-radius:8px; padding:1rem 1.2rem;
              margin-bottom:0.9rem; background:var(--cds-card);
              box-shadow: 0 1px 2px rgba(15,44,76,0.04);
          }
          .optcard.flagged {border-color:#f2c6c6; background:#fffdfc;}
          .stat-card {
              border:1px solid var(--cds-border); border-radius:8px;
              background:var(--cds-card); padding:0.95rem 1.1rem;
              box-shadow: 0 1px 2px rgba(15,44,76,0.04); height:100%;
          }
          .stat-card .label {
              font-size:0.72rem; text-transform:uppercase; letter-spacing:0.04em;
              color:var(--cds-muted); font-weight:600; margin-bottom:0.3rem;
          }
          .stat-card .value { font-size:1.35rem; font-weight:700; color:var(--cds-navy-dark); }
          .stat-card .sub { font-size:0.76rem; color:var(--cds-muted); margin-top:0.15rem; }

          /* Pills / badges */
          .pill {
              display:inline-block; padding:0.12rem 0.55rem; border-radius:999px;
              font-size:0.72rem; font-weight:600; margin-right:0.35rem;
          }
          .risk-badge {
              display:inline-flex; align-items:center; gap:0.35rem;
              padding:0.2rem 0.65rem; border-radius:6px;
              font-size:0.76rem; font-weight:700; letter-spacing:0.01em;
              border:1px solid transparent;
          }
          .risk-low { color:var(--cds-green-fg); background:var(--cds-green-bg); border-color:#bfe3cb; }
          .risk-low-moderate { color:var(--cds-blue-fg); background:var(--cds-blue-bg); border-color:#c3d6f7; }
          .risk-moderate { color:var(--cds-amber-fg); background:var(--cds-amber-bg); border-color:#f5dfa4; }
          .risk-high { color:var(--cds-red-fg); background:var(--cds-red-bg); border-color:#f3c3c3; }

          /* Wizard stepper */
          .cds-stepper { display:flex; gap:0.5rem; margin-bottom:1.1rem; }
          .cds-step {
              flex:1; text-align:center; padding:0.5rem 0.4rem; border-radius:6px;
              font-size:0.78rem; font-weight:600; border:1px solid var(--cds-border);
              background:var(--cds-card); color:var(--cds-muted);
          }
          .cds-step.active { background:var(--cds-navy); color:#ffffff; border-color:var(--cds-navy); }
          .cds-step.done { background:#eef4ee; color:var(--cds-green-fg); border-color:#bfe3cb; }

          .cds-nav-status {
              font-size:0.78rem; padding:0.4rem 0.1rem; display:flex;
              align-items:center; gap:0.4rem;
          }
          .cds-nav-status .dot {
              width:0.5rem; height:0.5rem; border-radius:999px; display:inline-block;
          }
          .cds-nav-status.ready .dot { background:#22c55e; }
          .cds-nav-status.down .dot { background:#ef4444; }

          .cds-eyebrow {
              font-size:0.75rem; text-transform:uppercase; letter-spacing:0.06em;
              color:var(--cds-muted); font-weight:700; margin-bottom:0.15rem;
          }

          [data-testid="stMetricValue"] { color: var(--cds-navy-dark); }
          div[data-testid="stDataFrame"] { border:1px solid var(--cds-border); border-radius:8px; }
        </style>
        """,
        unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Risk-category display bucketing
# ---------------------------------------------------------------------------
# NOTE: This buckets the model's own predicted probability into a display
# label for the ranking table. It is a UI convenience, not a separate
# clinical claim or a new model output — the underlying number is always
# shown alongside it.
_RISK_BANDS = [
    (0.15, "LOW RESISTANCE RISK", "risk-low", "●"),
    (0.30, "LOW-MODERATE RESISTANCE RISK", "risk-low-moderate", "●"),
    (0.50, "MODERATE RESISTANCE RISK", "risk-moderate", "▲"),
    (1.01, "HIGH RESISTANCE RISK", "risk-high", "▲"),
]


def risk_category(p_nonsusceptible: float) -> tuple[str, str]:
    """(label, css-class) for a display bucketing of P(resistant/non-susceptible)."""
    p = 0.0 if p_nonsusceptible is None else float(p_nonsusceptible)
    for upper, label, css_class, _icon in _RISK_BANDS:
        if p < upper:
            return label, css_class
    return _RISK_BANDS[-1][1], _RISK_BANDS[-1][2]


def risk_badge_html(p_nonsusceptible: float) -> str:
    label, css_class = risk_category(p_nonsusceptible)
    icon = next(i for u, l, c, i in _RISK_BANDS if c == css_class)
    return (f'<span class="risk-badge {css_class}">{icon} {label} '
            f'&middot; {float(p_nonsusceptible):.0%}</span>')


def nav_status(ready: bool, label: str | None = None) -> str:
    cls = "ready" if ready else "down"
    text = label or ("Model System Ready" if ready else "Model System Unavailable")
    return f'<div class="cds-nav-status {cls}"><span class="dot"></span>{text}</div>'


def stat_card_html(label: str, value: str, sub: str | None = None) -> str:
    sub_html = f'<div class="sub">{sub}</div>' if sub else ""
    return (f'<div class="stat-card"><div class="label">{label}</div>'
            f'<div class="value">{value}</div>{sub_html}</div>')


def stepper_html(steps: list, current_index: int) -> str:
    parts = []
    for i, name in enumerate(steps):
        cls = "active" if i == current_index else ("done" if i < current_index else "")
        parts.append(f'<div class="cds-step {cls}">{i + 1}. {name}</div>')
    return f'<div class="cds-stepper">{"".join(parts)}</div>'


def safety_banner(compact: bool = False) -> None:
    st.markdown(
        f'<div class="safety-banner">{SHORT_DISCLAIMER if compact else SAFETY_DISCLAIMER}</div>',
        unsafe_allow_html=True)


def render_flag(flag: dict) -> None:
    icon, colour, bg = SEVERITY_STYLE.get(flag.get("severity", "info"),
                                          SEVERITY_STYLE["info"])
    st.markdown(
        f'<div class="flag" style="background:{bg}; border-left:4px solid {colour};">'
        f'<b>{icon} {flag.get("message", "")}</b><br>'
        f'<span style="color:#374151;">{flag.get("detail", "")}</span><br>'
        f'<span style="font-size:0.74rem;color:#6b7280;">Source: '
        f'{flag.get("source", "—")}</span></div>',
        unsafe_allow_html=True)


def provenance(source: str | None, url: str | None, retrieved: str | None,
               cached: bool = False, extra: str | None = None) -> None:
    bits = []
    if source:
        bits.append(f"Source: {source}")
    if retrieved:
        bits.append(f"retrieved {retrieved} UTC")
    if cached:
        bits.append("served from local cache")
    if url:
        bits.append(f'<a href="{url}" target="_blank">endpoint</a>')
    if extra:
        bits.append(extra)
    st.markdown(f'<div class="prov">{" &nbsp;·&nbsp; ".join(bits)}</div>',
                unsafe_allow_html=True)


def pill(text: str, colour: str = "#e5e7eb", fg: str = "#374151") -> str:
    return (f'<span class="pill" style="background:{colour};color:{fg};">'
            f'{text}</span>')


def aware_pill(category: str) -> str:
    colours = {"Access": ("#dcfce7", "#166534"),
               "Watch": ("#fef3c7", "#92400e"),
               "Reserve": ("#fee2e2", "#991b1b")}
    bg, fg = colours.get(category, ("#e5e7eb", "#374151"))
    return pill(f"AWaRe: {category}", bg, fg)


def evidence_pill(strength: str) -> str:
    if strength.startswith("strong"):
        return pill("evidence: strong", "#dcfce7", "#166534")
    if strength.startswith("moderate"):
        return pill("evidence: moderate", "#e0f2fe", "#075985")
    if strength.startswith("none"):
        return pill("evidence: none for this pair", "#fee2e2", "#991b1b")
    return pill(f"evidence: {strength}", "#fef3c7", "#92400e")


def assumed_fields_notice(assumed: list) -> None:
    if not assumed:
        return
    st.markdown(
        '<div class="assumed"><b>Fields not supplied — a training-set default '
        'was used.</b> The prediction is correspondingly less specific to this '
        f'patient.<br>{", ".join(f"<code>{a}</code>" for a in assumed)}</div>',
        unsafe_allow_html=True)


def unavailable(message: str, reason: str | None = None) -> None:
    st.info(f"**{message}**" + (f"\n\n{reason}" if reason else ""))
