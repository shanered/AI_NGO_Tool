# app.py — NGO AI Tool (Streamlit, resilient DOCX fallback)
# If python-docx is missing, app still runs and offers .txt downloads + banner to install it.
# Install to enable Word downloads:
#   pip install python-docx PyPDF2

import io
import re
import time
from datetime import datetime
from typing import Dict, List, Tuple

import streamlit as st

# ------- Optional packages -------
try:
    from docx import Document
    from docx.shared import Pt
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    HAS_DOCX = True
except Exception:
    HAS_DOCX = False

try:
    import PyPDF2  # type: ignore
    HAS_PYPDF2 = True
except Exception:
    HAS_PYPDF2 = False

# ------------------------- App Config -------------------------
st.set_page_config(page_title="NGO AI Tool (Mock)", page_icon="🌍", layout="wide")

WORDS_PRESETS = {"Short": 300, "Standard": 600, "Long": 1000}
PASTE_WORD_LIMIT = 5000

DEFAULT_SECTIONS = [
    ("executive_summary",       "Executive Summary",                         True, 600),
    ("problem_statement",       "Problem Statement / Needs",                 True, 600),
    ("context_trends",          "Context & Trends (incl. aid cuts)",         True, 600),
    ("objectives_results",      "Objectives & Results (logframe-lite)",      True, 600),
    ("methodology",             "Methodology / Workplan",                    True, 600),
    ("safeguarding_inclusion",  "Safeguarding, Gender & Inclusion",          True, 400),
    ("m_e_learning",            "M&E & Learning",                            True, 400),
    ("risk_mgmt",               "Risks & Mitigation",                        False, 300),
    ("budget_summary",          "Budget Summary & VfM",                      False, 300),
    ("org_capacity",            "Org Capacity & Past Performance",           True, 300),
]

NAV = ["Home", "Grant / Tender Scanner", "Aid Trends", "Concept Note Builder", "Exports", "Settings"]

# ------------------------- Session State -------------------------
if "sections" not in st.session_state:
    st.session_state.sections = {k: {"label": lbl, "on": on, "words": words}
                                 for k, lbl, on, words in DEFAULT_SECTIONS}

for key in ["scanner_bytes", "scanner_name", "trends_bytes", "trends_name", "concept_bytes", "concept_name"]:
    if key not in st.session_state:
        st.session_state[key] = None

if "banner_dismissed" not in st.session_state:
    st.session_state.banner_dismissed = False

# ------------------------- Helpers & CSS -------------------------
def clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def word_count(s: str) -> int:
    return len([w for w in re.split(r"\s+", s.strip()) if w])

def enforce_soft_limit(text: str, limit_words: int) -> str:
    words = text.split()
    if len(words) <= limit_words:
        return text
    return " ".join(words[:limit_words])

def pill_css():
    st.markdown("""
    <style>
      .pill {display:inline-block;padding:.25rem .6rem;border-radius:999px;font-size:.75rem;margin:.125rem;}
      .pill.gray{background:#f3f4f6;color:#111827}
      .pill.green{background:#dcfce7;color:#065f46}
      .pill.red{background:#fee2e2;color:#991b1b}
      .pill.blue{background:#e0f2fe;color:#075985}
      .card{background:#1118270d;border:1px solid #2a2f3a;border-radius:16px;padding:16px}
      .muted{color:#9ca3af}
      .lead{font-size:1.05rem}
      .banner{background:#fff3cd;color:#7a5d00;border:1px solid #ffe69c;border-radius:12px;padding:12px 14px;margin-bottom:16px;}
    </style>
    """, unsafe_allow_html=True)

def pill(text: str, tone: str="gray"):
    st.markdown(f"<span class='pill {tone}'>{text}</span>", unsafe_allow_html=True)

def prototype_banner():
    if st.session_state.banner_dismissed:
        return
    text = "This is a local prototype. Data isn’t saved and features are limited."
    if not HAS_DOCX:
        text += " Word downloads are disabled because `python-docx` isn’t installed. Run: `pip install python-docx`."
    st.markdown(f"<div class='banner'><b>Prototype:</b> {text}</div>", unsafe_allow_html=True)
    st.button("Dismiss", key="dismiss_banner", on_click=lambda: st.session_state.update(banner_dismissed=True))

def extract_text_from_upload(upload) -> str:
    name = upload.name.lower()
    data = upload.read()

    if name.endswith(".txt"):
        try:
            return data.decode("utf-8", errors="ignore")
        except Exception:
            return data.decode("latin-1", errors="ignore")

    if name.endswith(".docx"):
        try:
            from docx import Document as DocxDocument  # local import to avoid hard dependency
            f = io.BytesIO(data)
            doc = DocxDocument(f)
            return "\n".join(p.text for p in doc.paragraphs)
        except Exception:
            return ""

    if name.endswith(".pdf"):
        if not HAS_PYPDF2:
            return ""
        try:
            text = []
            reader = PyPDF2.PdfReader(io.BytesIO(data))
            for page in reader.pages:
                text.append(page.extract_text() or "")
            return "\n".join(text)
        except Exception:
            return ""

    return ""

def mock_extract_meta(consolidated_text: str) -> Dict[str, List[str]]:
    t = clean_text(consolidated_text).lower()
    sectors = [k for k in [
        "livelihoods","agriculture","cash","voucher","climate","tvet","education",
        "wash","protection","nutrition","governance","market systems","sme","health"
    ] if k in t]
    geos = [g.title() for g in ["kenya","somalia","tanzania","uganda","rwanda","ethiopia","ukraine","indonesia"] if g in t]
    donors = [d.upper() for d in ["usaid","fcdo","sida","eu","eib","world bank","jica","giz","adb"] if d.lower() in t]
    red = []
    if "48 hours" in t or "24 hours" in t: red.append("Tight timeline")
    if "mandatory" in t and "all experts" in t: red.append("All roles mandatory")
    if "security clearance" in t: red.append("Security clearance")
    if not red: red = ["None flagged"]
    if not sectors: sectors = ["general development"]
    if not geos: geos = ["Global / LMICs"]
    if not donors: donors = ["Institutional / Foundations"]
    return {"sectors": sectors, "geos": geos, "donors": donors, "red": red}

def words_to_bullets(word_target: int) -> int:
    if word_target <= 350: return 4
    if word_target <= 700: return 7
    return 10

def section_template(title: str, word_target: int, context: str, trends_hint: str) -> str:
    bullets = words_to_bullets(word_target)
    ctx_line = clean_text(context) if context else "Context provided by the organisation."
    hint = f" Trend signal: {clean_text(trends_hint)}." if trends_hint else ""

    blocks = []
    blocks.append(f"{title}\n")
    blocks.append(f"{ctx_line}{hint}\n")
    blocks.append("Purpose\n- Clarify the problem and desired change in plain language.\n- Align with community priorities and national frameworks.\n")
    blocks.append("Expected Results\n" + "\n".join([f"- Result {i}: measurable, time-bound." for i in range(1, min(3, bullets)+1)]) + "\n")
    act_count = max(3, bullets - 3)
    blocks.append("Core Activities\n" + "\n".join([f"- Activity {i}: description; local partner role; timeframe." for i in range(1, act_count+1)]) + "\n")
    blocks.append("Monitoring, Evaluation & Learning (MEL)\n- Indicators: output & outcome, simple tools.\n- Learning loops: quarterly reviews.\n- Data protection & safeguarding integrated.\n")
    blocks.append("Risks & Value for Money (VfM)\n- Risks: capacity, access, compliance, shocks.\n- Mitigation: clear roles, contingencies, adaptive plans.\n- VfM: economy, efficiency, effectiveness, equity.\n")
    return "\n".join(blocks).strip()

def build_concept_note(sections: List[Dict], context: str, trends_hint: str) -> str:
    parts = [f"CONCEPT NOTE (DRAFT)\n\nGenerated: {datetime.utcnow().isoformat()}Z\n"]
    for s in sections:
        parts.append(section_template(s["label"], s["words"], context, trends_hint))
        parts.append("")
    return "\n\n".join(parts)

def build_trends_brief(keywords: str, geography: str) -> str:
    kws = [k.strip() for k in (keywords or "").split(",") if k.strip()]
    kw_line = ", ".join(kws) if kws else "education, livelihoods, climate resilience"
    geo_line = geography or "target region"
    parts = []
    parts.append("AID TRENDS BRIEF — FOR NGO PLANNING\n")
    parts.append(f"Keywords: {kw_line}\nGeography: {geo_line}\nDate: {datetime.utcnow().date()}\n")
    parts.append("Executive Summary\nDonor budgets are under pressure and increasingly performance-oriented. Funding is tilting toward climate resilience, localisation, and results-based delivery. For NGOs, this means sharper value-for-money narratives, stronger local partnerships, and credible MEL.")
    parts.append("Key Trends\n• Real-terms aid contractions; competition for fewer, larger awards.\n• Climate & resilience mainstreamed; cross-cutting outcomes expected.\n• Localisation: higher sub-grant shares, due-diligence support, joint MEL.\n• Digital/AI for inclusion, measurement, cost control.\n• Blended finance piloted in livelihoods/MSD, with equity safeguards.")
    parts.append("Implications for NGOs\n• Position proposals with measurable outcomes and realistic unit costs.\n• Structure consortia around local leadership and delivery proximity.\n• Build climate co-benefits across social sectors.\n• Invest in MEL that provides decision-ready feedback.")
    parts.append("Funding Outlook\n• Fewer, larger frameworks; partner-led entry.\n• Philanthropy/private donors seek catalytic pilots with scale pathways.\n• Emphasis on VfM and adaptive management.")
    parts.append(f"Recommendations for NGOs in {geo_line}\n• Map local partners in {kw_line}; co-design MoUs/capacity plans.\n• Prepare 2–3 page ‘concept packs’ with objectives, results, and costs.\n• Lightweight pipeline tracker linked to donor signals.\n• Include climate/gender/inclusion indicators.\n• Agree early data-sharing and learning rhythms.")
    return "\n\n".join(parts)

# -------- DOC GENERATORS --------
def make_docx(title: str, body_text: str) -> bytes:
    """Return bytes for a DOCX if available, else TXT."""
    if HAS_DOCX:
        doc = Document()
        p = doc.add_paragraph()
        run = p.add_run(title)
        run.bold = True
        run.font.size = Pt(18)
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        doc.add_paragraph("")
        for block in body_text.split("\n\n"):
            if block.strip():
                doc.add_paragraph(block)
        buf = io.BytesIO()
        doc.save(buf)
        buf.seek(0)
        return buf.read()
    else:
        # Fallback to TXT bytes
        return body_text.encode("utf-8")

def extract_labels() -> List[str]:
    return [meta["label"] for _, meta in st.session_state.sections.items()]

# ------------------------- Pages -------------------------
def page_home():
    prototype_banner()
    st.title("Welcome 👋")
    st.write("""
    **Our pitch:** Our AI tool helps NGOs and development teams analyse **grants, tenders, and donor trends in seconds**.  
    It tracks **aid cuts, shifting priorities, and funding patterns** so you can target the right opportunities.  
    Think of it as a **full-time BD analyst**—faster, smarter, and built for today’s evolving aid landscape.
    """)
    st.info("Use the left sidebar to try the Grant/Tender Scanner, explore Aid Trends, or build a Concept Note.")

def page_scanner():
    prototype_banner()
    st.title("Grant / Tender Scanner (mock)")
    col1, col2 = st.columns([1,1])

    with col1:
        st.write("**Paste ToR or core call text** (optional if you upload files).")
        pasted = st.text_area("Paste text (up to 5,000 words)", height=220, placeholder="Eligibility, scope, geography…")
        wc = word_count(pasted)
        st.caption(f"Words: {wc}/{PASTE_WORD_LIMIT}")
        if wc > PASTE_WORD_LIMIT:
            st.warning("Text exceeds 5,000 words. It will be truncated for processing.")

    with col2:
        st.write("**Attach documents** (multiple):")
        uploads = st.file_uploader("Supported: .txt, .docx, .pdf", type=["txt", "docx", "pdf"], accept_multiple_files=True)
        st.caption("Add ToR, needs assessments, partner profiles, annual reports, etc.")
        geography = st.text_input("Target geography (optional)", "", help="e.g., Kenya; East Africa")
        focus = st.text_input("Technical focus (optional)", "", help="e.g., livelihoods, climate resilience, education")
        org = st.text_input("Organisation name (optional)", "", help="Shown in the summary header")

    if st.button("Scan"):
        with st.spinner("Scanning documents and building summary…"):
            collected_texts = []
            files_info: List[Tuple[str, int, bool]] = []

            if clean_text(pasted):
                processed = enforce_soft_limit(pasted, PASTE_WORD_LIMIT)
                collected_texts.append(processed)
                files_info.append(("Pasted text", len(processed), True))

            if uploads:
                for up in uploads:
                    txt = extract_text_from_upload(up)
                    ok = len(clean_text(txt)) > 0
                    files_info.append((up.name, len(txt), ok))
                    if ok:
                        collected_texts.append(txt)

            if not collected_texts:
                msg = "Please paste text or attach at least one parsable file (.txt/.docx"
                if HAS_PYPDF2: msg += "/.pdf"
                msg += ")."
                st.warning(msg)
                return

            consolidated = "\n\n".join(collected_texts)
            meta = mock_extract_meta(consolidated)

            lines = []
            lines.append("GRANT / TENDER SCAN — SUMMARY\n")
            if org: lines.append(f"Organisation: {org}")
            if geography: lines.append(f"Geography: {geography}")
            if focus: lines.append(f"Technical focus: {focus}")
            lines.append(f"Sectors: {', '.join(meta['sectors'])}")
            lines.append(f"Likely geographies: {', '.join(meta['geos'])}")
            lines.append(f"Donor alignment: {', '.join(meta['donors'])}")
            lines.append(f"Red flags: {', '.join(meta['red'])}")
            lines.append("\nHeadline analysis:")
            lines.append("Feasible if partnered locally with clear outcomes, realistic unit costs, "
                         "and early risk mitigation on timelines and due diligence.")
            summary_text = "\n".join(lines)
            time.sleep(0.2)

        st.success("Scan complete!")
        st.markdown("### Results")
        st.markdown(f"<div class='card'><div class='lead'>{summary_text.replace(chr(10), '<br/>')}</div></div>", unsafe_allow_html=True)

        # Build download (DOCX if available else TXT)
        bytes_out = make_docx("Grant / Tender Scan — Summary", summary_text)
        st.session_state.scanner_bytes = bytes_out
        base = "scanner_summary_" + datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        st.session_state.scanner_name = base + (".docx" if HAS_DOCX else ".txt")

        st.download_button(
            f"Download summary ({'.docx' if HAS_DOCX else '.txt'})",
            data=bytes_out,
            file_name=st.session_state.scanner_name,
            mime=("application/vnd.openxmlformats-officedocument.wordprocessingml.document" if HAS_DOCX else "text/plain"),
        )

        if uploads and any(u.name.lower().endswith(".pdf") for u in uploads) and not HAS_PYPDF2:
            st.caption("Note: PDF text extraction requires `PyPDF2`. TXT/DOCX were processed normally.")

def page_trends():
    prototype_banner()
    st.title("Aid Trends (mock)")
    c1, c2 = st.columns([2,1])
    with c1:
        keywords = st.text_input("Enter trend keywords (comma-separated)", "aid cuts, localisation, climate resilience, digital, blended finance")
        geography = st.text_input("Geography / region (optional)", "East Africa")
    with c2:
        st.markdown("<div class='card muted'>Generates a succinct **two-page brief** to paste into proposals or share internally.</div>", unsafe_allow_html=True)

    if st.button("Generate two-page trends brief"):
        with st.spinner("Generating trends brief…"):
            brief = build_trends_brief(keywords, geography)
            bytes_out = make_docx("Aid Trends Brief", brief)
            st.session_state.trends_bytes = bytes_out
            base = "trends_brief_" + datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            st.session_state.trends_name = base + (".docx" if HAS_DOCX else ".txt")
            time.sleep(0.2)
        st.success("Trends brief ready!")

    if st.session_state.trends_bytes:
        st.markdown("### Aid Trends Brief")
        st.download_button(
            f"Download trends_brief ({'.docx' if HAS_DOCX else '.txt'})",
            data=st.session_state.trends_bytes,
            file_name=st.session_state.trends_name,
            mime=("application/vnd.openxmlformats-officedocument.wordprocessingml.document" if HAS_DOCX else "text/plain"),
        )

def page_concept():
    prototype_banner()
    st.title("Concept Note Builder")

    left, right = st.columns([1,1])
    with left:
        context = st.text_area(
            "Project context / brief (used to seed each section)",
            (
                "Rural Kenya; tribal/pastoralist communities. Donor-ready concept note.\n"
                "Objectives: access, quality, inclusion, resilience; integrate aid-cuts/trends.\n"
                "Style: UK English, concise."
            ),
            height=140,
            help="This seeds each section. Keep it concise."
        )
        st.caption(f"Words: {word_count(context)}")
        trends_hint = st.text_input("Optional one-liner to weave in (e.g., trends insight)", "", help="Short phrase referenced across sections")

    with right:
        st.write("**Word presets**")
        b1, b2, b3 = st.columns(3)
        preset_clicked = None
        if b1.button("Short (300)"): preset_clicked = "Short"
        if b2.button("Standard (600)"): preset_clicked = "Standard"
        if b3.button("Long (1,000)"): preset_clicked = "Long"

    st.markdown("#### Sections & word counts")
    selected_sections = []
    for key, meta in st.session_state.sections.items():
        colA, colB = st.columns([3,1])
        with colA:
            on_now = st.checkbox(meta["label"], value=meta["on"], key=f"on_{key}")
        with colB:
            default_words = WORDS_PRESETS.get(preset_clicked, meta["words"])
            if preset_clicked:
                meta["words"] = default_words
            words = st.number_input("Words", min_value=150, max_value=1500, value=int(meta["words"]), step=50, key=f"words_{key}")
        st.session_state.sections[key]["on"] = on_now
        st.session_state.sections[key]["words"] = words
        if on_now:
            selected_sections.append({"key": key, "label": meta["label"], "words": int(words)})

    total_words = sum(s["words"] for s in selected_sections)
    st.caption(f"Approx. total length: **{total_words:,} words**")

    if st.button("Generate Concept Note"):
        if not selected_sections:
            st.warning("Select at least one section.")
        else:
            with st.spinner("Composing concept note…"):
                note_txt = build_concept_note(selected_sections, context, trends_hint)
                bytes_out = make_docx("Concept Note (Draft)", note_txt)
                st.session_state.concept_bytes = bytes_out
                base = "concept_note_" + datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                st.session_state.concept_name = base + (".docx" if HAS_DOCX else ".txt")
                time.sleep(0.2)
            st.success("Concept note ready!")

    if st.session_state.concept_bytes:
        st.markdown("### Concept Note")
        st.download_button(
            f"Download concept_note ({'.docx' if HAS_DOCX else '.txt'})",
            data=st.session_state.concept_bytes,
            file_name=st.session_state.concept_name,
            mime=("application/vnd.openxmlformats-officedocument.wordprocessingml.document" if HAS_DOCX else "text/plain"),
        )

def page_exports():
    prototype_banner()
    st.title("Exports")
    st.write("Download the latest documents generated in other tabs.")

    def chip(ok: bool):
        pill("ready" if ok else "empty", "green" if ok else "red")

    def block(label: str, data_key: str, name_key: str, fallback: str):
        st.subheader(label)
        data = st.session_state.get(data_key)
        name = st.session_state.get(name_key) or (fallback + (".docx" if HAS_DOCX else ".txt"))
        chip(bool(data))
        if not data:
            st.caption("Nothing here yet — generate it first.")
            return
        st.download_button(f"Download {name}", data=data, file_name=name,
                           mime=("application/vnd.openxmlformats-officedocument.wordprocessingml.document" if HAS_DOCX else "text/plain"))

    block("Scanner Summary", "scanner_bytes", "scanner_name", "scanner_summary")
    st.markdown("---")
    block("Aid Trends Brief", "trends_bytes", "trends_name", "trends_brief")
    st.markdown("---")
    block("Concept Note", "concept_bytes", "concept_name", "concept_note")

def page_settings():
    prototype_banner()
    st.title("Settings")
    st.write("Default section toggles and word counts.")
    for key, meta in st.session_state.sections.items():
        colA, colB = st.columns([3,1])
        with colA:
            on_now = st.checkbox(meta["label"], value=meta["on"], key=f"def_on_{key}")
        with colB:
            words_now = st.number_input("Words", min_value=150, max_value=1500, value=int(meta["words"]), step=50, key=f"def_words_{key}")
        st.session_state.sections[key]["on"] = on_now
        st.session_state.sections[key]["words"] = int(words_now)
    st.success("Settings updated.")

# ------------------------- Router -------------------------
pill_css()
st.sidebar.title("NGO AI Tool (Mock)")
nav = st.sidebar.radio("Navigate", NAV, index=0)
st.sidebar.caption("Local prototype, no external APIs.")

if nav == "Home":
    page_home()
elif nav == "Grant / Tender Scanner":
    page_scanner()
elif nav == "Aid Trends":
    page_trends()
elif nav == "Concept Note Builder":
    page_concept()
elif nav == "Exports":
    page_exports()
elif nav == "Settings":
    page_settings()