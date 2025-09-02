import io
import re
import string
from collections import Counter
from datetime import datetime

import pandas as pd
import streamlit as st

# -------- Optional deps we try to import gracefully --------
# PDF / DOCX extraction
try:
    from pypdf import PdfReader  # modern PyPDF2 fork
except Exception:
    PdfReader = None

try:
    from docx import Document  # python-docx
except Exception:
    Document = None


# =========================
# Utilities & Persistence
# =========================
def ss_init():
    """Initialize all session_state keys once."""
    if "nav" not in st.session_state:
        st.session_state.nav = "Home"

    # Scanner
    st.session_state.setdefault("scanner_paste", "")
    st.session_state.setdefault("scanner_summary_md", "")
    st.session_state.setdefault("scanner_text_raw", "")

    # Trends
    st.session_state.setdefault("trends_region", "Global")
    st.session_state.setdefault("trends_theme", "Climate")
    st.session_state.setdefault("trends_horizon", "Long (3–5y)")
    st.session_state.setdefault("trends_audience", "Programme Team")
    st.session_state.setdefault("trends_notes", "")
    st.session_state.setdefault("trends_brief_md", "")

    # Funding feed (mock)
    st.session_state.setdefault("feed_region", "All")
    st.session_state.setdefault("feed_sector", [])
    st.session_state.setdefault("feed_country", "")
    st.session_state.setdefault("feed_grant_min", 50_000)
    st.session_state.setdefault("feed_grant_max", 500_000)
    st.session_state.setdefault("feed_size", "All")
    st.session_state.setdefault("feed_df", pd.DataFrame())
    st.session_state.setdefault("feed_shortlist_df", pd.DataFrame())

    # Concept note
    st.session_state.setdefault("cn_sections", {})  # generated text per section
    st.session_state.setdefault("cn_full_md", "")

    # Exports store: dict[name] = {"bytes":..., "filename":...}
    st.session_state.setdefault("exports", {})


def clean_text(s: str) -> str:
    """Squash whitespace; remove repeated spaces; trim weird breaks."""
    s = s.replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def extract_text_from_upload(file) -> str:
    """Try to read TXT / DOCX / PDF. Return plain text (or '')."""
    name = (file.name or "").lower()

    # TXT
    if name.endswith(".txt"):
        try:
            return file.read().decode("utf-8", errors="ignore")
        except Exception:
            return file.read().decode("latin-1", errors="ignore")

    # DOCX
    if name.endswith(".docx") and Document is not None:
        try:
            doc = Document(file)
            return "\n".join(p.text for p in doc.paragraphs)
        except Exception:
            return ""

    # PDF
    if name.endswith(".pdf") and PdfReader is not None:
        text_parts = []
        try:
            reader = PdfReader(file)
            for page in reader.pages:
                t = page.extract_text() or ""
                text_parts.append(t)
            return "\n".join(text_parts)
        except Exception:
            return ""

    return ""


def naive_keywords(text: str, topn=12):
    """Quick keyword calc: drop urls, numbers, stopwords, punctuation."""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = [t for t in text.split() if len(t) > 2]
    stop = set(
        """
        the a an and for from with into upon over under between among out off on in to of by
        we you they them our your their within across there here this that those these
        is are was were be been being it its as at or if but so not than then such
        project programmes program programme tender grant call terms reference
        will shall may also make take get set due new year years
        """.split()
    )
    tokens = [t for t in tokens if t not in stop]
    cnt = Counter(tokens)
    return [w for w, _ in cnt.most_common(topn)]


def detect_bullets(text: str, max_lines=10):
    """Pull first few bullet-like lines as a quick sample list."""
    lines = [l.strip() for l in text.splitlines()]
    bullets = []
    for l in lines:
        if re.match(r"^[-•*]\s+", l) and len(l) > 5:
            bullets.append(re.sub(r"^[-•*]\s+", "", l))
        if len(bullets) >= max_lines:
            break
    return bullets


def md_code_block(md: str) -> str:
    """Wrap markdown in a code block for mono display w/o syntax coloring gimmicks."""
    return f"```\n{md.strip()}\n```"


def make_docx_bytes(markdown_text: str, title="Document"):
    """Very light DOCX writer (no external APIs)."""
    if Document is None:
        # Fallback to TXT if python-docx isn't installed
        return io.BytesIO(markdown_text.encode("utf-8")), f"{title}.txt"

    doc = Document()
    for para in markdown_text.split("\n"):
        doc.add_paragraph(para)
    bio = io.BytesIO()
    doc.save(bio)
    bio.seek(0)
    return bio, f"{title}.docx"


def store_export(name: str, markdown_text: str):
    bio, fname = make_docx_bytes(markdown_text, title=name.replace(" ", "_"))
    st.session_state.exports[name] = {"bytes": bio.getvalue(), "filename": fname}


# =========================
# UI Bits
# =========================
def sidebar_nav():
    st.sidebar.title("NGO AI Tool (Mock)")
    st.sidebar.caption("Local prototype, no external APIs.")
    nav = st.sidebar.radio(
        "Navigate",
        ["Home", "Funding Feed", "Grant / Tender Scanner", "Aid Trends", "Concept Note Builder", "Exports", "Settings"],
        index=["Home", "Funding Feed", "Grant / Tender Scanner", "Aid Trends", "Concept Note Builder", "Exports", "Settings"].index(
            st.session_state.nav
        ),
    )
    st.session_state.nav = nav


# =========================
# PAGES
# =========================
def page_home():
    st.title("NGO AI Grant Assistant (Local Prototype)")
    st.caption("Local prototype, no external APIs.")

    st.subheader("What this tool does")
    st.markdown(
        """
- **Find & focus** on relevant grants/tenders (mock feed for now).
- **Digest ToRs quickly** — paste or upload PDF/DOCX/TXT and get an **inline, clean summary**.
- **Understand donor trends** — a concise 2-page style brief with indicative sources.
- **Draft a concept note** — section-by-section, with **optional key data/keywords** and your own word-length sliders.
- **Export everything** as DOCX from the **Exports** tab.
        """
    )

    st.info(
        "Data persists across tabs in this session. Use **Settings → Reset all** if you want to start over.",
        icon="💾",
    )


def page_funding_feed():
    st.title("Funding Feed (mock)")
    st.caption("Local prototype, no external APIs.")

    with st.expander("Filters", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.feed_region = st.selectbox(
                "Region", ["All", "Africa", "Asia", "MENA", "Europe", "LAC"], index=["All", "Africa", "Asia", "MENA", "Europe", "LAC"].index(st.session_state.feed_region)
            )
            st.session_state.feed_country = st.text_input(
                "Country (type to filter)", st.session_state.feed_country
            )
        with col2:
            st.session_state.feed_sector = st.multiselect(
                "Sector", ["Climate", "Smart Agriculture", "Health", "Education", "Protection", "WASH"], default=st.session_state.feed_sector
            )
        col3, col4 = st.columns([2, 1])
        with col3:
            st.session_state.feed_grant_min, st.session_state.feed_grant_max = st.slider(
                "Grant size (USD)", 10_000, 2_000_000, (st.session_state.feed_grant_min, st.session_state.feed_grant_max), step=10_000
            )
        with col4:
            st.session_state.feed_size = st.selectbox(
                "Size of NGO", ["All", "Small", "Medium", "Large"], index=["All", "Small", "Medium", "Large"].index(st.session_state.feed_size)
            )

        if st.button("Generate shortlist"):
            # Build a tiny mock dataset and then filter
            df = pd.DataFrame(
                [
                    {"Opportunity": "Climate Resilience Challenge Fund", "Funder": "Acme Foundation", "Region": "Africa", "Country": "Kenya", "Sector": "Climate", "GrantUSD": 200_000, "Deadline": "2025-10-15"},
                    {"Opportunity": "Urban Food Systems Innovation", "Funder": "Green Cities Alliance", "Region": "Asia", "Country": "Indonesia", "Sector": "Smart Agriculture", "GrantUSD": 350_000, "Deadline": "2025-11-01"},
                    {"Opportunity": "Education Equity Small Grants", "Funder": "Open Learning Trust", "Region": "MENA", "Country": "Jordan", "Sector": "Education", "GrantUSD": 80_000, "Deadline": "2025-09-30"},
                ]
            )
            st.session_state.feed_df = df.copy()

            dd = df
            if st.session_state.feed_region != "All":
                dd = dd[dd["Region"] == st.session_state.feed_region]
            if st.session_state.feed_country.strip():
                dd = dd[dd["Country"].str.contains(st.session_state.feed_country.strip(), case=False, na=False)]
            if st.session_state.feed_sector:
                dd = dd[dd["Sector"].isin(st.session_state.feed_sector)]
            dd = dd[(dd["GrantUSD"] >= st.session_state.feed_grant_min) & (dd["GrantUSD"] <= st.session_state.feed_grant_max)]
            st.session_state.feed_shortlist_df = dd.reset_index(drop=True)

    if not st.session_state.feed_shortlist_df.empty:
        st.subheader("Shortlist")
        st.dataframe(st.session_state.feed_shortlist_df, use_container_width=True)
        # Save export
        csv_io = io.StringIO()
        st.session_state.feed_shortlist_df.to_csv(csv_io, index=False)
        shortlist_md = f"""# Funding Shortlist
Generated: {datetime.utcnow().isoformat()}Z

{st.session_state.feed_shortlist_df.to_markdown(index=False)}
"""
        store_export("Funding Shortlist", shortlist_md)
        st.success("Shortlist generated. A DOCX is ready in **Exports**.")
    else:
        st.info("No shortlist yet — set filters and click **Generate shortlist**.")


def page_scanner():
    st.title("Grant / Tender Scanner")
    st.caption("Paste or upload. Results display below; you can download DOCX and continue to the Concept Note.")

    colA, colB = st.columns([2, 1])

    with colA:
        st.session_state.scanner_paste = st.text_area(
            "PASTE TOR / PROJECT DESCRIPTION",
            st.session_state.scanner_paste,
            height=300,
            placeholder="Paste text here… (no character limit)",
        )
        c1, c2 = st.columns([1, 1])
        with c1:
            run_scan = st.button("Scan", type="primary")
        with c2:
            if st.button("Clear"):
                st.session_state.scanner_paste = ""
                st.session_state.scanner_summary_md = ""
                st.session_state.scanner_text_raw = ""

    with colB:
        uploads = st.file_uploader(
            "OR UPLOAD FILE(S)",
            type=["txt", "docx", "pdf"],
            accept_multiple_files=True,
            help="TXT, DOCX, PDF; up to 200MB per file.",
        )

    # ----- Scan logic -----
    if run_scan:
        chunks = []
        if st.session_state.scanner_paste.strip():
            chunks.append(st.session_state.scanner_paste)

        if uploads:
            for f in uploads:
                txt = extract_text_from_upload(f)
                if not txt:
                    st.warning(f"Could not extract text from **{f.name}** (scanned PDF or unsupported). Try an OCR’d PDF or a DOCX/TXT.", icon="⚠️")
                else:
                    chunks.append(txt)

        combined = clean_text("\n\n".join(chunks))
        st.session_state.scanner_text_raw = combined

        if not combined:
            st.error("No readable text found. Paste text or upload OCR’d/real-text PDFs or DOCX/TXT files.")
        else:
            # Summarize heuristically
            words = [w for w in re.findall(r"\b[\w-]+\b", combined)]
            kw = naive_keywords(combined, topn=12)
            bullets = detect_bullets(combined, max_lines=8)
            maybe_bullets = bullets if bullets else ["(No bullet structure detected; see full text for details.)"]

            # Very light heuristics to create a more concise structure
            summary = f"""
**GRANT / TENDER SCAN — SUMMARY**
----------------------------------

**Detected length:** ~{len(words)} words  
**Prominent keywords (naive):** {", ".join(kw)}

**Likely Requirements / Highlights:**
- Clear deliverables, partners/stakeholders, timeline, and eligibility criteria typically expected.
- Emphasis on measurable outcomes and value for money.
- Ensure safeguarding/data-privacy provisions are addressed.

**Detected bullet points (sample):**
- """ + "\n- ".join(maybe_bullets) + """

**Potential Risks / Considerations (heuristic):**
- Tight timelines and eligibility constraints may limit competition.
- Budget ceilings and any co-finance expectations should be confirmed.
- Ensure alignment with donor priorities and local partner capacity.
            """
            st.session_state.scanner_summary_md = summary
            st.success("Scan complete.")

            # Save for exports
            store_export("Scanner Summary", summary)

    # ----- Show results -----
    if st.session_state.scanner_summary_md:
        st.subheader("Summary")
        st.markdown(md_code_block(st.session_state.scanner_summary_md))

        dl_bytes, dl_name = make_docx_bytes(st.session_state.scanner_summary_md, "Scanner_Summary")
        st.download_button("Download DOCX", data=dl_bytes.getvalue(), file_name=dl_name, type="primary")

        if st.session_state.scanner_text_raw:
            with st.expander("Show extracted raw text"):
                st.text_area("Extracted Text", st.session_state.scanner_text_raw, height=250)


def page_trends():
    st.title("Aid Trends")
    st.caption("Local prototype, no external APIs.")

    col1, col2 = st.columns(2)
    with col1:
        st.session_state.trends_region = st.selectbox("Region focus", ["Global", "Africa", "Asia", "MENA", "Europe", "LAC"], index=["Global","Africa","Asia","MENA","Europe","LAC"].index(st.session_state.trends_region))
        st.session_state.trends_theme = st.selectbox("Theme focus", ["Climate", "Smart Agriculture", "Health", "Education", "Protection", "WASH"], index=["Climate","Smart Agriculture","Health","Education","Protection","WASH"].index(st.session_state.trends_theme))
    with col2:
        st.session_state.trends_horizon = st.selectbox("Time horizon", ["Short (6–12m)", "Medium (1–3y)", "Long (3–5y)"], index=["Short (6–12m)","Medium (1–3y)","Long (3–5y)"].index(st.session_state.trends_horizon))
        st.session_state.trends_audience = st.selectbox("Audience", ["Programme Team", "Board / Exec", "Donor Relations"], index=["Programme Team","Board / Exec","Donor Relations"].index(st.session_state.trends_audience))

    st.session_state.trends_notes = st.text_area("Optional analyst notes", st.session_state.trends_notes, placeholder="Any nuance you want to inject…")

    if st.button("Generate Trends Brief", type="primary"):
        r = st.session_state.trends_region
        t = st.session_state.trends_theme
        h = st.session_state.trends_horizon
        a = st.session_state.trends_audience
        n = st.session_state.trends_notes.strip()

        sources = [
            ("OECD CRS (2022–2024)", "https://stats.oecd.org/"),
            ("Donor press releases & pipelines", "https://www.devex.com/news"),
            ("GIIN/IFC blended finance insights", "https://thegiin.org/"),
            ("Philanthropy data (Candid)", "https://candid.org/")
        ]

        brief = f"""
AID TRENDS BRIEF — {r.upper()} / {t.upper()}
--------------------------------------------

Audience: {a} | Horizon: {h}

1) Donor Shifts (illustrative):
- Bilateral donors are consolidating portfolios; larger multi-country calls; fewer small awards.
- In {t}, windows increasingly emphasise outcomes and co-finance hooks.
- In {r}, flexible instruments (windows, challenge funds) prioritise catalytic pilots.

2) Private Funding Pipelines (illustrative):
- Corporate philanthropy in {r} is increasing around {t.lower()} & resilience.
- Impact funds seek ‘pay-for-results’ structures with KPIs tied to climate/social outcomes.
- Blended finance vehicles are emerging for place-based projects with measurable outcomes.

3) Implications for NGOs:
- Position consortia to deliver at scale; clarify unique role and local legitimacy.
- Strengthen MEL for results-based disbursement; define 3–5 clear KPIs.
- Prepare a pipeline of ‘shovel-ready’ concepts (12–24 months) with co-finance hooks.

Indicative Sources:
- """ + "\n- ".join([f"[{label}]({url})" for label, url in sources]) + (f"\n\nAnalyst Notes:\n- {n}" if n else "")

        st.session_state.trends_brief_md = brief
        st.success("Generated.")
        store_export("Aid Trends Brief", brief)

    if st.session_state.trends_brief_md:
        st.subheader("Brief")
        st.markdown(md_code_block(st.session_state.trends_brief_md))


def page_concept_note():
    st.title("Concept Note Builder")

    st.caption(
        "Provide optional **key data/keywords** per section. The tool will also reuse your **Scanner** and **Trends** content when helpful. "
        "Use sliders to choose approximate word counts."
    )

    sections = [
        ("Background / Problem", 120),
        ("Objectives", 100),
        ("Approach / Theory of Change", 180),
        ("Activities & Workplan", 160),
        ("Geography & Beneficiaries", 120),
        ("Partnerships & Governance", 120),
        ("MEL & KPIs", 120),
        ("Risk & Safeguarding", 120),
        ("Budget Summary", 100),
        ("Sustainability / Exit", 110),
    ]

    # Helper: assemble text using light templates and feed-through info
    def synthesize(section_name, words_target, hints):
        scan_bits = st.session_state.scanner_summary_md.strip()
        trends_bits = st.session_state.trends_brief_md.strip()

        # Trim to target-ish length without an LLM: we compose bullet-like prose
        base = []
        if section_name == "Background / Problem":
            base = [
                "Context: the organisation seeks to address priority needs identified in recent funding calls and trends.",
                "Problem statement: demand for measurable outcomes and scalable delivery is rising, alongside tighter eligibility and timelines.",
                "Relevance: the proposed work aligns with donor priorities and local partner capacity in the target geography."
            ]
        elif section_name == "Objectives":
            base = [
                "Overall objective: improve outcomes for target groups through a results-based programme aligned with donor priorities.",
                "Specific objectives: 3–5 measurable goals covering access, quality, and sustainability."
            ]
        elif section_name == "Approach / Theory of Change":
            base = [
                "We apply a theory-of-change rooted in evidence, incentives for performance, and learning loops.",
                "Inputs lead to outputs (capacity, services, data), which lead to outcomes (improved practices, resilience) and impact.",
            ]
        elif section_name == "Activities & Workplan":
            base = [
                "Workstreams: WS1 capacity & systems; WS2 service delivery pilots; WS3 learning & scale-up.",
                "Workplan includes inception, pilot, adaptation, and consolidation phases with clear milestones."
            ]
        elif section_name == "Geography & Beneficiaries":
            base = [
                "Target geography: focal regions selected for need and feasibility.",
                "Beneficiaries: priority groups defined with clear inclusion criteria and reach estimates."
            ]
        elif section_name == "Partnerships & Governance":
            base = [
                "Partnership model: roles for local NGOs, government counterparts, private actors and technical partners.",
                "Governance: a light PMU, advisory oversight, and working groups for delivery."
            ]
        elif section_name == "MEL & KPIs":
            base = [
                "MEL plan: results framework with indicators, baselines and targets; timely monitoring and learning events.",
                "KPIs reflect outcomes and co-finance leverage where applicable."
            ]
        elif section_name == "Risk & Safeguarding":
            base = [
                "Key risks: delivery timelines, eligibility, data-privacy/safeguarding, and partner capacity.",
                "Mitigation includes clear protocols, training, and escalation paths."
            ]
        elif section_name == "Budget Summary":
            base = [
                "Budget envelope covers personnel, delivery costs, MEL, and contingency.",
                "Value-for-money approach: efficient management and leverage of co-finance where relevant."
            ]
        elif section_name == "Sustainability / Exit":
            base = [
                "Exit strategy: transfer of capacities, institutional ownership, and financing options.",
                "Sustainability: co-design with local partners and progressive handover plan."
            ]

        # weave in scanner/trends snippets if available
        if scan_bits:
            base.append("Relevance to the call: informed by the tender/ToR scan (requirements, timelines, outcomes).")
        if trends_bits:
            base.append("Positioning: aligned with current donor shifts and private funding pipelines as highlighted in the trends brief.")

        if hints.strip():
            base.append(f"Key data/keywords from user: {hints.strip()}")

        # Join and crop roughly
        text = " ".join(base)
        # crude crop to approx words_target
        words = text.split()
        if len(words) > words_target:
            text = " ".join(words[:words_target]) + "…"
        return text

    # Draw UI for each section
    built_sections = {}
    for sec_name, default_words in sections:
        st.markdown(f"### {sec_name}")
        col1, col2 = st.columns([2, 1])
        with col1:
            hints = st.text_area(
                "Add key data / keywords (optional)",
                value=st.session_state.cn_sections.get(sec_name, {}).get("hints", ""),
                key=f"hints_{sec_name}",
                placeholder="E.g., locations, partner names, target figures, learning priorities…",
            )
        with col2:
            tgt = st.slider("Approx. words", 60, 400, st.session_state.cn_sections.get(sec_name, {}).get("words", default_words), key=f"words_{sec_name}")

        if st.button(f"Generate {sec_name}", key=f"gen_{sec_name}", use_container_width=True):
            text = synthesize(sec_name, tgt, hints)
            st.session_state.cn_sections[sec_name] = {"text": text, "words": tgt, "hints": hints}
            st.success("Section generated.")

        # show current section text if any
        cur = st.session_state.cn_sections.get(sec_name, {}).get("text", "")
        if cur:
            st.text_area("Draft", value=cur, height=160, key=f"show_{sec_name}")

        st.divider()

    if st.button("Assemble full Concept Note", type="primary"):
        parts = []
        for sec_name, _ in sections:
            t = st.session_state.cn_sections.get(sec_name, {}).get("text", "")
            if t:
                parts.append(f"## {sec_name}\n\n{t}\n")
        full = f"# Concept Note Draft\nGenerated: {datetime.utcnow().isoformat()}Z\n\n" + "\n".join(parts)
        st.session_state.cn_full_md = full
        st.success("Assembled. A DOCX is also saved to Exports.")
        store_export("Concept Note", full)

    if st.session_state.cn_full_md:
        st.subheader("Full draft")
        st.markdown(md_code_block(st.session_state.cn_full_md))


def page_exports():
    st.title("Exports")

    if not st.session_state.exports:
        st.info("No exported documents yet. Generate a shortlist, scan, trends brief, or concept note first.")
        return

    for name, blob in st.session_state.exports.items():
        st.write(f"**{name}**")
        st.download_button(
            "Download",
            data=blob["bytes"],
            file_name=blob["filename"],
            key=f"dl_{name}",
        )
        st.divider()


def page_settings():
    st.title("Settings")

    if st.button("Reset ALL stored content", type="secondary"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        ss_init()
        st.success("All content cleared.")


# =========================
# Main
# =========================
def main():
    ss_init()
    sidebar_nav()

    pages = {
        "Home": page_home,
        "Funding Feed": page_funding_feed,
        "Grant / Tender Scanner": page_scanner,
        "Aid Trends": page_trends,
        "Concept Note Builder": page_concept_note,
        "Exports": page_exports,
        "Settings": page_settings,
    }
    pages[st.session_state.nav]()


if __name__ == "__main__":
    main()
