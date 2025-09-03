# NGO AI Grant Assistant — Local Prototype (No external APIs)
# -------------------------------------------------------------------
# Pages:
#   - Home
#   - ToR / Tender Scanner
#   - Donor Intelligence Panel
#   - Aid Trends
#   - Concept Note Builder
#   - Exports
#   - Settings
#
# Features:
#   * Session-persistent state across pages
#   * Upload TXT/DOCX/PDF; fallback if libs missing (OCR’d PDF best)
#   * “Reset this page” button (every page except Home)
#   * Exports for: tor_scan, donor_intel, aid_trends, concept_note
#
# Install:
#   pip install streamlit python-docx PyPDF2

import io
import re
from datetime import datetime
from typing import Dict, List, Tuple, Any

import streamlit as st

# Optional parsers
HAS_DOCX = False
HAS_PDF = False
try:
    from docx import Document
    HAS_DOCX = True
except Exception:
    HAS_DOCX = False

try:
    from PyPDF2 import PdfReader
    HAS_PDF = True
except Exception:
    HAS_PDF = False


# ========================= Session & Styling =========================

def ss_init() -> None:
    ss = st.session_state

    ss.setdefault("nav", "Home")

    # ---------- ToR / Tender Scanner ----------
    ss.setdefault("tor_text", "")
    ss.setdefault("tor_summary_md", "")
    ss.setdefault("tor_last_uploaded", 0)

    # ---------- Donor Intelligence ----------
    ss.setdefault("donor_filters", {
        "region": "Global",
        "country": "All",
        "sector": [],
        "modality": [],
        "grant_min": 0,
        "grant_max": 50_000_000,
        "keyword": "",
    })
    ss.setdefault("donor_selected", None)
    ss.setdefault("donor_brief_md", "")

    # ---------- Aid Trends ----------
    ss.setdefault("trends", {
        "region": "Global",
        "theme": "Climate",
        "horizon": "Long (3–5y)",
        "audience": "Programme Team",
        "notes": ""
    })
    ss.setdefault("trends_brief_md", "")

    # ---------- Concept Note ----------
    ss.setdefault("cn_sections", {})   # sec -> { text, words, hints }
    ss.setdefault("cn_full_md", "")

    # ---------- Exports ----------
    ss.setdefault("exports", {
        "tor_scan": None,
        "donor_intel": None,
        "aid_trends": None,
        "concept_note": None
    })


def inject_style() -> None:
    st.markdown(
        """
        <style>
          .stApp { background: #0f172a; color:#e5e7eb; }
          .block-container { padding-top: 1rem; }
          .smallcaps{ text-transform:uppercase; letter-spacing:.06em; font-size:.82rem; color:#9ca3af;}
          .muted{ color:#94a3b8; }
          .good{ background:#123326; color:#d1fae5; padding:.6rem .8rem; border-radius:8px;}
          .warn{ background:#3b2f1e; color:#fef3c7; padding:.6rem .8rem; border-radius:8px;}
          textarea{ font-size:.98rem !important; line-height:1.45 !important; }
          code, pre{ font-size:.94rem !important; }
          .tight { line-height: 1.35; }
        </style>
        """,
        unsafe_allow_html=True
    )


# ========================= Export Helpers =========================

def make_docx_bytes(text: str, title: str) -> Tuple[bytes, str]:
    """DOCX if python-docx available, else TXT."""
    if HAS_DOCX:
        try:
            doc = Document()
            for line in text.split("\n"):
                doc.add_paragraph(line)
            bio = io.BytesIO()
            doc.save(bio)
            return bio.getvalue(), f"{title}.docx"
        except Exception:
            pass
    return text.encode("utf-8"), f"{title}.txt"


def store_export(bucket: str, markdown_text: str, title: str) -> None:
    b, fname = make_docx_bytes(markdown_text, title.replace(" ", "_"))
    st.session_state.exports[bucket] = {"bytes": b, "filename": fname, "ts": datetime.utcnow().isoformat() + "Z"}


def md_code(md: str) -> str:
    return f"```\n{md.strip()}\n```"


# ========================= Utilities =========================

STOP = set("""
the a an and or for from with this that into over under across more less very much many most among about above below
to of in on at by as be is are was were been being it its their there here our your his her they them which who whose
""".split())

def top_keywords(text: str, n: int = 14) -> List[str]:
    t = re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())
    words = [w for w in t.split() if len(w) > 3 and w not in STOP]
    freq: Dict[str, int] = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    return [w for w, _ in sorted(freq.items(), key=lambda kv: kv[1], reverse=True)[:n]]


def read_txt_bytes(b: bytes) -> str:
    try:
        return b.decode("utf-8")
    except Exception:
        try:
            return b.decode("latin-1")
        except Exception:
            return ""


def read_docx_bytes(b: bytes) -> str:
    if not HAS_DOCX:
        return ""
    try:
        bio = io.BytesIO(b)
        doc = Document(bio)
        return "\n".join([p.text for p in doc.paragraphs])
    except Exception:
        return ""


def read_pdf_bytes(b: bytes) -> str:
    if not HAS_PDF:
        return ""
    try:
        bio = io.BytesIO(b)
        reader = PdfReader(bio)
        out = []
        for page in reader.pages:
            try:
                out.append(page.extract_text() or "")
            except Exception:
                pass
        return "\n".join(out)
    except Exception:
        return ""


# ========================= Donor Data =========================

DONORS: Dict[str, Dict[str, Any]] = {
    # Bilaterals / agencies requested
    "BMZ / GIZ (Germany)": {
        "overview": "BMZ sets policy; GIZ implements German development cooperation. Focus on governance, climate, TVET, energy, health.",
        "priorities": ["governance", "climate", "tvet", "energy", "health"],
        "geographies": ["Africa","Asia","MENA","Europe","LAC"],
        "modalities": ["Contract","Grant"],
        "typical_grant_usd": (300_000, 30_000_000),
        "compliance": ["Audits", "Safeguarding", "Procurement standards"],
        "portals": [{"name":"GIZ Procurement","url":"https://www.giz.de/en/workingwithgiz/284.html"}],
        "example_kpis": ["institutions supported","policy reforms","skills certifications"],
        "keywords": ["giz","bmz","german","cooperation","governance","tvet","energy"],
    },
    "JICA / MOFA (Japan)": {
        "overview": "Japan’s JICA and MOFA fund infrastructure, disaster risk reduction, health, education; tenders and grants.",
        "priorities": ["infrastructure","drm","health","education"],
        "geographies": ["Asia","Africa","Global"],
        "modalities": ["Contract","Grant","Loan"],
        "typical_grant_usd": (500_000, 50_000_000),
        "compliance": ["Audits","Procurement rules"],
        "portals": [{"name":"JICA Procurement","url":"https://www.jica.go.jp/english/index.html"}],
        "example_kpis": ["km of road","facilities constructed","people protected"],
        "keywords": ["jica","mofa","japan","infrastructure","disaster","health","education"],
    },
    "FCDO (UK)": {
        "overview": "FCDO funds governance, climate, health, education, humanitarian via commercial tenders and grants.",
        "priorities": ["governance","climate resilience","education","health","humanitarian"],
        "geographies": ["Africa","Asia","MENA","Europe","LAC","Global"],
        "modalities": ["Contract","Grant","Challenge Fund"],
        "typical_grant_usd": (500_000, 50_000_000),
        "compliance": ["Audited financial statements","Safeguarding","Anti-fraud","Data protection"],
        "portals": [
            {"name":"Find a Tender (UK)","url":"https://www.find-tender.service.gov.uk/"},
            {"name":"Contracts Finder","url":"https://www.contractsfinder.service.gov.uk/"},
        ],
        "example_kpis": ["% reached","learning outcomes","resilience index"],
        "keywords": ["fcdo","uk","british","governance","girls","education","resilience"],
    },
    "AFD (France)": {
        "overview": "AFD finances development and climate transition projects via grants, loans, and TA.",
        "priorities": ["climate","infrastructure","health","education"],
        "geographies": ["Africa","MENA","Asia","LAC"],
        "modalities": ["Grant","Loan","Service Contract"],
        "typical_grant_usd": (300_000, 30_000_000),
        "compliance": ["Audits","PRAG-like procurement"],
        "portals": [{"name":"AFD","url":"https://www.afd.fr/en"}],
        "example_kpis": ["institutions strengthened","pop reached","policies adopted"],
        "keywords": ["afd","france","infra","climate","service contract","grant"],
    },
    "Netherlands MFA": {
        "overview": "Netherlands MFA directs development policy and aid budget with strong civil society programs.",
        "priorities": ["governance","srhr","climate","food systems"],
        "geographies": ["Africa","Asia","MENA","LAC","Global"],
        "modalities": ["Grant","Contract"],
        "typical_grant_usd": (500_000, 50_000_000),
        "compliance": ["Audits","Safeguarding","Due diligence"],
        "portals": [{"name":"Netherlands MFA","url":"https://www.government.nl/ministries/ministry-of-foreign-affairs"}],
        "example_kpis": ["cso capacity scores","service coverage","policy change"],
        "keywords": ["netherlands","mfa","srhr","governance","food","climate"],
    },
    "Sida (Sweden)": {
        "overview": "Sida funds human rights, climate & environment, gender equality, health, and education.",
        "priorities": ["human rights","gender","climate","health","education"],
        "geographies": ["Africa","Asia","MENA","Global"],
        "modalities": ["Grant","Framework Agreement","Contract"],
        "typical_grant_usd": (300_000, 30_000_000),
        "compliance": ["Audits","Safeguarding","Anti-corruption"],
        "portals": [{"name":"Sida","url":"https://www.sida.se/en"}],
        "example_kpis": ["rights fulfilled","access improved","capacity strengthened"],
        "keywords": ["sida","sweden","gender","rights","climate","health","education"],
    },
    "Global Affairs Canada": {
        "overview": "Canada funds inclusive governance, gender equality, health, education, climate, humanitarian response.",
        "priorities": ["gender","health","education","governance","climate"],
        "geographies": ["Global"],
        "modalities": ["Grant","Contract"],
        "typical_grant_usd": (300_000, 30_000_000),
        "compliance": ["Audits","Safeguarding"],
        "portals": [{"name":"GAC","url":"https://www.international.gc.ca/"}],
        "example_kpis": ["women/children reached","policy reforms","learning outcomes"],
        "keywords": ["canada","gac","gender","education","health","governance"],
    },
    "Norad (Norway)": {
        "overview": "Norad funds civil society, education, climate, oceans, and governance; strong results-based focus.",
        "priorities": ["education","climate","oceans","governance","human rights"],
        "geographies": ["Global"],
        "modalities": ["Grant","Contract"],
        "typical_grant_usd": (200_000, 20_000_000),
        "compliance": ["Audits","Results framework","Safeguarding"],
        "portals": [{"name":"Norad","url":"https://www.norad.no/en/front/"}],
        "example_kpis": ["children learning","emissions avoided","institutions strengthened"],
        "keywords": ["norad","norway","education","climate","oceans","governance"],
    },
    "AICS (Italy)": {
        "overview": "AICS funds development cooperation in health, education, cultural heritage, climate and migration.",
        "priorities": ["health","education","climate","migration"],
        "geographies": ["Africa","MENA","Balkans"],
        "modalities": ["Grant","Contract"],
        "typical_grant_usd": (200_000, 15_000_000),
        "compliance": ["Audits","Due diligence","Safeguarding"],
        "portals": [{"name":"AICS","url":"https://www.aics.gov.it/"}],
        "example_kpis": ["access improved","institutions supported","youth reached"],
        "keywords": ["aics","italy","health","education","migration","climate"],
    },

    # Foundations requested
    "Mastercard Foundation": {
        "overview": "Mastercard Foundation supports youth livelihoods, financial inclusion, and education in Africa.",
        "priorities": ["youth employment","livelihoods","education","financial inclusion"],
        "geographies": ["Africa"],
        "modalities": ["Grant","Partnership"],
        "typical_grant_usd": (500_000, 50_000_000),
        "compliance": ["Audits","Safeguarding"],
        "portals": [{"name":"Mastercard Foundation","url":"https://mastercardfdn.org/"}],
        "example_kpis": ["jobs created","youth trained","enterprises supported"],
        "keywords": ["mastercard","youth","employment","education","financial inclusion"],
    },
    "Novo Nordisk Foundation": {
        "overview": "Novo Nordisk Foundation funds health, life sciences, and humanitarian initiatives.",
        "priorities": ["health","life sciences","humanitarian"],
        "geographies": ["Global"],
        "modalities": ["Grant"],
        "typical_grant_usd": (200_000, 10_000_000),
        "compliance": ["Audits","Ethics"],
        "portals": [{"name":"NNF","url":"https://novonordiskfonden.dk/en/"}],
        "example_kpis": ["research outputs","patients reached","capacity built"],
        "keywords": ["novo","nordisk","health","science","foundation"],
    },
    "Tata Trusts": {
        "overview": "Tata Trusts fund health, livelihoods, education, and rural development in India and beyond.",
        "priorities": ["health","livelihoods","education","rural development"],
        "geographies": ["India","South Asia"],
        "modalities": ["Grant","Partnership"],
        "typical_grant_usd": (100_000, 5_000_000),
        "compliance": ["Audits","Local registration (India)"],
        "portals": [{"name":"Tata Trusts","url":"https://www.tatatrusts.org/"}],
        "example_kpis": ["households reached","students supported","jobs created"],
        "keywords": ["tata","india","livelihoods","health","education"],
    },
    "Gates Foundation": {
        "overview": "Gates Foundation funds global health, agriculture, and education innovations.",
        "priorities": ["global health","agriculture","education","innovation"],
        "geographies": ["Global"],
        "modalities": ["Grant","PRI"],
        "typical_grant_usd": (200_000, 20_000_000),
        "compliance": ["Audits","Safeguarding"],
        "portals": [{"name":"Gates Grants","url":"https://www.gatesfoundation.org/about/committed-grants"}],
        "example_kpis": ["coverage rates","yield improvements","learning outcomes"],
        "keywords": ["gates","foundation","innovation","health","agriculture","education"],
    },
    "Wellcome Trust": {
        "overview": "Wellcome Trust funds health research, epidemics, mental health, and climate-health links.",
        "priorities": ["health research","epidemics","mental health","climate-health"],
        "geographies": ["Global"],
        "modalities": ["Grant"],
        "typical_grant_usd": (200_000, 15_000_000),
        "compliance": ["Audits","Ethics"],
        "portals": [{"name":"Wellcome","url":"https://wellcome.org/grant-funding"}],
        "example_kpis": ["research outputs","policy uptake","health outcomes"],
        "keywords": ["wellcome","trust","health","research","epidemic","climate"],
    },
    "Stichting INGKA Foundation": {
        "overview": "INGKA Foundation (IKEA) supports livelihoods, refugees’ economic inclusion, and climate action.",
        "priorities": ["livelihoods","refugees","climate action"],
        "geographies": ["Global"],
        "modalities": ["Grant","Partnership"],
        "typical_grant_usd": (200_000, 8_000_000),
        "compliance": ["Audits","Safeguarding"],
        "portals": [{"name":"IKEA Foundation","url":"https://ikeafoundation.org/"}],
        "example_kpis": ["jobs supported","enterprises created","emissions avoided"],
        "keywords": ["ingka","ikea","livelihoods","refugees","climate"],
    },
}

REGIONS = sorted({g for d in DONORS.values() for g in d["geographies"]} | {"Global"})
ALL_SECTORS = sorted(set(sum([d["priorities"] for d in DONORS.values()], [])))
ALL_MODALITIES = sorted(set(sum([d["modalities"] for d in DONORS.values()], [])))

COUNTRIES = [
    "All","Afghanistan","Albania","Algeria","Angola","Argentina","Australia","Bangladesh","Belgium","Benin","Bolivia",
    "Bosnia and Herzegovina","Botswana","Brazil","Bulgaria","Burkina Faso","Burundi","Cambodia","Cameroon","Canada",
    "Central African Republic","Chad","Chile","China","Colombia","Congo","Costa Rica","Cote d'Ivoire","Croatia","Cuba",
    "Denmark","DR Congo","Dominican Republic","Ecuador","Egypt","El Salvador","Ethiopia","Finland","France","Gabon",
    "Gambia","Georgia","Germany","Ghana","Greece","Guatemala","Guinea","Haiti","Honduras","Hungary","Iceland","India",
    "Indonesia","Iran","Iraq","Ireland","Israel","Italy","Jamaica","Japan","Jordan","Kenya","Kuwait","Laos","Lebanon",
    "Lesotho","Liberia","Libya","Madagascar","Malawi","Malaysia","Mali","Mauritania","Mexico","Moldova","Mongolia",
    "Morocco","Mozambique","Myanmar","Namibia","Nepal","Netherlands","New Zealand","Nicaragua","Niger","Nigeria",
    "North Macedonia","Norway","Pakistan","Panama","Papua New Guinea","Paraguay","Peru","Philippines","Poland",
    "Portugal","Romania","Rwanda","Saudi Arabia","Senegal","Serbia","Sierra Leone","Somalia","South Africa","South Sudan",
    "Spain","Sri Lanka","Sudan","Sweden","Switzerland","Syria","Tanzania","Thailand","Togo","Tunisia","Turkey","Uganda",
    "Ukraine","United Arab Emirates","United Kingdom","United States","Uruguay","Venezuela","Vietnam","Yemen","Zambia","Zimbabwe"
]

COUNTRY_TO_REGION = {
    # quick mapping for filtering; not exhaustive
    "Canada":"Global","United States":"Global",
    "United Kingdom":"Europe","France":"Europe","Germany":"Europe","Netherlands":"Europe","Sweden":"Europe","Italy":"Europe",
    "Norway":"Europe","Japan":"Asia","India":"Asia","Bangladesh":"Asia","Pakistan":"Asia","Indonesia":"Asia",
    "Kenya":"Africa","Uganda":"Africa","Tanzania":"Africa","Rwanda":"Africa","Nigeria":"Africa","Ghana":"Africa",
    "Ethiopia":"Africa","Somalia":"Africa","South Sudan":"Africa","Morocco":"Africa","Tunisia":"Africa","Algeria":"Africa",
    "Brazil":"LAC","Peru":"LAC","Colombia":"LAC","Mexico":"LAC","Haiti":"LAC","Bolivia":"LAC","Chile":"LAC","Argentina":"LAC",
}


# ========================= Donor Logic =========================

def donor_matches_filters(donor_key: str, f: Dict[str, Any]) -> bool:
    d = DONORS[donor_key]

    # Region filter
    if f["region"] != "Global" and f["region"] not in d["geographies"]:
        return False

    # Country filter -> map to region if possible
    if f["country"] != "All":
        region_guess = COUNTRY_TO_REGION.get(f["country"], "Global")
        if region_guess != "Global" and region_guess not in d["geographies"]:
            return False

    # sectors
    if f["sector"]:
        if not any(s in d["priorities"] for s in f["sector"]):
            return False

    # modality
    if f["modality"]:
        if not any(m in d["modalities"] for m in f["modality"]):
            return False

    # grant size
    gmin, gmax = d["typical_grant_usd"]
    if gmax < f["grant_min"] or gmin > f["grant_max"]:
        return False

    # keyword
    kw = f["keyword"].strip().lower()
    if kw:
        hay = (d["overview"] + " " + " ".join(d["priorities"]) + " " + " ".join(d["keywords"])).lower()
        if kw not in hay:
            return False

    return True


def render_donor_brief(donor_key: str) -> str:
    d = DONORS[donor_key]
    gmin, gmax = d["typical_grant_usd"]
    portals = "\n".join([f"- {p['name']}: {p['url']}" for p in d["portals"]])
    priorities = ", ".join(d["priorities"])
    comps = ", ".join(d["compliance"])
    geos = ", ".join(d["geographies"])
    mods = ", ".join(d["modalities"])

    md = f"""
DONOR BRIEF — {donor_key}
-------------------------

Overview:
{d['overview']}

Focus Areas:
- Priorities: {priorities}
- Geographies: {geos}
- Modalities: {mods}
- Typical Award Size (USD): {gmin:,} – {gmax:,}

Key Compliance:
- {comps}

Indicative KPIs:
- {", ".join(d["example_kpis"])}

Procurement / Portals:
{portals}
""".strip()
    return md


# ========================= Trends & CN =========================

def render_trends_brief(region: str, theme: str, horizon: str, audience: str, notes: str) -> str:
    sources = [
        ("OECD Aid Data", "https://stats.oecd.org/"),
        ("Devex Funding News", "https://www.devex.com/news"),
        ("GIIN Insights", "https://thegiin.org/"),
        ("Candid (Philanthropy Data)", "https://candid.org/"),
    ]
    md = f"""
AID TRENDS BRIEF — {region.upper()} / {theme.upper()}
-----------------------------------------------------

Audience: {audience}
Horizon:  {horizon}

1) Donor Shifts (illustrative):
- Larger multi-country programs; fewer small awards.
- Measurable outcomes and co-finance increasingly required.
- In {region}, calls in {theme.lower()} emphasise clear MEL and partnerships.

2) Private Funding Pipelines (illustrative):
- Corporate/philanthropy interest rising in {theme.lower()} and resilience.
- Blended finance: outcomes-based grants + concessional capital.
- Family offices: place-based projects with measurable outcomes.

3) Implications for NGOs:
- Position consortia for scale; clarify local legitimacy.
- Strengthen MEL (3–5 KPIs) and results-based readiness.
- Maintain a pipeline of ‘shovel-ready’ concepts (12–24 months).

Indicative Sources:
- """ + "\n- ".join([f"[{label}]({url})" for (label, url) in sources]) + f"""

Analyst Notes:
- {notes.strip() if notes.strip() else 'N/A'}
""".strip()
    return md


CN_SECTIONS = [
    "Background / Problem Statement",
    "Objectives",
    "Approach / Theory of Change",
    "Key Activities",
    "Geography & Beneficiaries",
    "Partnerships & Governance",
    "MEL & KPIs",
    "Risk & Safeguarding",
    "Budget & Value for Money",
    "Sustainability / Exit",
]

def synthesize_section_text(section: str, words: int, hints: str, donor_brief: str, trends_brief: str, tor_summary: str) -> str:
    """
    Stronger synthesis:
      - seeded template paragraph
      - weave in ToR + Donor + Trends references
      - expand until approx words
    """
    base = {
        "Background / Problem Statement": (
            "The programme addresses clearly evidenced needs and systemic constraints in the target geographies. "
            "The context highlights gaps in delivery capacity, coordination, and the need for measurable, durable outcomes."
        ),
        "Objectives": (
            "We propose a concise set of measurable objectives that align with donor focus areas and community priorities. "
            "Objectives emphasise equity, quality of service, and resilience to shocks."
        ),
        "Approach / Theory of Change": (
            "Our approach links inputs to outputs and outcomes through evidence-based activities, "
            "feedback loops, and adaptive management. Learning informs iteration across delivery cycles."
        ),
        "Key Activities": (
            "Delivery centres on practical, locally-led workstreams with clear milestones, "
            "capacity transfer, and value for money. A light-touch PMU ensures coordination and risk control."
        ),
        "Geography & Beneficiaries": (
            "The action targets prioritised locations and population groups, with attention to inclusion, "
            "gender equality, disability, and conflict sensitivity where relevant."
        ),
        "Partnerships & Governance": (
            "The consortium combines local legitimacy with technical expertise. Governance is simple and accountable, "
            "with clear roles, escalation routes, and an embedded safeguarding culture."
        ),
        "MEL & KPIs": (
            "We deploy a lean MEL framework with a concise results chain and 3–5 KPIs. "
            "Data quality and protection are safeguarded; findings inform adaptation and accountability."
        ),
        "Risk & Safeguarding": (
            "Risks are assessed and mitigated through safeguarding, Do No Harm, and data protection protocols. "
            "Procurement and financial control prevent fraud and inefficiency."
        ),
        "Budget & Value for Money": (
            "Resources prioritise delivery, local capacity, and learning. VfM is demonstrated via economy, efficiency, and effectiveness, "
            "with clear unit costs and a rationale for scale."
        ),
        "Sustainability / Exit": (
            "The exit is planned from inception. We build institutional capacity, peer learning, and financing pathways "
            "to sustain outcomes beyond the grant period."
        ),
    }

    # Context anchors
    donor_line = donor_brief.splitlines()[0].strip() if donor_brief else ""
    trends_line = trends_brief.splitlines()[0].strip().lower() if trends_brief else ""
    tor_line = tor_summary.splitlines()[0].strip().lower() if tor_summary else ""

    seed = base.get(section, "")
    if hints.strip():
        seed += f" Key data/keywords from the proponent: {hints.strip()}."
    if donor_line:
        seed += f" Alignment: {donor_line}."
    if trends_line:
        seed += f" Trends reference: {trends_line}."
    if tor_line:
        seed += f" ToR summary anchor: {tor_line}."

    # Expand heuristically to reach target words
    expansions = [
        " Local partners co-design and lead implementation, ensuring legitimacy and contextual fit.",
        " The approach enables rapid learning cycles and course-correction through simple reflection points.",
        " Cross-cutting priorities include gender equality, disability inclusion, and climate-smart methods.",
        " Stakeholder engagement clarifies incentives and supports durable coalitions for change.",
        " Where feasible, co-financing or blended arrangements crowd-in additional capital.",
        " Clear roles, RACI lines, and an escalation path maintain accountability and momentum.",
        " Evidence and stories of change are communicated with respect for data privacy and safeguarding.",
    ]
    out_words = seed.split()
    i = 0
    while len(out_words) < words:
        out_words.extend(expansions[i % len(expansions)].split())
        i += 1
        if i > 50:  # safety
            break
    text = " ".join(out_words[:words])
    if not text.endswith((".", "!", "?")):
        text += "."
    return text


# ========================= ToR Summariser (deeper) =========================

SECTION_HINTS = [
    r"(objectives|purpose|goals)",
    r"(scope|description|work packages|tasks|activities)",
    r"(timeline|schedule|deliverables|milestones)",
    r"(eligibility|requirements|qualification|experience|compliance)",
    r"(budget|value for money|ceiling|cost|price|financial)",
]

def extract_numeric_cues(text: str) -> List[str]:
    cues = []
    for m in re.finditer(r"\b(\d{1,3}(?:,\d{3})+|\d+)\s*(USD|eur|gbp|usd|€|\$)\b", text, re.I):
        cues.append(f"Budget mention: {m.group(0)}")
    for m in re.finditer(r"\b(\d{1,2})\s*(months?|years?)\b", text, re.I):
        cues.append(f"Timeline mention: {m.group(0)}")
    for m in re.finditer(r"\bdeadline[:\s]+([A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4}|\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4})", text, re.I):
        cues.append(f"Deadline: {m.group(1)}")
    return list(dict.fromkeys(cues))[:6]


def best_lines(text: str, limit: int = 6) -> List[str]:
    """Pick strong lines resembling headings/bullets."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    bullets = [ln for ln in lines if re.match(r"^[-•\u2022]", ln)]
    if bullets:
        cleaned = [re.sub(r"^[-•\u2022]\s*", "", ln) for ln in bullets]
        return cleaned[:limit]
    # fallback: pick longer lines with verbs
    scored = []
    for ln in lines:
        score = len(ln)
        if re.search(r"\b(will|shall|deliver|develop|support|provide|report|train)\b", ln, re.I):
            score += 40
        scored.append((score, ln))
    return [ln for _, ln in sorted(scored, reverse=True)[:limit]]


def summarise_tor_deep(text: str) -> str:
    if not text.strip():
        return ""

    wc = len(text.split())
    kws = top_keywords(text, 14)
    nums = extract_numeric_cues(text)

    # Heuristic sections
    lower = text.lower()
    sections: Dict[str, List[str]] = {
        "Likely Objectives / Outcomes": [],
        "Eligibility & Compliance": [],
        "Deliverables & Timeline": [],
        "Budget / Value for Money": [],
        "Activities / Scope (sample)": [],
    }

    # capture lines around hits
    lines = text.splitlines()
    L = len(lines)
    for i, ln in enumerate(lines):
        l = ln.lower()
        for hint in SECTION_HINTS:
            if re.search(hint, l):
                window = "\n".join(lines[max(0, i-3):min(L, i+6)])
                if "objective" in hint or "purpose" in hint or "goals" in hint:
                    sections["Likely Objectives / Outcomes"].extend(best_lines(window, 5))
                elif "eligibility" in hint or "qualification" in hint or "compliance" in hint:
                    sections["Eligibility & Compliance"].extend(best_lines(window, 6))
                elif "timeline" in hint or "deliverables" in hint or "milestones" in hint:
                    sections["Deliverables & Timeline"].extend(best_lines(window, 6))
                elif "budget" in hint or "value for money" in hint or "cost" in hint:
                    sections["Budget / Value for Money"].extend(best_lines(window, 4))
                elif "scope" in hint or "activities" in hint or "work packages" in hint or "tasks" in hint:
                    sections["Activities / Scope (sample)"].extend(best_lines(window, 8))

    # tidy
    for k in sections:
        # de-dup + keep top
        seen = []
        clean = []
        for x in sections[k]:
            x = re.sub(r"^[-•\u2022]\s*", "", x).strip()
            if x and x not in seen:
                seen.append(x); clean.append(x)
            if len(clean) >= (8 if k == "Activities / Scope (sample)" else 6):
                break
        sections[k] = clean

    overview = [
        "The document outlines a competitive process requiring measurable outcomes, clear deliverables, and strong governance.",
        "Partnerships and value for money are emphasised alongside safeguarding and MEL requirements.",
    ]

    if wc > 2000:  # bigger doc -> add extra overview lines
        overview.extend([
            "Given the breadth, bidders should propose a lean PMU, adaptive management, and clear escalation paths.",
            "Co-finance or blended arrangements may be advantageous if feasible."
        ])

    risks = [
        "Ambitious timelines relative to scope and expected coordination.",
        "Eligibility filters and compliance needs may limit competition.",
        "Data privacy & safeguarding requirements call for robust controls."
    ]

    next_steps = [
        "Clarify budget ceiling and contracting modality with the donor.",
        "Confirm deliverables schedule and acceptance criteria.",
        "Map local partners and roles; prepare a 3–5 KPI MEL mini-framework."
    ]

    md = f"""
**ToR / TENDER SCAN — ENHANCED SUMMARY**
----------------------------------------

Detected length: ~{wc} words
Prominent keywords (naive): {", ".join(kws) if kws else "N/A"}

Overview:
- """ + "\n- ".join(overview) + """

Numeric / Date cues (heuristic):
- """ + ("\n- ".join(nums) if nums else "None detected.") + "\n\n"

    for title, items in sections.items():
        md += f"**{title}:**\n"
        if items:
            md += "- " + "\n- ".join(items) + "\n\n"
        else:
            md += "- Not explicitly detected; refer to full text.\n\n"

    md += "**Potential Risks:**\n- " + "\n- ".join(risks) + "\n\n"
    md += "**Suggested Next Steps:**\n- " + "\n- ".join(next_steps)

    return md.strip()


# ========================= Reset helpers =========================

def reset_page(page: str) -> None:
    if page == "ToR":
        st.session_state.tor_text = ""
        st.session_state.tor_summary_md = ""
        st.session_state.exports["tor_scan"] = None
    elif page == "Donor":
        st.session_state.donor_filters = {
            "region": "Global",
            "country": "All",
            "sector": [],
            "modality": [],
            "grant_min": 0,
            "grant_max": 50_000_000,
            "keyword": "",
        }
        st.session_state.donor_selected = None
        st.session_state.donor_brief_md = ""
        st.session_state.exports["donor_intel"] = None
    elif page == "Trends":
        st.session_state.trends = {
            "region": "Global",
            "theme": "Climate",
            "horizon": "Long (3–5y)",
            "audience": "Programme Team",
            "notes": ""
        }
        st.session_state.trends_brief_md = ""
        st.session_state.exports["aid_trends"] = None
    elif page == "CN":
        st.session_state.cn_sections = {}
        st.session_state.cn_full_md = ""
        st.session_state.exports["concept_note"] = None


# ========================= Pages =========================

def page_home() -> None:
    st.title("NGO AI Grant Assistant (Local Prototype)")
    st.caption("No external APIs • Everything stays in this browser session.")

    st.markdown(
        """
### What this does  
**Turn a ToR into a funder-ready concept** — fast.  
Scan ToRs, research likely **donors**, generate a clean **trends brief**, and assemble a **concept note** with your own key data injected where it matters.

### Workflow
1. **ToR / Tender Scanner** – paste or upload (TXT, DOCX, PDF). Get an enhanced summary and export.  
2. **Donor Intelligence Panel** – filter donors (bilateral & foundations) and export a donor brief.  
3. **Aid Trends** – create a short trends memo for your region/theme.  
4. **Concept Note Builder** – add key data per section; we reuse ToR/Donor/Trends context; export the note.  
5. **Exports** – one-click download for each artefact.

Use **Settings → Reset** to clear everything.
        """
    )


def page_tor_scanner() -> None:
    st.title("ToR / Tender Scanner")
    st.caption("Paste or upload. Results display below; export as DOCX/TXT.")

    cA, cB = st.columns([2,1], vertical_alignment="top")
    with cA:
        st.session_state.tor_text = st.text_area(
            "PASTE TOR / PROJECT DESCRIPTION",
            value=st.session_state.tor_text,
            height=340,
            placeholder="Paste text here… (no character limit)"
        )
    with cB:
        st.markdown("**OR UPLOAD FILE(S)**")
        st.caption("Limit 200MB per file • TXT, DOCX, PDF (OCR’d)")
        files = st.file_uploader(" ", type=["txt", "docx", "pdf"], accept_multiple_files=True, label_visibility="collapsed")

    # Merge uploads into tor_text
    if files:
        parts = []
        for f in files:
            raw = f.read()
            if f.type == "text/plain" or f.name.lower().endswith(".txt"):
                parts.append(read_txt_bytes(raw))
            elif f.name.lower().endswith(".docx"):
                t = read_docx_bytes(raw)
                if not t:
                    st.warning("DOCX parser unavailable. Install `python-docx` for richer parsing.")
                parts.append(t)
            elif f.name.lower().endswith(".pdf"):
                t = read_pdf_bytes(raw)
                if not t:
                    st.warning("PDF parser unavailable OR PDF is scanned. Use an OCR’d PDF for text.")
                parts.append(t)
        if parts:
            st.session_state.tor_text = (st.session_state.tor_text + "\n\n" + "\n\n".join(parts)).strip()

    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        if st.button("Scan ToR", type="primary"):
            if not st.session_state.tor_text.strip():
                st.warning("Provide text or upload at least one file.")
            else:
                st.session_state.tor_summary_md = summarise_tor_deep(st.session_state.tor_text)
                store_export("tor_scan", st.session_state.tor_summary_md, "ToR_Scan_Summary")
                st.success("Enhanced summary created.")
    with c2:
        if st.button("Reset this page"):
            reset_page("ToR")
            st.experimental_rerun()
    with c3:
        st.caption(" ")

    if st.session_state.tor_summary_md:
        st.subheader("Enhanced Summary")
        st.markdown(md_code(st.session_state.tor_summary_md))
        blob = st.session_state.exports["tor_scan"]
        if blob:
            st.download_button("Download ToR Summary (DOCX)", data=blob["bytes"], file_name=blob["filename"], type="primary")


def page_donor_panel() -> None:
    st.title("Donor Intelligence Panel")

    f = st.session_state.donor_filters
    with st.expander("Filters", expanded=True):
        c1, c2, c3 = st.columns([1,1,1])
        with c1:
            f["region"] = st.selectbox("Region", REGIONS, index=REGIONS.index(f["region"]))
            f["country"] = st.selectbox("Country (A–Z)", COUNTRIES, index=COUNTRIES.index(f["country"]))
        with c2:
            f["sector"] = st.multiselect("Priority (sector)", ALL_SECTORS, default=f["sector"])
            f["modality"] = st.multiselect("Modality", ALL_MODALITIES, default=f["modality"])
        with c3:
            f["grant_min"], f["grant_max"] = st.slider("Typical award size (USD)", 0, 50_000_000, (f["grant_min"], f["grant_max"]), step=100_000)
            f["keyword"] = st.text_input("Keyword (optional)", f["keyword"], placeholder="e.g., resilience, education…")

        if st.button("Reset filters", use_container_width=True):
            reset_page("Donor")
            st.experimental_rerun()

    matches = [k for k in DONORS.keys() if donor_matches_filters(k, f)]
    if not matches:
        st.info("No donors match these filters. Broaden your search or clear keyword.")
        return

    st.subheader("Matched donors")
    choice = st.radio("Select a donor", matches, index=0, horizontal=True)
    st.session_state.donor_selected = choice

    st.markdown("---")
    st.subheader(f"Profile: {choice}")
    brief = render_donor_brief(choice)
    st.session_state.donor_brief_md = brief
    st.markdown(md_code(brief))

    # Export donor brief
    store_export("donor_intel", brief, f"Donor_Brief_{choice}")
    blob = st.session_state.exports["donor_intel"]
    if blob:
        st.download_button("Download Donor Brief (DOCX)", data=blob["bytes"], file_name=blob["filename"], type="primary")


def page_aid_trends() -> None:
    st.title("Aid Trends")
    t = st.session_state.trends
    c1, c2 = st.columns([1,1])
    with c1:
        t["region"] = st.selectbox("Region focus", ["Global","Africa","Asia","MENA","Europe","LAC"], index=["Global","Africa","Asia","MENA","Europe","LAC"].index(t["region"]))
        t["theme"]  = st.selectbox("Theme focus", ["Climate","Smart Agriculture","Health","Education","Protection","WASH"], index=["Climate","Smart Agriculture","Health","Education","Protection","WASH"].index(t["theme"]))
    with c2:
        t["horizon"]  = st.selectbox("Time horizon", ["Short (6–12m)","Medium (1–3y)","Long (3–5y)"], index=["Short (6–12m)","Medium (1–3y)","Long (3–5y)"].index(t["horizon"]))
        t["audience"] = st.selectbox("Audience", ["Programme Team","Board / Exec","Donor Relations"], index=["Programme Team","Board / Exec","Donor Relations"].index(t["audience"]))
    t["notes"] = st.text_area("Analyst notes (optional)", t["notes"], height=110)

    cA, cB = st.columns([1,1])
    with cA:
        if st.button("Generate Trends Brief", type="primary"):
            brief = render_trends_brief(t["region"], t["theme"], t["horizon"], t["audience"], t["notes"])
            st.session_state.trends_brief_md = brief
            store_export("aid_trends", brief, "Aid_Trends_Brief")
            st.success("Trends brief created.")
    with cB:
        if st.button("Reset this page"):
            reset_page("Trends")
            st.experimental_rerun()

    if st.session_state.trends_brief_md:
        st.subheader("Brief")
        st.markdown(md_code(st.session_state.trends_brief_md))
        blob = st.session_state.exports["aid_trends"]
        if blob:
            st.download_button("Download Trends Brief (DOCX)", data=blob["bytes"], file_name=blob["filename"], type="primary")


def page_concept_note() -> None:
    st.title("Concept Note Builder")
    st.caption("Add key data/keywords per section and choose word counts. We reuse ToR / Donor / Trends context automatically.")

    donor_ref = st.session_state.donor_brief_md
    trends_ref = st.session_state.trends_brief_md
    tor_summary = st.session_state.tor_summary_md

    cols = st.columns(2)
    for i, sec in enumerate(CN_SECTIONS):
        with cols[i % 2]:
            st.markdown(f"### {sec}")
            hints = st.text_area("Add key data / keywords (optional)",
                                 value=st.session_state.cn_sections.get(sec, {}).get("hints",""),
                                 key=f"hints_{sec}", height=100)
            words = st.slider("Approx. words", 90, 450, st.session_state.cn_sections.get(sec,{}).get("words",180),
                              10, key=f"words_{sec}")
            if st.button(f"Generate {sec}", key=f"gen_{sec}", use_container_width=True):
                text = synthesize_section_text(sec, words, hints, donor_ref, trends_ref, tor_summary)
                st.session_state.cn_sections[sec] = {"text": text, "words": words, "hints": hints}
                st.success("Section generated.")

            cur = st.session_state.cn_sections.get(sec,{}).get("text","")
            if cur:
                st.text_area("Draft", value=cur, height=160, key=f"show_{sec}")

    st.markdown("---")
    cA, cB = st.columns([1,1])
    with cA:
        if st.button("Assemble Full Concept Note", type="primary"):
            blocks = []
            for sec in CN_SECTIONS:
                t = st.session_state.cn_sections.get(sec,{}).get("text","")
                if t:
                    blocks.append(f"## {sec}\n{t}\n")
            if not blocks:
                st.warning("Generate at least one section first.")
            else:
                full = f"# Concept Note Draft\nGenerated: {datetime.utcnow().isoformat()}Z\n\n" + "\n".join(blocks)
                st.session_state.cn_full_md = full
                store_export("concept_note", full, "Concept_Note")
                st.success("Concept Note assembled. Download below or from Exports.")
    with cB:
        if st.button("Reset this page"):
            reset_page("CN")
            st.experimental_rerun()

    if st.session_state.cn_full_md:
        st.subheader("Full Draft")
        st.markdown(md_code(st.session_state.cn_full_md))
        blob = st.session_state.exports["concept_note"]
        if blob:
            st.download_button("Download Concept Note (DOCX)", data=blob["bytes"], file_name=blob["filename"], type="primary")


def page_exports() -> None:
    st.title("Exports")
    st.caption("One-click downloads for the latest artefacts.")

    def show_row(label: str, key: str):
        st.subheader(label)
        item = st.session_state.exports.get(key)
        if not item:
            st.info("Nothing here yet.")
        else:
            st.caption(f"Latest: {item.get('ts','')}")
            st.download_button("Download", data=item["bytes"], file_name=item["filename"], type="primary")
        st.divider()

    show_row("ToR / Tender Scan", "tor_scan")
    show_row("Donor Intelligence", "donor_intel")
    show_row("Aid Trends Brief", "aid_trends")
    show_row("Concept Note", "concept_note")


def page_settings() -> None:
    st.title("Settings")
    st.caption("Data is stored only in this browser session.")
    if st.button("Reset all data"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        ss_init()
        st.success("All data cleared. Go to Home to start fresh.")


# ========================= App =========================

def main() -> None:
    st.set_page_config(page_title="NGO AI Grant Assistant", page_icon="🌍", layout="wide")
    ss_init()
    inject_style()

    with st.sidebar:
        st.title("Navigate")
        st.session_state.nav = st.radio(
            "",
            ["Home","ToR / Tender Scanner","Donor Intelligence Panel","Aid Trends","Concept Note Builder","Exports","Settings"],
            index=["Home","ToR / Tender Scanner","Donor Intelligence Panel","Aid Trends","Concept Note Builder","Exports","Settings"].index(st.session_state.nav),
        )
        st.caption("Local prototype, no external APIs.")

    page = st.session_state.nav
    if page == "Home":
        page_home()
    elif page == "ToR / Tender Scanner":
        page_tor_scanner()
    elif page == "Donor Intelligence Panel":
        page_donor_panel()
    elif page == "Aid Trends":
        page_aid_trends()
    elif page == "Concept Note Builder":
        page_concept_note()
    elif page == "Exports":
        page_exports()
    elif page == "Settings":
        page_settings()


if __name__ == "__main__":
