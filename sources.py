"""
sources.py — Knowledge-base sources for the GRADE and RoB2 chatbots.

Single place for source URLs, PDF folders, cache paths and the up-to-date status
of each GRADE source as labelled on the GRADE Living Map
(https://book.gradepro.org/map). See docs/findings/grade_source_audit.md.

Sources marked "Outdated" on the map are not part of the knowledge base.
"""

STATUS_CURRENT = "current"
STATUS_PARTIAL = "partially_up_to_date"

# ---------------------------------------------------------------------------
# GRADE
# ---------------------------------------------------------------------------

GRADE_BOOK_URL = "https://book.gradepro.org/guideline/"

GRADE_URLS = [
    GRADE_BOOK_URL + "overview-of-the-grade-approach",
    GRADE_BOOK_URL + "the-development-methods-of-grade",
    GRADE_BOOK_URL + "requirements-for-claiming-the-use-of-grade",
    GRADE_BOOK_URL + "questions-about-interventions-diagnostic-test-prognosis-and-exposures",
    GRADE_BOOK_URL + "risk-of-bias-randomized-trials",
    GRADE_BOOK_URL + "inconsistency",
    GRADE_BOOK_URL + "imprecision",
    GRADE_BOOK_URL + "dissemination-bias",
    GRADE_BOOK_URL + "indirectness",
    GRADE_BOOK_URL + "decision-thresholds",
    GRADE_BOOK_URL + "outcomes",
    GRADE_BOOK_URL + "the-history-and-evolution-of-grade",
    GRADE_BOOK_URL + "introduction-to-the-evidence-to-decision-frameworks",
    GRADE_BOOK_URL + "grade-recommendations",
]

GRADE_PDF_DIR = "data/grade_pdfs"
PARSED_GRADE_PDF_DIR = "data/parsed_grade"
GRADE_CACHE_DIR = ".faiss_cache/grade"

# Sources not listed here are Current on the map.
# 9.pdf and 11.pdf are Partially Up-To-Date under one topic and Outdated under another;
# they are kept as partially up-to-date until the instructor decides otherwise.
GRADE_SOURCE_STATUS = {
    "9.pdf": STATUS_PARTIAL,
    "10.pdf": STATUS_PARTIAL,
    "11.pdf": STATUS_PARTIAL,
    "12.pdf": STATUS_PARTIAL,
    "13.pdf": STATUS_PARTIAL,
    "19.pdf": STATUS_PARTIAL,
    "20.pdf": STATUS_PARTIAL,
    "32.pdf": STATUS_PARTIAL,
    "33.pdf": STATUS_PARTIAL,
}

# ---------------------------------------------------------------------------
# RoB2
# ---------------------------------------------------------------------------

ROB2_PDF_DIR = "data/rob2_pdfs"
PARSED_ROB2_PDF_DIR = "data/parsed_rob2"
ROB2_CACHE_DIR = ".faiss_cache/rob2"


def source_status(source: str, status_map: dict[str, str]) -> str:
    """Return the up-to-date status of a source (file name or URL)."""
    return status_map.get(source, STATUS_CURRENT)
