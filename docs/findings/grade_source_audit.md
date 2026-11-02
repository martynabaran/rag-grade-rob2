# GRADE Source Audit

Comparison of the GRADE knowledge base (`data/grade_pdfs/` and the GRADE Book pages scraped in
`grade_rag.py`) with the status labels on the [GRADE Living Map](https://book.gradepro.org/map).

- **Map version checked:** 13 June 2026 (checked on 2026-11-02)
- **How to reproduce:** open the map, check the *Repository* view (Current and Partially
  Up-To-Date articles) and the *Archive* view (Outdated articles), and match titles
  against the first chunk of each file in `data/parsed_grade/*.json`.

## Summary

| Status | PDFs | Count |
|---|---|---|
| Outdated | 1, 3, 4, 5, 6, 7, 8, 14, 15 | 9 |
| Mixed (Partially Up-To-Date in one topic, Outdated in another) | 9, 11 | 2 |
| Partially Up-To-Date | 10, 12, 13, 19, 20, 32, 33 | 7 |
| Current (in at least one topic) | 2, 16, 17, 18, 21_1, 21_2, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 34, 35, 36, 37, 38, 39, 40, 41, 44 | 25 |

All **8 scraped GRADE Book pages are Current**.

> An earlier check concluded that none of our PDFs are outdated. That check only looked at the
> default *Repository* view, which hides Outdated articles. They are listed in the *Archive*
> view.

## Per-PDF status

| File | Article | Map status |
|---|---|---|
| 1.pdf | GRADE guidelines 1. Introduction — GRADE evidence profiles and summary of findings tables | **Outdated** |
| 2.pdf | GRADE guidelines 2. Framing the question and deciding on important outcomes | Current |
| 3.pdf | GRADE guidelines 3. Rating the quality of evidence | **Outdated** |
| 4.pdf | GRADE guidelines 4. Rating the quality of evidence — study limitations (risk of bias) | **Outdated** |
| 5.pdf | GRADE guidelines 5. Rating the quality of evidence — publication bias | **Outdated** |
| 6.pdf | GRADE guidelines 6. Rating the quality of evidence — imprecision | **Outdated** (superseded by Guidance 34/35) |
| 7.pdf | GRADE guidelines 7. Rating the quality of evidence — inconsistency | **Outdated** (superseded by Guidance 36) |
| 8.pdf | GRADE guidelines 8. Rating the quality of evidence — indirectness | **Outdated** |
| 9.pdf | GRADE guidelines 9. Rating up the quality of evidence | Partially Up-To-Date (Large effect) / **Outdated** (Dose-response, superseded by Guidance 38) |
| 10.pdf | GRADE guidelines 10. Considering resource use and rating the quality of economic evidence | Partially Up-To-Date |
| 11.pdf | GRADE guidelines 11. Making an overall rating of confidence in effect estimates | Partially Up-To-Date (SoF tables) / **Outdated** (Overall certainty) |
| 12.pdf | GRADE guidelines 12. Preparing Summary of Findings tables — binary outcomes | Partially Up-To-Date |
| 13.pdf | GRADE guidelines 13. Preparing Summary of Findings tables — continuous outcomes | Partially Up-To-Date |
| 14.pdf | GRADE guidelines 14. Going from evidence to recommendations: significance and presentation | **Outdated** |
| 15.pdf | GRADE guidelines 15. Going from evidence to recommendation — direction and strength | **Outdated** |
| 16.pdf | GRADE guidelines 16. EtD frameworks for tests in clinical practice and public health | Current |
| 17.pdf | GRADE guidelines 17. Risk of bias from missing participant outcome data | Current |
| 18.pdf | GRADE guidelines 18. ROBINS-I and other tools for non-randomized studies | Partially Up-To-Date (Risk of bias) / Current (RCT–NRS integration) |
| 19.pdf | GRADE guidelines 19. Certainty in importance of outcomes / values — risk of bias and indirectness | Partially Up-To-Date |
| 20.pdf | GRADE guidelines 20. Certainty in importance of outcomes / values — inconsistency, imprecision | Partially Up-To-Date |
| 21_1.pdf | GRADE guidelines 21 part 1. Test accuracy — study design, risk of bias, indirectness | Current |
| 21_2.pdf | GRADE guidelines 21 part 2. Test accuracy — inconsistency, imprecision, publication bias | Current |
| 22.pdf | GRADE guidelines 22. Tests and strategies — from test accuracy to patient-important outcomes | Current (DTA) / Partially Up-To-Date (Modelled evidence) |
| 23.pdf | GRADE guidance 23. Cost-effectiveness evidence | Current |
| 24.pdf | GRADE guidance 24. Integrating randomized and non-randomized studies | Current |
| 25.pdf | GRADE approach for network meta-analysis: addressing incoherence | Current |
| 26.pdf | GRADE guidelines 26. Informative statements | Current |
| 27.pdf | GRADE guidelines 27. Absolute effects for time-to-event outcomes | Current |
| 28.pdf | GRADE guidelines 28. Prognostic factors | Current |
| 29.pdf | GRADE guidelines 29. Time-to-event outcomes — censoring | Current |
| 30.pdf | GRADE guidelines 30. Certainty of modelled evidence | Current |
| 31.pdf | GRADE guidance 31. Comparative test accuracy | Current |
| 32.pdf | GRADE guidelines 32. Choosing targets of certainty ratings | Partially Up-To-Date |
| 33.pdf | GRADE guidelines 33. Imprecision in network meta-analysis | Partially Up-To-Date |
| 34.pdf | GRADE guidance 34. Imprecision — minimally contextualized approach | Current |
| 35.pdf | GRADE guidance 35. Imprecision — contextualized approach | Current |
| 36.pdf | GRADE guidance 36. Updates to addressing inconsistency | Current |
| 37.pdf | GRADE guidance 37. Imprecision in test accuracy | Current |
| 38.pdf | GRADE guidance 38. Rating up for dose-response gradient | Current |
| 39.pdf | GRADE guidance 39. GRADE-ADOLOPMENT | Current |
| 40.pdf | GRADE guidance 40. EtD for environmental and occupational health | Current |
| 41.pdf | GRADE guidance 41. Certainty ratings: thresholds rather than categories | Current |
| 44.pdf | GRADE guidance 44. Utilization of randomized and non-randomized studies | Current |

## Scraped GRADE Book pages

| URL (`book.gradepro.org/guideline/...`) | Map status |
|---|---|
| overview-of-the-grade-approach | Current |
| the-development-methods-of-grade | Current |
| requirements-for-claiming-the-use-of-grade | Current |
| questions-about-interventions-diagnostic-test-prognosis-and-exposures | Current |
| risk-of-bias-randomized-trials | Current |
| inconsistency | Current |
| imprecision | Current |
| dissemination-bias | Current |

## Current sources missing from our corpus

Removing outdated PDFs leaves some topics thinner (e.g. indirectness: PDF 8 is outdated and
we do not scrape the GRADE Book *Indirectness* page). Candidates to add:

**GRADE Book pages (scrapable, same pipeline as the existing 8 URLs):**

- `the-history-and-evolution-of-grade`
- `decision-thresholds`
- `indirectness`
- `outcomes`
- `introduction-to-the-evidence-to-decision-frameworks`
- `grade-recommendations`

**Guidance articles (PDFs):**

- GRADE Guidance 42 — thresholds for judgments on benefits and harms
- GRADE Guidance 43 — good practice statements
- GRADE Guidance 45 — EtD framework for pairwise and multiple comparisons
- GRADE Guidance 46 — planetary health in guidelines

## Recommended actions

Status: 1, 3 and the GRADE Book pages from 4 are done (see the project plan, Phase 1).

1. Remove PDFs 1, 3, 4, 5, 6, 7, 8, 14, 15 and their `data/parsed_grade/*.json` caches,
   then rebuild `.faiss_cache/grade`.
2. Decide on 9 and 11 with the instructor (mixed status).
3. Keep Partially Up-To-Date sources for now; store a `status` field in chunk metadata so the
   app can flag answers based on partially outdated sources.
4. Add the missing Current GRADE Book pages (and, if time allows, Guidance 42–46).
5. Re-check evaluation questions that were generated from removed sources.
