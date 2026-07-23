# ESG Claim Verification via Temporal NLP

Detecting corporate greenwashing by checking whether sustainability commitments made in one year's report actually show up in the next year's regulatory filings, rather than scoring a single report's language in isolation.

## Problem

Corporate ESG disclosures are self-reported and largely unverified. Existing NLP approaches to greenwashing detection typically analyze a single sustainability report for vague or hedging language. This project treats greenwashing as a temporal verification problem instead: does a company's 2022 forward-looking emissions commitment show up, get addressed, or go silent in its 2023 SEC 10-K filing?

*Originally developed as a team project for a graduate NLP course at Drexel University.*

## Approach

**Data:** 2022 sustainability reports and 2023 SEC 10-K filings for 8 S&P 500 companies (MSFT, AAPL, JPM, WMT, CVX, XOM, NEE, PG), spanning tech, finance, energy, retail, and consumer goods.

**Claim extraction (3 stages):**
1. Sentence splitting with spaCy, filtered to 30-500 characters to drop headers and fragments
2. Regex pre-filter requiring all three of: commitment language ("will," "plan," "by 2030"), a quantifiable target (percentage, year, "net-zero"), and an emissions/climate scope term
3. Zero-shot classification with `facebook/bart-large-mnli` (4 labels), keeping only claims classified as forward-looking commitments with confidence above 0.5

**Cross-referencing:** Chunked each 10-K into paragraphs, encoded claims and paragraphs with `all-MiniLM-L6-v2` sentence embeddings, and retrieved the best-matching passage per claim by cosine similarity. A similarity below 0.55 was labeled "silent," meaning the company stopped mentioning the commitment.

**Validation:** Two independent annotators labeled 30 claim-disclosure pairs by hand, and we swept the similarity threshold from 0.35 to 0.65 to find the best empirical cutoff.

## Results

- 39 high-quality forward-commitment claims extracted from Microsoft's report alone, including specific commitments like "Scope 3 cut in half by 2030" and "carbon negative by 2030"
- Across 153 claims total, 89 were classified silent and 64 addressed in the following year's 10-K
- Raw inter-annotator agreement of 87%, Cohen's kappa of 0.77 (substantial agreement)
- Best similarity threshold of 0.55 (kappa 0.41, 70% accuracy against human labels)
- Spearman correlation of 0.50 (p = 0.005) between the divergence score and human judgment, independent of the specific threshold chosen

**Notable finding:** oil and gas companies showed the lowest divergence scores, not because they deliver on commitments better, but because their 10-Ks discuss emissions and climate extensively as regulatory risk language. The metric captures topical persistence, not actual fulfillment, and this distinction matters for interpreting the results correctly.

## Limitations

- The quantitative divergence component (comparing specific promised percentages against reported percentages) was designed but found zero valid comparable pairs, since 2023 10-Ks predated mandatory climate disclosure rules
- Some companies had too few claims for stable rates (XOM n=2, WMT n=5, PG n=6)
- The 30-pair benchmark was used for both calibrating and evaluating the similarity threshold, which is too small for a proper held-out validation split
- Validated only on large-cap US companies reporting in English

## Tech Stack

Python, spaCy, Transformers (BART), Sentence-Transformers, pandas, scikit-learn

## Repository Structure

```
esg-claim-verification/
├── README.md
├── requirements.txt
├── notebooks/
│   └── esg_finalcode.py
├── data/
│   └── (SEC 10-K filings and sustainability reports not included; see Data Sources below)
└── outputs/
    └── (extracted claims, cross-reference results, divergence summaries)
```

## Data Sources

- SEC 10-K filings via [sec-edgar-downloader](https://pypi.org/project/sec-edgar-downloader/)
- 2022 sustainability reports collected as PDFs from each company's investor relations site

## How to Run

```bash
pip install -r requirements.txt
python notebooks/esg_finalcode.py
```

Note: the notebook was originally developed in Google Colab and references Google Drive paths that will need to be updated for a local or alternate cloud environment.
