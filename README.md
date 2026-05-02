# LangOps Sanitizer

Enterprise-style QA and cleanup platform for localization assets.

A professional Streamlit application to sanitize, review, merge, edit, and export multilingual localization files such as TMX, XLSX, CSV, XLIFF, XLF, TXLF, and XLZ.

Built for translators, reviewers, localization teams, vendors, and language operations specialists.

---

## Core Features

### File Support
- TMX
- XLSX
- CSV
- XLIFF / XLF
- TXLF
- XLZ

### QA & Sanitization
- Unicode normalization
- Double spaces
- Double punctuation
- Space before punctuation
- Placeholder mismatch detection
- Number mismatch detection
- Source = Target detection
- Suspicious length ratio checks
- Malformed tag detection
- Typography checks
- German language micro-QA
- Glossary violation checks
- Brand protection rules

### LQA Dashboard
- Quality Score (100-point model)
- Critical / Major / Minor issue counts
- Penalty scoring
- Segment severity overview
- Issue category charts
- AI review metrics

### Review & Edit Workspace
- Review issue queues
- Edit target segments directly
- Apply AI suggestions
- Re-run QA instantly
- Mark segments as reviewed

### AI Review Engine
Bring your own AI provider:

- OpenAI
- Ollama (local LLMs)
- Azure OpenAI

Use AI for:
- grammar checks
- fluency checks
- idiomatic improvements
- rewrite suggestions
- reviewer support

### Merge Center
Merge multiple uploaded files into one export.

Options:
- No deduplication
- Source + Target deduplication
- Source-only deduplication

### Export Options
- Clean TMX
- XLSX QA report
- CSV
- XLIFF
- Native grouped exports

---

## Ideal Use Cases

- Translation QA
- Linguistic review
- Glossary compliance
- Terminology governance
- TM cleanup
- Localization vendor review
- Pre-delivery checks
- AI-assisted language QA
- Batch multilingual asset cleanup

--- use at your own risk, hope you can get a use out of it, it is still in developement so it has still a lot of bugs :-)

## Installation

```bash
pip install -r requirements.txt
streamlit run app.py
