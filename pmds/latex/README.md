# Main-story forecasting presentation

The rebuilt presentation is `chronos_vs_classical_main_story.tex`. It contains
the motivation, model descriptions, the 12-dataset benchmark, evaluation
metrics, statistical comparison methods, and initial experimental results for
MASE, WQL, and CRPS. The results section reports empirical winners alongside
Holm-adjusted SPA and Diebold--Mariano comparison counts and 95% Model
Confidence Set sizes, followed by model-level win counts for each loss. It
contains no appendix.

The result tables are based on the CSV outputs in
`../results/11_12_3_rep/`, with low-power datasets and the exploratory status
of the Diebold--Mariano analysis marked directly on the statistical slides.

Build from this directory with:

```bash
pdflatex chronos_vs_classical_main_story.tex
pdflatex chronos_vs_classical_main_story.tex
```

Or build from the repository root with:

```bash
pdflatex -output-directory=pmds/latex pmds/latex/chronos_vs_classical_main_story.tex
pdflatex -output-directory=pmds/latex pmds/latex/chronos_vs_classical_main_story.tex
```

The three model diagrams in `assets/` are cropped from the original papers.
Presentation-facing source lines were intentionally removed for a cleaner slide
layout; `SOURCES.md` retains the full provenance and figure-selection notes.
