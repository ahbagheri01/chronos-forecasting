# Main-story forecasting presentation

The rebuilt presentation is `chronos_vs_classical_main_story.tex`. It contains
the motivation, model descriptions, the 12-dataset benchmark, evaluation
metrics, and statistical comparison methods. It intentionally contains no
experimental results and no appendix.

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
