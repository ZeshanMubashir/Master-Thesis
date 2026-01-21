Master's Thesis – Autonomous Monitoring of Floating Solar PVs

## Overview
This repository contains the LaTeX source for the master's thesis titled **“Autonomous Monitoring of Floating Solar PVs Anomalies using Deep Learning Methods.”** It includes the main manuscript (`main.tex`), bibliography (`bibliography.bib`), figures, and generated auxiliary files.

## Structure
- `main.tex` – entry point for the thesis document
- `Chapters/` – chapter content split into separate files
- `Figures/` – images and diagrams referenced in the thesis
- `bibliography.bib` – BibLaTeX references in IEEE style
- `main.pdf` / `main_updated.pdf` – compiled outputs

## Building the PDF
This project uses `latexmk` with `biber` for bibliography handling.

```bash
latexmk -pdf main.tex   # builds PDF and runs biber automatically
latexmk -c              # cleans auxiliary files (optional)
```

If `latexmk` is not available, compile manually:
```bash
pdflatex main.tex
biber main
pdflatex main.tex
pdflatex main.tex
```

## Notes
- The document uses IEEE-style citations and NTNU-themed colors defined in `main.tex`.
- Ensure all images referenced in `Figures/` are present before compiling.
