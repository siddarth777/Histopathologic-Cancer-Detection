# EDA

This folder contains the exploratory data analysis pipeline and historical outputs.

## Folder Layout

```text
EDA/
├── src/                 # EDA implementation modules
├── outputs/             # Historical plots/reports
└── .gitignore
```

## Primary Entry Point

The top-level dispatcher uses `EDA/src/eda.py`.

From repository root:

```bash
python main.py --eda
```

Direct run:

```bash
python -m EDA.src.main
```

## Key Pipeline Stages

- class-distribution visualization
- pixel statistics
- color analysis
- texture analysis
- morphology analysis
- KL-divergence reports
- edge-density reports
- PCA and LDA analysis

## Related Docs

- `src/README.md`
- `../README.md`