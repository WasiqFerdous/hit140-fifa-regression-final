# HIT140 Assessment 3 FIFA World Cup 2026 Regression

Python code and datasets for two regression tasks:
- Task 2.1: predict match goal difference using 104 match records.
- Task 2.2: predict team goals using 208 team records.

The analysis compares ordinary least squares, a training-mean baseline and Ridge regression.

## Run the notebook

Keep the notebook and CSV files in the same folder. Open
`FIFA_World_Cup_2026_Regression_Final.ipynb` and run the cells from top to bottom.

## Run the Python script

Install the packages and run the script:

```bash
python -m pip install -r requirements.txt
python run_analysis.py
```

Figures, model datasets and result tables are saved in `analysis_outputs/`.

## Main files

- `FIFA_World_Cup_2026_Regression_Final.ipynb`: executed notebook.
- `run_analysis.py`: Python analysis script.
- `requirements.txt`: package versions.
- The five CSV files: supplied datasets.
- `analysis_outputs/`: figures and numerical results.
- `FIFA_Data_Source_Checks.md`: dataset sources and limitations.
