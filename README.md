# Interactive Data Cleaning App

A modular, interactive Streamlit app for step-by-step data cleaning, profiling, and visualization. Designed for usability, real-time feedback, and extensibility.

## Features
- Upload or use a default CSV dataset
- Data profiling: summary stats, missing values, outlier detection
- Step-by-step cleaning wizard (imputation, outlier handling, duplicate removal, etc.)
- Advanced and standard cleaning modes
- Interactive visualizations (Altair only)
- Per-action review and apply workflow
- Export cleaned data and cleaning reports
- Modular codebase for easy extension

## File Structure
- `data_cleaning_app.py` — Main Streamlit app UI and workflow
- `enhanced_data_cleaning.py` — Core cleaning logic and anomaly detection
- `data_utils.py` — Helper functions for cleaning and profiling
- `viz_utils.py` — Visualization helpers (Altair)
- `requirements.txt` — Python dependencies
- `default_data.csv` — (Optional) Default dataset loaded if no file is uploaded


## Usage
Run the Streamlit app:
```bash
streamlit run data_cleaning_app.py
```
- Use the sidebar to upload a CSV or work with the default dataset.
- Explore tabs for profiling, cleaning, review, export, and visualization.
- Apply cleaning actions step-by-step and export results as needed.

## Contributing
- Fork the repo and create a feature branch.
- Keep logic modular (UI in `data_cleaning_app.py`, logic in helpers).
- Submit a pull request with a clear description.

## License
MIT License 