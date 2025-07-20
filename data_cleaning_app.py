import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import io
from enhanced_data_cleaning import (
    IntelligentDataCleaner,
    display_general_info,
    column_info,suggest_group_impute
)
#from streamlit_sortables import sort_items
#from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
from pandas.api.types import is_numeric_dtype, CategoricalDtype
from data_utils import  count_outliers, detect_missing_runs, detect_case_issues, normalize_case, parse_money_series
from viz_utils import plot_missing_bar, plot_histogram, plot_bar_top_values, plot_missing_heatmap, plot_missing_timeline, plot_value_counts,plot_boxplot


st.set_page_config(page_title="Interactive Data Cleaning App", layout="wide")
st.title("🧹 Interactive Data Cleaning App")

# Initialize session state
if 'cleaned_df' not in st.session_state:
    st.session_state.cleaned_df = None
if 'report' not in st.session_state:
    st.session_state.report = None
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
# Remove pipeline state initialization
# if 'pipeline' not in st.session_state:
#     st.session_state.pipeline = [
#         "Detect missing values",
#         "Impute missing values",
#         "Detect outliers",
#         "Drop duplicates"
#     ]
if 'custom_rules' not in st.session_state:
    st.session_state.custom_rules = []

with st.sidebar:
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    # Remove local file selection
    # local_files = [f for f in os.listdir('.') if f.endswith('.csv')]
    # selected_local = st.selectbox("Or select a local CSV", [None] + local_files)
    sample_frac = st.slider("Sample fraction", 0.01, 1.0, 1.0, 0.01, key="sample_frac")
    mode = st.radio("Mode", ["Standard", "Advanced"], key="mode")
    st.markdown("---")
    # Save/Load config buttons
    import json
    if st.button("Save config"):
        st.download_button("Download config", json.dumps(dict(st.session_state)), file_name="config.json")
    config_file = st.file_uploader("Load config", type=["json"], key="load_config")
    if config_file:
        config_data = json.load(config_file)
        for k, v in config_data.items():
            st.session_state[k] = v

# Remove pipeline preview and reordering in sidebar
# st.sidebar.subheader("Pipeline Preview (drag to reorder):")
# st.session_state.pipeline = sort_items(
#     st.session_state.pipeline, direction="vertical"
# )
# for i, step_name in enumerate(st.session_state.pipeline):
#     st.sidebar.checkbox(step_name, value=True, key=f"step_{i}")

# Load DataFrame
import os

df = None
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success(f"Loaded uploaded file: {uploaded_file.name}")
else:
    # Load default dataset
    default_path = "default_data.csv"
    if os.path.exists(default_path):
        df = pd.read_csv(default_path)
        st.info("Loaded default dataset.")
    else:
        st.info("Please upload a CSV file to begin.")

# Tab structure: Profile, Clean, Review, Export, Visualization
tab_profile, tab_clean, tab_review, tab_export, tab_viz = st.tabs(["Profile", "Clean", "Review", "Export", "Visualization"])

with tab_profile:
    st.subheader("General Dataset Info")
    if df is not None:
        st.dataframe(display_general_info(df))
    else:
        st.info("Please upload or select a CSV file to begin.")
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Column-wise Summary")
    if df is not None:
        st.dataframe(column_info(df))
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Data Preview")
    if df is not None:
        st.dataframe(df.head(20))
    st.markdown("<br>", unsafe_allow_html=True)
    missing_pct = df.isnull().mean().mean() * 100
    outliers_count = count_outliers(df)
    dupe_count = df.duplicated().sum()
    m1, m2, m3 = st.columns(3)
    m1.metric("Missing %", f"{missing_pct:.1f}%")
    m2.metric("Outliers", f"{outliers_count}")
    m3.metric("Duplicates", f"{dupe_count}")
    st.text_area("Notes/Comments for this step", key="notes_tab1")

with tab_clean:
    st.header("🧹 Data Cleaning & Anomaly Detection")
    if df is not None:
        # Global duplicate row check
        dupe_row_count = df.duplicated().sum()
        dupe_row_pct = (dupe_row_count / len(df) * 100) if len(df) > 0 else 0
        if dupe_row_count > 0:
            st.warning(f"Duplicate rows found: {dupe_row_count} ({dupe_row_pct:.1f}%)")
            if st.button("Drop duplicate rows", key="drop_dupe_rows"):
                df = df.drop_duplicates()
                st.session_state.cleaned_df = df.copy()
                st.success("Duplicate rows dropped.")
        else:
            st.info("No duplicate rows found.")

        # Global duplicate column check
        dupe_col_mask = df.T.duplicated()
        dupe_col_count = dupe_col_mask.sum()
        if dupe_col_count > 0:
            dupe_cols = df.columns[dupe_col_mask].tolist()
            st.warning(f"Duplicate columns found: {dupe_cols}")
            if st.button("Drop duplicate columns", key="drop_dupe_cols"):
                df = df.loc[:, ~dupe_col_mask]
                st.session_state.cleaned_df = df.copy()
                st.success(f"Duplicate columns dropped: {dupe_cols}")
        else:
            st.info("No duplicate columns found.")

        # Drop rows with high nulls (store choice only)
        thresh = st.slider("Row null % threshold", 0, 100, 50, key="row_null_thresh") / 100
        if st.button("Mark rows above threshold for drop", key="mark_high_nulls_row"):
            st.session_state.rows_to_drop = df.index[df.isnull().mean(axis=1) > thresh].tolist()

        # Drop columns with high nulls (store choice only)
        col_thresh = st.slider("Column null % threshold", 0, 100, 50, key="col_null_thresh") / 100
        if st.button("Mark columns above threshold for drop", key="mark_high_nulls_col"):
            st.session_state.cols_to_drop = [col for col in df.columns if df[col].isnull().mean() > col_thresh]

        # Single per-column expander for all actions
        for col in df.columns:
            with st.expander(f"Column: {col}", expanded=False):
                # Visualizations (Plotly only)
                st.altair_chart(plot_missing_timeline(df[col]), use_container_width=True)
                st.altair_chart(plot_value_counts(df[col]), use_container_width=True)

                # Missing run stats
                runs = detect_missing_runs(df[col])
                st.write(f"Longest contiguous gap: {runs['longest_gap']} rows")
                st.write(f"Isolated missing entries: {runs['isolated_count']}")

                # Imputation controls (store choice only)
                if df[col].isnull().sum() > 0:
                    strat = st.selectbox(
                        f"Imputation strategy for '{col}'",
                        ["mean", "median", "mode", "ffill", "bfill", "custom"],
                        key=f"impute_strategy_{col}"
                    )
                    if 'impute_choices' not in st.session_state:
                        st.session_state.impute_choices = {}
                    st.session_state.impute_choices[col] = strat
                    if strat == "custom":
                        custom_val = st.text_input(f"Custom value for '{col}'", key=f"impute_custom_{col}")
                        if 'impute_custom_values' not in st.session_state:
                            st.session_state.impute_custom_values = {}
                        st.session_state.impute_custom_values[col] = custom_val
                else:
                    st.info("No missing values in this column.")

                # Outlier threshold (store choice only)
                # if mode == "Advanced" and is_numeric_dtype(df[col]):
                #     outlier_thresh = st.slider(f"Outlier threshold (IQR multiplier) for '{col}'", 1.0, 3.0, 1.5, 0.1, key=f"outlier_thresh_{col}")
                #     if 'outlier_thresholds' not in st.session_state:
                #         st.session_state.outlier_thresholds = {}
                #     st.session_state.outlier_thresholds[col] = outlier_thresh

                # Group-aware imputation (store choice only, Advanced mode only)
                if mode == "Advanced" and is_numeric_dtype(df[col]):
                    group_mode = st.radio(
                        f"Apply group impute for '{col}' on:",
                        ["Missing values", "Outliers above threshold"],
                        key=f"groupimpute_mode_{col}"
                    )
                    if group_mode == "Outliers above threshold":
                        outlier_thresh = st.slider(
                            f"Outlier IQR multiplier for '{col}'", 1.0, 3.0, 1.5, 0.1, key=f"groupimpute_outlier_thresh_{col}"
                        )
                    group_cols = st.multiselect("Group by:", df.columns.tolist(), key=f"groupcols_{col}")
                    # Store in session_state
                    if 'group_impute_mode' not in st.session_state:
                        st.session_state.group_impute_mode = {}
                    st.session_state.group_impute_mode[col] = group_mode
                    if group_mode == "Outliers above threshold":
                        if 'group_impute_outlier_thresh' not in st.session_state:
                            st.session_state.group_impute_outlier_thresh = {}
                        st.session_state.group_impute_outlier_thresh[col] = outlier_thresh
                    if 'group_impute' not in st.session_state:
                        st.session_state.group_impute = {}
                    st.session_state.group_impute[col] = group_cols

                # Normalize text case and parse money (for object columns, store choice only)
                if df[col].dtype == 'object':
                    pct = detect_case_issues(df[col])
                    st.write(f"Mixed-case %: {pct*100:.1f}%")
                    if st.checkbox("Normalize to lower", key=f"norm_{col}"):
                        if 'normalize_case_cols' not in st.session_state:
                            st.session_state.normalize_case_cols = set()
                        st.session_state.normalize_case_cols.add(col)
                    if st.checkbox("Parse money strings", key=f"money_{col}"):
                        if 'money_parse_cols' not in st.session_state:
                            st.session_state.money_parse_cols = set()
                        st.session_state.money_parse_cols.add(col)

    st.text_area("Notes/Comments for this step", key="notes_tab_clean")

with tab_review:
    st.header("Review Cleaning Actions")
    if df is not None:
        st.write("### Pending Actions:")
        # Preview duplicate rows/columns if present
        cleaned = st.session_state.cleaned_df if st.session_state.cleaned_df is not None else df
        # Duplicate rows
        dupe_row_mask = cleaned.duplicated()
        dupe_row_count = dupe_row_mask.sum()
        if dupe_row_count > 0:
            st.warning(f"Duplicate rows found: {dupe_row_count}")
            st.write("Preview of duplicate rows:")
            st.dataframe(cleaned[dupe_row_mask].head(5))
        # Duplicate columns
        dupe_col_mask = cleaned.T.duplicated()
        dupe_col_count = dupe_col_mask.sum()
        if dupe_col_count > 0:
            dupe_cols = cleaned.columns[dupe_col_mask].tolist()
            st.warning(f"Duplicate columns found: {dupe_cols}")
        # Track which actions have been applied
        if 'applied_actions' not in st.session_state:
            st.session_state.applied_actions = set()
        # Imputation
        if 'impute_choices' in st.session_state and st.session_state.impute_choices:
            for col, strat in st.session_state.impute_choices.items():
                action_key = f"impute_{col}"
                if action_key not in st.session_state.applied_actions:
                    st.write(f"Impute '{col}' using {strat}")
                    if st.button(f"Apply Imputation: {col}", key=f"apply_impute_{col}"):
                        cleaned = st.session_state.cleaned_df.copy() if st.session_state.cleaned_df is not None else df.copy()
                        s = cleaned[col]
                        if strat == "mean" and is_numeric_dtype(s):
                            cleaned[col] = s.fillna(s.mean())
                        elif strat == "median" and is_numeric_dtype(s):
                            cleaned[col] = s.fillna(s.median())
                        elif strat == "mode":
                            mode_val = s.mode()
                            if not mode_val.empty:
                                cleaned[col] = s.fillna(mode_val[0])
                        elif strat == "ffill":
                            cleaned[col] = s.fillna(method='ffill')
                        elif strat == "bfill":
                            cleaned[col] = s.fillna(method='bfill')
                        elif strat == "custom":
                            val = st.session_state.impute_custom_values.get(col, None) if 'impute_custom_values' in st.session_state else None
                            if val is not None:
                                cleaned[col] = s.fillna(val)
                        st.session_state.cleaned_df = cleaned
                        st.session_state.applied_actions.add(action_key)
                        st.success(f"Imputation applied for {col}")
        # Money parsing
        if 'money_parse_cols' in st.session_state and st.session_state.money_parse_cols:
            for col in st.session_state.money_parse_cols:
                action_key = f"money_{col}"
                if action_key not in st.session_state.applied_actions:
                    st.write(f"Parse money for '{col}'")
                    if st.button(f"Apply Money Parse: {col}", key=f"apply_money_{col}"):
                        cleaned = st.session_state.cleaned_df.copy() if st.session_state.cleaned_df is not None else df.copy()
                        if col in cleaned.columns:
                            cleaned[col] = parse_money_series(cleaned[col])
                            st.session_state.cleaned_df = cleaned
                            st.session_state.applied_actions.add(action_key)
                            st.success(f"Money parsing applied for {col}")
        # Drop rows
        if 'rows_to_drop' in st.session_state and st.session_state.rows_to_drop:
            if 'drop_rows' not in st.session_state.applied_actions:
                st.write(f"Rows to drop: {st.session_state.rows_to_drop}")
                if st.button("Apply Drop Rows", key="apply_drop_rows"):
                    cleaned = st.session_state.cleaned_df.copy() if st.session_state.cleaned_df is not None else df.copy()
                    cleaned = cleaned.drop(index=st.session_state.rows_to_drop)
                    st.session_state.cleaned_df = cleaned
                    st.session_state.applied_actions.add('drop_rows')
                    st.success("Rows dropped.")
        # Drop columns
        if 'cols_to_drop' in st.session_state and st.session_state.cols_to_drop:
            if 'drop_cols' not in st.session_state.applied_actions:
                st.write(f"Columns to drop: {st.session_state.cols_to_drop}")
                if st.button("Apply Drop Columns", key="apply_drop_cols"):
                    cleaned = st.session_state.cleaned_df.copy() if st.session_state.cleaned_df is not None else df.copy()
                    cleaned = cleaned.drop(columns=st.session_state.cols_to_drop)
                    st.session_state.cleaned_df = cleaned
                    st.session_state.applied_actions.add('drop_cols')
                    st.success("Columns dropped.")
        # Group-aware impute
        if 'group_impute' in st.session_state and st.session_state.group_impute:
            for col, group_cols in st.session_state.group_impute.items():
                action_key = f"groupimpute_{col}"
                if action_key not in st.session_state.applied_actions:
                    group_mode = st.session_state.group_impute_mode.get(col, "Missing values")
                    st.write(f"Group impute for '{col}' by {group_cols} ({group_mode})")
                    if st.button(f"Apply Group Impute: {col}", key=f"apply_groupimpute_{col}"):
                        cleaned = st.session_state.cleaned_df.copy() if st.session_state.cleaned_df is not None else df.copy()
                        if col in cleaned.columns and group_cols:
                            if group_mode == "Missing values":
                                cleaned[col] = suggest_group_impute(cleaned, col, group_cols)
                            elif group_mode == "Outliers above threshold":
                                # Detect outliers using IQR
                                thresh = st.session_state.group_impute_outlier_thresh.get(col, 1.5)
                                s = cleaned[col]
                                q1 = s.quantile(0.25)
                                q3 = s.quantile(0.75)
                                iqr = q3 - q1
                                lower = q1 - thresh * iqr
                                upper = q3 + thresh * iqr
                                outlier_mask = (s < lower) | (s > upper)
                                # Replace outliers with group mean
                                group_means = cleaned.groupby(group_cols)[col].transform('mean')
                                s_update = s.copy()
                                s_update[outlier_mask] = group_means[outlier_mask]
                                cleaned[col] = s_update
                            st.session_state.cleaned_df = cleaned
                            st.session_state.applied_actions.add(action_key)
                            st.success(f"Group imputation applied for {col} ({group_mode})")
        # Normalize case
        if 'normalize_case_cols' in st.session_state and st.session_state.normalize_case_cols:
            for col in st.session_state.normalize_case_cols:
                action_key = f"norm_{col}"
                if action_key not in st.session_state.applied_actions:
                    st.write(f"Normalize case for '{col}'")
                    if st.button(f"Apply Normalize Case: {col}", key=f"apply_norm_{col}"):
                        cleaned = st.session_state.cleaned_df.copy() if st.session_state.cleaned_df is not None else df.copy()
                        if col in cleaned.columns:
                            cleaned[col] = normalize_case(cleaned[col], 'lower')
                            st.session_state.cleaned_df = cleaned
                            st.session_state.applied_actions.add(action_key)
                            st.success(f"Case normalization applied for {col}")
        # Optionally, keep the ability to apply all actions at once
        if st.button("Apply All Cleaning Actions", key="apply_cleaning_all"):
            cleaned = df.copy()
            # Drop columns
            if 'cols_to_drop' in st.session_state:
                cleaned = cleaned.drop(columns=st.session_state.cols_to_drop)
            # Drop rows
            if 'rows_to_drop' in st.session_state:
                cleaned = cleaned.drop(index=st.session_state.rows_to_drop)
            # Impute
            if 'impute_choices' in st.session_state:
                for col, strat in st.session_state.impute_choices.items():
                    if col not in cleaned.columns:
                        continue
                    if strat == "mean" and is_numeric_dtype(cleaned[col]):
                        cleaned[col] = cleaned[col].fillna(cleaned[col].mean())
                    elif strat == "median" and is_numeric_dtype(cleaned[col]):
                        cleaned[col] = cleaned[col].fillna(cleaned[col].median())
                    elif strat == "mode":
                        mode_val = cleaned[col].mode()
                        if not mode_val.empty:
                            cleaned[col] = cleaned[col].fillna(mode_val[0])
                    elif strat == "ffill":
                        cleaned[col] = cleaned[col].fillna(method='ffill')
                    elif strat == "bfill":
                        cleaned[col] = cleaned[col].fillna(method='bfill')
                    elif strat == "custom":
                        val = st.session_state.impute_custom_values.get(col, None) if 'impute_custom_values' in st.session_state else None
                        if val is not None:
                            cleaned[col] = cleaned[col].fillna(val)
            # Outlier handling (optional: just show threshold, not apply)
            # Group-aware impute
            if 'group_impute' in st.session_state:
                for col, group_cols in st.session_state.group_impute.items():
                    if col in cleaned.columns and group_cols:
                        cleaned[col] = suggest_group_impute(cleaned, col, group_cols)
            # Normalize case
            if 'normalize_case_cols' in st.session_state:
                for col in st.session_state.normalize_case_cols:
                    if col in cleaned.columns:
                        cleaned[col] = normalize_case(cleaned[col], 'lower')
            # Money parsing
            if 'money_parse_cols' in st.session_state:
                for col in st.session_state.money_parse_cols:
                    if col in cleaned.columns:
                        cleaned[col] = parse_money_series(cleaned[col])
            st.session_state.cleaned_df = cleaned
            st.session_state.applied_actions = set([
                f"impute_{col}" for col in st.session_state.impute_choices.keys()
            ])
            if 'money_parse_cols' in st.session_state:
                st.session_state.applied_actions.update([f"money_{col}" for col in st.session_state.money_parse_cols])
            if 'rows_to_drop' in st.session_state:
                st.session_state.applied_actions.add('drop_rows')
            if 'cols_to_drop' in st.session_state:
                st.session_state.applied_actions.add('drop_cols')
            if 'group_impute' in st.session_state:
                st.session_state.applied_actions.update([f"groupimpute_{col}" for col in st.session_state.group_impute.keys()])
            if 'normalize_case_cols' in st.session_state:
                st.session_state.applied_actions.update([f"norm_{col}" for col in st.session_state.normalize_case_cols])
            st.success("All cleaning actions applied!")
        # Show before/after preview
        if st.session_state.get('cleaned_df') is not None:
            # Find changed rows
            before = df.copy()
            after = st.session_state.cleaned_df.copy()
            # Align indices and columns
            before, after = before.align(after, join='inner', axis=1)
            before, after = before.align(after, join='inner', axis=0)
            changed_mask = (before != after) & ~(before.isnull() & after.isnull())
            changed_rows = changed_mask.any(axis=1)
            if changed_rows.any():
                st.write("Rows with changes (before):")
                st.dataframe(before[changed_rows])
                st.write("Rows with changes (after):")
                st.dataframe(after[changed_rows])
            else:
                st.info("No rows have changed after cleaning.")
    else:
        st.info("No cleaned data available. Please apply corrections first.")
    st.text_area("Notes/Comments for this step", key="notes_tab4")

with tab_export:
    st.header("📥 Export Results")
    if st.session_state.report is not None:
        # Export cleaning report
        if st.button("Export Cleaning Report (JSON)"):
            cleaner = IntelligentDataCleaner()
            filename = cleaner.export_report(st.session_state.report)
            with open(filename, "rb") as f:
                st.download_button(
                    "Download Report", 
                    f, 
                    file_name=filename, 
                    mime="application/json"
                )
            st.success(f"Report saved as: {filename}")
    if st.session_state.cleaned_df is not None and isinstance(st.session_state.cleaned_df, pd.DataFrame):
        # Export cleaned data
        st.write("**Export Cleaned Data:**")
        csv_str = st.session_state.cleaned_df.to_csv(index=False)
        if csv_str is not None:
            csv = csv_str.encode('utf-8')
            st.download_button(
                "Download Cleaned CSV", 
                csv, 
                file_name="cleaned_data.csv", 
                mime="text/csv"
            )
    else:
        st.info("No cleaned data available. Please apply corrections first.")
    st.text_area("Notes/Comments for this step", key="notes_tab5")

with tab_viz:
    st.header("📈 Data Visualization")
    st.subheader("Missing Value Percentage by Column")
    fig = plot_missing_bar(df)
    st.altair_chart(fig, use_container_width=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Column-wise Distributions and Outliers")
    col_to_plot = st.selectbox("Select a column to visualize", df.columns, key="viz_col_tab_viz")
    s = df[col_to_plot]
    if is_numeric_dtype(s):
        st.write("**Histogram (Altair):**")
        hist = plot_histogram(df, col_to_plot)
        st.altair_chart(hist, use_container_width=True)
        st.write("**Boxplot (Altair):**")
        box = plot_boxplot(df, col_to_plot)
        st.altair_chart(box, use_container_width=True)
    elif isinstance(s.dtype, CategoricalDtype) or s.dtype == 'object':
        st.write("**Bar Chart of Top 10 Values (Altair):**")
        vc = s.value_counts(dropna=False)
        if vc.empty:
            st.warning("No values to display for this column.")
        else:
            st.altair_chart(plot_bar_top_values(s, col_to_plot), use_container_width=True)
    else:
        st.info("Selected column is not suitable for visualization.")
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Missingness Heatmap (Altair)")
    fig_heatmap = plot_missing_heatmap(df)
    st.altair_chart(fig_heatmap, use_container_width=True)
    st.text_area("Notes/Comments for this step", key="notes_tab6") 