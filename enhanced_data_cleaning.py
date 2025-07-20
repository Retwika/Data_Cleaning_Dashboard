import pandas as pd
import numpy as np
import json
import re
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import logging
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class DataAnomaly:
    """Represents a data anomaly with context and suggested corrections."""
    row_index: int
    column: str
    original_value: Any
    anomaly_type: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    confidence: float
    suggested_correction: Any
    reasoning: str
    context: Dict[str, Any]

@dataclass
class CleaningReport:
    """Comprehensive report of the data cleaning process."""
    dataset_info: Dict[str, Any]
    anomalies_found: List[DataAnomaly]
    cleaning_suggestions: Dict[str, List[str]]
    data_quality_score: float
    timestamp: str

class LLMDataAnalyzer:
    """Simulates LLM-powered data analysis for anomaly detection."""
    
    def __init__(self):
        self.column_patterns = {}
        self.data_types = {}
        
    def analyze_column_context(self, df: pd.DataFrame, column: str) -> Dict[str, Any]:
        """Analyze column context to understand data patterns."""
        col_data = df[column].dropna()
        
        context = {
            'data_type': str(df[column].dtype),
            'null_count': df[column].isnull().sum(),
            'unique_count': df[column].nunique(),
            'sample_values': col_data.head(10).tolist() if len(col_data) > 0 else [],
            'statistical_summary': {}
        }
        
        # Add statistical summary for numeric columns
        if pd.api.types.is_numeric_dtype(df[column]):
            context['statistical_summary'] = {
                'mean': df[column].mean(),
                'median': df[column].median(),
                'std': df[column].std(),
                'min': df[column].min(),
                'max': df[column].max(),
                'q1': df[column].quantile(0.25),
                'q3': df[column].quantile(0.75)
            }
        
        # Detect patterns in string columns
        if df[column].dtype == 'object':
            context['string_patterns'] = self._detect_string_patterns(col_data)
            context['common_values'] = col_data.value_counts().head(10).to_dict()
        
        return context
    
    def _detect_string_patterns(self, series: pd.Series) -> Dict[str, Any]:
        """Detect common patterns in string data."""
        patterns = {
            'email': 0.0,
            'phone': 0.0,
            'date': 0.0,
            'url': 0.0,
            'numeric_string': 0.0,
            'contains_digits': 0.0,
            'all_caps': 0.0,
            'mixed_case': 0.0,
            'average_length': 0.0
        }
        
        if len(series) == 0:
            return patterns
        
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        phone_pattern = r'(\+?1?[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})'
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        
        for value in series.astype(str):
            if pd.isna(value) or value == 'nan':
                continue
                
            patterns['email'] += bool(re.search(email_pattern, value))
            patterns['phone'] += bool(re.search(phone_pattern, value))
            patterns['url'] += bool(re.search(url_pattern, value))
            patterns['numeric_string'] += value.replace('.', '').replace('-', '').isdigit()
            patterns['contains_digits'] += bool(re.search(r'\d', value))
            patterns['all_caps'] += value.isupper() and value.isalpha()
            patterns['mixed_case'] += bool(re.search(r'[a-z]', value)) and bool(re.search(r'[A-Z]', value))
        
        patterns['average_length'] = series.str.len().mean()
        
        # Convert counts to percentages
        total = len(series)
        for key in patterns:
            if key != 'average_length':
                patterns[key] = (patterns[key] / total) * 100
        
        return patterns

class AnomalyDetector(ABC):
    """Abstract base class for anomaly detectors."""
    
    @abstractmethod
    def detect(self, df: pd.DataFrame, column: str, context: Dict[str, Any]) -> List[DataAnomaly]:
        pass

class OutlierDetector(AnomalyDetector):
    """Detects statistical outliers in numeric data."""
    
    def detect(self, df: pd.DataFrame, column: str, context: Dict[str, Any]) -> List[DataAnomaly]:
        anomalies = []
        
        if not pd.api.types.is_numeric_dtype(df[column]):
            return anomalies
        
        stats = context.get('statistical_summary', {})
        if not stats:
            return anomalies
        
        q1, q3 = stats.get('q1', 0), stats.get('q3', 0)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outlier_mask = (df[column] < lower_bound) | (df[column] > upper_bound)
        outlier_indices = df[outlier_mask].index
        
        for idx in outlier_indices:
            value = df.loc[idx, column]
            if pd.isna(value):
                continue
                
            # Determine severity based on how extreme the outlier is
            z_score = abs((value - stats['mean']) / stats['std']) if stats['std'] > 0 else 0
            severity = 'low' if z_score < 3 else 'medium' if z_score < 5 else 'high'
            
            # Suggest correction (median or mean)
            suggested_correction = stats['median']
            
            anomalies.append(DataAnomaly(
                row_index=int(idx),
                column=column,
                original_value=value,
                anomaly_type='statistical_outlier',
                severity=severity,
                confidence=min(z_score / 5, 1.0),
                suggested_correction=suggested_correction,
                reasoning=f"Value {value} is {z_score:.2f} standard deviations from mean. Outside IQR bounds [{lower_bound:.2f}, {upper_bound:.2f}]",
                context={'z_score': z_score, 'iqr_bounds': [lower_bound, upper_bound]}
            ))
        
        return anomalies

class PatternAnomalyDetector(AnomalyDetector):
    """Detects pattern inconsistencies in string data."""
    
    def detect(self, df: pd.DataFrame, column: str, context: Dict[str, Any]) -> List[DataAnomaly]:
        anomalies = []
        
        if df[column].dtype != 'object':
            return anomalies
        
        patterns = context.get('string_patterns', {})
        common_values = context.get('common_values', {})
        
        # Detect email format inconsistencies
        if patterns.get('email', 0) > 50:  # If >50% look like emails
            email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            for idx, value in df[column].items():
                if pd.isna(value) or str(value) == '':
                    continue
                if not re.match(email_pattern, str(value)):
                    anomalies.append(DataAnomaly(
                        row_index=int(idx),
                        column=column,
                        original_value=value,
                        anomaly_type='pattern_mismatch',
                        severity='medium',
                        confidence=0.8,
                        suggested_correction='REVIEW_NEEDED',
                        reasoning=f"Value doesn't match expected email pattern in column where {patterns['email']:.1f}% are emails",
                        context={'expected_pattern': 'email'}
                    ))
        
        # Detect case inconsistencies
        if len(common_values) > 0:
            most_common_case = self._determine_dominant_case(list(common_values.keys()))
            for idx, value in df[column].items():
                if pd.isna(value) or str(value) == '':
                    continue
                if not self._matches_case_pattern(str(value), most_common_case):
                    suggested_correction = self._apply_case_correction(str(value), most_common_case)
                    anomalies.append(DataAnomaly(
                        row_index=int(idx),
                        column=column,
                        original_value=value,
                        anomaly_type='case_inconsistency',
                        severity='low',
                        confidence=0.6,
                        suggested_correction=suggested_correction,
                        reasoning=f"Case doesn't match dominant pattern ({most_common_case})",
                        context={'dominant_case': most_common_case}
                    ))
        
        return anomalies
    
    def _determine_dominant_case(self, values: List[str]) -> str:
        """Determine the dominant case pattern in a list of strings."""
        case_counts = {'upper': 0, 'lower': 0, 'title': 0, 'mixed': 0}
        
        for value in values:
            if value.isupper():
                case_counts['upper'] += 1
            elif value.islower():
                case_counts['lower'] += 1
            elif value.istitle():
                case_counts['title'] += 1
            else:
                case_counts['mixed'] += 1
        
        return max(case_counts.items(), key=lambda x: x[1])[0]
    
    def _matches_case_pattern(self, value: str, pattern: str) -> bool:
        """Check if a value matches the expected case pattern."""
        if pattern == 'upper':
            return value.isupper()
        elif pattern == 'lower':
            return value.islower()
        elif pattern == 'title':
            return value.istitle()
        return True  # Mixed case always matches
    
    def _apply_case_correction(self, value: str, pattern: str) -> str:
        """Apply case correction based on pattern."""
        if pattern == 'upper':
            return value.upper()
        elif pattern == 'lower':
            return value.lower()
        elif pattern == 'title':
            return value.title()
        return value

class MissingValueDetector(AnomalyDetector):
    """Detects and suggests corrections for missing values."""
    
    def detect(self, df: pd.DataFrame, column: str, context: Dict[str, Any]) -> List[DataAnomaly]:
        anomalies = []
        null_indices = df[df[column].isnull()].index
        
        for idx in null_indices:
            suggested_correction = self._suggest_missing_value_correction(df, column, idx, context)
            severity = self._assess_missing_value_severity(context)
            
            anomalies.append(DataAnomaly(
                row_index=int(idx),
                column=column,
                original_value=None,
                anomaly_type='missing_value',
                severity=severity,
                confidence=0.7,
                suggested_correction=suggested_correction,
                reasoning=f"Missing value in column with {context['null_count']} total nulls ({(context['null_count']/len(df)*100):.1f}%)",
                context={'null_percentage': context['null_count']/len(df)*100}
            ))
        
        return anomalies
    
    def _suggest_missing_value_correction(self, df: pd.DataFrame, column: str, idx: int, context: Dict[str, Any]) -> Any:
        """Suggest appropriate correction for missing value."""
        if pd.api.types.is_numeric_dtype(df[column]):
            stats = context.get('statistical_summary', {})
            return stats.get('median', stats.get('mean', 0))
        else:
            common_values = context.get('common_values', {})
            if common_values:
                return list(common_values.keys())[0]  # Most common value
            return 'UNKNOWN'
    
    def _assess_missing_value_severity(self, context: Dict[str, Any]) -> str:
        """Assess severity of missing values based on percentage."""
        null_percentage = context['null_count'] / context.get('total_rows', 1) * 100
        if null_percentage < 5:
            return 'low'
        elif null_percentage < 20:
            return 'medium'
        else:
            return 'high'

class DuplicateDetector(AnomalyDetector):
    """Detects duplicate values and rows."""
    
    def detect(self, df: pd.DataFrame, column: str, context: Dict[str, Any]) -> List[DataAnomaly]:
        anomalies = []
        
        # Find duplicate values in the column
        duplicates = df[df.duplicated(subset=[column], keep=False)]
        
        for idx in duplicates.index:
            value = df.loc[idx, column]
            duplicate_count = df[df[column] == value].shape[0]
            
            anomalies.append(DataAnomaly(
                row_index=int(idx),
                column=column,
                original_value=value,
                anomaly_type='duplicate_value',
                severity='medium' if duplicate_count > 5 else 'low',
                confidence=0.9,
                suggested_correction='REVIEW_FOR_REMOVAL',
                reasoning=f"Value appears {duplicate_count} times in column",
                context={'duplicate_count': duplicate_count}
            ))
        
        return anomalies

class IntelligentDataCleaner:
    """Main class orchestrating the intelligent data cleaning process."""
    
    def __init__(self):
        self.llm_analyzer = LLMDataAnalyzer()
        self.detectors = [
            OutlierDetector(),
            PatternAnomalyDetector(),
            MissingValueDetector(),
            DuplicateDetector()
        ]
        self.cleaning_history = []

    def _get_relevant_detectors(self, col_dtype):
        """Return only detectors relevant for the column dtype."""
        relevant = []
        for detector in self.detectors:
            if isinstance(detector, OutlierDetector) and not pd.api.types.is_numeric_dtype(col_dtype):
                continue
            if isinstance(detector, PatternAnomalyDetector) and col_dtype != 'object':
                continue
            if isinstance(detector, MissingValueDetector):
                relevant.append(detector)  # Always check for missing values
                continue
            if isinstance(detector, DuplicateDetector):
                relevant.append(detector)  # Always check for duplicates
                continue
            relevant.append(detector)
        return relevant

    def _append_anomalies(self, all_anomalies, column_anomalies):
        if column_anomalies:
            all_anomalies.extend(column_anomalies)

    def analyze_dataset(self, df: pd.DataFrame) -> CleaningReport:
        """Perform comprehensive analysis of the dataset (efficient version)."""
        logger.info(f"Starting analysis of dataset with shape {df.shape}")
        all_anomalies = []
        dataset_info = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum(),
            'total_nulls': df.isnull().sum().sum()
        }
        for column in df.columns:
            logger.info(f"Analyzing column: {column}")
            col_dtype = df[column].dtype
            context = self.llm_analyzer.analyze_column_context(df, column)
            context['total_rows'] = len(df)
            # Only run relevant detectors
            for detector in self._get_relevant_detectors(col_dtype):
                try:
                    column_anomalies = detector.detect(df, column, context)
                    self._append_anomalies(all_anomalies, column_anomalies)
                except Exception as e:
                    logger.error(f"Error in {detector.__class__.__name__} for column {column}: {e}")
        cleaning_suggestions = self._generate_cleaning_suggestions(all_anomalies)
        quality_score = self._calculate_data_quality_score(df, all_anomalies)
        report = CleaningReport(
            dataset_info=dataset_info,
            anomalies_found=all_anomalies,
            cleaning_suggestions=cleaning_suggestions,
            data_quality_score=quality_score,
            timestamp=datetime.now().isoformat()
        )
        logger.info(f"Analysis complete. Found {len(all_anomalies)} anomalies. Quality score: {quality_score:.2f}")
        return report
    
    def _generate_cleaning_suggestions(self, anomalies: List[DataAnomaly]) -> Dict[str, List[str]]:
        """Generate high-level cleaning suggestions based on anomalies."""
        suggestions = {
            'immediate_actions': [],
            'review_required': [],
            'preventive_measures': []
        }
        
        # Group anomalies by type
        anomaly_types = {}
        for anomaly in anomalies:
            if anomaly.anomaly_type not in anomaly_types:
                anomaly_types[anomaly.anomaly_type] = 0
            anomaly_types[anomaly.anomaly_type] += 1
        
        # Generate suggestions based on anomaly patterns
        if 'missing_value' in anomaly_types:
            count = anomaly_types['missing_value']
            suggestions['immediate_actions'].append(f"Address {count} missing values using appropriate imputation strategies")
        
        if 'statistical_outlier' in anomaly_types:
            count = anomaly_types['statistical_outlier']
            suggestions['review_required'].append(f"Review {count} statistical outliers to determine if they're errors or valid extreme values")
        
        if 'pattern_mismatch' in anomaly_types:
            count = anomaly_types['pattern_mismatch']
            suggestions['immediate_actions'].append(f"Standardize {count} values that don't match expected patterns")
        
        if 'duplicate_value' in anomaly_types:
            count = anomaly_types['duplicate_value']
            suggestions['review_required'].append(f"Review {count} duplicate values for potential removal")
        
        # Add preventive measures
        suggestions['preventive_measures'].extend([
            "Implement data validation rules at data entry points",
            "Set up regular data quality monitoring",
            "Establish data governance standards",
            "Create automated data quality dashboards"
        ])
        
        return suggestions
    
    def _calculate_data_quality_score(self, df: pd.DataFrame, anomalies: List[DataAnomaly]) -> float:
        """Calculate overall data quality score (0-100)."""
        total_cells = df.shape[0] * df.shape[1]
        
        # Weight anomalies by severity
        severity_weights = {'low': 0.1, 'medium': 0.3, 'high': 0.6, 'critical': 1.0}
        weighted_anomalies = sum(severity_weights.get(anomaly.severity, 0.5) for anomaly in anomalies)
        
        # Calculate score (higher is better)
        if total_cells == 0:
            return 0.0
        
        anomaly_ratio = weighted_anomalies / total_cells
        quality_score = max(0, 100 * (1 - anomaly_ratio))
        
        return min(100, quality_score)
    
    def apply_corrections(self, df: pd.DataFrame, anomalies: List[DataAnomaly], 
                         auto_apply: bool = False, severity_threshold: str = 'medium') -> pd.DataFrame:
        """Apply corrections to the dataset with human oversight."""
        df_cleaned = df.copy()
        applied_corrections = []
        
        severity_order = {'low': 1, 'medium': 2, 'high': 3, 'critical': 4}
        threshold_level = severity_order.get(severity_threshold, 2)
        
        for anomaly in anomalies:
            # Skip if below severity threshold
            if severity_order.get(anomaly.severity, 0) < threshold_level:
                continue
            
            # Skip if suggested correction is a review marker
            if anomaly.suggested_correction in ['REVIEW_NEEDED', 'REVIEW_FOR_REMOVAL']:
                continue
            
            if auto_apply or self._get_user_approval(anomaly):
                try:
                    df_cleaned.loc[anomaly.row_index, anomaly.column] = anomaly.suggested_correction
                    applied_corrections.append(anomaly)
                    logger.info(f"Applied correction: {anomaly.original_value} -> {anomaly.suggested_correction}")
                except Exception as e:
                    logger.error(f"Failed to apply correction: {e}")
        
        logger.info(f"Applied {len(applied_corrections)} corrections out of {len(anomalies)} anomalies")
        return df_cleaned
    
    def _get_user_approval(self, anomaly: DataAnomaly) -> bool:
        """Get user approval for applying a correction (simplified for demo)."""
        # In a real implementation, this would present the anomaly to the user
        # and get their approval through a UI or command line interface
        print(f"\nAnomaly found:")
        print(f"  Row: {anomaly.row_index}, Column: {anomaly.column}")
        print(f"  Original: {anomaly.original_value}")
        print(f"  Suggested: {anomaly.suggested_correction}")
        print(f"  Reasoning: {anomaly.reasoning}")
        print(f"  Severity: {anomaly.severity}, Confidence: {anomaly.confidence:.2f}")
        
        response = input("Apply this correction? (y/n): ").lower().strip()
        return response == 'y'
    
    def export_report(self, report: CleaningReport, filename: Optional[str] = None) -> str:
        """Export cleaning report to JSON file."""
        if filename is None:
            filename = f"data_cleaning_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Convert report to serializable format
        report_dict = {
            'dataset_info': report.dataset_info,
            'anomalies_found': [
                {
                    'row_index': a.row_index,
                    'column': a.column,
                    'original_value': str(a.original_value),
                    'anomaly_type': a.anomaly_type,
                    'severity': a.severity,
                    'confidence': a.confidence,
                    'suggested_correction': str(a.suggested_correction),
                    'reasoning': a.reasoning,
                    'context': a.context
                }
                for a in report.anomalies_found
            ],
            'cleaning_suggestions': report.cleaning_suggestions,
            'data_quality_score': report.data_quality_score,
            'timestamp': report.timestamp
        }
        
        with open(filename, 'w') as f:
            json.dump(report_dict, f, indent=2, default=str)
        
        logger.info(f"Report exported to {filename}")
        return filename

# Enhanced utility functions
def display_general_info(df):
    """Enhanced general info display with more metrics."""
    info = {
        "Number of rows": len(df),
        "Number of columns": len(df.columns),
        "Number of duplicated rows": df.duplicated().sum(),
        "Number of duplicated columns": df.T.duplicated().sum(),
        "Total missing values": df.isnull().sum().sum(),
        "Memory usage (MB)": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
        "Data types": df.dtypes.value_counts().to_dict()
    }
    gen = pd.DataFrame(list(info.items()), columns=["Metric", "Value"])
    return gen

def column_info(df):
    if df is None:
        return None
    # Outlier detection (IQR method for numeric columns)
    def count_outliers(col):
        if pd.api.types.is_numeric_dtype(col):
            q1 = col.quantile(0.25)
            q3 = col.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            return int(((col < lower) | (col > upper)).sum())
        return ''
    outliers = [count_outliers(df[c]) for c in df.columns]
    # Helper to convert numpy types to native Python types
    def to_native(val):
        if hasattr(val, 'item'):
            return val.item()
        return val
    # Build summary DataFrame
    summary = pd.DataFrame({
        'Column': df.columns,
        'Data Type': df.dtypes.astype(str),
        'Missing Values': df.isnull().sum(),
        'Missing %': (df.isnull().mean() * 100).round(2),
        'Unique Values': [str(df[c].nunique(dropna=True)) for c in df.columns],
        'Sample Values': [str([to_native(v) for v in df[c].dropna().unique()[:5]]) for c in df.columns],
        'Mean': [str(df[c].mean()) if pd.api.types.is_numeric_dtype(df[c]) else '' for c in df.columns],
        'Median': [str(df[c].median()) if pd.api.types.is_numeric_dtype(df[c]) else '' for c in df.columns],
        'Std': [str(df[c].std()) if pd.api.types.is_numeric_dtype(df[c]) else '' for c in df.columns],
        'Min': [str(df[c].min()) if pd.api.types.is_numeric_dtype(df[c]) else '' for c in df.columns],
        'Max': [str(df[c].max()) if pd.api.types.is_numeric_dtype(df[c]) else '' for c in df.columns],
        'Mode': [str(df[c].mode().iloc[0]) if not df[c].mode().empty else '' for c in df.columns],
        'Zeros': [str((df[c] == 0).sum()) if pd.api.types.is_numeric_dtype(df[c]) else '' for c in df.columns],
        'Negatives': [str((df[c] < 0).sum()) if pd.api.types.is_numeric_dtype(df[c]) else '' for c in df.columns],
        'Outliers': [str(o) for o in outliers],
    })
    # Add flag column
    flag = []
    for i, row in summary.iterrows():
        f = ''
        if float(row['Missing %']) > 50:
            f += 'High missing; '
        if row['Unique Values'] == '0' or row['Unique Values'] == '1':
            f += 'Low uniqueness; '
        if row['Outliers'] not in ('', '0'):
            f += 'Has outliers; '
        flag.append(f)
    summary['Flag'] = flag
    # Ensure all columns are stringified if they are not scalar
    for col in summary.columns:
        summary[col] = summary[col].apply(lambda x: str(x) if isinstance(x, (list, dict, np.ndarray)) else x)
    return summary

def parse_currency(val):
    """Enhanced currency parsing with better error handling."""
    if isinstance(val, str):
        val = val.replace("$", "").replace(",", "").strip()
        match = re.match(r"([\d.]+)([KMB]?)", val, re.IGNORECASE)
        if match:
            num, suffix = match.groups()
            num = float(num)
            if suffix.upper() == 'K':
                return num * 1e3
            elif suffix.upper() == 'M':
                return num * 1e6
            elif suffix.upper() == 'B':
                return num * 1e9
            else:
                return num
        else:
            return None
    return val

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enhanced missing value handling with intelligent strategies.
    """
    df_filled = df.copy()
    for col in df_filled.columns:
        if df_filled[col].isnull().any():
            missing_pct = df_filled[col].isnull().mean() * 100
            
            if missing_pct < 5:
                # Drop rows with missing values for low missing percentage
                df_filled = df_filled.dropna(subset=[col])
            elif missing_pct < 30:
                # Fill with median/mean for numerical, mode for categorical
                if np.issubdtype(df_filled[col].dtype, np.number):
                    df_filled[col].fillna(df_filled[col].median(), inplace=True)
                else:
                    mode_val = df_filled[col].mode()
                    if not mode_val.empty:
                        df_filled[col].fillna(mode_val[0], inplace=True)
            else:
                # High missing percentage - use more sophisticated methods
                if np.issubdtype(df_filled[col].dtype, np.number):
                    df_filled[col].fillna(df_filled[col].median(), inplace=True)
                else:
                    df_filled[col].fillna('MISSING', inplace=True)
    
    return df_filled

def summarize_column(df: pd.DataFrame, col: str) -> dict:
    """Return a summary of the column for UI display."""
    s = df[col]
    summary = {
        'dtype': str(s.dtype),
        'missing_pct': s.isnull().mean() * 100,
        'n_missing': s.isnull().sum(),
        'n_unique': s.nunique(dropna=True),
        'sample_values': s.dropna().unique()[:5].tolist(),
        'min': s.min() if pd.api.types.is_numeric_dtype(s) else None,
        'max': s.max() if pd.api.types.is_numeric_dtype(s) else None,
        'mode': s.mode().iloc[0] if not s.mode().empty else None,
        'mean': s.mean() if pd.api.types.is_numeric_dtype(s) else None,
        'median': s.median() if pd.api.types.is_numeric_dtype(s) else None
    }
    return summary

def impute_missing_per_column(df: pd.DataFrame, strategies: dict, custom_values: dict = None) -> pd.DataFrame:
    """
    Impute missing values per column using the specified strategies.
    strategies: dict of {col: strategy}, where strategy is one of:
      'mean', 'median', 'mode', 'ffill', 'bfill', 'custom', or a callable
    custom_values: dict of {col: value} for 'custom' strategy
    """
    df_filled = df.copy()
    custom_values = custom_values or {}
    for col, strat in strategies.items():
        if col not in df_filled.columns:
            continue
        s = df_filled[col]
        if not s.isnull().any():
            continue
        if strat == 'mean' and pd.api.types.is_numeric_dtype(s):
            df_filled[col] = s.fillna(s.mean())
        elif strat == 'median' and pd.api.types.is_numeric_dtype(s):
            df_filled[col] = s.fillna(s.median())
        elif strat == 'mode':
            mode_val = s.mode()
            if not mode_val.empty:
                df_filled[col] = s.fillna(mode_val[0])
        elif strat == 'ffill':
            df_filled[col] = s.fillna(method='ffill')
        elif strat == 'bfill':
            df_filled[col] = s.fillna(method='bfill')
        elif strat == 'custom':
            val = custom_values.get(col, None)
            if val is not None:
                df_filled[col] = s.fillna(val)
        elif callable(strat):
            df_filled[col] = s.fillna(strat(s))
        elif strat == 'groupby':
            # For groupby, expect custom_values[col] = (group_col, agg_func)
            group_col, agg_func = custom_values.get(col, (None, None))
            if group_col and agg_func:
                df_filled[col] = s.fillna(
                    s.groupby(df_filled[group_col]).transform(agg_func)
                )
        # else: skip or warn
    return df_filled

def suggest_group_impute(df: pd.DataFrame, target: str, groups: list) -> pd.Series:
    return df[target].fillna(df.groupby(groups)[target].transform('mean'))

def drop_duplicates(df, ignore_columns=None):
    ignore_columns = ignore_columns or []
    cols_to_check = [col for col in df.columns if col not in ignore_columns]
    return df.drop_duplicates(subset=cols_to_check)

def demonstrate_system():
    """Demonstrate the enhanced intelligent data cleaning system."""
    
    # Load datasets
    fed = pd.read_csv("fedreserve_Kaggle.csv")
    sas = pd.read_csv("Kaggle_saas.csv")
    
    # Initialize the intelligent cleaner
    cleaner = IntelligentDataCleaner()
    
    print("=== ENHANCED INTELLIGENT DATA CLEANING SYSTEM ===\n")
    
    # Analyze Federal Reserve dataset
    print("🔍 Analyzing Federal Reserve dataset...")
    fed_report = cleaner.analyze_dataset(fed)
    
    print(f"\n📊 FEDERAL RESERVE ANALYSIS RESULTS:")
    print(f"Data Quality Score: {fed_report.data_quality_score:.2f}/100")
    print(f"Total Anomalies Found: {len(fed_report.anomalies_found)}")
    
    # Show anomaly breakdown
    anomaly_types = {}
    severity_counts = {'low': 0, 'medium': 0, 'high': 0, 'critical': 0}
    
    for anomaly in fed_report.anomalies_found:
        anomaly_types[anomaly.anomaly_type] = anomaly_types.get(anomaly.anomaly_type, 0) + 1
        severity_counts[anomaly.severity] += 1
    
    print(f"\n📋 Anomaly Breakdown:")
    for atype, count in anomaly_types.items():
        print(f"  {atype}: {count}")
    
    print(f"\n⚠️ Severity Distribution:")
    for severity, count in severity_counts.items():
        print(f"  {severity}: {count}")
    
    # Show cleaning suggestions
    print(f"\n💡 CLEANING SUGGESTIONS:")
    for category, suggestions in fed_report.cleaning_suggestions.items():
        print(f"\n{category.upper()}:")
        for suggestion in suggestions:
            print(f"  • {suggestion}")
    
    # Show sample anomalies
    print(f"\n🔍 SAMPLE ANOMALIES (first 3):")
    for i, anomaly in enumerate(fed_report.anomalies_found[:3]):
        print(f"\n{i+1}. {anomaly.anomaly_type.upper()} in '{anomaly.column}'")
        print(f"   Row {anomaly.row_index}: '{anomaly.original_value}' -> '{anomaly.suggested_correction}'")
        print(f"   Severity: {anomaly.severity}, Confidence: {anomaly.confidence:.2f}")
        print(f"   Reasoning: {anomaly.reasoning}")
    
    # Export detailed report
    report_file = cleaner.export_report(fed_report)
    print(f"\n📄 Detailed report exported to: {report_file}")
    
    return cleaner, fed, fed_report

if __name__ == "__main__":
    # Run demonstration
    cleaner, original_df, cleaning_report = demonstrate_system()
    
    print(f"\n" + "="*60)
    print("ENHANCED SYSTEM READY FOR PRODUCTION USE")
    print("="*60)
    print("\nTo use with your own data:")
    print("1. Load your CSV: df = pd.read_csv('your_data.csv')")
    print("2. Analyze: report = cleaner.analyze_dataset(df)")
    print("3. Apply corrections: cleaned_df = cleaner.apply_corrections(df, report.anomalies_found)")
    print("4. Export report: cleaner.export_report(report)")
