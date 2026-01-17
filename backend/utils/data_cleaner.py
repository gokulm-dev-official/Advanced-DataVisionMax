import pandas as pd
import numpy as np
from datetime import datetime
import re

class DataCleaner:
    def __init__(self):
        self.cleaning_log = []
    
    def clean_data_comprehensive(self, df):
        """Comprehensive data cleaning with detailed reporting"""
        self.cleaning_log = []
        original_shape = df.shape
        cleaned_df = df.copy()
        
        # 1. Handle missing values
        cleaned_df = self._handle_missing_values(cleaned_df)
        
        # 2. Remove duplicates
        cleaned_df = self._remove_duplicates(cleaned_df)
        
        # 3. Fix data types
        cleaned_df = self._fix_data_types(cleaned_df)
        
        # 4. Handle outliers
        cleaned_df = self._handle_outliers(cleaned_df)
        
        # 5. Standardize text data
        cleaned_df = self._standardize_text(cleaned_df)
        
        # 6. Validate and fix dates
        cleaned_df = self._fix_dates(cleaned_df)
        
        # Generate cleaning report
        cleaning_report = {
            'original_shape': original_shape,
            'cleaned_shape': cleaned_df.shape,
            'rows_removed': original_shape[0] - cleaned_df.shape[0],
            'cleaning_steps': self.cleaning_log,
            'data_quality_score': self._calculate_quality_score(cleaned_df),
            'recommendations': self._generate_recommendations(cleaned_df)
        }
        
        return cleaned_df, cleaning_report
    
    def _handle_missing_values(self, df):
        """Handle missing values intelligently"""
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            missing_percentage = (missing_count / len(df)) * 100
            
            if missing_count > 0:
                if missing_percentage > 50:
                    # Drop columns with more than 50% missing values
                    df = df.drop(columns=[col])
                    self.cleaning_log.append(f"Dropped column '{col}' (>{missing_percentage:.1f}% missing)")
                else:
                    if df[col].dtype in ['int64', 'float64']:
                        # Fill numeric with median
                        df[col].fillna(df[col].median(), inplace=True)
                        self.cleaning_log.append(f"Filled {missing_count} missing values in '{col}' with median")
                    else:
                        # Fill categorical with mode or 'Unknown'
                        mode_value = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
                        df[col].fillna(mode_value, inplace=True)
                        self.cleaning_log.append(f"Filled {missing_count} missing values in '{col}' with mode/Unknown")
        
        return df
    
    def _remove_duplicates(self, df):
        """Remove duplicate rows"""
        initial_count = len(df)
        df = df.drop_duplicates()
        removed_count = initial_count - len(df)
        
        if removed_count > 0:
            self.cleaning_log.append(f"Removed {removed_count} duplicate rows")
        
        return df
    
    def _fix_data_types(self, df):
        """Fix and optimize data types"""
        for col in df.columns:
            # Try to convert to appropriate data types
            if col.lower() in ['date', 'time', 'created', 'updated'] or 'date' in col.lower():
                try:
                    df[col] = pd.to_datetime(df[col])
                    self.cleaning_log.append(f"Converted '{col}' to datetime")
                except:
                    pass
            
            # Convert object columns that are actually numeric
            if df[col].dtype == 'object':
                # Check if it's actually numeric
                try:
                    # Remove common non-numeric characters
                    test_series = df[col].astype(str).str.replace('[^0-9.-]', '', regex=True)
                    test_series = pd.to_numeric(test_series, errors='coerce')
                    if test_series.notna().sum() / len(test_series) > 0.8:  # 80% can be converted
                        df[col] = test_series
                        self.cleaning_log.append(f"Converted '{col}' to numeric")
                except:
                    pass
        
        return df
    
    def _handle_outliers(self, df):
        """Handle outliers using IQR method"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            
            if outliers_count > 0:
                # Cap outliers instead of removing them
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
                self.cleaning_log.append(f"Capped {outliers_count} outliers in '{col}'")
        
        return df
    
    def _standardize_text(self, df):
        """Standardize text data"""
        text_cols = df.select_dtypes(include=['object']).columns
        
        for col in text_cols:
            if df[col].dtype == 'object':
                # Remove leading/trailing whitespace
                df[col] = df[col].astype(str).str.strip()
                
                # Standardize case for categorical-looking data
                if df[col].nunique() < len(df) * 0.5:  # Likely categorical
                    df[col] = df[col].str.title()
                    self.cleaning_log.append(f"Standardized text format in '{col}'")
        
        return df
    
    def _fix_dates(self, df):
        """Fix and validate date columns"""
        date_cols = df.select_dtypes(include=['datetime64']).columns
        
        for col in date_cols:
            # Remove future dates if they seem unrealistic
            current_date = datetime.now()
            future_dates = df[col] > current_date
            
            if future_dates.sum() > 0:
                df = df[~future_dates]
                self.cleaning_log.append(f"Removed {future_dates.sum()} future dates from '{col}'")
        
        return df
    
    def _calculate_quality_score(self, df):
        """Calculate overall data quality score"""
        scores = []
        
        # Completeness score
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isnull().sum().sum()
        completeness = ((total_cells - missing_cells) / total_cells) * 100
        scores.append(completeness)
        
        # Consistency score (based on data types)
        type_consistency = (df.dtypes != 'object').sum() / len(df.columns) * 100
        scores.append(type_consistency)
        
        # Uniqueness score (absence of duplicates)
        uniqueness = (1 - df.duplicated().sum() / len(df)) * 100
        scores.append(uniqueness)
        
        return round(np.mean(scores), 2)
    
    def _generate_recommendations(self, df):
        """Generate data quality recommendations"""
        recommendations = []
        
        # Check for potential issues
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if df[col].std() == 0:
                recommendations.append(f"Column '{col}' has no variance - consider removing")
        
        # Check for high cardinality categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].nunique() > len(df) * 0.9:
                recommendations.append(f"Column '{col}' has very high cardinality - consider encoding")
        
        if not recommendations:
            recommendations.append("Data quality is good - no major issues detected")
        
        return recommendations