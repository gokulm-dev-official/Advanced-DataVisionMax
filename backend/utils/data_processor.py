import pandas as pd
import numpy as np
# Removed heavy sklearn dependencies to optimize for Vercel (250MB limit)
# from sklearn.linear_model import LinearRegression
# from sklearn.preprocessing import StandardScaler 
# from sklearn.ensemble import IsolationForest
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_absolute_error, r2_score
import json
import os
from datetime import datetime, timedelta
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Lightweight shim for LinearRegression using numpy
class LinearRegression:
    def __init__(self):
        self.slope = 0
        self.intercept = 0
        self.coef_ = [0]
        
    def fit(self, X, y):
        # Flatten input to 1D arrays for polyfit
        X_flat = np.array(X).flatten()
        y_flat = np.array(y).flatten()
        if len(X_flat) > 0:
            self.slope, self.intercept = np.polyfit(X_flat, y_flat, 1)
            self.coef_ = [self.slope]
        return self
        
    def predict(self, X):
        X_flat = np.array(X).flatten()
        return self.slope * X_flat + self.intercept
        
    def score(self, X, y):
        try:
            y_pred = self.predict(X)
            y_true = np.array(y).flatten()
            u = ((y_true - y_pred) ** 2).sum()
            v = ((y_true - y_true.mean()) ** 2).sum()
            return 1 - (u/v) if v != 0 else 0
        except:
            return 0

class DataProcessor:
    def __init__(self):
        # self.scaler = StandardScaler() # Removed
        self.models = {}
    
    def load_data(self, file_path):
        """Load data from various file formats with enhanced error handling"""
        try:
            print(f"📂 Loading file: {file_path}")
            
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            file_size = os.path.getsize(file_path)
            print(f"📏 File size: {file_size} bytes")
            
            if file_size == 0:
                raise ValueError("File is empty")
            
            # Determine file type and load accordingly
            file_extension = os.path.splitext(file_path)[1].lower()
            
            if file_extension == '.csv':
                # Try different encodings and separators
                encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
                separators = [',', ';', '\t', '|']
                
                df = None
                for encoding in encodings:
                    for sep in separators:
                        try:
                            df = pd.read_csv(file_path, encoding=encoding, sep=sep)
                            if len(df.columns) > 1:  # Valid separation found
                                print(f"✅ CSV loaded successfully with {encoding} encoding and '{sep}' separator")
                                break
                        except (UnicodeDecodeError, pd.errors.EmptyDataError):
                            continue
                    if df is not None and len(df.columns) > 1:
                        break
                
                if df is None or len(df.columns) <= 1:
                    # Fallback: try with default settings and error handling
                    df = pd.read_csv(file_path, encoding='utf-8', errors='ignore')
                    print("⚠️ CSV loaded with basic settings, some issues may exist")
                    
            elif file_extension in ['.xlsx', '.xls']:
                try:
                    # Try reading all sheets and use the first non-empty one
                    excel_file = pd.ExcelFile(file_path)
                    sheet_names = excel_file.sheet_names
                    
                    df = None
                    for sheet in sheet_names:
                        try:
                            temp_df = pd.read_excel(file_path, sheet_name=sheet)
                            if not temp_df.empty and len(temp_df.columns) > 1:
                                df = temp_df
                                print(f"✅ Excel sheet '{sheet}' loaded successfully")
                                break
                        except:
                            continue
                    
                    if df is None:
                        df = pd.read_excel(file_path)  # Default attempt
                        
                except Exception as e:
                    print(f"⚠️ Excel reading issue: {e}, trying with openpyxl engine")
                    df = pd.read_excel(file_path, engine='openpyxl')
                
                print(f"✅ Excel file loaded successfully")
                
            elif file_extension == '.json':
                # Handle different JSON structures
                try:
                    df = pd.read_json(file_path)
                    print(f"✅ JSON loaded successfully")
                except ValueError:
                    try:
                        # Try loading as lines of JSON
                        df = pd.read_json(file_path, lines=True)
                        print(f"✅ JSON Lines loaded successfully")
                    except:
                        # Try loading as a list of records
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                        df = pd.DataFrame(data)
                        print(f"✅ JSON records loaded successfully")
                        
            else:
                raise ValueError(f"Unsupported file format: {file_extension}. Supported formats: .csv, .xlsx, .xls, .json")
            
            if df.empty:
                raise ValueError("The loaded dataset is empty")
            
            # Basic cleanup
            df = df.dropna(how='all')  # Remove completely empty rows
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]  # Remove unnamed columns
            
            if df.empty:
                raise ValueError("Dataset is empty after removing blank rows/columns")
            
            print(f"📊 Dataset loaded: {len(df)} rows × {len(df.columns)} columns")
            print(f"🏷️ Columns: {list(df.columns)}")
            
            return df
            
        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            raise e
    
    def get_comprehensive_stats(self, df):
        """Generate comprehensive statistics for any dataset"""
        try:
            print("📊 Generating comprehensive statistics...")
            
            stats = {
                'total_rows': len(df),
                'total_columns': len(df.columns),
                'missing_values': df.isnull().sum().to_dict(),
                'data_types': df.dtypes.astype(str).to_dict(),
                'memory_usage': f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB",
                'numeric_summary': {},
                'categorical_summary': {},
                'date_summary': {},
                'correlation_summary': {},
                'data_quality_metrics': {}
            }
            
            # Automatically detect and convert date columns
            df = self._auto_detect_dates(df)
            
            # Numeric columns analysis
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            print(f"🔢 Found {len(numeric_cols)} numeric columns")
            
            for col in numeric_cols:
                if df[col].notna().sum() > 0:  # Only process if has data
                    try:
                        stats['numeric_summary'][col] = {
                            'count': int(df[col].count()),
                            'mean': float(df[col].mean()),
                            'median': float(df[col].median()),
                            'std': float(df[col].std()) if df[col].std() is not np.nan else 0,
                            'min': float(df[col].min()),
                            'max': float(df[col].max()),
                            'q25': float(df[col].quantile(0.25)),
                            'q75': float(df[col].quantile(0.75)),
                            'skewness': float(df[col].skew()) if not df[col].skew() is np.nan else 0,
                            'kurtosis': float(df[col].kurtosis()) if not df[col].kurtosis() is np.nan else 0,
                            'outliers_count': self._count_outliers(df[col]),
                            'zero_count': int((df[col] == 0).sum()),
                            'negative_count': int((df[col] < 0).sum())
                        }
                    except Exception as e:
                        print(f"⚠️ Error processing numeric column {col}: {e}")
                        continue
            
            # Categorical columns analysis
            categorical_cols = df.select_dtypes(include=['object']).columns
            print(f"📝 Found {len(categorical_cols)} categorical columns")
            
            for col in categorical_cols:
                try:
                    value_counts = df[col].value_counts()
                    stats['categorical_summary'][col] = {
                        'unique_count': int(df[col].nunique()),
                        'most_frequent': str(value_counts.index[0]) if not value_counts.empty else None,
                        'most_frequent_count': int(value_counts.iloc[0]) if not value_counts.empty else 0,
                        'top_5_values': {str(k): int(v) for k, v in value_counts.head(5).to_dict().items()},
                        'null_count': int(df[col].isnull().sum()),
                        'empty_string_count': int((df[col] == '').sum())
                    }
                except Exception as e:
                    print(f"⚠️ Error processing categorical column {col}: {e}")
                    continue
            
            # Date columns analysis
            date_cols = df.select_dtypes(include=['datetime64']).columns
            print(f"📅 Found {len(date_cols)} date columns")
            
            for col in date_cols:
                try:
                    if df[col].notna().sum() > 0:
                        stats['date_summary'][col] = {
                            'min_date': str(df[col].min()),
                            'max_date': str(df[col].max()),
                            'date_range_days': int((df[col].max() - df[col].min()).days),
                            'unique_dates': int(df[col].nunique()),
                            'null_count': int(df[col].isnull().sum())
                        }
                except Exception as e:
                    print(f"⚠️ Error processing date column {col}: {e}")
                    continue
            
            # Correlation analysis (only if we have multiple numeric columns)
            if len(numeric_cols) > 1:
                try:
                    corr_matrix = df[numeric_cols].corr()
                    corr_pairs = []
                    
                    for i in range(len(corr_matrix.columns)):
                        for j in range(i+1, len(corr_matrix.columns)):
                            corr_val = corr_matrix.iloc[i, j]
                            if not np.isnan(corr_val) and abs(corr_val) > 0.3:  # Only significant correlations
                                corr_pairs.append({
                                    'col1': corr_matrix.columns[i],
                                    'col2': corr_matrix.columns[j],
                                    'correlation': float(corr_val)
                                })
                    
                    stats['correlation_summary'] = {
                        'strong_correlations': sorted(corr_pairs, key=lambda x: abs(x['correlation']), reverse=True)[:5]
                    }
                except Exception as e:
                    print(f"⚠️ Error calculating correlations: {e}")
                    stats['correlation_summary'] = {'strong_correlations': []}
            
            # Data quality metrics
            total_cells = df.shape[0] * df.shape[1]
            missing_cells = df.isnull().sum().sum()
            duplicate_rows = df.duplicated().sum()
            
            stats['data_quality_metrics'] = {
                'completeness': float(((total_cells - missing_cells) / total_cells) * 100),
                'uniqueness': float(((len(df) - duplicate_rows) / len(df)) * 100),
                'consistency': self._calculate_consistency_score(df),
                'overall_quality_score': self._calculate_overall_quality(df)
            }
            
            print("✅ Statistics generation completed")
            return stats
            
        except Exception as e:
            print(f"❌ Error generating statistics: {e}")
            # Return basic stats if comprehensive analysis fails
            return {
                'total_rows': len(df),
                'total_columns': len(df.columns),
                'missing_values': {},
                'data_types': {},
                'memory_usage': "Unknown",
                'numeric_summary': {},
                'categorical_summary': {},
                'date_summary': {},
                'correlation_summary': {},
                'data_quality_metrics': {'overall_quality_score': 50}
            }
    
    def _auto_detect_dates(self, df):
        """Automatically detect and convert date columns"""
        df = df.copy()
        
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check if column looks like dates
                sample_values = df[col].dropna().head(10)
                
                if len(sample_values) > 0:
                    date_like_count = 0
                    for val in sample_values:
                        val_str = str(val)
                        # Check for common date patterns
                        if any(pattern in val_str.lower() for pattern in ['/', '-', 'jan', 'feb', 'mar', 'apr', 'may', 'jun', 
                                                                         'jul', 'aug', 'sep', 'oct', 'nov', 'dec']):
                            date_like_count += 1
                    
                    # If more than 50% look like dates, try to convert
                    if date_like_count / len(sample_values) > 0.5:
                        try:
                            df[col] = pd.to_datetime(df[col], errors='coerce')
                            print(f"📅 Auto-converted column '{col}' to datetime")
                        except:
                            continue
        
        return df
    
    def predict_comprehensive_trends(self, df):
        """Generate comprehensive predictions for suitable columns"""
        try:
            print("🔮 Generating predictions...")
            predictions = {}
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            date_cols = df.select_dtypes(include=['datetime64']).columns
            
            # Time-based predictions if we have date columns
            if len(date_cols) > 0 and len(numeric_cols) > 0:
                date_col = date_cols[0]
                df_sorted = df.sort_values(date_col)
                
                for num_col in numeric_cols[:3]:  # Limit to top 3 numeric columns
                    try:
                        prediction_result = self._predict_time_series(df_sorted, date_col, num_col)
                        if prediction_result:
                            predictions[num_col] = prediction_result
                    except Exception as e:
                        print(f"⚠️ Prediction error for {num_col}: {e}")
                        continue
            
            # Simple trend predictions for numeric columns
            for col in numeric_cols[:5]:  # Limit to 5 columns
                if col not in predictions:
                    try:
                        simple_prediction = self._predict_simple_trend(df, col)
                        if simple_prediction:
                            predictions[col] = simple_prediction
                    except Exception as e:
                        print(f"⚠️ Simple prediction error for {col}: {e}")
                        continue
            
            # Business-specific predictions
            try:
                business_predictions = self._generate_business_predictions(df)
                predictions.update(business_predictions)
            except Exception as e:
                print(f"⚠️ Business predictions error: {e}")
            
            print(f"✅ Generated predictions for {len(predictions)} metrics")
            return predictions
            
        except Exception as e:
            print(f"❌ Error in predictions: {e}")
            return {}
    
    def _predict_time_series(self, df, date_col, value_col):
        """Predict time series with enhanced error handling"""
        try:
            if len(df) < 10:
                return None
            
            # Prepare data
            df_clean = df[[date_col, value_col]].dropna()
            if len(df_clean) < 5:
                return None
            
            df_clean = df_clean.sort_values(date_col)
            df_clean['days_from_start'] = (df_clean[date_col] - df_clean[date_col].min()).dt.days
            
            X = df_clean['days_from_start'].values.reshape(-1, 1)
            y = df_clean[value_col].values
            
            # Linear regression
            linear_model = LinearRegression()
            linear_model.fit(X, y)
            
            # Generate future predictions
            last_day = df_clean['days_from_start'].max()
            future_days = np.arange(last_day + 1, last_day + 31).reshape(-1, 1)  # Next 30 days
            future_predictions = linear_model.predict(future_days)
            
            # Calculate metrics
            r2 = linear_model.score(X, y)
            slope = linear_model.coef_[0]
            trend = 'increasing' if slope > 0 else 'decreasing'
            
            # Seasonal analysis if enough data
            seasonal_pattern = None
            if len(df_clean) > 30:
                seasonal_pattern = self._detect_seasonal_pattern(df_clean, date_col, value_col)
            
            return {
                'trend': trend,
                'slope': float(slope),
                'predictions': [max(0, pred) for pred in future_predictions.tolist()],  # Ensure non-negative
                'confidence_score': max(0, min(1, r2)),  # Ensure 0-1 range
                'model_used': 'linear_regression',
                'seasonal_pattern': seasonal_pattern,
                'data_points': len(df_clean)
            }
            
        except Exception as e:
            print(f"⚠️ Time series prediction error: {e}")
            return None
    
    def _predict_simple_trend(self, df, col):
        """Simple trend prediction for numeric columns"""
        try:
            values = df[col].dropna()
            if len(values) < 5:
                return None
            
            # Use index as time proxy
            X = np.arange(len(values)).reshape(-1, 1)
            y = values.values
            
            model = LinearRegression()
            model.fit(X, y)
            
            # Predict next 10 points
            future_X = np.arange(len(values), len(values) + 10).reshape(-1, 1)
            future_predictions = model.predict(future_X)
            
            slope = model.coef_[0]
            r2 = max(0, min(1, model.score(X, y)))  # Ensure 0-1 range
            
            return {
                'trend': 'increasing' if slope > 0 else 'decreasing',
                'slope': float(slope),
                'predictions': [max(0, pred) for pred in future_predictions.tolist()],  # Ensure non-negative
                'confidence_score': r2,
                'model_used': 'simple_linear_regression',
                'data_points': len(values)
            }
            
        except Exception as e:
            print(f"⚠️ Simple trend prediction error: {e}")
            return None
    
    def _detect_seasonal_pattern(self, df, date_col, value_col):
        """Detect seasonal patterns in time series data"""
        try:
            if len(df) < 30:
                return None
            
            df = df.copy()
            df['month'] = df[date_col].dt.month
            df['day_of_week'] = df[date_col].dt.dayofweek
            df['quarter'] = df[date_col].dt.quarter
            
            # Monthly seasonality
            monthly_avg = df.groupby('month')[value_col].mean()
            monthly_std = df.groupby('month')[value_col].std().fillna(0)
            
            # Weekly seasonality
            weekly_avg = df.groupby('day_of_week')[value_col].mean()
            
            # Quarterly seasonality
            quarterly_avg = df.groupby('quarter')[value_col].mean()
            
            # Calculate variation coefficients safely
            monthly_variation = (monthly_std.mean() / monthly_avg.mean()) if monthly_avg.mean() != 0 else 0
            weekly_variation = (weekly_avg.std() / weekly_avg.mean()) if weekly_avg.mean() != 0 else 0
            quarterly_variation = (quarterly_avg.std() / quarterly_avg.mean()) if quarterly_avg.mean() != 0 else 0
            
            strongest_pattern = max([
                ('monthly', monthly_variation),
                ('weekly', weekly_variation),
                ('quarterly', quarterly_variation)
            ], key=lambda x: x[1])
            
            return {
                'strongest_pattern': strongest_pattern[0],
                'pattern_strength': float(strongest_pattern[1]),
                'monthly_peaks': int(monthly_avg.idxmax()),
                'monthly_lows': int(monthly_avg.idxmin()),
                'best_day_of_week': int(weekly_avg.idxmax()),
                'best_quarter': int(quarterly_avg.idxmax())
            }
            
        except Exception as e:
            print(f"⚠️ Seasonal pattern detection error: {e}")
            return None
    
    def _generate_business_predictions(self, df):
        """Generate business-specific predictions based on column names"""
        predictions = {}
        
        try:
            # Look for common business columns
            sales_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ['sales', 'revenue', 'amount', 'price'])]
            category_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ['category', 'product', 'type', 'segment'])]
            region_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ['region', 'location', 'area', 'country', 'state'])]
            customer_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ['customer', 'client', 'user', 'id'])]
            date_cols = df.select_dtypes(include=['datetime64']).columns
            
            # Product/Category seasonality analysis
            if len(sales_cols) > 0 and len(category_cols) > 0 and len(date_cols) > 0:
                try:
                    product_seasonal = self._analyze_product_seasonality(df, sales_cols[0], category_cols[0], date_cols[0])
                    if product_seasonal:
                        predictions['product_seasonality'] = product_seasonal
                except Exception as e:
                    print(f"⚠️ Product seasonality error: {e}")
            
            # Regional performance analysis
            if len(region_cols) > 0 and len(sales_cols) > 0:
                try:
                    regional_predictions = self._analyze_regional_performance(df, region_cols[0], sales_cols[0])
                    if regional_predictions:
                        predictions['regional_performance'] = regional_predictions
                except Exception as e:
                    print(f"⚠️ Regional analysis error: {e}")
            
            # Customer behavior analysis
            if len(customer_cols) > 0 and len(sales_cols) > 0:
                try:
                    customer_insights = self._analyze_customer_behavior(df, customer_cols[0], sales_cols[0])
                    if customer_insights:
                        predictions['customer_insights'] = customer_insights
                except Exception as e:
                    print(f"⚠️ Customer analysis error: {e}")
            
        except Exception as e:
            print(f"⚠️ Business predictions error: {e}")
        
        return predictions
    
    def _analyze_product_seasonality(self, df, sales_col, category_col, date_col):
        """Analyze which products/categories perform best in which periods"""
        try:
            df = df.copy()
            df['Month'] = df[date_col].dt.month
            df['Month_Name'] = df[date_col].dt.strftime('%B')
            
            # Product performance by month
            product_monthly = df.groupby([category_col, 'Month_Name'])[sales_col].sum().reset_index()
            
            # Find best month for each product/category
            product_best_months = {}
            for product in df[category_col].unique()[:10]:  # Limit to top 10
                product_data = product_monthly[product_monthly[category_col] == product]
                if not product_data.empty and len(product_data) > 1:
                    best_month = product_data.loc[product_data[sales_col].idxmax(), 'Month_Name']
                    best_sales = product_data[sales_col].max()
                    avg_sales = product_data[sales_col].mean()
                    
                    product_best_months[str(product)] = {
                        'best_month': best_month,
                        'best_sales': float(best_sales),
                        'average_sales': float(avg_sales),
                        'performance_prediction': f"{best_month} is optimal for {product} with {best_sales:,.0f} in sales"
                    }
            
            return product_best_months
            
        except Exception as e:
            print(f"⚠️ Product seasonality analysis error: {e}")
            return None
    
    def _analyze_regional_performance(self, df, region_col, sales_col):
        """Analyze regional performance patterns"""
        try:
            regional_performance = df.groupby(region_col)[sales_col].agg([
                'sum', 'mean', 'count', 'std'
            ]).fillna(0).round(2)
            
            # Identify top performing regions
            top_region = regional_performance['sum'].idxmax()
            most_consistent = regional_performance['std'].idxmin()
            
            return {
                'top_performing_region': str(top_region),
                'most_consistent_region': str(most_consistent),
                'regional_rankings': {str(k): v for k, v in regional_performance.sort_values('sum', ascending=False).to_dict().items()},
                'performance_insights': {
                    'highest_total': f"{top_region} leads in total {sales_col.replace('_', ' ').lower()}",
                    'most_stable': f"{most_consistent} shows most consistent performance"
                }
            }
            
        except Exception as e:
            print(f"⚠️ Regional analysis error: {e}")
            return None
    
    def _analyze_customer_behavior(self, df, customer_col, sales_col):
        """Analyze customer behavior patterns"""
        try:
            # Customer metrics
            customer_metrics = df.groupby(customer_col).agg({
                sales_col: ['sum', 'count', 'mean']
            }).round(2)
            
            customer_metrics.columns = ['total_sales', 'transaction_count', 'avg_transaction']
            
            # Customer segments
            high_value_threshold = customer_metrics['total_sales'].quantile(0.8)
            frequent_buyer_threshold = customer_metrics['transaction_count'].quantile(0.8)
            
            segments = {
                'high_value_customers': int(len(customer_metrics[customer_metrics['total_sales'] > high_value_threshold])),
                'frequent_buyers': int(len(customer_metrics[customer_metrics['transaction_count'] > frequent_buyer_threshold])),
                'average_customer_value': float(customer_metrics['total_sales'].mean()),
                'top_customer_value': float(customer_metrics['total_sales'].max()),
                'total_customers': int(len(customer_metrics))
            }
            
            return segments
            
        except Exception as e:
            print(f"⚠️ Customer behavior analysis error: {e}")
            return None
    
    def detect_comprehensive_anomalies(self, df):
        """Comprehensive anomaly detection with enhanced error handling"""
        try:
            print("🔍 Detecting anomalies...")
            anomalies = {}
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                try:
                    if len(df) > 10 and df[col].notna().sum() > 5:
                        col_anomalies = self._detect_column_anomalies(df, col)
                        if col_anomalies and col_anomalies.get('count', 0) > 0:
                            anomalies[col] = col_anomalies
                except Exception as e:
                    print(f"⚠️ Anomaly detection error for {col}: {e}")
                    continue
            
            # Business-specific anomaly detection
            try:
                business_anomalies = self._detect_business_anomalies(df)
                if business_anomalies:
                    anomalies.update(business_anomalies)
            except Exception as e:
                print(f"⚠️ Business anomaly detection error: {e}")
            
            print(f"✅ Anomaly detection completed. Found issues in {len(anomalies)} columns")
            return anomalies
            
        except Exception as e:
            print(f"❌ Error in anomaly detection: {e}")
            return {}
    
    def _detect_column_anomalies(self, df, col):
        """Detect anomalies in a specific column using multiple methods"""
        try:
            data = df[col].dropna()
            if len(data) < 5:
                return None
            
            # Statistical outliers (IQR method)
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            
            if IQR == 0:  # Handle case where all values are the same
                return {'count': 0, 'indices': [], 'values': [], 'percentage': 0}
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            statistical_outliers = data[(data < lower_bound) | (data > upper_bound)]
            
            # Z-score method (only if we have enough data and variation)
            zscore_outliers = pd.Series([])
            if len(data) > 20 and data.std() > 0:
                try:
                    z_scores = np.abs(stats.zscore(data))
                    zscore_outliers = data[z_scores > 3]
                except:
                    pass
            
            # Combine methods
            all_outliers = pd.concat([statistical_outliers, zscore_outliers]).drop_duplicates()
            
            return {
                'count': len(all_outliers),
                'indices': all_outliers.index.tolist()[:20],  # Limit to 20 for performance
                'values': all_outliers.tolist()[:20],
                'percentage': (len(all_outliers) / len(data)) * 100,
                'detection_methods': {
                    'statistical': len(statistical_outliers),
                    'zscore': len(zscore_outliers)
                }
            }
            
        except Exception as e:
            print(f"⚠️ Column anomaly detection error for {col}: {e}")
            return None
    
    def _detect_business_anomalies(self, df):
        """Detect business-specific anomalies"""
        anomalies = {}
        
        try:
            # Look for business columns
            sales_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ['sales', 'revenue', 'amount'])]
            date_cols = df.select_dtypes(include=['datetime64']).columns
            customer_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ['customer', 'client', 'user'])]
            
            # Sales pattern anomalies
            if len(sales_cols) > 0 and len(date_cols) > 0:
                try:
                    sales_anomalies = self._detect_sales_anomalies(df, sales_cols[0], date_cols[0])
                    if sales_anomalies:
                        anomalies['sales_patterns'] = sales_anomalies
                except Exception as e:
                    print(f"⚠️ Sales anomaly detection error: {e}")
            
            # Customer behavior anomalies
            if len(customer_cols) > 0 and len(sales_cols) > 0:
                try:
                    customer_anomalies = self._detect_customer_anomalies(df, customer_cols[0], sales_cols[0])
                    if customer_anomalies:
                        anomalies['customer_behavior'] = customer_anomalies
                except Exception as e:
                    print(f"⚠️ Customer anomaly detection error: {e}")
            
        except Exception as e:
            print(f"⚠️ Business anomaly detection error: {e}")
        
        return anomalies
    
    def _detect_sales_anomalies(self, df, sales_col, date_col):
        """Detect anomalies in sales patterns"""
        try:
            df = df.copy()
            daily_sales = df.groupby(date_col)[sales_col].sum()
            
            if len(daily_sales) < 10:
                return None
            
            # Detect unusual sales days
            Q1 = daily_sales.quantile(0.25)
            Q3 = daily_sales.quantile(0.75)
            IQR = Q3 - Q1
            
            if IQR == 0:
                return None
            
            unusual_low = daily_sales[daily_sales < Q1 - 1.5 * IQR]
            unusual_high = daily_sales[daily_sales > Q3 + 1.5 * IQR]
            
            return {
                'unusual_low_sales_days': len(unusual_low),
                'unusual_high_sales_days': len(unusual_high),
                'low_sales_dates': unusual_low.index.strftime('%Y-%m-%d').tolist()[:10],
                'high_sales_dates': unusual_high.index.strftime('%Y-%m-%d').tolist()[:10],
                'count': len(unusual_low) + len(unusual_high),
                'values': unusual_low.tolist()[:10] + unusual_high.tolist()[:10]
            }
            
        except Exception as e:
            print(f"⚠️ Sales anomaly detection error: {e}")
            return None
    
    def _detect_customer_anomalies(self, df, customer_col, sales_col):
        """Detect anomalies in customer behavior"""
        try:
            customer_totals = df.groupby(customer_col)[sales_col].sum()
            
            if len(customer_totals) < 10:
                return None
            
            # Very high spenders (potential data errors or VIP customers)
            high_threshold = customer_totals.quantile(0.95)
            high_spenders = customer_totals[customer_totals > high_threshold]
            
            return {
                'extremely_high_spenders': len(high_spenders),
                'high_spender_values': high_spenders.tolist()[:10],
                'count': len(high_spenders),
                'values': high_spenders.tolist()[:10]
            }
            
        except Exception as e:
            print(f"⚠️ Customer anomaly detection error: {e}")
            return None
    
    def _count_outliers(self, series):
        """Count outliers in a series using IQR method"""
        try:
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            
            if IQR == 0:
                return 0
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            return len(series[(series < lower_bound) | (series > upper_bound)])
        except:
            return 0
    
    def _calculate_consistency_score(self, df):
        """Calculate data consistency score"""
        try:
            scores = []
            
            # Check for consistent data types
            type_score = 0
            for col in df.columns:
                try:
                    if df[col].dtype in ['int64', 'float64', 'datetime64[ns]']:
                        type_score += 1
                    elif df[col].dtype == 'object':
                        # Check if object column has consistent format
                        if df[col].nunique() / len(df) < 0.5:  # Likely categorical
                            type_score += 0.8
                        else:
                            type_score += 0.3
                except:
                    type_score += 0
            
            if len(df.columns) > 0:
                scores.append((type_score / len(df.columns)) * 100)
            else:
                scores.append(0)
            
            return np.mean(scores) if scores else 0
            
        except:
            return 50  # Default score
    
    def _calculate_overall_quality(self, df):
        """Calculate overall data quality score"""
        try:
            scores = []
            
            # Completeness
            total_cells = df.shape[0] * df.shape[1]
            missing_cells = df.isnull().sum().sum()
            completeness = ((total_cells - missing_cells) / total_cells) * 100 if total_cells > 0 else 0
            scores.append(completeness)
            
            # Uniqueness
            duplicate_rows = df.duplicated().sum()
            uniqueness = ((len(df) - duplicate_rows) / len(df)) * 100 if len(df) > 0 else 0
            scores.append(uniqueness)
            
            # Consistency
            consistency = self._calculate_consistency_score(df)
            scores.append(consistency)
            
            # Validity (check for reasonable values)
            validity_scores = []
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                try:
                    outlier_percentage = (self._count_outliers(df[col]) / len(df)) * 100
                    validity_scores.append(max(0, 100 - outlier_percentage * 2))
                except:
                    validity_scores.append(80)  # Default score
            
            if validity_scores:
                scores.append(np.mean(validity_scores))
            
            return round(np.mean(scores), 2) if scores else 50
            
        except Exception as e:
            print(f"⚠️ Quality calculation error: {e}")
            return 50  # Default score