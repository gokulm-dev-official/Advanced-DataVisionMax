import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import numpy as np
from datetime import datetime, timedelta

class ChartGenerator:
    def __init__(self):
        self.color_schemes = {
            'primary': ['#3b82f6', '#8b5cf6', '#ec4899', '#10b981', '#f59e0b'],
            'business': ['#1e40af', '#7c3aed', '#db2777', '#059669', '#d97706'],
            'gradient': ['#06b6d4', '#3b82f6', '#8b5cf6', '#ec4899', '#f97316']
        }
    
    def generate_comprehensive_charts(self, df):
        """Generate all types of charts for comprehensive analysis"""
        charts = {}
        
        numeric_cols = df.select_dtypes(include=['number']).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        date_cols = df.select_dtypes(include=['datetime64']).columns
        
        # 1. KPI Summary
        charts['kpis'] = self._create_kpi_summary(df)
        
        # 2. Time series charts
        if len(date_cols) > 0 and len(numeric_cols) > 0:
            charts['time_series'] = self._create_time_series_charts(df, date_cols[0])
            charts['revenue_trend'] = self._create_revenue_trend_chart(df, date_cols[0])
        
        # 3. Distribution charts
        if len(numeric_cols) > 0:
            charts['distributions'] = self._create_distribution_charts(df, numeric_cols)
        
        # 4. Categorical analysis
        if len(categorical_cols) > 0:
            charts['categorical'] = self._create_categorical_charts(df, categorical_cols)
        
        # 5. Correlation analysis
        if len(numeric_cols) > 1:
            charts['correlation'] = self._create_correlation_chart(df, numeric_cols)
        
        # 6. Advanced business charts
        charts.update(self._create_advanced_business_charts(df))
        
        # 7. Performance metrics
        charts['performance_metrics'] = self._create_performance_metrics(df)
        
        return charts
    
    def _create_kpi_summary(self, df):
        """Create comprehensive KPI summary"""
        kpis = {}
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        for col in numeric_cols:
            if df[col].notna().sum() > 0:  # Only process columns with data
                kpis[col] = {
                    'total': float(df[col].sum()),
                    'average': float(df[col].mean()),
                    'median': float(df[col].median()),
                    'max': float(df[col].max()),
                    'min': float(df[col].min()),
                    'std': float(df[col].std()),
                    'count': int(df[col].count()),
                    'growth_rate': self._calculate_growth_rate(df, col)
                }
        
        return kpis
    
    def _calculate_growth_rate(self, df, col):
        """Calculate growth rate for a column"""
        try:
            if len(df) < 2:
                return 0
            
            # If there's a date column, use it for chronological order
            date_cols = df.select_dtypes(include=['datetime64']).columns
            if len(date_cols) > 0:
                df_sorted = df.sort_values(date_cols[0])
            else:
                df_sorted = df
            
            first_half = df_sorted[col].iloc[:len(df_sorted)//2].mean()
            second_half = df_sorted[col].iloc[len(df_sorted)//2:].mean()
            
            if first_half != 0:
                growth_rate = ((second_half - first_half) / first_half) * 100
                return round(growth_rate, 2)
            return 0
        except:
            return 0
    
    def _create_time_series_charts(self, df, date_col):
        """Create comprehensive time series charts"""
        charts = []
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        df_sorted = df.sort_values(date_col)
        
        for col in numeric_cols[:3]:  # Top 3 numeric columns
            # Daily trend
            fig = px.line(df_sorted, x=date_col, y=col, 
                         title=f'{col.replace("_", " ").title()} Trend Over Time',
                         color_discrete_sequence=self.color_schemes['primary'])
            
            # Add moving average
            df_sorted[f'{col}_ma7'] = df_sorted[col].rolling(window=7, min_periods=1).mean()
            fig.add_scatter(x=df_sorted[date_col], y=df_sorted[f'{col}_ma7'],
                          mode='lines', name='7-day Moving Average',
                          line=dict(dash='dash', color='orange'))
            
            # Enhance layout
            fig.update_layout(
                template='plotly_dark',
                hovermode='x unified',
                showlegend=True,
                xaxis_title='Date',
                yaxis_title=col.replace('_', ' ').title()
            )
            
            charts.append({
                'type': 'time_series',
                'column': col,
                'chart': json.loads(fig.to_json())
            })
        
        return charts
    
    def _create_revenue_trend_chart(self, df, date_col):
        """Create specific revenue trend chart"""
        # Look for revenue/sales columns
        value_cols = [col for col in df.columns if any(keyword in col.lower() 
                     for keyword in ['sales', 'revenue', 'profit', 'amount'])]
        
        if not value_cols:
            value_cols = df.select_dtypes(include=['number']).columns[:1]
        
        if len(value_cols) == 0:
            return None
        
        value_col = value_cols[0]
        df_sorted = df.sort_values(date_col)
        
        # Create subplot with secondary y-axis
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=[f'{value_col.replace("_", " ").title()} Trend', 'Monthly Summary'],
            vertical_spacing=0.15,
            row_heights=[0.7, 0.3]
        )
        
        # Main trend line
        fig.add_trace(
            go.Scatter(x=df_sorted[date_col], y=df_sorted[value_col],
                      mode='lines+markers', name=value_col.replace('_', ' ').title(),
                      line=dict(color='#3b82f6', width=3),
                      marker=dict(size=4)),
            row=1, col=1
        )
        
        # Monthly aggregation
        df_sorted['month_year'] = df_sorted[date_col].dt.to_period('M')
        monthly_data = df_sorted.groupby('month_year')[value_col].sum().reset_index()
        monthly_data['month_year_str'] = monthly_data['month_year'].astype(str)
        
        fig.add_trace(
            go.Bar(x=monthly_data['month_year_str'], y=monthly_data[value_col],
                  name='Monthly Total', marker_color='#10b981'),
            row=2, col=1
        )
        
        fig.update_layout(
            template='plotly_dark',
            showlegend=True,
            height=600
        )
        
        return json.loads(fig.to_json())
    
    def _create_distribution_charts(self, df, numeric_cols):
        """Create distribution charts for numeric columns"""
        charts = []
        
        for col in numeric_cols[:4]:  # Limit to 4 charts
            # Histogram with KDE
            fig = px.histogram(df, x=col, nbins=30, 
                             title=f'Distribution of {col.replace("_", " ").title()}',
                             color_discrete_sequence=self.color_schemes['primary'])
            
            # Add statistics annotations
            mean_val = df[col].mean()
            median_val = df[col].median()
            
            fig.add_vline(x=mean_val, line_dash="dash", line_color="red", 
                         annotation_text=f"Mean: {mean_val:.2f}")
            fig.add_vline(x=median_val, line_dash="dash", line_color="green", 
                         annotation_text=f"Median: {median_val:.2f}")
            
            fig.update_layout(
                template='plotly_dark',
                showlegend=False,
                xaxis_title=col.replace('_', ' ').title(),
                yaxis_title='Frequency'
            )
            
            charts.append({
                'type': 'histogram',
                'column': col,
                'chart': json.loads(fig.to_json())
            })
        
        return charts
    
    def _create_categorical_charts(self, df, categorical_cols):
        """Create comprehensive categorical analysis charts"""
        charts = []
        
        for col in categorical_cols[:3]:  # Limit to 3 categorical columns
            value_counts = df[col].value_counts().head(10)
            
            # Pie chart
            fig_pie = px.pie(values=value_counts.values, names=value_counts.index,
                           title=f'{col.replace("_", " ").title()} Distribution',
                           color_discrete_sequence=self.color_schemes['gradient'])
            
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            fig_pie.update_layout(template='plotly_dark')
            
            charts.append({
                'type': 'pie',
                'column': col,
                'chart': json.loads(fig_pie.to_json())
            })
            
            # Bar chart
            fig_bar = px.bar(x=value_counts.index, y=value_counts.values,
                           title=f'{col.replace("_", " ").title()} Count',
                           color=value_counts.values,
                           color_continuous_scale='viridis')
            
            fig_bar.update_layout(
                template='plotly_dark',
                xaxis_title=col.replace('_', ' ').title(),
                yaxis_title='Count',
                showlegend=False
            )
            
            charts.append({
                'type': 'bar',
                'column': col,
                'chart': json.loads(fig_bar.to_json())
            })
        
        return charts
    
    def _create_correlation_chart(self, df, numeric_cols):
        """Create enhanced correlation heatmap"""
        corr_matrix = df[numeric_cols].corr()
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        corr_matrix_masked = corr_matrix.mask(mask)
        
        fig = px.imshow(corr_matrix_masked,
                       title="Correlation Heatmap",
                       color_continuous_scale="RdBu_r",
                       aspect="auto",
                       text_auto=True)
        
        fig.update_layout(
            template='plotly_dark',
            width=600,
            height=600
        )
        
        # Add annotations for strong correlations
        annotations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:
                    annotations.append(
                        dict(x=j, y=i, text=f"{corr_val:.2f}",
                             showarrow=False, font=dict(color="white", size=12))
                    )
        
        fig.update_layout(annotations=annotations)
        
        return json.loads(fig.to_json())
    
    def _create_advanced_business_charts(self, df):
        """Create advanced business-specific charts"""
        charts = {}
        
        # Monthly performance analysis
        if 'Date' in df.columns:
            monthly_chart = self._create_monthly_performance_chart(df)
            if monthly_chart:
                charts['monthly_performance'] = monthly_chart
        
        # Product category performance
        if 'Product_Category' in df.columns and 'Sales_Amount' in df.columns:
            category_chart = self._create_category_performance_chart(df)
            if category_chart:
                charts['category_performance'] = category_chart
        
        # Regional analysis
        if 'Region' in df.columns:
            regional_chart = self._create_regional_analysis_chart(df)
            if regional_chart:
                charts['regional_analysis'] = regional_chart
        
        # Customer analysis
        if 'Customer_ID' in df.columns:
            customer_chart = self._create_customer_analysis_chart(df)
            if customer_chart:
                charts['customer_analysis'] = customer_chart
        
        return charts
    
    def _create_monthly_performance_chart(self, df):
        """Create monthly performance analysis"""
        if 'Date' not in df.columns:
            return None
        
        df = df.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        df['Month'] = df['Date'].dt.strftime('%B')
        df['Year'] = df['Date'].dt.year
        
        # Find the best numeric column for analysis
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) == 0:
            return None
        
        value_col = numeric_cols[0]
        if 'Sales_Amount' in numeric_cols:
            value_col = 'Sales_Amount'
        elif 'Profit' in numeric_cols:
            value_col = 'Profit'
        
        # Monthly aggregation
        monthly_data = df.groupby(['Month', 'Year'])[value_col].sum().reset_index()
        monthly_avg = df.groupby('Month')[value_col].mean().reset_index()
        
        # Create subplot
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=[f'Monthly {value_col.replace("_", " ").title()}', 'Monthly Average'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Monthly trend by year
        for year in monthly_data['Year'].unique():
            year_data = monthly_data[monthly_data['Year'] == year]
            fig.add_trace(
                go.Scatter(x=year_data['Month'], y=year_data[value_col],
                          mode='lines+markers', name=str(year)),
                row=1, col=1
            )
        
        # Monthly averages
        month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                      'July', 'August', 'September', 'October', 'November', 'December']
        monthly_avg['Month'] = pd.Categorical(monthly_avg['Month'], categories=month_order, ordered=True)
        monthly_avg = monthly_avg.sort_values('Month')
        
        fig.add_trace(
            go.Bar(x=monthly_avg['Month'], y=monthly_avg[value_col],
                  name='Average', marker_color='lightblue'),
            row=1, col=2
        )
        
        fig.update_layout(
            template='plotly_dark',
            showlegend=True,
            height=400
        )
        
        return json.loads(fig.to_json())
    
    def _create_category_performance_chart(self, df):
        """Create product category performance chart"""
        category_performance = df.groupby('Product_Category').agg({
            'Sales_Amount': ['sum', 'mean', 'count'],
            'Profit': 'sum' if 'Profit' in df.columns else 'Sales_Amount'
        }).round(2)
        
        category_performance.columns = ['Total_Sales', 'Avg_Sales', 'Transaction_Count', 'Total_Profit']
        category_performance = category_performance.reset_index()
        
        # Create bubble chart
        fig = px.scatter(category_performance, 
                        x='Total_Sales', y='Total_Profit',
                        size='Transaction_Count',
                        color='Product_Category',
                        hover_name='Product_Category',
                        title='Product Category Performance Analysis',
                        labels={'Total_Sales': 'Total Sales', 'Total_Profit': 'Total Profit'})
        
        fig.update_layout(
            template='plotly_dark',
            showlegend=True
        )
        
        return json.loads(fig.to_json())
    
    def _create_regional_analysis_chart(self, df):
        """Create regional analysis chart"""
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) == 0:
            return None
        
        value_col = numeric_cols[0]
        if 'Sales_Amount' in numeric_cols:
            value_col = 'Sales_Amount'
        
        regional_data = df.groupby('Region')[value_col].agg(['sum', 'mean', 'count']).reset_index()
        regional_data.columns = ['Region', 'Total', 'Average', 'Count']
        
        # Create horizontal bar chart
        fig = px.bar(regional_data, x='Total', y='Region', orientation='h',
                    title=f'Regional {value_col.replace("_", " ").title()} Analysis',
                    color='Total', color_continuous_scale='viridis')
        
        fig.update_layout(
            template='plotly_dark',
            yaxis={'categoryorder': 'total ascending'}
        )
        
        return json.loads(fig.to_json())
    
    def _create_customer_analysis_chart(self, df):
        """Create customer analysis chart"""
        if 'Sales_Amount' not in df.columns:
            return None
        
        customer_data = df.groupby('Customer_ID')['Sales_Amount'].agg(['sum', 'count']).reset_index()
        customer_data.columns = ['Customer_ID', 'Total_Sales', 'Transaction_Count']
        
        # Customer segmentation
        customer_data['Segment'] = pd.cut(customer_data['Total_Sales'], 
                                        bins=[0, customer_data['Total_Sales'].quantile(0.33),
                                             customer_data['Total_Sales'].quantile(0.67),
                                             customer_data['Total_Sales'].max()],
                                        labels=['Low Value', 'Medium Value', 'High Value'])
        
        # Create scatter plot
        fig = px.scatter(customer_data, x='Transaction_Count', y='Total_Sales',
                        color='Segment', title='Customer Segmentation Analysis',
                        hover_data=['Customer_ID'])
        
        fig.update_layout(
            template='plotly_dark',
            xaxis_title='Number of Transactions',
            yaxis_title='Total Sales'
        )
        
        return json.loads(fig.to_json())
    
    def _create_performance_metrics(self, df):
        """Create performance metrics summary"""
        metrics = {}
        
        # Data overview metrics
        metrics['data_overview'] = {
            'total_records': len(df),
            'total_columns': len(df.columns),
            'date_range': self._get_date_range(df),
            'data_quality_score': self._calculate_data_quality_score(df)
        }
        
        # Business metrics
        if 'Sales_Amount' in df.columns:
            metrics['business_metrics'] = {
                'total_revenue': float(df['Sales_Amount'].sum()),
                'average_transaction': float(df['Sales_Amount'].mean()),
                'revenue_growth': self._calculate_growth_rate(df, 'Sales_Amount'),
                'top_sales_day': self._get_top_sales_day(df)
            }
        
        # Customer metrics
        if 'Customer_ID' in df.columns:
            metrics['customer_metrics'] = {
                'total_customers': df['Customer_ID'].nunique(),
                'average_customer_value': float(df.groupby('Customer_ID')['Sales_Amount'].sum().mean()) if 'Sales_Amount' in df.columns else 0,
                'repeat_customers': self._calculate_repeat_customers(df)
            }
        
        return metrics
    
    def _get_date_range(self, df):
        """Get date range from dataframe"""
        date_cols = df.select_dtypes(include=['datetime64']).columns
        if len(date_cols) > 0:
            date_col = date_cols[0]
            return {
                'start': str(df[date_col].min()),
                'end': str(df[date_col].max()),
                'days': (df[date_col].max() - df[date_col].min()).days
            }
        return None
    
    def _calculate_data_quality_score(self, df):
        """Calculate overall data quality score"""
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isnull().sum().sum()
        completeness = ((total_cells - missing_cells) / total_cells) * 100
        
        duplicate_rows = df.duplicated().sum()
        uniqueness = ((len(df) - duplicate_rows) / len(df)) * 100
        
        return round((completeness + uniqueness) / 2, 2)
    
    def _get_top_sales_day(self, df):
        """Get the day with highest sales"""
        if 'Date' not in df.columns or 'Sales_Amount' not in df.columns:
            return None
        
        daily_sales = df.groupby('Date')['Sales_Amount'].sum()
        top_day = daily_sales.idxmax()
        top_sales = daily_sales.max()
        
        return {
            'date': str(top_day),
            'sales': float(top_sales)
        }
    
    def _calculate_repeat_customers(self, df):
        """Calculate percentage of repeat customers"""
        if 'Customer_ID' not in df.columns:
            return 0
        
        customer_transactions = df.groupby('Customer_ID').size()
        repeat_customers = (customer_transactions > 1).sum()
        total_customers = len(customer_transactions)
        
        return round((repeat_customers / total_customers) * 100, 2)