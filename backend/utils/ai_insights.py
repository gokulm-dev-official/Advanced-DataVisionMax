import openai
import pandas as pd
import json
import os
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

class AIInsights:
    def __init__(self):
        openai.api_key = os.getenv('OPENAI_API_KEY')
    
    def generate_comprehensive_insights(self, df):
        """Generate comprehensive AI-powered insights"""
        try:
            # Analyze the data comprehensively
            data_analysis = self._analyze_data_patterns(df)
            
            prompt = f"""
            As a senior business analyst, analyze this dataset and provide detailed insights:
            
            Dataset Analysis:
            {json.dumps(data_analysis, indent=2)}
            
            Provide comprehensive insights in the following format:
            
            1. EXECUTIVE SUMMARY (3-4 key points)
            2. BUSINESS INSIGHTS (5-6 detailed findings)
            3. SEASONAL TRENDS (specific patterns by month/quarter)
            4. PRODUCT PERFORMANCE (if applicable - which products perform best when)
            5. REGIONAL ANALYSIS (if applicable - geographical patterns)
            6. RISK FACTORS (potential concerns)
            7. GROWTH OPPORTUNITIES (actionable recommendations)
            8. PREDICTIONS (specific forecasts with examples like "March will be best for Electronics")
            
            Be specific with numbers, percentages, and concrete examples.
            """
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1500,
                temperature=0.7
            )
            
            ai_response = response.choices[0].message.content
            
            # Structure the response
            insights = self._structure_insights(ai_response, df)
            
            return insights
            
        except Exception as e:
            return self._generate_enhanced_fallback_insights(df)
    
    def advanced_chat_query(self, df, query):
        """Handle advanced chat queries with data context"""
        try:
            # Analyze query intent
            query_context = self._analyze_query_intent(query, df)
            
            # Prepare comprehensive data context
            data_context = self._prepare_comprehensive_context(df)
            
            prompt = f"""
            You are an expert data analyst for DataVision-AI. Answer the user's question about their dataset.
            
            Dataset Context:
            {data_context}
            
            User Question: {query}
            
            Provide a detailed, accurate answer based on the actual data. If the question requires calculations, perform them and show the results. Be specific with numbers and insights.
            
            Format your response professionally and include relevant statistics when applicable.
            """
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=800,
                temperature=0.3
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            # Enhanced fallback with actual data analysis
            return self._generate_contextual_fallback(query, df)
    
    def _analyze_data_patterns(self, df):
        """Analyze data patterns for AI insights"""
        analysis = {
            'dataset_overview': {
                'total_records': len(df),
                'total_columns': len(df.columns),
                'date_range': None,
                'numeric_columns': list(df.select_dtypes(include=[np.number]).columns),
                'categorical_columns': list(df.select_dtypes(include=['object']).columns)
            }
        }
        
        # Date analysis
        date_cols = [col for col in df.columns if 'date' in col.lower() or df[col].dtype == 'datetime64[ns]']
        if date_cols:
            date_col = date_cols[0]
            analysis['date_range'] = {
                'start': str(df[date_col].min()),
                'end': str(df[date_col].max()),
                'span_days': (df[date_col].max() - df[date_col].min()).days
            }
            
            # Monthly patterns
            if len(df) > 30:
                df['month'] = pd.to_datetime(df[date_col]).dt.month
                monthly_stats = {}
                
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                for col in numeric_cols[:3]:  # Top 3 numeric columns
                    monthly_avg = df.groupby('month')[col].mean()
                    monthly_stats[col] = {
                        'best_month': int(monthly_avg.idxmax()),
                        'best_month_value': float(monthly_avg.max()),
                        'worst_month': int(monthly_avg.idxmin()),
                        'worst_month_value': float(monthly_avg.min())
                    }
                
                analysis['monthly_patterns'] = monthly_stats
        
        # Categorical analysis
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            analysis['categorical_insights'] = {}
            for col in categorical_cols[:3]:  # Top 3 categorical columns
                value_counts = df[col].value_counts().head(5)
                analysis['categorical_insights'][col] = {
                    'top_categories': value_counts.to_dict(),
                    'total_categories': df[col].nunique()
                }
        
        # Numeric analysis
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            analysis['numeric_insights'] = {}
            for col in numeric_cols[:5]:  # Top 5 numeric columns
                analysis['numeric_insights'][col] = {
                    'mean': float(df[col].mean()),
                    'median': float(df[col].median()),
                    'std': float(df[col].std()),
                    'min': float(df[col].min()),
                    'max': float(df[col].max()),
                    'growth_trend': self._calculate_trend(df, col, date_cols[0] if date_cols else None)
                }
        
        return analysis
    
    def _calculate_trend(self, df, numeric_col, date_col):
        """Calculate trend for numeric column over time"""
        if date_col is None or len(df) < 10:
            return "insufficient_data"
        
        try:
            # Simple linear trend calculation
            df_sorted = df.sort_values(date_col)
            x = np.arange(len(df_sorted))
            y = df_sorted[numeric_col].values
            
            # Calculate correlation coefficient
            correlation = np.corrcoef(x, y)[0, 1]
            
            if correlation > 0.3:
                return "increasing"
            elif correlation < -0.3:
                return "decreasing"
            else:
                return "stable"
        except:
            return "unknown"
    
    def _structure_insights(self, ai_response, df):
        """Structure AI response into organized insights"""
        insights = {
            'executive_summary': [],
            'business_insights': [],
            'seasonal_trends': [],
            'product_performance': [],
            'regional_analysis': [],
            'risk_factors': [],
            'growth_opportunities': [],
            'specific_predictions': []
        }
        
        # Parse AI response and categorize
        lines = ai_response.split('\n')
        current_section = 'business_insights'
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Detect section headers
            if 'EXECUTIVE SUMMARY' in line.upper():
                current_section = 'executive_summary'
            elif 'BUSINESS INSIGHTS' in line.upper():
                current_section = 'business_insights'
            elif 'SEASONAL' in line.upper() or 'TRENDS' in line.upper():
                current_section = 'seasonal_trends'
            elif 'PRODUCT' in line.upper():
                current_section = 'product_performance'
            elif 'REGIONAL' in line.upper():
                current_section = 'regional_analysis'
            elif 'RISK' in line.upper():
                current_section = 'risk_factors'
            elif 'GROWTH' in line.upper() or 'OPPORTUNITIES' in line.upper():
                current_section = 'growth_opportunities'
            elif 'PREDICTION' in line.upper():
                current_section = 'specific_predictions'
            elif line.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '-', '•')):
                insights[current_section].append(line)
        
        # Add data-driven insights
        self._add_data_driven_insights(insights, df)
        
        return insights
    
    def _add_data_driven_insights(self, insights, df):
        """Add specific data-driven insights"""
        try:
            # Add specific business insights based on actual data
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if 'Sales_Amount' in df.columns or 'sales' in str(df.columns).lower():
                sales_col = 'Sales_Amount' if 'Sales_Amount' in df.columns else [col for col in df.columns if 'sales' in col.lower()][0]
                total_sales = df[sales_col].sum()
                avg_sales = df[sales_col].mean()
                
                insights['business_insights'].append(f"Total sales revenue: ${total_sales:,.2f} with average transaction of ${avg_sales:.2f}")
            
            # Product category insights
            if 'Product_Category' in df.columns:
                product_sales = df.groupby('Product_Category')['Sales_Amount'].sum().sort_values(ascending=False)
                top_product = product_sales.index[0]
                top_sales = product_sales.iloc[0]
                
                insights['product_performance'].append(f"{top_product} is the top-performing category with ${top_sales:,.2f} in total sales")
            
            # Monthly insights
            if 'Date' in df.columns:
                df['Month_Name'] = pd.to_datetime(df['Date']).dt.strftime('%B')
                monthly_sales = df.groupby('Month_Name')['Sales_Amount'].sum()
                best_month = monthly_sales.idxmax()
                best_sales = monthly_sales.max()
                
                insights['seasonal_trends'].append(f"{best_month} shows the highest sales performance with ${best_sales:,.2f}")
                
                # Specific product-month insights
                if 'Product_Category' in df.columns:
                    for product in df['Product_Category'].unique()[:3]:
                        product_monthly = df[df['Product_Category'] == product].groupby('Month_Name')['Sales_Amount'].sum()
                        if not product_monthly.empty:
                            best_month_product = product_monthly.idxmax()
                            insights['specific_predictions'].append(f"{best_month_product} is the optimal month for {product} sales")
            
        except Exception as e:
            pass
    
    def _prepare_comprehensive_context(self, df):
        """Prepare comprehensive context for chat queries"""
        context = {
            'dataset_summary': {
                'total_rows': len(df),
                'total_columns': len(df.columns),
                'columns': list(df.columns)
            }
        }
        
        # Add sample data
        context['sample_data'] = df.head(3).to_dict('records')
        
        # Add statistical summary
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            context['numeric_summary'] = {}
            for col in numeric_cols:
                context['numeric_summary'][col] = {
                    'mean': float(df[col].mean()),
                    'max': float(df[col].max()),
                    'min': float(df[col].min()),
                    'total': float(df[col].sum())
                }
        
        return json.dumps(context, indent=2)
    
    def _analyze_query_intent(self, query, df):
        """Analyze the intent of user query"""
        query_lower = query.lower()
        
        intent = {
            'type': 'general',
            'columns_mentioned': [],
            'aggregation': None,
            'time_based': False
        }
        
        # Check for specific columns mentioned
        for col in df.columns:
            if col.lower() in query_lower:
                intent['columns_mentioned'].append(col)
        
        # Check for aggregation intent
        if any(word in query_lower for word in ['total', 'sum', 'average', 'mean', 'max', 'min', 'count']):
            intent['aggregation'] = 'summary'
        
        # Check for time-based queries
        if any(word in query_lower for word in ['month', 'year', 'date', 'time', 'when', 'trend']):
            intent['time_based'] = True
        
        return intent
    
    def _generate_contextual_fallback(self, query, df):
        """Generate contextual fallback response based on actual data"""
        try:
            query_lower = query.lower()
            
            # Try to answer based on data
            if 'total' in query_lower or 'sum' in query_lower:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    totals = []
                    for col in numeric_cols[:3]:
                        total = df[col].sum()
                        totals.append(f"{col}: {total:,.2f}")
                    return f"Based on your data, here are the key totals: {', '.join(totals)}"
            
            elif 'average' in query_lower or 'mean' in query_lower:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    averages = []
                    for col in numeric_cols[:3]:
                        avg = df[col].mean()
                        averages.append(f"{col}: {avg:.2f}")
                    return f"Average values in your dataset: {', '.join(averages)}"
            
            elif 'highest' in query_lower or 'maximum' in query_lower or 'best' in query_lower:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    col = numeric_cols[0]
                    max_val = df[col].max()
                    return f"The highest value in {col} is {max_val:,.2f}"
            
            elif 'month' in query_lower and 'Date' in df.columns:
                df_temp = df.copy()
                df_temp['Month'] = pd.to_datetime(df_temp['Date']).dt.strftime('%B')
                monthly_summary = df_temp.groupby('Month').size()
                best_month = monthly_summary.idxmax()
                return f"Based on your data, {best_month} shows the highest activity with {monthly_summary.max()} records."
            
            else:
                return f"I can see your dataset has {len(df)} records with {len(df.columns)} columns: {', '.join(df.columns[:5])}{'...' if len(df.columns) > 5 else ''}. Could you please be more specific about what you'd like to know?"
                
        except Exception as e:
            return "I'm analyzing your data. Could you please rephrase your question or be more specific about what you'd like to know?"
    
    def _generate_enhanced_fallback_insights(self, df):
        """Generate enhanced fallback insights without AI"""
        insights = {
            'executive_summary': [],
            'business_insights': [],
            'seasonal_trends': [],
            'product_performance': [],
            'regional_analysis': [],
            'risk_factors': [],
            'growth_opportunities': [],
            'specific_predictions': []
        }
        
        # Executive Summary
        insights['executive_summary'] = [
            f"Dataset contains {len(df):,} records across {len(df.columns)} dimensions",
            f"Data spans from {df.select_dtypes(include=['datetime64']).min().min() if len(df.select_dtypes(include=['datetime64']).columns) > 0 else 'N/A'} to {df.select_dtypes(include=['datetime64']).max().max() if len(df.select_dtypes(include=['datetime64']).columns) > 0 else 'N/A'}",
            "Comprehensive analysis reveals key business patterns and opportunities"
        ]
        
        # Business Insights
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols[:3]:
            total = df[col].sum()
            avg = df[col].mean()
            insights['business_insights'].append(f"{col}: Total of {total:,.2f} with average of {avg:.2f}")
        
        # Seasonal Trends (if date data exists)
        date_cols = [col for col in df.columns if 'date' in col.lower() or df[col].dtype == 'datetime64[ns]']
        if date_cols and len(numeric_cols) > 0:
            date_col = date_cols[0]
            df_temp = df.copy()
            df_temp['Month'] = pd.to_datetime(df_temp[date_col]).dt.month
            
            for col in numeric_cols[:2]:
                monthly_avg = df_temp.groupby('Month')[col].mean()
                best_month = monthly_avg.idxmax()
                insights['seasonal_trends'].append(f"Month {best_month} shows peak performance for {col}")
        
        # Product Performance
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0 and len(numeric_cols) > 0:
            cat_col = categorical_cols[0]
            num_col = numeric_cols[0]
            
            category_performance = df.groupby(cat_col)[num_col].sum().sort_values(ascending=False)
            top_category = category_performance.index[0]
            insights['product_performance'].append(f"{top_category} leads in {num_col} performance")
        
        # Growth Opportunities
        insights['growth_opportunities'] = [
            "Focus on high-performing periods identified in seasonal analysis",
            "Optimize resource allocation based on regional performance data",
            "Implement predictive analytics for demand forecasting"
        ]
        
        return insights