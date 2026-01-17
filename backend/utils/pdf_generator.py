from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.graphics.shapes import Drawing, Rect
from reportlab.graphics.charts.linecharts import HorizontalLineChart
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics import renderPDF
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
import os
import io
import base64
from datetime import datetime
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

class PDFGenerator:
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self.setup_custom_styles()
        os.makedirs("reports", exist_ok=True)
    
    def setup_custom_styles(self):
        """Setup custom styles for professional report"""
        # Title style
        self.title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=28,
            spaceAfter=30,
            textColor=colors.HexColor('#1e40af'),
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        )
        
        # Subtitle style
        self.subtitle_style = ParagraphStyle(
            'CustomSubtitle',
            parent=self.styles['Heading2'],
            fontSize=18,
            spaceAfter=20,
            textColor=colors.HexColor('#3b82f6'),
            alignment=TA_LEFT,
            fontName='Helvetica-Bold'
        )
        
        # Section header style
        self.section_style = ParagraphStyle(
            'SectionHeader',
            parent=self.styles['Heading3'],
            fontSize=14,
            spaceAfter=12,
            spaceBefore=20,
            textColor=colors.HexColor('#1f2937'),
            fontName='Helvetica-Bold',
            backColor=colors.HexColor('#f3f4f6'),
            borderPadding=8
        )
        
        # Body text style
        self.body_style = ParagraphStyle(
            'CustomBody',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceAfter=8,
            textColor=colors.HexColor('#374151'),
            fontName='Helvetica',
            alignment=TA_JUSTIFY,
            leading=14
        )
        
        # Highlight style
        self.highlight_style = ParagraphStyle(
            'Highlight',
            parent=self.styles['Normal'],
            fontSize=12,
            spaceAfter=10,
            textColor=colors.HexColor('#059669'),
            fontName='Helvetica-Bold',
            backColor=colors.HexColor('#ecfdf5'),
            borderPadding=6
        )
        
        # Warning style
        self.warning_style = ParagraphStyle(
            'Warning',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceAfter=10,
            textColor=colors.HexColor('#dc2626'),
            fontName='Helvetica-Bold',
            backColor=colors.HexColor('#fef2f2'),
            borderPadding=6
        )
    
    def create_professional_report(self, df, charts, insights, predictions, anomalies, cleaning_report, filename):
        """Create comprehensive professional PDF report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_path = f"reports/DataVision_AI_Professional_Report_{timestamp}.pdf"
        
        doc = SimpleDocTemplate(
            pdf_path, 
            pagesize=A4,
            topMargin=0.5*inch,
            bottomMargin=0.5*inch,
            leftMargin=0.75*inch,
            rightMargin=0.75*inch
        )
        
        story = []
        
        # Add cover page
        story.extend(self._create_cover_page(filename, df))
        story.append(PageBreak())
        
        # Add executive summary
        story.extend(self._create_executive_summary(df, insights, cleaning_report))
        story.append(PageBreak())
        
        # Add table of contents
        story.extend(self._create_table_of_contents())
        story.append(PageBreak())
        
        # Add dataset overview
        story.extend(self._create_dataset_overview(df, cleaning_report))
        story.append(PageBreak())
        
        # Add data quality assessment
        story.extend(self._create_data_quality_section(cleaning_report))
        story.append(PageBreak())
        
        # Add key insights
        story.extend(self._create_insights_section(insights))
        story.append(PageBreak())
        
        # Add business analytics
        story.extend(self._create_business_analytics_section(df, charts))
        story.append(PageBreak())
        
        # Add predictive analysis
        story.extend(self._create_predictive_analysis_section(predictions))
        story.append(PageBreak())
        
        # Add risk assessment
        story.extend(self._create_risk_assessment_section(anomalies))
        story.append(PageBreak())
        
        # Add recommendations
        story.extend(self._create_recommendations_section(insights, predictions))
        story.append(PageBreak())
        
        # Add appendix
        story.extend(self._create_appendix_section(df))
        
        # Build PDF
        doc.build(story)
        return pdf_path
    
    def _create_cover_page(self, filename, df):
        """Create professional cover page"""
        cover_elements = []
        
        # Add logo placeholder (you can add actual logo here)
        cover_elements.append(Spacer(1, 50))
        
        # Main title
        cover_elements.append(Paragraph("DataVision-AI", self.title_style))
        cover_elements.append(Spacer(1, 10))
        
        # Subtitle
        subtitle_text = "Professional Analytics Report"
        cover_elements.append(Paragraph(subtitle_text, self.subtitle_style))
        cover_elements.append(Spacer(1, 30))
        
        # Dataset information box
        dataset_info = f"""
        <para align="center">
        <b>Dataset Analysis</b><br/>
        <br/>
        <b>Source:</b> {filename}<br/>
        <b>Records:</b> {len(df):,}<br/>
        <b>Columns:</b> {len(df.columns)}<br/>
        <b>Analysis Date:</b> {datetime.now().strftime('%B %d, %Y')}<br/>
        </para>
        """
        
        info_style = ParagraphStyle(
            'DatasetInfo',
            parent=self.body_style,
            fontSize=12,
            alignment=TA_CENTER,
            backColor=colors.HexColor('#f8fafc'),
            borderColor=colors.HexColor('#3b82f6'),
            borderWidth=2,
            borderPadding=20
        )
        
        cover_elements.append(Paragraph(dataset_info, info_style))
        cover_elements.append(Spacer(1, 50))
        
        # Company information
        company_info = """
        <para align="center">
        <b>Powered by DataVision-AI</b><br/>
        Created by GOKUL M<br/>
        Advanced Analytics & Business Intelligence Platform<br/>
        <br/>
        <i>Transforming Data into Actionable Insights</i>
        </para>
        """
        
        cover_elements.append(Paragraph(company_info, self.body_style))
        cover_elements.append(Spacer(1, 100))
        
        # Add disclaimer
        disclaimer = """
        <para align="center" fontSize="8" textColor="#6b7280">
        This report is generated automatically by DataVision-AI platform.<br/>
        All analysis and insights are based on the provided dataset.<br/>
        Please verify critical business decisions with domain experts.
        </para>
        """
        
        cover_elements.append(Paragraph(disclaimer, self.body_style))
        
        return cover_elements
    
    def _create_table_of_contents(self):
        """Create table of contents"""
        toc_elements = []
        
        toc_elements.append(Paragraph("Table of Contents", self.subtitle_style))
        toc_elements.append(Spacer(1, 20))
        
        toc_items = [
            ("1. Executive Summary", "Page 3"),
            ("2. Dataset Overview", "Page 4"),
            ("3. Data Quality Assessment", "Page 5"),
            ("4. Key Business Insights", "Page 6"),
            ("5. Business Analytics", "Page 7"),
            ("6. Predictive Analysis", "Page 8"),
            ("7. Risk Assessment", "Page 9"),
            ("8. Strategic Recommendations", "Page 10"),
            ("9. Technical Appendix", "Page 11")
        ]
        
        toc_data = [['Section', 'Page']]
        for item, page in toc_items:
            toc_data.append([item, page])
        
        toc_table = Table(toc_data, colWidths=[4*inch, 1*inch])
        toc_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3b82f6')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8fafc')),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e5e7eb')),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f9fafb')])
        ]))
        
        toc_elements.append(toc_table)
        
        return toc_elements
    
    def _create_executive_summary(self, df, insights, cleaning_report):
        """Create executive summary section"""
        summary_elements = []
        
        summary_elements.append(Paragraph("1. Executive Summary", self.subtitle_style))
        summary_elements.append(Spacer(1, 20))
        
        # Key metrics overview
        key_metrics_text = f"""
        This comprehensive analysis of the dataset reveals significant business insights and opportunities. 
        The dataset contains {len(df):,} records across {len(df.columns)} dimensions, with a data quality 
        score of {cleaning_report.get('data_quality_score', 85)}%.
        """
        
        summary_elements.append(Paragraph(key_metrics_text, self.body_style))
        summary_elements.append(Spacer(1, 15))
        
        # Key findings
        summary_elements.append(Paragraph("Key Findings:", self.section_style))
        
        executive_insights = insights.get('executive_summary', [])
        if not executive_insights:
            executive_insights = insights.get('business_insights', [])[:3]
        
        for i, insight in enumerate(executive_insights[:5], 1):
            insight_text = f"{i}. {insight}"
            summary_elements.append(Paragraph(insight_text, self.body_style))
            summary_elements.append(Spacer(1, 8))
        
        # Business impact
        summary_elements.append(Spacer(1, 20))
        summary_elements.append(Paragraph("Business Impact:", self.section_style))
        
        impact_text = """
        The analysis identifies key performance drivers, seasonal trends, and growth opportunities that can 
        inform strategic decision-making. Specific attention should be paid to the identified high-performing 
        segments and potential risk factors highlighted in the detailed analysis.
        """
        
        summary_elements.append(Paragraph(impact_text, self.highlight_style))
        
        return summary_elements
    
    def _create_dataset_overview(self, df, cleaning_report):
        """Create dataset overview section"""
        overview_elements = []
        
        overview_elements.append(Paragraph("2. Dataset Overview", self.subtitle_style))
        overview_elements.append(Spacer(1, 20))
        
        # Basic statistics table
        basic_stats = [
            ['Metric', 'Value', 'Description'],
            ['Total Records', f"{len(df):,}", 'Number of data points analyzed'],
            ['Total Columns', str(len(df.columns)), 'Number of data dimensions'],
            ['Missing Values', str(df.isnull().sum().sum()), 'Total missing data points'],
            ['Duplicate Records', str(df.duplicated().sum()), 'Duplicate entries identified'],
            ['Memory Usage', f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB", 'Dataset size in memory'],
            ['Data Quality Score', f"{cleaning_report.get('data_quality_score', 85)}%", 'Overall quality assessment']
        ]
        
        stats_table = Table(basic_stats, colWidths=[2*inch, 1.5*inch, 2.5*inch])
        stats_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e40af')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('ALIGN', (1, 0), (1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8fafc')),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e5e7eb')),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f9fafb')])
        ]))
        
        overview_elements.append(stats_table)
        overview_elements.append(Spacer(1, 20))
        
        # Column analysis
        overview_elements.append(Paragraph("Column Analysis:", self.section_style))
        
        # Numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        date_cols = df.select_dtypes(include=['datetime64']).columns
        
        column_summary = f"""
        <b>Numeric Columns:</b> {len(numeric_cols)} ({', '.join(numeric_cols[:5])}{'...' if len(numeric_cols) > 5 else ''})<br/>
        <b>Categorical Columns:</b> {len(categorical_cols)} ({', '.join(categorical_cols[:5])}{'...' if len(categorical_cols) > 5 else ''})<br/>
        <b>Date Columns:</b> {len(date_cols)} ({', '.join(date_cols[:3])}{'...' if len(date_cols) > 3 else ''})
        """
        
        overview_elements.append(Paragraph(column_summary, self.body_style))
        
        return overview_elements
    
    def _create_data_quality_section(self, cleaning_report):
        """Create data quality assessment section"""
        quality_elements = []
        
        quality_elements.append(Paragraph("3. Data Quality Assessment", self.subtitle_style))
        quality_elements.append(Spacer(1, 20))
        
        # Quality score
        quality_score = cleaning_report.get('data_quality_score', 85)
        score_color = colors.HexColor('#059669') if quality_score >= 80 else colors.HexColor('#dc2626') if quality_score < 60 else colors.HexColor('#d97706')
        
        score_text = f"""
        <para align="center" fontSize="18" textColor="{score_color}">
        <b>Overall Quality Score: {quality_score}%</b>
        </para>
        """
        
        quality_elements.append(Paragraph(score_text, self.body_style))
        quality_elements.append(Spacer(1, 20))
        
        # Data cleaning steps
        quality_elements.append(Paragraph("Data Cleaning Process:", self.section_style))
        
        cleaning_steps = cleaning_report.get('cleaning_steps', [])
        for step in cleaning_steps:
            quality_elements.append(Paragraph(f"• {step}", self.body_style))
            quality_elements.append(Spacer(1, 5))
        
        # Quality metrics table
        quality_elements.append(Spacer(1, 15))
        quality_elements.append(Paragraph("Quality Metrics:", self.section_style))
        
        original_shape = cleaning_report.get('original_shape', [0, 0])
        cleaned_shape = cleaning_report.get('cleaned_shape', [0, 0])
        
        quality_metrics = [
            ['Metric', 'Before Cleaning', 'After Cleaning', 'Improvement'],
            ['Rows', f"{original_shape[0]:,}", f"{cleaned_shape[0]:,}", f"{cleaned_shape[0] - original_shape[0]:+,}"],
            ['Columns', str(original_shape[1]), str(cleaned_shape[1]), f"{cleaned_shape[1] - original_shape[1]:+}"],
            ['Completeness', 'Variable', f"{quality_score}%", 'Optimized'],
        ]
        
        quality_table = Table(quality_metrics, colWidths=[1.5*inch, 1.5*inch, 1.5*inch, 1.5*inch])
        quality_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#059669')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f0fdf4')),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#bbf7d0')),
            ('FONTSIZE', (0, 1), (-1, -1), 9)
        ]))
        
        quality_elements.append(quality_table)
        
        # Recommendations
        quality_elements.append(Spacer(1, 20))
        quality_elements.append(Paragraph("Quality Recommendations:", self.section_style))
        
        recommendations = cleaning_report.get('recommendations', [])
        for rec in recommendations:
            quality_elements.append(Paragraph(f"• {rec}", self.body_style))
            quality_elements.append(Spacer(1, 5))
        
        return quality_elements
    
    def _create_insights_section(self, insights):
        """Create key insights section"""
        insights_elements = []
        
        insights_elements.append(Paragraph("4. Key Business Insights", self.subtitle_style))
        insights_elements.append(Spacer(1, 20))
        
        # Business insights
        if 'business_insights' in insights and insights['business_insights']:
            insights_elements.append(Paragraph("Business Intelligence:", self.section_style))
            for insight in insights['business_insights'][:5]:
                insights_elements.append(Paragraph(f"• {insight}", self.highlight_style))
                insights_elements.append(Spacer(1, 8))
        
        # Seasonal trends
        if 'seasonal_trends' in insights and insights['seasonal_trends']:
            insights_elements.append(Spacer(1, 15))
            insights_elements.append(Paragraph("Seasonal Patterns:", self.section_style))
            for trend in insights['seasonal_trends'][:3]:
                insights_elements.append(Paragraph(f"• {trend}", self.body_style))
                insights_elements.append(Spacer(1, 5))
        
        # Product performance
        if 'product_performance' in insights and insights['product_performance']:
            insights_elements.append(Spacer(1, 15))
            insights_elements.append(Paragraph("Product Performance:", self.section_style))
            for performance in insights['product_performance'][:3]:
                insights_elements.append(Paragraph(f"• {performance}", self.body_style))
                insights_elements.append(Spacer(1, 5))
        
        # Regional analysis
        if 'regional_analysis' in insights and insights['regional_analysis']:
            insights_elements.append(Spacer(1, 15))
            insights_elements.append(Paragraph("Regional Analysis:", self.section_style))
            for regional in insights['regional_analysis'][:3]:
                insights_elements.append(Paragraph(f"• {regional}", self.body_style))
                insights_elements.append(Spacer(1, 5))
        
        return insights_elements
    
    def _create_business_analytics_section(self, df, charts):
        """Create business analytics section"""
        analytics_elements = []
        
        analytics_elements.append(Paragraph("5. Business Analytics", self.subtitle_style))
        analytics_elements.append(Spacer(1, 20))
        
        # KPI Summary
        if 'kpis' in charts:
            analytics_elements.append(Paragraph("Key Performance Indicators:", self.section_style))
            
            kpi_data = [['Metric', 'Total', 'Average', 'Growth Rate']]
            for metric, values in list(charts['kpis'].items())[:5]:
                growth = values.get('growth_rate', 0)
                growth_display = f"{growth:+.1f}%" if growth != 0 else "N/A"
                kpi_data.append([
                    metric.replace('_', ' ').title(),
                    f"{values['total']:,.0f}",
                    f"{values['average']:,.2f}",
                    growth_display
                ])
            
            kpi_table = Table(kpi_data, colWidths=[2*inch, 1.5*inch, 1.5*inch, 1*inch])
            kpi_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3b82f6')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#eff6ff')),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#bfdbfe')),
                ('FONTSIZE', (0, 1), (-1, -1), 9),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8fafc')])
            ]))
            
            analytics_elements.append(kpi_table)
            analytics_elements.append(Spacer(1, 20))
        
        # Performance metrics
        if 'performance_metrics' in charts:
            metrics = charts['performance_metrics']
            
            analytics_elements.append(Paragraph("Performance Overview:", self.section_style))
            
            if 'business_metrics' in metrics:
                business_metrics = metrics['business_metrics']
                metrics_text = f"""
                <b>Total Revenue:</b> ${business_metrics.get('total_revenue', 0):,.2f}<br/>
                <b>Average Transaction:</b> ${business_metrics.get('average_transaction', 0):.2f}<br/>
                <b>Revenue Growth:</b> {business_metrics.get('revenue_growth', 0):+.1f}%<br/>
                """
                analytics_elements.append(Paragraph(metrics_text, self.body_style))
        
        return analytics_elements
    
    def _create_predictive_analysis_section(self, predictions):
        """Create predictive analysis section"""
        predictive_elements = []
        
        predictive_elements.append(Paragraph("6. Predictive Analysis", self.subtitle_style))
        predictive_elements.append(Spacer(1, 20))
        
        predictive_elements.append(Paragraph("Forecast Summary:", self.section_style))
        
        # Trend predictions table
        if predictions:
            trend_data = [['Metric', 'Trend Direction', 'Confidence', 'Next Period Forecast']]
            
            for metric, prediction in list(predictions.items())[:5]:
                if isinstance(prediction, dict):
                    trend = prediction.get('trend', 'Unknown')
                    confidence = prediction.get('confidence_score', 0)
                    next_pred = prediction.get('predictions', [0])[0] if prediction.get('predictions') else 0
                    
                    trend_data.append([
                        metric.replace('_', ' ').title(),
                        trend.title(),
                        f"{confidence:.1%}" if confidence <= 1 else f"{confidence:.1f}%",
                        f"{next_pred:.2f}"
                    ])
            
            if len(trend_data) > 1:
                trend_table = Table(trend_data, colWidths=[2*inch, 1.5*inch, 1*inch, 1.5*inch])
                trend_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#7c3aed')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#faf5ff')),
                    ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#c4b5fd')),
                    ('FONTSIZE', (0, 1), (-1, -1), 9),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f3f4f6')])
                ]))
                
                predictive_elements.append(trend_table)
                predictive_elements.append(Spacer(1, 20))
        
        # Business-specific predictions
        if 'product_seasonality' in predictions:
            predictive_elements.append(Paragraph("Product Seasonality Insights:", self.section_style))
            
            seasonality = predictions['product_seasonality']
            for product, data in list(seasonality.items())[:3]:
                insight_text = f"• {data.get('performance_prediction', f'{product} analysis')}"
                predictive_elements.append(Paragraph(insight_text, self.body_style))
                predictive_elements.append(Spacer(1, 5))
        
        # Model information
        predictive_elements.append(Spacer(1, 15))
        predictive_elements.append(Paragraph("Methodology:", self.section_style))
        
        methodology_text = """
        Predictions are generated using advanced machine learning algorithms including Linear Regression, 
        Random Forest, and time series analysis. Confidence scores indicate the reliability of each forecast 
        based on historical data patterns and statistical significance.
        """
        
        predictive_elements.append(Paragraph(methodology_text, self.body_style))
        
        return predictive_elements
    
    def _create_risk_assessment_section(self, anomalies):
        """Create risk assessment section"""
        risk_elements = []
        
        risk_elements.append(Paragraph("7. Risk Assessment", self.subtitle_style))
        risk_elements.append(Spacer(1, 20))
        
        if not anomalies or all(anomaly.get('count', 0) == 0 for anomaly in anomalies.values()):
            risk_elements.append(Paragraph("✅ No Critical Risks Identified", self.highlight_style))
            risk_elements.append(Spacer(1, 15))
            
            low_risk_text = """
            The analysis indicates that your data falls within normal operational ranges. 
            Continue monitoring key metrics for any emerging patterns that might require attention.
            """
            risk_elements.append(Paragraph(low_risk_text, self.body_style))
        else:
            risk_elements.append(Paragraph("Risk Factors Identified:", self.section_style))
            
            # Anomaly summary table
            anomaly_data = [['Risk Category', 'Anomalies Found', 'Severity', 'Recommendation']]
            
            for category, anomaly in anomalies.items():
                if anomaly.get('count', 0) > 0:
                    count = anomaly['count']
                    severity = 'High' if count > 20 else 'Medium' if count > 5 else 'Low'
                    severity_color = colors.HexColor('#dc2626') if severity == 'High' else colors.HexColor('#d97706') if severity == 'Medium' else colors.HexColor('#059669')
                    
                    recommendation = self._get_anomaly_recommendation(category, count)
                    
                    anomaly_data.append([
                        category.replace('_', ' ').title(),
                        str(count),
                        severity,
                        recommendation
                    ])
            
            if len(anomaly_data) > 1:
                anomaly_table = Table(anomaly_data, colWidths=[1.5*inch, 1*inch, 1*inch, 2.5*inch])
                anomaly_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#dc2626')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('ALIGN', (1, 0), (2, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#fef2f2')),
                    ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#fca5a5')),
                    ('FONTSIZE', (0, 1), (-1, -1), 9),
                    ('VALIGN', (0, 0), (-1, -1), 'TOP')
                ]))
                
                risk_elements.append(anomaly_table)
                risk_elements.append(Spacer(1, 20))
        
        # Risk mitigation
        risk_elements.append(Paragraph("Risk Mitigation Strategy:", self.section_style))
        
        mitigation_text = """
        1. <b>Regular Monitoring:</b> Implement automated alerts for key metrics<br/>
        2. <b>Data Validation:</b> Establish data quality checks at source<br/>
        3. <b>Trend Analysis:</b> Monitor patterns for early warning signals<br/>
        4. <b>Business Rules:</b> Define acceptable ranges for critical metrics
        """
        
        risk_elements.append(Paragraph(mitigation_text, self.body_style))
        
        return risk_elements
    
    def _get_anomaly_recommendation(self, category, count):
        """Get recommendation based on anomaly type and count"""
        recommendations = {
            'sales_amount': 'Review transaction validation rules',
            'profit': 'Investigate cost structure variations',
            'customer_behavior': 'Analyze customer segmentation',
            'sales_patterns': 'Review seasonal adjustments',
            'default': 'Investigate data collection process'
        }
        
        return recommendations.get(category.lower(), recommendations['default'])
    
    def _create_recommendations_section(self, insights, predictions):
        """Create strategic recommendations section"""
        recommendations_elements = []
        
        recommendations_elements.append(Paragraph("8. Strategic Recommendations", self.subtitle_style))
        recommendations_elements.append(Spacer(1, 20))
        
        # Short-term recommendations
        recommendations_elements.append(Paragraph("Immediate Actions (0-30 days):", self.section_style))
        
        immediate_actions = [
            "Implement data quality monitoring dashboards",
            "Focus resources on high-performing segments identified in analysis",
            "Address any critical anomalies flagged in risk assessment",
            "Set up automated alerts for key performance indicators"
        ]
        
        for action in immediate_actions:
            recommendations_elements.append(Paragraph(f"• {action}", self.body_style))
            recommendations_elements.append(Spacer(1, 5))
        
        # Medium-term recommendations
        recommendations_elements.append(Spacer(1, 15))
        recommendations_elements.append(Paragraph("Strategic Initiatives (1-6 months):", self.section_style))
        
        strategic_actions = []
        
        # Add insights-based recommendations
        if 'growth_opportunities' in insights:
            strategic_actions.extend(insights['growth_opportunities'][:3])
        
        if 'specific_predictions' in insights:
            strategic_actions.extend([f"Leverage insight: {pred}" for pred in insights['specific_predictions'][:2]])
        
        # Add default recommendations if none from insights
        if not strategic_actions:
            strategic_actions = [
                "Develop predictive models for demand forecasting",
                "Optimize resource allocation based on seasonal patterns",
                "Implement customer segmentation strategies",
                "Create performance benchmarking frameworks"
            ]
        
        for action in strategic_actions:
            recommendations_elements.append(Paragraph(f"• {action}", self.body_style))
            recommendations_elements.append(Spacer(1, 5))
        
        # Success metrics
        recommendations_elements.append(Spacer(1, 20))
        recommendations_elements.append(Paragraph("Success Metrics to Track:", self.section_style))
        
        success_metrics = [
            "Data quality score improvement (target: >90%)",
            "Reduction in anomaly detection rates",
            "Improvement in forecast accuracy",
            "Increase in identified growth opportunities"
        ]
        
        for metric in success_metrics:
            recommendations_elements.append(Paragraph(f"• {metric}", self.highlight_style))
            recommendations_elements.append(Spacer(1, 5))
        
        return recommendations_elements
    
    def _create_appendix_section(self, df):
        """Create technical appendix"""
        appendix_elements = []
        
        appendix_elements.append(Paragraph("9. Technical Appendix", self.subtitle_style))
        appendix_elements.append(Spacer(1, 20))
        
        # Technical specifications
        appendix_elements.append(Paragraph("Technical Specifications:", self.section_style))
        
        tech_specs = f"""
        <b>Analysis Platform:</b> DataVision-AI v2.0<br/>
        <b>Processing Engine:</b> Python with Pandas, NumPy, Scikit-learn<br/>
        <b>Visualization:</b> Plotly, ReportLab<br/>
        <b>AI Engine:</b> OpenAI GPT-3.5 for insights generation<br/>
        <b>Report Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>
        <b>Dataset Size:</b> {len(df):,} rows × {len(df.columns)} columns<br/>
        <b>Memory Usage:</b> {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB
        """
        
        appendix_elements.append(Paragraph(tech_specs, self.body_style))
        appendix_elements.append(Spacer(1, 20))
        
        # Data dictionary
        appendix_elements.append(Paragraph("Data Dictionary:", self.section_style))
        
        # Column information table
        col_data = [['Column Name', 'Data Type', 'Non-Null Count', 'Unique Values']]
        
        for col in df.columns:
            col_data.append([
                col,
                str(df[col].dtype),
                str(df[col].count()),
                str(df[col].nunique())
            ])
        
        col_table = Table(col_data, colWidths=[2*inch, 1.5*inch, 1.5*inch, 1*inch])
        col_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#374151')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f9fafb')),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#d1d5db')),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f3f4f6')])
        ]))
        
        appendix_elements.append(col_table)
        appendix_elements.append(Spacer(1, 20))
        
        # Footer
        footer_text = """
        <para align="center" fontSize="10" textColor="#6b7280">
        <br/><br/>
        <b>End of Report</b><br/>
        <br/>
        This report was generated automatically by DataVision-AI.<br/>
        For questions or additional analysis, please contact: support@datavision-ai.com<br/>
        <br/>
        © 2024 DataVision-AI by GOKUL M. All rights reserved.
        </para>
        """
        
        appendix_elements.append(Paragraph(footer_text, self.body_style))
        
        return appendix_elements