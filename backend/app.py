import os
import json
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import google.generativeai as genai
from dotenv import load_dotenv
from fpdf import FPDF
import io
import datetime

# Load environment variables
load_dotenv(dotenv_path="../.env")

app = Flask(__name__, static_folder="../frontend", static_url_path="")
CORS(app)

# Configuration
import tempfile

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Use system temp directory for file operations (required for Vercel/Lambda)
TEMP_DIR = tempfile.gettempdir()
UPLOAD_FOLDER = os.path.join(TEMP_DIR, 'uploads')
REPORTS_FOLDER = os.path.join(TEMP_DIR, 'reports')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORTS_FOLDER, exist_ok=True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Global In-Memory Storage
DATASETS = {} # {session_id: dataframe}
METADATA = {} # {session_id: {filename, upload_time}}

def get_gemini_response(prompt):
    if not GEMINI_API_KEY:
        return "⚠️ Gemini API Key is missing."
    try:
        # Use standard flash model (lowest quota usage)
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        try:
             # Fallback to standard pro model if flash fails
            model = genai.GenerativeModel('gemini-1.5-pro')
            response = model.generate_content(prompt)
            return response.text
        except Exception as e2:
            return f"AI Error: Quota Exceeded. Please try again in 1 minute."

@app.route('/')
def home():
    return send_from_directory('../frontend', 'index.html')

@app.route('/dashboard')
def dashboard():
    return send_from_directory('../frontend', 'dashboard.html')

@app.route('/about')
def about():
    return send_from_directory('../frontend', 'about.html')

@app.route('/profile-config', methods=['GET'])
def get_profile_config():
    return jsonify({
        "name": os.getenv("AUTHOR_NAME", "GOKUL M"),
        "role": os.getenv("AUTHOR_ROLE", "Lead Architect"),
        "email": os.getenv("AUTHOR_EMAIL", "gokulxmg26@gmail.com"),
        "phone": os.getenv("AUTHOR_PHONE", "9342099745"),
        "bio": os.getenv("AUTHOR_BIO", "Passionate AI Developer"),
        "focus": os.getenv("PROJECT_FOCUS", "AI Systems"),
        "skills": os.getenv("SKILLS", "Python,AI").split(","),
        "github": os.getenv("GITHUB_URL", "#"),
        "linkedin": os.getenv("LINKEDIN_URL", "#")
    })

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        session_id = request.form.get('session_id', 'default')
        print(f"DEBUG: Receiving upload for session {session_id}")
        
        if 'file' not in request.files:
            return jsonify({"error": "No file part received"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        print(f"DEBUG: File saved to {filepath}")

        # Read File
        df = None
        if filename.lower().endswith('.csv'):
            try:
                df = pd.read_csv(filepath, encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(filepath, encoding='latin1')
        elif filename.lower().endswith(('.xls', '.xlsx')):
            df = pd.read_excel(filepath)
        elif filename.lower().endswith('.json'):
            df = pd.read_json(filepath)
        else:
            return jsonify({"error": "Unsupported file format. Please upload CSV, Excel, or JSON."}), 400

        if df is None:
             return jsonify({"error": "Failed to read dataframe (unknown error)"}), 500

        # Auto-Clean Data
        df = clean_dataframe(df)
        
        # Store in memory
        DATASETS[session_id] = df
        METADATA[session_id] = {
            "filename": filename,
            "uploaded_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        print(f"DEBUG: Dataset stored. Shape: {df.shape}")

        return jsonify({
            "message": "File uploaded and processed successfully",
            "filename": filename,
            "columns": df.columns.tolist(),
            "row_count": len(df),
            "preview": df.head(5).to_dict(orient='records')
        })

    except Exception as e:
        print(f"ERROR: Upload failed - {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Server Error: {str(e)}"}), 500

@app.route('/google-sheet', methods=['POST'])
def google_sheet():
    session_id = request.form.get('session_id', 'default')
    sheet_url = request.json.get('url')
    
    if not sheet_url:
        return jsonify({"error": "No URL provided"}), 400

    try:
        if "/d/" in sheet_url:
            sheet_id = sheet_url.split("/d/")[1].split("/")[0]
            csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
        else:
            return jsonify({"error": "Invalid Google Sheet URL"}), 400
            
        df = pd.read_csv(csv_url)
        df = clean_dataframe(df)

        DATASETS[session_id] = df
        METADATA[session_id] = {
            "filename": "Google Sheet",
            "uploaded_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        return jsonify({
            "message": "Google Sheet linked successfully",
            "columns": df.columns.tolist(),
            "row_count": len(df),
            "preview": df.head(5).to_dict(orient='records')
        })

    except Exception as e:
        return jsonify({"error": f"Failed to fetch Google Sheet: {str(e)}"}), 500

def clean_dataframe(df):
    # Remove columns with > 80% missing values
    threshold = len(df) * 0.2
    df = df.dropna(thresh=threshold, axis=1)
    
    # Drop duplicates
    if not df.empty:
        df = df.drop_duplicates()
    
    # Fill numeric NaNs with mean
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].mean())
        
    # Fill text NaNs with "Unknown"
    object_cols = df.select_dtypes(include=['object']).columns
    for col in object_cols:
        df[col] = df[col].fillna("Unknown")
        
    # Attempt to drop potential ID columns (high cardinality string columns that are unique)
    for col in object_cols:
        if df[col].nunique() == len(df) and "name" not in col.lower():
             try:
                 df = df.drop(columns=[col])
             except:
                 pass
                 
    return df

def calculate_health_score(df):
    score = 100
    if df.isnull().sum().sum() > 0: score -= 10
    if df.duplicated().sum() > 0: score -= 10
    # Penalty for generic ID columns
    for col in df.columns:
        if df[col].nunique() == len(df): score -= 5
    return max(0, score)

@app.route('/analysis', methods=['GET'])
def get_analysis():
    session_id = request.args.get('session_id', 'default')
    if session_id not in DATASETS:
        return jsonify({"error": "No dataset found"}), 404
    
    df = DATASETS[session_id].copy()
    
    # Filter Logic
    filters = request.args.get('filters')
    if filters:
        filter_dict = json.loads(filters)
        for col, val in filter_dict.items():
            if col in df.columns and val and val != "All":
                if df[col].dtype == 'object':
                    df = df[df[col] == val]
    
    # KPIs
    kpis = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
         if not df.empty:
            kpis[col] = {
                "sum": float(df[col].sum()),
                "mean": float(df[col].mean()),
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "trend": "up" if len(df) > 1 and df[col].iloc[-1] > df[col].mean() else "down"
            }
         else:
            kpis[col] = {"sum": 0, "mean": 0, "min": 0, "max": 0, "trend": "flat"}
            
    health_score = calculate_health_score(DATASETS[session_id]) # Score based on original data

    return jsonify({
        "kpis": kpis,
        "columns": df.columns.tolist(),
        "data": df.fillna(0).to_dict(orient='list'),
        "preview": df.head(10).to_dict(orient='records'),
        "row_count": len(df),
        "health_score": health_score
    })

@app.route('/export-report', methods=['POST'])
def export_report():
    session_id = request.json.get('session_id', 'default')
    if session_id not in DATASETS:
        return jsonify({"error": "No dataset found"}), 404
    
    df = DATASETS[session_id]
    
    class PDF(FPDF):
        def header(self):
            self.set_font('Arial', 'B', 10)
            self.cell(0, 10, 'DataVision-AI Enterprise Report', 0, 0, 'R')
            self.ln(10)
        
        def footer(self):
            self.set_y(-15)
            self.set_font('Arial', 'I', 8)
            self.cell(0, 10, f'Page {self.page_no()} - Generated by {os.getenv("AUTHOR_NAME")}', 0, 0, 'C')

    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # -- COVER PAGE --
    pdf.add_page()
    pdf.set_fill_color(15, 23, 42) # Dark background
    pdf.rect(0, 0, 210, 297, 'F')
    
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Arial", 'B', 36)
    pdf.ln(80)
    pdf.cell(0, 20, "EXECUTIVE", 0, 1, 'C')
    pdf.cell(0, 20, "DATA REPORT", 0, 1, 'C')
    
    pdf.set_font("Arial", '', 14)
    pdf.ln(20)
    pdf.cell(0, 10, f"Prepared for: Enterprise Stakeholders", 0, 1, 'C')
    pdf.cell(0, 10, f"Date: {datetime.datetime.now().strftime('%B %d, %Y')}", 0, 1, 'C')
    
    pdf.ln(40)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, f"Authored By: {os.getenv('AUTHOR_NAME')}", 0, 1, 'C')
    pdf.set_font("Arial", '', 10)
    pdf.cell(0, 10, f"{os.getenv('AUTHOR_ROLE')}", 0, 1, 'C')
    
    # -- CONTENT PAGE --
    pdf.add_page()
    pdf.set_text_color(0, 0, 0)
    
    # AI Summary
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Executive Summary", 0, 1, 'L')
    pdf.ln(5)
    pdf.set_font("Arial", '', 11)
    
    # Generate quick summary for report if not existing
    summary_prompt = f"Write a 4 sentence executive summary for a dataset with columns: {list(df.columns)}. Focus on business value."
    summary_text = get_gemini_response(summary_prompt)
    pdf.multi_cell(0, 7, summary_text.replace('*', ''))
    
    pdf.ln(10)
    
    # Data Health
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Data Health & Quality", 0, 1)
    pdf.set_font("Arial", '', 11)
    score = calculate_health_score(df)
    pdf.cell(0, 8, f"Health Score: {score}/100", 0, 1)
    pdf.cell(0, 8, f"Total Rows: {len(df)}", 0, 1)
    pdf.cell(0, 8, f"Columns Processed: {len(df.columns)}", 0, 1)

    pdf.ln(10)
    
    # Statistics
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Key Statistical Metrics", 0, 1)
    stats = df.describe().to_string()
    pdf.set_font("Courier", '', 8)
    pdf.multi_cell(0, 5, stats)
    
    # Contact
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Contact Information", 0, 1)
    pdf.ln(10)
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 10, f"Name: {os.getenv('AUTHOR_NAME')}", 0, 1)
    pdf.cell(0, 10, f"Email: {os.getenv('AUTHOR_EMAIL')}", 0, 1)
    pdf.cell(0, 10, f"Phone: {os.getenv('AUTHOR_PHONE')}", 0, 1)
    pdf.cell(0, 10, f"Website: DataVision AI", 0, 1)

    # Save
    report_path = os.path.join(REPORTS_FOLDER, f"Professional_Report_{session_id}.pdf")
    pdf.output(report_path)
    
    return send_file(report_path, as_attachment=True)

@app.route('/get-filters', methods=['GET'])
def get_filters():
    session_id = request.args.get('session_id', 'default')
    if session_id not in DATASETS:
        return jsonify({"error": "No dataset found"}), 404
        
    df = DATASETS[session_id]
    dataset_filters = {}
    
    # Get unique values for object columns (limit to top 50)
    object_cols = df.select_dtypes(include=['object']).columns
    for col in object_cols:
        if df[col].nunique() < 50: 
            dataset_filters[col] = ["All"] + sorted(df[col].unique().astype(str).tolist())
            
    return jsonify(dataset_filters)

@app.route('/perform_cleaning', methods=['POST'])
def perform_cleaning():
    session_id = request.json.get('session_id', 'default')
    action = request.json.get('action')
    target = request.json.get('target')
    
    if session_id not in DATASETS:
        return jsonify({"error": "No dataset found"}), 404
    
    df = DATASETS[session_id]
    
    try:
        if action == 'drop_column':
            if target in df.columns:
                df.drop(columns=[target], inplace=True)
        elif action == 'drop_duplicates':
            df.drop_duplicates(inplace=True)
        elif action == 'fill_missing_mean':
            num_cols = df.select_dtypes(include=np.number).columns
            df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
        elif action == 'fill_missing_unknown':
             obj_cols = df.select_dtypes(include='object').columns
             df[obj_cols] = df[obj_cols].fillna("Unknown")
             
        # Update dataset in memory
        DATASETS[session_id] = df
        return jsonify({"message": "Cleaning action applied successfully", "columns": df.columns.tolist()})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/download-data', methods=['GET'])
def download_data():
    session_id = request.args.get('session_id', 'default')
    if session_id not in DATASETS:
        return "No dataset found", 404
    
    df = DATASETS[session_id]
    output = io.StringIO()
    df.to_csv(output, index=False)
    
    mem = io.BytesIO()
    mem.write(output.getvalue().encode('utf-8'))
    mem.seek(0)
    
    return send_file(
        mem, 
        as_attachment=True, 
        download_name=f'cleaned_data_{session_id}.csv',
        mimetype='text/csv'
    )

@app.route('/generate-story', methods=['POST'])
def generate_story():
    session_id = request.json.get('session_id', 'default')
    if session_id not in DATASETS:
        return jsonify({"error": "No dataset found"}), 404
    
    df = DATASETS[session_id]
    # TRUNCATE to simple stats to save tokens
    summary = df.describe().to_string()
    if len(summary) > 2000: summary = summary[:2000] 
    
    prompt = f"""
    You are a Data Storyteller.
    Analyze this dataset statistics:
    {summary}
    Write a 3-paragraph "Data Story": Big Picture, Climax, Resolution.
    """
    story = get_gemini_response(prompt)
    
    # Offline Fallback if AI fails
    if "AI Error" in story or "Quota" in story:
         try:
             # Manual Story Generation
             num_cols = df.select_dtypes(include=np.number).columns
             top_col = num_cols[0] if len(num_cols) > 0 else "N/A"
             max_val = df[top_col].max() if len(num_cols) > 0 else 0
             story = f"""
             ### 📊 Automated Insight (Offline Mode)
             
             **The Big Picture**: We are looking at a dataset with **{len(df)} rows** and **{len(df.columns)} columns**.
             
             **Key Trends**: The primary metric **{top_col}** reaches a maximum of **{max_val}**, showing significant variance.
             
             *(Note: AI features are currently rate-limited. This is a computed summary.)*
             """
         except:
             story = "Could not generate story."

    return jsonify({"story": story})

@app.route('/ai-insight', methods=['POST'])
def ai_insight():
    session_id = request.json.get('session_id', 'default')
    query = request.json.get('query', '')
    
    # Command Parsing
    if "remove" in query.lower() and "column" in query.lower():
        if session_id in DATASETS:
            df = DATASETS[session_id]
            for col in df.columns:
                if col.lower() in query.lower():
                    df.drop(columns=[col], inplace=True)
                    DATASETS[session_id] = df
                    return jsonify({"response": f"✅ Removed column **'{col}'**."})

    # Context
    app_context = "DataVision AI: Analytics, KPI, AI Storytelling, Data Cleaning."
    dataset_context = "No dataset."
    if session_id in DATASETS:
        df = DATASETS[session_id]
        # Limit context size significantly
        dataset_context = f"Columns: {list(df.columns)}\nData Sample:\n{df.head(3).to_string()}"
        if len(dataset_context) > 1500: dataset_context = dataset_context[:1500]

    prompt = f"""
    You are DataVision Assistant.
    {app_context}
    Dataset:
    {dataset_context}
    Query: "{query}"
    Be professional.
    """
    response = get_gemini_response(prompt)
    
    if "AI Error" in response:
        response = """⚠️ **AI Quota Exceeded**
        
        The free Gemini API limit has been reached. To fix this immediately:
        1. Get a new free key from [Google AI Studio](https://aistudio.google.com/app/apikey).
        2. Open `.env` file and replace `GEMINI_API_KEY`.
        3. Save and restart.
        
        *Falling back to offline mode...*"""
        
    return jsonify({"response": response})
if __name__ == '__main__':
    app.run(debug=True, port=5000)