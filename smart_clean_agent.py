import streamlit as st
import pandas as pd
import numpy as np
import io
import re
import os
from scipy import stats

# Page Config (Premium Cyber-SaaS Layout)
st.set_page_config(page_title="Smart-Clean AI Data Agent Pro", page_icon="🛡️", layout="wide")

# Advanced UI: Premium Matrix Canvas & Glassmorphism Theme CSS Injection
st.markdown("""
    <style>
    .stApp { background-color: #020617; color: #f8fafc; }
    #matrixCanvas { position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; z-index: -1; opacity: 0.12; pointer-events: none; }
    div[data-testid="stMetricBlock"] { background: rgba(15, 23, 42, 0.7); backdrop-filter: blur(8px); border: 1px solid rgba(56, 189, 248, 0.2); border-radius: 12px; padding: 15px; }
    .stButton>button { background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%); color: white; border-radius: 8px; width: 100%; font-weight: bold; border: none; }
    .stDownloadButton>button { background: linear-gradient(135deg, #16a34a 0%, #15803d 100%); color: white; border-radius: 8px; width: 100%; font-weight: bold; border: none; }
    div[data-testid="stMetricValue"] { font-size: 40px; color: #38bdf8; font-weight: bold; }
    </style>
    <canvas id="matrixCanvas"></canvas>
    <script>
        const canvas = document.getElementById('matrixCanvas');
        const ctx = canvas.getContext('2d');
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
        const letters = '010101🛡️💻📊AI_PRO';
        const alphabet = letters.split('');
        const fontSize = 14;
        const columns = canvas.width / fontSize;
        const rainDrops = [];
        for(let x = 0; x < columns; x++) { rainDrops[x] = 1; }
        function drawMatrix() {
            ctx.fillStyle = 'rgba(2, 6, 23, 0.05)';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            ctx.fillStyle = '#06b6d4';
            ctx.font = fontSize + 'px monospace';
            for(let i = 0; i < rainDrops.length; i++) {
                const text = alphabet[Math.floor(Math.random() * alphabet.length)];
                ctx.fillText(text, i * fontSize, rainDrops[i] * fontSize);
                if(rainDrops[i] * fontSize > canvas.height && Math.random() > 0.975) { rainDrops[i] = 0; }
                rainDrops[i]++;
            }
        }
        setInterval(drawMatrix, 33);
    </script>
    """, unsafe_allow_html=True)

st.title("🛡️ Smart-Clean AI Data Agent Enterprise Pro")
st.subheader("Autonomous Machine Learning Imputation & Semantic NLP Pipeline")
st.write("---")

# Sidebar Controls
st.sidebar.title("⚙️ AI Pipeline Controls")
remove_whitespace = st.sidebar.checkbox("Trim Text Whitespaces", value=True)
convert_case = st.sidebar.selectbox("Text Case Standardize", ["No Change", "Title Case", "Uppercase", "Lowercase"])
mask_pii = st.sidebar.checkbox("Mask Sensitive PII Fields (GDPR/CCPA)", value=False)

handle_strategy = st.sidebar.selectbox("Numeric Imputation Strategy", ["Median Value", "Mean Value", "Drop Missing Rows"])
text_strategy = st.sidebar.selectbox("Categorical Imputation Strategy", ["Most Frequent (Mode)", "Fill as 'Unknown'"])
outlier_threshold = st.sidebar.slider("Outlier Detection Sensitivity (Z-Score)", 2.0, 4.0, 3.0, step=0.5)

chart_theme = st.sidebar.selectbox("Analytics Chart Color Accent", ["Matrix Blue", "Neon Coral", "Emerald Mint"])
theme_colors = {"Matrix Blue": "#38bdf8", "Neon Coral": "#f43f5e", "Emerald Mint": "#10b981"}
selected_color = theme_colors[chart_theme]

uploaded_files = st.file_uploader("Upload Spreadsheets for AI Data Profiling...", type=["csv", "xlsx", "xls"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        file_name = uploaded_file.name
        st.markdown(f"### 📂 Processing: `{file_name}`")
        
        df = pd.read_csv(uploaded_file) if file_name.endswith('.csv') else pd.read_excel(uploaded_file)

        total_rows, total_cols = df.shape
        total_cells = total_rows * total_cols
        
        missing_series = df.isnull().sum()
        total_missing = missing_series.sum()
        total_duplicates = df.duplicated().sum()
        
        email_cols, phone_cols, numeric_features = [], [], []
        for col in df.columns:
            col_lower = col.lower()
            if any(key in col_lower for key in ['email', 'mail', 'addr_email']): email_cols.append(col)
            elif any(key in col_lower for key in ['phone', 'mobile', 'contact', 'ph_no']): phone_cols.append(col)
            if df[col].dtype in ['int64', 'float64']: numeric_features.append(col)

        invalid_emails = sum(df[col].dropna().astype(str).apply(lambda x: 0 if re.match(r"[^@]+@[^@]+\.[^@]+", x) else 1).sum() for col in email_cols)
        invalid_phones = sum(df[col].dropna().astype(str).apply(lambda x: 0 if re.match(r"^\+?[-0-9\s\(]{7,20}$", x) else 1).sum() for col in phone_cols)

        total_outliers = 0
        outlier_info = {}
        for col in numeric_features:
            if df[col].dropna().size > 5:
                try:
                    z_scores = np.abs(stats.zscore(df[col].dropna()))
                    outliers_count = np.sum(z_scores > outlier_threshold)
                    total_outliers += outliers_count
                    if outliers_count > 0: outlier_info[col] = int(outliers_count)
                except: pass

        if total_cells > 0:
            base_score = ((total_cells - total_missing) / total_cells) * 100
            deductions = (total_duplicates + invalid_emails + invalid_phones + total_outliers) / total_rows * 100
            health_score = round(max(0, base_score - deductions), 2)
        else: health_score = 0

        # UI Dashboard Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric(label="System Health Index", value=f"{health_score}%")
        with col2: st.metric(label="Total Mapped Rows", value=f"{total_rows:,}")
        with col3: st.metric(label="Null Fields Detected", value=f"{total_missing:,}")
        with col4: st.metric(label="Data Integrity Violations", value=f"{(total_duplicates + invalid_emails + invalid_phones + total_outliers):,}")

        # Visual Dashboard Dynamic Engine
        st.write("### 🎛️ Enterprise Visual Diagnostic Core")
        row1_col1, row1_col2 = st.columns(2)
        with row1_col1:
            if total_missing > 0:
                st.write("**Chart 1: Missing Fields Distribution**")
                m_df = missing_series[missing_series > 0].reset_index()
                m_df.columns = ['Attribute Name', 'Empty Records']
                st.bar_chart(data=m_df, x='Attribute Name', y='Empty Records', color=selected_color)
            else:
                st.write("**Chart 1: Unique Value Cardinality Profile (100% Data Clean Fallback)**")
                cardinality_df = pd.DataFrame({
                    'Attribute Name': df.columns,
                    'Unique Value Logs': [df[c].nunique() for c in df.columns]
                })
                st.bar_chart(data=cardinality_df, x='Attribute Name', y='Unique Value Logs', color=selected_color)
                
        with row1_col2:
            st.write("**Chart 2: Statistical Variance Anomaly Curve**")
            if outlier_info:
                out_df = pd.DataFrame(list(outlier_info.items()), columns=['Numeric Features', 'Outlier Volume'])
                st.area_chart(data=out_df, x='Numeric Features', y='Outlier Volume', color="#f59e0b")
            else: st.success("✅ Perfect Matrix: No numeric structural outliers logged.")
        
        row2_col1, row2_col2 = st.columns(2)
        with row2_col1:
            st.write("**Chart 3: Data Completeness Trend Profile**")
            completeness_df = pd.DataFrame({
                'Features Grid': df.columns,
                'Completeness Ratio (%)': [(1 - (df[c].isnull().sum() / total_rows)) * 100 for c in df.columns]
            })
            st.line_chart(data=completeness_df, x='Features Grid', y='Completeness Ratio (%)', color="#a855f7")
            
        with row2_col2:
            st.write("**Chart 4: Threat Vector Incidents Mapping**")
            risk_factors = {'Missing Loss': total_missing, 'Redundant Duplicates': total_duplicates, 'Email Drift': invalid_emails, 'Phone Drift': invalid_phones, 'Statistical Outliers': total_outliers}
            risk_df = pd.DataFrame(list(risk_factors.items()), columns=['Threat Vector', 'Incident Log Count'])
            st.bar_chart(data=risk_df, x='Threat Vector', y='Incident Log Count', color="#e11d48", horizontal=True)

        st.write("---")

        # ---- THE CLEANING PIPELINE ENGINE ----
        df_clean = df.copy()
        if total_duplicates > 0: df_clean = df_clean.drop_duplicates()
        
        for col in df_clean.columns:
            if df_clean[col].dtype == 'object':
                if remove_whitespace: df_clean[col] = df_clean[col].astype(str).str.strip()
                if convert_case == "Uppercase": df_clean[col] = df_clean[col].astype(str).str.upper()
                elif convert_case == "Lowercase": df_clean[col] = df_clean[col].astype(str).str.lower()
                elif convert_case == "Title Case": df_clean[col] = df_clean[col].astype(str).str.title()

        for col in df_clean.columns:
            if df_clean[col].isnull().any():
                if col in numeric_features:
                    if handle_strategy == "Median Value": df_clean[col] = df_clean[col].fillna(df_clean[col].median())
                    elif handle_strategy == "Mean Value": df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
                    elif handle_strategy == "Drop Missing Rows": df_clean = df_clean.dropna(subset=[col])
                else:
                    if text_strategy == "Most Frequent (Mode)":
                        df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0] if not df_clean[col].mode().empty else "Unknown")
                    else: df_clean[col] = df_clean[col].fillna("Unknown")

        if mask_pii:
            for col in email_cols: df_clean[col] = df_clean[col].astype(str).apply(lambda x: re.sub(r'(?<=.).(?=.*@)', '*', x) if '@' in x else x)
            for col in phone_cols: df_clean[col] = df_clean[col].astype(str).apply(lambda x: x[:3] + '*' * (len(x) - 5) + x[-2:] if len(x) > 5 else '*****')

        # AI Local Text Summary Report Builder
        ai_insights = f"### 🧠 AI Automated Natural Language Insights Report\n"
        if total_missing > 0:
            top_missing_col = missing_series.idxmax()
            ai_insights += f"- **Structural Deficiency Alert:** Column `{top_missing_col}` possesses the highest vulnerability footprint with `{missing_series[top_missing_col]}` null elements.\n"
        else:
            ai_insights += f"- **Database Integrity Assessment:** Column architecture is 100% physically complete. No row dropouts found.\n"
        if total_outliers > 0:
            top_outlier_col = max(outlier_info, key=outlier_info.get) if outlier_info else "None"
            ai_insights += f"- **Statistical Skew Warning:** Feature column `{top_outlier_col}` has extreme mathematical outliers detected.\n"

        missing_labels = [str(k) for k in missing_series[missing_series > 0].index.tolist()] if total_missing > 0 else []
        missing_data = [int(v) for v in missing_series[missing_series > 0].tolist()] if total_missing > 0 else []
        table_name = re.sub(r'[^a-zA-Z0-9]', '_', os.path.splitext(file_name)[0].lower())

        # Standalone Static HTML Snapshot Template Code
        html_report = f"""<!DOCTYPE html><html><head><script src="https://cdn.jsdelivr.net/npm/chart.js"></script></head><body style="background-color:#0b0f19;color:#fff;padding:40px;font-family:sans-serif;"><h3>🛡️ AI Agent Executive Audit Snapshot</h3><p>Target Source: {file_name}</p><p>Integrity Index: {health_score}%</p><canvas id="c1" width="400" height="150"></canvas><script>new Chart(document.getElementById('c1').getContext('2d'),{{type:'bar',data:{{labels:{missing_labels if total_missing > 0 else list(df.columns)},datasets:[{{label:'Data Scaler Balance',data:{missing_data if total_missing > 0 else [int(df[c].nunique()) for c in df.columns]},backgroundColor:'#38bdf8'}}]}}}});</script></body></html>"""

        # Tabs Navigation Suite
        tab1, tab2, tab_ai, tab3, tab4, tab5, tab6 = st.tabs([
            "👀 Raw Input", "⚙️ AI Cleaned", "🧠 AI Insights Engine", "💻 Advanced SQL", "📡 Cross-Platform APIs", "📋 Download Reports", "💼 B2B Pitch"
        ])

        with tab1: st.dataframe(df.head(15), use_container_width=True)
        
        with tab2:
            st.dataframe(df_clean.head(15), use_container_width=True)
            output_excel = io.BytesIO()
            with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
                df_clean.to_excel(writer, index=False, sheet_name='Cleaned_Data')
            st.download_button(label="📥 Download Cleaned Dataset (EXCEL .xlsx)", data=output_excel.getvalue(), file_name=f"Cleaned_{file_name}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key=f"dl_xl_{file_name}")

        with tab_ai: st.markdown(ai_insights)

        with tab3:
            st.write("#### 💻 Target Multi-Engine Relational DDL Script")
            sql_engine = st.selectbox("Select Target Database Flavor:", ["PostgreSQL", "MySQL", "MS SQL Server"], key=f"eng_{file_name}")
            sql_lines = [f"CREATE TABLE {table_name} ("]
            for col in df_clean.columns:
                col_cleaned = re.sub(r'[^a-zA-Z0-9_]', '', col.lower())
                dtype = "INT" if df_clean[col].dtype in ['int64'] else "DOUBLE PRECISION" if (df_clean[col].dtype in ['float64'] and sql_engine == "PostgreSQL") else "FLOAT" if df_clean[col].dtype in ['float64'] else "TEXT" if sql_engine == "PostgreSQL" else "VARCHAR(MAX)" if sql_engine == "MS SQL Server" else "VARCHAR(255)"
                sql_lines.append(f"    {col_cleaned} {dtype},")
            sql_lines[-1] = sql_lines[-1].rstrip(',')
            sql_lines.append(");\n")
            for _, row in df_clean.head(5).iterrows():
                vals = [f"'{str(v).replace(chr(39), chr(39)+chr(39))}'" if not isinstance(v, (int, float)) or pd.isna(v) else str(v) for v in row]
                sql_lines.append(f"INSERT INTO {table_name} VALUES ({', '.join(vals)});")
            st.code("\n".join(sql_lines), language="sql")

        with tab4:
            st.write("#### 📡 Direct Endpoint API Sync Codes")
            api_lang = st.selectbox("Select Integration Language:", ["Python (Requests)", "Node.js (Axios)", "cURL Shell"], key=f"api_{file_name}")
            if api_lang == "Python (Requests)":
                st.code(f"import requests\nurl = 'https://api.clientdomain.com/v1/data-stream'\nheaders = {{'Authorization': 'Bearer SEC_TOKEN'}}\nresponse = requests.post(url, json={{'table': '{table_name}', 'rows': {total_rows}}})\nprint(response.status_code)", language="python")
            elif api_lang == "Node.js (Axios)":
                st.code(f"const axios = require('axios');\naxios.post('https://api.clientdomain.com/v1/data-stream', {{table: '{table_name}', rows: {total_rows}}}, {{headers: {{Authorization: 'Bearer SEC_TOKEN'}}}});", language="javascript")
            else:
                st.code(f"curl -X POST https://api.clientdomain.com/v1/data-stream -H 'Authorization: Bearer SEC_TOKEN' -d '{{\"table\":\"{table_name}\",\"rows\":{total_rows}}}'", language="bash")

        with tab5:
            st.write("#### 📋 Export Centralized Quality Assets")
            col_d1, col_d2 = st.columns(2)
            with col_d1: st.download_button(label="📄 Download Official Audit Log (.txt)", data=f"Database Account Target: {file_name}\nHealth Index Base: {health_score}%\nTotal Rows Mapped: {total_rows}", file_name=f"Audit_Report_{file_name}.txt", key=f"txt_{file_name}")
            with col_d2: st.download_button(label="🌐 Download Multi-Chart HTML Executive Report", data=html_report, file_name=f"Interactive_Dashboard_{table_name}.html", mime="text/html", key=f"html_{file_name}")

        with tab6:
            st.write("#### 💼 Dynamic High-Converting Outbound Value Proposal")
            pitch_text = f"""Hi [Client Name],

I conducted an advanced structural validation audit on your dataset ({file_name}).

Your System Health Index is currently at {health_score}% due to {total_missing} empty/null data nodes, {total_duplicates} duplicate records, and {total_outliers} extreme statistical anomalies. Left unresolved, these formatting drifts heavily skew standard BI metrics and analytical algorithms.

I have deployed a custom automated pipeline that handles numeric imputer parameters, verifies records structure metrics, and prepares cross-platform backend schema targets (MySQL/PostgreSQL) with 100% execution safety.

Attached are the diagnostic summary blueprints and preview logs. Let me know if you want to deploy this automated cleaning pipeline live straight into your core production database!

Best regards,
Salar Ahsan
AI Systems Developer & Data Solutions Specialist"""
            st.text_area("", pitch_text, height=300, key=f"p_{file_name}")
        st.write("---")