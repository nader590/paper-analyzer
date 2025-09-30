import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
import base64

# --- إعداد الصفحة ---
st.set_page_config(
    page_title="Scientific Correlation Analyzer | محلل العلاقات العلمية",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- اختيار اللغة ---
lang = st.sidebar.radio("🌐 Language | اللغة", ["English", "العربية"])

# --- النصوص ---
TEXTS = {
    "English": {
        "title": "🔬 Scientific Correlation Analyzer",
        "sidebar_header": "📁 Data Configuration",
        "upload": "Upload CSV File",
        "use_sample": "Use Sample Data",
        "rows": "Total Rows",
        "cols": "Total Columns",
        "num_cols": "Numeric Columns",
        "preview": "📋 Data Preview",
        "analysis": "📊 Correlation Analysis",
        "run": "🚀 Run Comprehensive Analysis",
        "report": "📄 Report Generation",
        "download": "📥 Download Comprehensive Report",
        "success": "✅ PDF report generated successfully!",
        "interp": "Interpretation",
        "sample": "Sample Data (First 5 rows)",
        "plot": "Scatter Plot"
    },
    "العربية": {
        "title": "🔬 محلل العلاقات العلمية",
        "sidebar_header": "📁 إعدادات البيانات",
        "upload": "ارفع ملف CSV",
        "use_sample": "استخدم بيانات تجريبية",
        "rows": "عدد الصفوف",
        "cols": "عدد الأعمدة",
        "num_cols": "الأعمدة الرقمية",
        "preview": "📋 معاينة البيانات",
        "analysis": "📊 تحليل الارتباط",
        "run": "🚀 ابدأ التحليل الشامل",
        "report": "📄 توليد التقرير",
        "download": "📥 تحميل التقرير الشامل",
        "success": "✅ تم توليد التقرير بنجاح!",
        "interp": "التفسير",
        "sample": "عينة من البيانات (أول 5 صفوف)",
        "plot": "المخطط المبعثر"
    }
}

T = TEXTS[lang]

# --- الدوال الأساسية ---
def generate_plot(df, x_col, y_col, r_value):
    chart_filename = f'plot_{x_col}_vs_{y_col}.png'
    plt.figure(figsize=(6, 4))
    plt.scatter(df[x_col], df[y_col], alpha=0.6, color='blue')
    plt.title(f'{x_col} vs {y_col} (r={r_value:.2f})')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig(chart_filename)
    plt.close()
    return chart_filename

def perform_analysis(df, x_col, y_col):
    correlation_r = df[x_col].corr(df[y_col])
    r_sq = correlation_r**2
    plot_path = generate_plot(df, x_col, y_col, correlation_r)
    small_table_df = df[[x_col, y_col]].head(5)
    
    # تفسير باللغتين
    if lang == "English":
        interp = f"The correlation between {x_col} and {y_col} is {correlation_r:.2f} (R²={r_sq:.2f})."
    else:
        interp = f"معامل الارتباط بين {x_col} و {y_col} هو {correlation_r:.2f} (R²={r_sq:.2f})."
    
    return small_table_df, correlation_r, plot_path, interp, r_sq

def create_download_link(file_path, link_text):
    with open(file_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    return f'<a href="data:file/pdf;base64,{b64}" download="{file_path}">{link_text}</a>'

# --- البرنامج الرئيسي ---
def main():
    st.markdown(f"## {T['title']}")
    st.sidebar.header(T['sidebar_header'])
    
    uploaded_file = st.sidebar.file_uploader(T['upload'], type=['csv'])
    use_sample = st.sidebar.checkbox(T['use_sample'], value=True)

    if use_sample:
        np.random.seed(42)
        df = pd.DataFrame({
            "X": np.random.rand(50) * 10,
            "Y": np.random.rand(50) * 5 + 2
        })
    elif uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        st.warning("⬅️ Please upload a file or use sample data | يرجى رفع ملف أو استخدام البيانات التجريبية")
        return
    
    # معلومات سريعة
    col1, col2, col3 = st.columns(3)
    with col1: st.metric(T["rows"], len(df))
    with col2: st.metric(T["cols"], len(df.columns))
    with col3: st.metric(T["num_cols"], sum(pd.api.types.is_numeric_dtype(df[c]) for c in df.columns))

    # معاينة
    st.markdown(f"### {T['preview']}")
    st.dataframe(df.head(), use_container_width=True)

    # اختيار الهدف
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not num_cols:
        st.error("No numeric columns detected | لا يوجد أعمدة رقمية")
        return
    y_col = st.sidebar.selectbox("Target (Y)", num_cols)

    # تحليل
    st.markdown(f"### {T['analysis']}")
    if st.button(T['run']):
        results = []
        for col in num_cols:
            if col == y_col: continue
            small_df, r, plot_path, interp, r_sq = perform_analysis(df, col, y_col)
            results.append((col, r, r_sq, interp, plot_path, small_df))
        
        for col, r, r_sq, interp, plot_path, small_df in results:
            st.subheader(f"{col} ↔ {y_col}")
            st.metric("r", f"{r:.2f}")
            st.metric("R²", f"{r_sq:.2f}")
            st.write(f"**{T['interp']}:** {interp}")
            st.dataframe(small_df)
            st.image(plot_path, caption=T['plot'], use_column_width=True)

if __name__ == "__main__":
    main()
