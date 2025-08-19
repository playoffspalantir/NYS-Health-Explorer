import streamlit as st
import pandas as pd
import altair as alt
import google.generativeai as genai
import markdown
from datetime import datetime
import json
import html
import io

# --- Page Configuration ---
st.set_page_config(page_title="NYS Health Data Explorer", page_icon="🏥", layout="wide")

# --- Universal API Configuration (SECURE METHOD) ---
# This will read the API key from Streamlit's secrets manager when deployed.
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
except Exception:
    # This error will only show up when the secret is not set in the Streamlit Community Cloud.
    st.error("🚨 Gemini API Key not found. Please add it to your Streamlit secrets.", icon="❗")

# --- Session State Initialization ---
if "saved_analyses" not in st.session_state:
    st.session_state.saved_analyses = []
if "current_ai_analysis" not in st.session_state:
    st.session_state.current_ai_analysis = None


# ==============================================================================
# --- Data Loading & AI Functions (Unchanged) ---
# ==============================================================================
@st.cache_data
def load_chirs_data(file_path):
    try:
        df = pd.read_excel(file_path, engine='openpyxl')
        for col in ['Geographic area', 'Year', 'Topic Area', 'Indicator Title', 'Data Source', 'Data Notes']:
            if col in df.columns: df[col] = df[col].astype(str).replace('nan', '')
        return df
    except Exception as e:
        st.error(f"Error loading CHIRS data: {e}");
        return None


def analyze_chirs_data(df, indicator_name):
    if df.empty: return "No data for analysis."
    data_string = df[['Geographic area', 'Year', 'Rate/Percent']].to_csv(index=False)
    prompt = (f"You are a professional epidemiologist providing an objective, data-driven summary. "
              f"Based *only* on the trend data for the indicator '{indicator_name}', write a concise analysis in one or two paragraphs of formal prose. "
              f"Do not use bullet points, markdown formatting (like bolding), or section titles. "
              f"Focus on the overall trend, identify any significant county-level outliers or divergences, and conclude with a statement on the general pattern observed.\n\n"
              f"Data:\n```{data_string}```")
    try:
        model = genai.GenerativeModel(model_name='gemini-1.5-flash');
        response = model.generate_content(prompt);
        return response.text
    except Exception as e:
        return f"AI Analysis Error: {str(e)}"


@st.cache_data
def load_prevention_data(file_path):
    try:
        df = pd.read_csv(file_path, encoding='latin-1')
        df.columns = df.columns.str.strip()
        df['Percentage/Rate/Ratio'] = pd.to_numeric(df['Percentage/Rate/Ratio'], errors='coerce')
        df['2024 Objective'] = pd.to_numeric(df['2024 Objective'], errors='coerce')
        df['Data Years'] = df['Data Years'].astype(str)
        return df
    except Exception as e:
        st.error(f"Error loading Prevention Agenda data: {e}");
        return None


def analyze_prevention_data(df, indicator_name):
    if df.empty: return "No data for analysis."
    data_for_ai = df[['County Name', 'Data Years', 'Percentage/Rate/Ratio', '2024 Objective']]
    data_string = data_for_ai.to_csv(index=False)
    prompt = (f"You are a professional epidemiologist providing an objective, data-driven summary. "
              f"Based *only* on the trend data for the indicator '{indicator_name}', write a concise analysis in one or two paragraphs of formal prose. "
              f"Do not use bullet points, markdown formatting (like bolding), or section titles. "
              f"Focus on the overall progress of the selected counties toward the 2024 objective, highlighting any counties with notable improvement or worsening trends.\n\n"
              f"Data:\n```{data_string}```")
    try:
        model = genai.GenerativeModel(model_name='gemini-1.5-flash');
        response = model.generate_content(prompt);
        return response.text
    except Exception as e:
        return f"AI Analysis Error: {str(e)}"


@st.cache_data
def load_mch_data(file_path):
    try:
        df = pd.read_excel(file_path, engine='openpyxl', header=0)
        df.columns = df.columns.str.strip()
        df['Percentage/Rate'] = pd.to_numeric(df['Percentage/Rate'], errors='coerce')
        df['MCH Objective'] = pd.to_numeric(df['MCH Objective'], errors='coerce')
        df['Data Years'] = df['Data Years'].astype(str)
        for col in ['Data Comments', 'Date Source']:
            if col in df.columns: df[col] = df[col].astype(str).replace('nan', '')
        return df
    except Exception as e:
        st.error(f"Error loading MCH data: {e}");
        return None


def analyze_mch_data(df, indicator_name):
    if df.empty: return "No data for analysis."
    data_for_ai = df[['County Name', 'Data Years', 'Percentage/Rate', 'MCH Objective']]
    data_string = data_for_ai.to_csv(index=False)
    prompt = (
        f"You are a professional epidemiologist specializing in Maternal and Child Health (MCH), providing an objective, data-driven summary. "
        f"Based *only* on the trend data for the indicator: '{indicator_name}', write a concise analysis in one or two paragraphs of formal prose. "
        f"Do not use bullet points, markdown formatting (like bolding), or section titles. "
        f"Focus on county-level progress towards the MCH Objective and mention if data quality comments (e.g., 'Unstable Estimate') warrant cautious interpretation of the trends.\n\n"
        f"Data:\n```{data_string}```")
    try:
        model = genai.GenerativeModel(model_name='gemini-1.5-flash');
        response = model.generate_content(prompt);
        return response.text
    except Exception as e:
        return f"AI Analysis Error: {str(e)}"


# ==============================================================================
# --- Reusable UI and Logic Functions ---
# ==============================================================================
def create_chart(df, config):
    chart = alt.Chart(df).mark_line(point=True).encode(
        x=alt.X(f"{config['year_col']}:N", title='Year', sort=alt.SortField(config['year_col'])),
        y=alt.Y(f"{config['value_col']}:Q", title=config['y_axis_label'], scale=alt.Scale(zero=False)),
        color=alt.Color(f"{config['county_col']}:N", title='County'),
        tooltip=[config['county_col'], config['year_col'], config['value_col']]
    )
    if config.get("objective_col"):
        objective_line = alt.Chart(df).mark_rule(color=config['objective_color'], strokeDash=[5, 5]).encode(
            y=f"mean({config['objective_col']}):Q")
        objective_text = objective_line.mark_text(align='left', baseline='middle', dx=7,
                                                  text=config['objective_label']).encode(
            color=alt.value(config['objective_color']))
        return (chart + objective_line + objective_text).interactive()
    return chart.interactive()


def render_dashboard(config, df):
    st.sidebar.header("Data Filters")
    filters = {}
    for i, f_config in enumerate(config["filters"]):
        temp_df = df.copy()
        for j in range(i):
            prev_filter_config = config["filters"][j]
            if isinstance(filters[prev_filter_config["label"]], list):
                temp_df = temp_df[temp_df[prev_filter_config["col"]].isin(filters[prev_filter_config["label"]])]
            else:
                temp_df = temp_df[temp_df[prev_filter_config["col"]] == filters[prev_filter_config["label"]]]
        options = sorted(temp_df[f_config["col"]].dropna().unique(), reverse=(f_config["col"] == config["year_col"]))
        if f_config["type"] == "selectbox":
            filters[f_config["label"]] = st.sidebar.selectbox(f"{i + 1}. {f_config['label']}", options)
        elif f_config["type"] == "multiselect":
            default_val = f_config.get("default", [])
            if default_val == "all": default_val = options
            default_selection = [d for d in default_val if d in options]
            filters[f_config["label"]] = st.sidebar.multiselect(f"{i + 1}. {f_config['label']}", options,
                                                                default=default_selection)
    filtered_df = df.copy()
    for f_config in config["filters"]:
        selected_val = filters[f_config["label"]]
        if not selected_val:
            st.warning(f"⬅️ Please select at least one {f_config['label']}.");
            return
        if isinstance(selected_val, list):
            filtered_df = filtered_df[filtered_df[f_config["col"]].isin(selected_val)]
        else:
            filtered_df = filtered_df[filtered_df[f_config["col"]] == selected_val]
    if config.get("value_col"):
        filtered_df = filtered_df.dropna(subset=[config["value_col"]])
    st.header(f"📈 Analysis for: {filters[config['indicator_label']]}")
    if filtered_df.empty:
        st.info("No data available for the current filter combination.");
        return
    final_chart = create_chart(filtered_df, config)
    st.altair_chart(final_chart, use_container_width=True)
    st.subheader("Data Context")
    sources = [s for s in filtered_df[config['source_col']].dropna().unique() if s]
    if sources: st.caption(f"Source: {', '.join(sources)}")
    notes = [n for n in filtered_df[config['notes_col']].dropna().unique() if n]
    if notes:
        st.markdown("**Data Comments/Notes:**")
        for note in notes: st.markdown(f"- {note}")
    st.divider()
    st.subheader("🤖 AI-Powered Analysis")
    if st.button(f"Generate Insights for {filters[config['indicator_label']]}"):
        with st.spinner("Analyzing..."):
            ai_text = config["analyzer_func"](filtered_df, filters[config['indicator_label']])
            st.session_state.current_ai_analysis = {
                "dashboard": config["title"], "indicator": filters[config['indicator_label']],
                "filters": {k: v for k, v in filters.items() if not (isinstance(v, list) and len(v) > 5)},
                "analysis_text": ai_text, "data_notes": notes, "data_source": sources,
                "raw_data": filtered_df.copy(), "config": config
            }
            st.rerun()
    if st.session_state.current_ai_analysis and st.session_state.current_ai_analysis["indicator"] == filters[
        config['indicator_label']]:
        current = st.session_state.current_ai_analysis
        st.markdown(current["analysis_text"])
        if st.button("💾 Save This Analysis", key="save_analysis"):
            st.session_state.saved_analyses.append(current)
            st.session_state.current_ai_analysis = None
            st.success(f"Saved analysis for '{current['indicator']}'")
            st.rerun()
    with st.expander("View Filtered Raw Data"):
        st.dataframe(filtered_df)


def view_saved_analyses():
    st.container();
    st.divider()
    st.header("📋 Report Builder")
    if not st.session_state.saved_analyses:
        st.info("No analyses saved yet. Use the '💾 Save This Analysis' button to add them here.");
        return
    html_parts = []
    indices_to_remove = []
    for i, snap in enumerate(st.session_state.saved_analyses):
        with st.container():
            st.subheader(f"{i + 1}. {snap['dashboard']}: {snap['indicator']}")
            if st.button(f"🗑️ Remove Analysis #{i + 1}", key=f"remove_{i}"):
                indices_to_remove.append(i)
            st.markdown(snap['analysis_text'])
            st.markdown("---")

        chart = create_chart(snap['raw_data'], snap['config']).properties(width='container')
        html_buffer = io.StringIO()
        chart.save(html_buffer, format='html')
        chart_html = html_buffer.getvalue()
        html_parts.append(f"<h2>{i + 1}. {snap['dashboard']}: {snap['indicator']}</h2>")
        html_parts.append(f"<p><strong>Filters:</strong> <code>{snap['filters']}</code></p>")
        html_parts.append(chart_html)
        if snap['data_source']: html_parts.append(f"<p><strong>Source:</strong> {', '.join(snap['data_source'])}</p>")
        if snap['data_notes']:
            notes_html = "<ul>" + "".join([f"<li>{note}</li>" for note in snap['data_notes']]) + "</ul>"
            html_parts.append(f"<div><strong>Data Notes:</strong>{notes_html}</div>")
        analysis_html = markdown.markdown(snap['analysis_text'])
        html_parts.append(f"<div><h3>AI-Generated Analysis</h3>{analysis_html}</div>")
        html_parts.append("<hr>")
    for i in sorted(indices_to_remove, reverse=True):
        st.session_state.saved_analyses.pop(i)
        st.rerun()
    html_style = "<style>body{font-family:sans-serif;line-height:1.6;}h1,h2{color:#2c3e50;border-bottom:2px solid #eee;padding-bottom:5px;}h3{color:#34495e;}code{background-color:#f4f4f4;padding:2px 5px;border-radius:4px;}</style>"
    final_html = f"<!DOCTYPE html><html><head><title>NYS Health Report</title>{html_style}</head><body><h1>NYS Health Data Report</h1><p><em>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</em></p>{''.join(html_parts)}</body></html>"
    st.download_button(
        label="📥 Download Full Report as HTML (Print to PDF from Browser)",
        data=final_html, file_name="NYS_Health_Data_Report.html", mime="text/html")


# ==============================================================================
# --- Main App Execution ---
# ==============================================================================
CONFIGS = {
    "CHIRS Indicators": {
        "loader_func": load_chirs_data,
        "file_path": "data/chir_county_trend.xlsx",
        "analyzer_func": analyze_chirs_data, "title": "CHIRS Indicators",
        "filters": [
            {"label": "Topic Area", "col": "Topic Area", "type": "selectbox"},
            {"label": "Indicator", "col": "Indicator Title", "type": "selectbox"},
            {"label": "Counties", "col": "Geographic area", "type": "multiselect",
             "default": ['Westchester County', 'Dutchess County', 'Putnam County', 'Sullivan County', 'Rockland County',
                         'Orange County', 'Ulster County']},
            {"label": "Years", "col": "Year", "type": "multiselect", "default": []}
        ],
        "indicator_label": "Indicator", "county_col": "Geographic area", "year_col": "Year",
        "value_col": "Rate/Percent", "y_axis_label": "Rate / Percent",
        "source_col": "Data Source", "notes_col": "Data Notes"
    },
    "Prevention Agenda Trends": {
        "loader_func": load_prevention_data,
        "file_path": "data/PreventionAgendaTrackingIndicators-CountyTrendData.csv",
        "analyzer_func": analyze_prevention_data, "title": "Prevention Agenda Trends",
        "filters": [
            {"label": "Priority Area", "col": "Priority Area", "type": "selectbox"},
            {"label": "Focus Area", "col": "Focus Area", "type": "selectbox"},
            {"label": "Indicator", "col": "Indicator", "type": "selectbox"},
            {"label": "Counties", "col": "County Name", "type": "multiselect",
             "default": ['Westchester', 'Dutchess', 'Putnam', 'Sullivan', 'Rockland', 'Orange', 'Ulster']},
            {"label": "Years", "col": "Data Years", "type": "multiselect", "default": "all"}
        ],
        "indicator_label": "Indicator", "county_col": "County Name", "year_col": "Data Years",
        "value_col": "Percentage/Rate/Ratio", "y_axis_label": "Percentage / Rate / Ratio",
        "source_col": "Date Source", "notes_col": "Data Comments",
        "objective_col": "2024 Objective", "objective_label": "2024 Objective", "objective_color": "red"
    },
    "MCH Dashboard": {
        "loader_func": load_mch_data,
        "file_path": "data/MCH-CountyTrendData.xlsx",
        "analyzer_func": analyze_mch_data, "title": "MCH Dashboard",
        "filters": [
            {"label": "Domain Area", "col": "Domain Area", "type": "selectbox"},
            {"label": "Indicator", "col": "Indicator", "type": "selectbox"},
            {"label": "Counties", "col": "County Name", "type": "multiselect",
             "default": ['Westchester', 'Dutchess', 'Putnam', 'Sullivan', 'Rockland', 'Orange', 'Ulster']},
            {"label": "Years", "col": "Data Years", "type": "multiselect", "default": "all"}
        ],
        "indicator_label": "Indicator", "county_col": "County Name", "year_col": "Data Years",
        "value_col": "Percentage/Rate", "y_axis_label": "Percentage / Rate",
        "source_col": "Date Source", "notes_col": "Data Comments",
        "objective_col": "MCH Objective", "objective_label": "MCH Objective", "objective_color": "green"
    }
}

st.sidebar.title("NYS Health Explorer")
app_selection = st.sidebar.radio("Choose a Dashboard:", list(CONFIGS.keys()))

st.title(f"🏥 {app_selection}")
config = CONFIGS[app_selection]
df = config["loader_func"](config["file_path"])

if df is not None:
    if "last_dashboard" not in st.session_state or st.session_state.last_dashboard != app_selection:
        st.session_state.current_ai_analysis = None
    st.session_state.last_dashboard = app_selection
    render_dashboard(config, df)

view_saved_analyses()