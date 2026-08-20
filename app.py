"""
Streamlit EDA app — converted from eda.ipynb
EMBER2024 win32 detection dataset: malware vs. benign EDA.

Run with:
    streamlit run app.py

Expects a parquet file at DATA_DIR / "win32_detection_train_20pct.parquet"
(path configurable in the sidebar), with columns including:
    sha256, label, imports, strings, general, ...
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

st.set_page_config(page_title="EMBER2024 Malware EDA", layout="wide")

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

DEFAULT_DATA_PATH = (
    Path(__file__).resolve().parent
    / "win32_data"
    / "win32_detection_train_20pct.parquet"
)


@st.cache_data(show_spinner="Loading dataset...")
def load_data(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


def parse_imports(value):
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return {}
    return {}


def count_imported_apis(import_data):
    if isinstance(import_data, dict):
        total = 0
        for apis in import_data.values():
            if isinstance(apis, list):
                total += len(apis)
        return total
    if isinstance(import_data, list):
        return len(import_data)
    return 0


def flatten_imported_apis(import_data):
    api_names = []
    if isinstance(import_data, dict):
        for apis in import_data.values():
            if isinstance(apis, list):
                api_names.extend(str(api).lower() for api in apis)
    elif isinstance(import_data, list):
        api_names.extend(str(api).lower() for api in import_data)
    return api_names


def count_matching_apis(api_names, search_terms):
    return sum(any(term in api_name for term in search_terms) for api_name in api_names)


NETWORK_TERMS = [
    "socket", "connect", "send", "recv", "bind", "listen", "accept",
    "internetopen", "internetconnect", "internetreadfile", "internetwritefile",
    "httpopenrequest", "httpsendrequest", "urlopen", "urldownloadtofile",
    "winhttpopen", "winhttpconnect", "winhttpsendrequest", "wsastartup",
]

API_CATEGORIES = {
    "Networking": ["socket", "connect", "send", "recv", "internetopen", "httpsendrequest", "winhttp", "wsastartup"],
    "File operations": ["createfile", "readfile", "writefile", "deletefile", "copyfile", "movefile"],
    "Registry": ["regopenkey", "regsetvalue", "regqueryvalue", "regcreatekey", "regdeletekey"],
    "Process and memory": ["createprocess", "openprocess", "virtualalloc", "virtualprotect", "writeprocessmemory", "createremotethread"],
    "Cryptography": ["cryptencrypt", "cryptdecrypt", "cryptacquirecontext", "bcrypt", "certopenstore"],
}

CATEGORY_MAP = {0: "Benign", 1: "Malware"}


# ---------------------------------------------------------------------------
# Sidebar: data source
# ---------------------------------------------------------------------------

st.sidebar.header("Data")
data_path = st.sidebar.text_input("Parquet file path", value=str(DEFAULT_DATA_PATH))

if not Path(data_path).exists():
    st.warning(f"File not found: `{data_path}`. Enter a valid path in the sidebar.")
    st.stop()

st.title("EMBER2024 Win32 Malware EDA")
 
model_tab, eda_tab = st.tabs(["🧪 Model Performance", "📊 EDA"])

# ---------------------------------------------------------------------------
# Model Tab
# ---------------------------------------------------------------------------
with model_tab:
    st.header("Detection Model Performance")
    st.caption(
        "Precomputed results from an LGBMClassifier trained on engineered "
        "import/string/general features. Run `export_model_results.py` "
        "after retraining to refresh this tab."
    )
 
    results_path = st.sidebar.text_input(
        "Model results JSON path", value=str(Path.cwd() / "model_results.json")
    )
 
    if not Path(results_path).exists():
        st.info(
            f"No results file found at `{results_path}`. "
            "Run `python export_model_results.py` first to generate it."
        )
    else:
        with open(results_path) as f:
            results = json.load(f)
 
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{results['accuracy']:.2%}")
        col2.metric("ROC AUC", f"{results['roc_auc']:.4f}")
        col3.metric("Train samples", f"{results['n_train']:,}")
        col4.metric("Test samples", f"{results['n_test']:,}")
 
        st.subheader("Confusion Matrix")
        labels = results["labels"]
        cm = results["confusion_matrix"]
        cm_df = pd.DataFrame(cm, index=labels, columns=labels).reset_index().melt(
            id_vars="index", var_name="predicted", value_name="count"
        ).rename(columns={"index": "actual"})
 
        cm_chart = alt.Chart(cm_df).mark_rect().encode(
            x=alt.X("predicted:N", title="Predicted label"),
            y=alt.Y("actual:N", title="Actual label"),
            color=alt.Color("count:Q", title="Count", scale=alt.Scale(scheme="blues")),
            tooltip=["actual", "predicted", "count"],
        ).properties(width=350, height=350, title="Confusion Matrix")
 
        cm_text = alt.Chart(cm_df).mark_text(baseline="middle", fontSize=16).encode(
            x="predicted:N",
            y="actual:N",
            text="count:Q",
            color=alt.condition(
                alt.datum.count > cm_df["count"].max() / 2,
                alt.value("white"),
                alt.value("black"),
            ),
        )
 
        st.altair_chart(cm_chart + cm_text, use_container_width=False)
 
        st.subheader("Classification Report")
        report_df = pd.DataFrame(results["classification_report"]).T.round(3)
        st.dataframe(report_df)
 
        if "feature_importance" in results:
            st.subheader("Top Feature Importances")
            fi_df = (
                pd.DataFrame(
                    results["feature_importance"].items(),
                    columns=["feature", "importance"],
                )
                .sort_values("importance", ascending=False)
                .head(15)
            )
            fi_chart = alt.Chart(fi_df).mark_bar().encode(
                x=alt.X("importance:Q", title="Importance"),
                y=alt.Y("feature:N", sort="-x", title="Feature"),
                tooltip=["feature", "importance"],
            ).properties(width=650, height=400, title="Top 15 Feature Importances")
            st.altair_chart(fi_chart, width='stretch')

# ---------------------------------------------------------------------------
# EDA Tab
# ---------------------------------------------------------------------------
with eda_tab:
    train_df = load_data(data_path)
    # ---------------------------------------------------------------------------
    # Overview
    # ---------------------------------------------------------------------------

    st.header("Dataset overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", f"{len(train_df):,}")
    col2.metric("Duplicate rows", f"{train_df.duplicated().sum():,}")
    col3.metric("Duplicate sha256", f"{train_df['sha256'].duplicated().sum():,}" if "sha256" in train_df else "n/a")

    with st.expander("Columns & dtypes"):
        st.write(train_df.columns.tolist())
        st.write(train_df.dtypes)

    st.subheader("Preview")
    st.dataframe(train_df.head().astype(str))

    if len(train_df) > 2:
        with st.expander("Sample record (row 2) — strings / general"):
            if "strings" in train_df:
                st.write(train_df.iloc[2]["strings"])
            if "general" in train_df:
                st.write(train_df.iloc[2]["general"])

    # ---------------------------------------------------------------------------
    # Label distribution
    # ---------------------------------------------------------------------------

    st.header("Label distribution")

    label_counts = (
        train_df["label"].value_counts().rename_axis("label").reset_index(name="count")
        .assign(category=lambda d: d["label"].map(CATEGORY_MAP))
    )

    chart = alt.Chart(label_counts).mark_bar().encode(
        x=alt.X("category:N", title="File Classification", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("count:Q", title="Number of Samples"),
        tooltip=["category", "count"],
    ).properties(title="Distribution of Malware and Benign Samples", width=500, height=350)

    st.altair_chart(chart, width="stretch")

    # ---------------------------------------------------------------------------
    # Imported API counts (cached, expensive parsing done once)
    # ---------------------------------------------------------------------------
    st.header("Imported API counts")

    @st.cache_data(show_spinner="Parsing imports...")
    def compute_api_data(df: pd.DataFrame):
        parsed_imports = df["imports"].apply(parse_imports)
        api_count_df = pd.DataFrame({
            "api_count": parsed_imports.apply(count_imported_apis),
            "label": df["label"],
        }).assign(category=lambda d: d["label"].map(CATEGORY_MAP))
        api_lists = parsed_imports.apply(flatten_imported_apis)
        return api_count_df, api_lists

    api_count_df, api_lists = compute_api_data(train_df)
    
    st.dataframe(
        api_count_df.groupby("category")["api_count"]
        .agg(["count", "mean", "median", "std", "max"])
        .round(2)
    )
    
    with st.expander("Sample imports (first record)"):
        sample_imports = train_df.iloc[0]["imports"]
        if isinstance(sample_imports, str):
            sample_imports = json.loads(sample_imports)
        st.write(sample_imports)
        st.caption("Flattened API names (first 20):")
        st.write(api_lists.iloc[0][:20])
    
    # Aggregate histogram before plotting
    upper_limit = api_count_df["api_count"].quantile(0.99)
    api_plot_df = api_count_df[api_count_df["api_count"] <= upper_limit].copy()
    bin_edges = np.linspace(api_plot_df["api_count"].min(), api_plot_df["api_count"].max(), 41)
    api_plot_df["api_bin"] = pd.cut(api_plot_df["api_count"], bins=bin_edges, include_lowest=True)
    
    api_histogram_df = (
        api_plot_df.groupby(["category", "api_bin"], observed=True).size().reset_index(name="count")
        .assign(bin_midpoint=lambda d: d["api_bin"].apply(lambda i: i.mid).astype(float))
    )
    api_histogram_df = api_histogram_df.assign(
        percentage=api_histogram_df.groupby("category")["count"].transform(lambda v: v / v.sum() * 100)
    )
    api_histogram_plot = api_histogram_df[["category", "bin_midpoint", "count", "percentage"]].copy()
    
    hist_chart = alt.Chart(api_histogram_plot).mark_line(point=True).encode(
        x=alt.X("bin_midpoint:Q", title="Number of imported APIs", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("percentage:Q", title="Percentage of files"),
        color=alt.Color("category:N", title="File type"),
        tooltip=[
            "category:N",
            alt.Tooltip("bin_midpoint:Q", title="Imported APIs", format=".0f"),
            alt.Tooltip("percentage:Q", title="Percentage", format=".2f"),
        ],
    ).properties(title="Imported API Count: Malware vs. Benign", width=650, height=400)
    
    st.altair_chart(hist_chart, width="stretch")
    
    # ---------------------------------------------------------------------------
    # Networking API usage
    # ---------------------------------------------------------------------------
    st.header("Networking API usage")
    
    @st.cache_data(show_spinner="Counting networking APIs...")
    def compute_network_summary(_api_lists: pd.Series, labels: pd.Series):
        network_df = pd.DataFrame({
            "network_api_count": _api_lists.apply(lambda names: count_matching_apis(names, NETWORK_TERMS)),
            "label": labels,
        }).assign(category=lambda d: d["label"].map(CATEGORY_MAP))
    
        network_summary = (
            network_df.assign(uses_network_api=network_df["network_api_count"] > 0)
            .groupby("category")
            .agg(
                files=("uses_network_api", "size"),
                files_using_networking=("uses_network_api", "sum"),
                average_network_apis=("network_api_count", "mean"),
                median_network_apis=("network_api_count", "median"),
            )
            .reset_index()
            .assign(percentage_using_networking=lambda d: d["files_using_networking"] / d["files"] * 100)
        )
        return network_summary
    
    
    network_summary = compute_network_summary(api_lists, train_df["label"])
    st.dataframe(network_summary.round(2))
    
    net_chart = alt.Chart(network_summary).mark_bar().encode(
        x=alt.X("category:N", title="File type", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("percentage_using_networking:Q", title="Files importing networking APIs (%)"),
        tooltip=[
            "category:N",
            alt.Tooltip("percentage_using_networking:Q", title="Percentage", format=".2f"),
            "files_using_networking:Q",
            "files:Q",
        ],
    ).properties(title="Networking API Usage: Malware vs. Benign", width=500, height=350)
    
    st.altair_chart(net_chart, width="stretch")
    
    # ---------------------------------------------------------------------------
    # API category comparison
    # ---------------------------------------------------------------------------
    
    st.header("API category comparison")
    
    
    @st.cache_data(show_spinner="Computing API category breakdown...")
    def compute_category_df(_api_lists: pd.Series, labels: pd.Series):
        category_rows = []
        for category_name, terms in API_CATEGORIES.items():
            counts = _api_lists.apply(lambda names: count_matching_apis(names, terms))
            for file_type, label_value in [("Benign", 0), ("Malware", 1)]:
                class_counts = counts[labels == label_value]
                category_rows.append({
                    "api_category": category_name,
                    "file_type": file_type,
                    "percentage_of_files": (class_counts > 0).mean() * 100,
                    "average_count": class_counts.mean(),
                })
        return pd.DataFrame(category_rows)
    
    
    api_category_df = compute_category_df(api_lists, train_df["label"])
    st.dataframe(api_category_df.round(2))
    
    cat_chart = alt.Chart(api_category_df).mark_bar().encode(
        x=alt.X("api_category:N", title="API category", axis=alt.Axis(labelAngle=0)),
        xOffset="file_type:N",
        y=alt.Y("percentage_of_files:Q", title="Files importing category (%)"),
        color=alt.Color("file_type:N", title="File type"),
        tooltip=[
            "api_category:N",
            "file_type:N",
            alt.Tooltip("percentage_of_files:Q", title="Percentage", format=".2f"),
            alt.Tooltip("average_count:Q", title="Average API count", format=".2f"),
        ],
    ).properties(title="Imported API Categories: Malware vs. Benign", width=700, height=400)
    
    st.altair_chart(cat_chart, width="stretch")
 