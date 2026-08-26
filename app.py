"""AI News Intelligence Platform - Main Streamlit Application."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split

from src.config import config
from src.data.loader import DataLoader
from src.data.validator import DataValidator
from src.data.profiler import DataProfiler
from src.nlp.column_detector import ColumnDetector
from src.nlp.cleaner import TextCleaner
from src.nlp.feature_extractor import FeatureExtractor
from src.nlp.text_statistics import TextStatistics
from src.models.sklearn_models import SklearnModels
from src.models.trainer import Trainer
from src.models.evaluator import Evaluator
from src.models.predictor import Predictor
from src.verification.llm_client import get_llm_provider
from src.verification.verifier import Verifier
from src.database.connection import db
from src.database.repository import Repository
from src.utils.metrics import compute_classification_metrics
from src.utils.export import export_predictions, export_verification_results
from src.utils.helpers import df_to_csv_bytes, df_to_json_bytes, format_metric

st.set_page_config(
    page_title="AI News Intelligence",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    .disclaimer {
        background-color: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .prediction-fake {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 8px;
        padding: 1.5rem;
    }
    .prediction-real {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 8px;
        padding: 1.5rem;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

if "df" not in st.session_state:
    st.session_state.df = None
if "cleaned_texts" not in st.session_state:
    st.session_state.cleaned_texts = None
if "text_column" not in st.session_state:
    st.session_state.text_column = None
if "target_column" not in st.session_state:
    st.session_state.target_column = None
if "label_mapping" not in st.session_state:
    st.session_state.label_mapping = {}
if "trained_models" not in st.session_state:
    st.session_state.trained_models = {}
if "evaluator" not in st.session_state:
    st.session_state.evaluator = Evaluator()
if "predictor" not in st.session_state:
    st.session_state.predictor = Predictor()
if "feature_extractor" not in st.session_state:
    st.session_state.feature_extractor = None
if "X_train" not in st.session_state:
    st.session_state.X_train = None
if "X_test" not in st.session_state:
    st.session_state.X_test = None
if "y_train" not in st.session_state:
    st.session_state.y_train = None
if "y_test" not in st.session_state:
    st.session_state.y_test = None
if "experiment_log" not in st.session_state:
    st.session_state.experiment_log = []
if "db_repository" not in st.session_state:
    st.session_state.db_repository = Repository()
if "verifier" not in st.session_state:
    st.session_state.verifier = None


def main():
    st.sidebar.markdown('<div class="main-header">📰 AI News Intelligence</div>', unsafe_allow_html=True)
    st.sidebar.markdown("---")

    pages = {
        "🏠 Overview": "overview",
        "📤 Upload Dataset": "upload",
        "🔍 Data Explorer": "explorer",
        "🧹 NLP Preprocessing": "preprocessing",
        "🧠 Feature Extraction": "features",
        "🤖 Model Training": "training",
        "📊 Evaluation": "evaluation",
        "🔎 News Analyzer": "analyzer",
        "🧾 Fact Verification": "verification",
        "📈 Analytics": "analytics",
        "🗄️ PostgreSQL": "database",
        "⚙️ Settings": "settings",
    }

    choice = st.sidebar.radio("Navigation", list(pages.keys()))
    page = pages[choice]

    if page == "overview":
        show_overview()
    elif page == "upload":
        show_upload()
    elif page == "explorer":
        show_explorer()
    elif page == "preprocessing":
        show_preprocessing()
    elif page == "features":
        show_features()
    elif page == "training":
        show_training()
    elif page == "evaluation":
        show_evaluation()
    elif page == "analyzer":
        show_analyzer()
    elif page == "verification":
        show_verification()
    elif page == "analytics":
        show_analytics()
    elif page == "database":
        show_database()
    elif page == "settings":
        show_settings()


def show_overview():
    st.markdown('<div class="main-header">📰 AI News Intelligence Platform</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Detect suspicious patterns, classify news content, '
        "and optionally perform AI-assisted claim verification.</div>",
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Dataset Loaded", "✅" if st.session_state.df is not None else "❌")
    with col2:
        st.metric("Models Trained", len(st.session_state.trained_models))
    with col3:
        st.metric("Text Column", st.session_state.text_column or "Not set")
    with col4:
        st.metric("Target Column", st.session_state.target_column or "Not set")

    st.markdown("---")
    st.subheader("Main Workflow")
    st.markdown("""
    1. **Upload** → Import CSV or JSON dataset
    2. **Explore** → Understand your data
    3. **Preprocess** → Clean and normalize text
    4. **Train** → Build ML/DL classification models
    5. **Predict** → Analyze new articles
    6. **Verify** → Optional LLM-assisted fact-checking
    7. **Analyze** → View results and export
    """)

    st.markdown("---")
    st.subheader("Supported Models")
    cols = st.columns(3)
    with cols[0]:
        st.info("**Classical ML**\n- Logistic Regression\n- Linear SVM\n- Naive Bayes\n- Random Forest")
    with cols[1]:
        st.info("**Deep Learning**\n- TensorFlow/Keras\n- PyTorch LSTM")
    with cols[2]:
        st.info("**LLM Verification**\n- OpenAI\n- Anthropic\n- Mock (offline)")

    st.markdown("---")
    st.markdown(
        '<div class="disclaimer">'
        "<strong>⚠️ Disclaimer:</strong> Model predictions are probabilistic and depend on the training data. "
        "A 'fake' prediction does not by itself establish that a claim is false. "
        "LLM-generated verification should be independently checked against reliable evidence "
        "before making consequential decisions.</div>",
        unsafe_allow_html=True,
    )


def show_upload():
    st.markdown('<div class="main-header">📤 Upload Dataset</div>', unsafe_allow_html=True)
    st.markdown("Import your news dataset in CSV or JSON format.")

    tab1, tab2 = st.tabs(["📊 CSV", "📄 JSON"])

    with tab1:
        csv_file = st.file_uploader("Upload News Dataset CSV", type=["csv"], key="csv_uploader")
        if csv_file:
            with st.spinner("Loading CSV..."):
                df, status = DataLoader.load_csv(csv_file)
            if df is not None:
                st.success(f"✅ Loaded CSV: {df.shape[0]} rows, {df.shape[1]} columns")
                st.session_state.df = df
                _show_dataset_summary(df)
            else:
                st.error(f"❌ {status}")

    with tab2:
        json_file = st.file_uploader("Upload News Dataset JSON", type=["json"], key="json_uploader")
        if json_file:
            with st.spinner("Loading JSON..."):
                df, status = DataLoader.load_json(json_file)
            if df is not None:
                st.success(f"✅ Loaded JSON: {df.shape[0]} rows, {df.shape[1]} columns")
                st.session_state.df = df
                _show_dataset_summary(df)
            else:
                st.error(f"❌ {status}")

    if st.session_state.df is not None:
        st.markdown("---")
        st.subheader("Quick Column Detection")
        df = st.session_state.df
        text_cols = ColumnDetector.detect_text_columns(df)
        target_cols = ColumnDetector.detect_target_columns(df)

        col1, col2 = st.columns(2)
        with col1:
            st.write("**Detected Text Columns:**")
            st.write(text_cols if text_cols else "None detected")
        with col2:
            st.write("**Detected Target Columns:**")
            st.write(target_cols if target_cols else "None detected")


def _show_dataset_summary(df):
    """Display dataset summary after upload."""
    is_valid, warnings, info = DataValidator.validate(df)

    cols = st.columns(4)
    cols[0].metric("Rows", info["rows"])
    cols[1].metric("Columns", info["columns"])
    cols[2].metric("Missing Total", info["missing_total"])
    cols[3].metric("Duplicates", info["duplicate_rows"])

    if warnings:
        for w in warnings:
            st.warning(w)

    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)


def show_explorer():
    st.markdown('<div class="main-header">🔍 Data Explorer</div>', unsafe_allow_html=True)

    if st.session_state.df is None:
        st.info("Please upload a dataset first.")
        return

    df = st.session_state.df
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Preview", "Statistics", "Missing Values", "Class Distribution", "NLP Statistics"])

    with tab1:
        st.dataframe(df.head(50), use_container_width=True)

    with tab2:
        profile = DataProfiler.profile(df)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Rows", profile["n_rows"])
        col2.metric("Columns", profile["n_columns"])
        col3.metric("Numeric Cols", len(profile["numeric_columns"]))
        col4.metric("Text Cols", len(profile["text_columns"]))

        st.subheader("Column Types")
        st.json(profile["dtypes"])

    with tab3:
        missing = df.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        if len(missing) > 0:
            fig = px.bar(x=missing.index, y=missing.values, labels={"x": "Column", "y": "Missing Count"})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("No missing values found.")

    with tab4:
        target_col = st.selectbox("Select target column for distribution", df.columns)
        if target_col:
            dist = df[target_col].value_counts()
            fig = px.pie(values=dist.values, names=dist.index, title=f"Distribution of {target_col}")
            st.plotly_chart(fig, use_container_width=True)

    with tab5:
        text_cols = [c for c in df.columns if df[c].dtype == "object"]
        if text_cols:
            selected = st.selectbox("Select text column for NLP stats", text_cols)
            if selected:
                stats = TextStatistics.compute_stats(df[selected])
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Documents", stats["document_count"])
                col2.metric("Avg Words", stats["avg_word_count"])
                col3.metric("Vocabulary", stats["vocabulary_size"])
                col4.metric("Duplicates", stats["duplicate_documents"])
                st.json(stats)
        else:
            st.info("No text columns found.")


def show_preprocessing():
    st.markdown('<div class="main-header">🧹 NLP Preprocessing</div>', unsafe_allow_html=True)

    if st.session_state.df is None:
        st.info("Please upload a dataset first.")
        return

    df = st.session_state.df
    text_cols = ColumnDetector.detect_text_columns(df)

    st.subheader("Column Selection")
    col1, col2 = st.columns(2)
    with col1:
        title_col = st.selectbox("Title Column", ["None"] + text_cols)
    with col2:
        main_text_col = st.selectbox("Main Text Column", text_cols if text_cols else ["None"])

    if main_text_col == "None" or (not text_cols and title_col == "None"):
        st.warning("No text column detected. Please select a text column manually.")
        return

    st.session_state.text_column = main_text_col if main_text_col != "None" else title_col

    st.subheader("Text Cleaning Controls")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        lowercase = st.checkbox("Lowercase", value=True)
        remove_urls = st.checkbox("Remove URLs", value=True)
    with col2:
        remove_html = st.checkbox("Remove HTML", value=True)
        remove_punct = st.checkbox("Remove punctuation", value=False)
    with col3:
        remove_stop = st.checkbox("Remove stopwords", value=False)
    with col4:
        stemming = st.checkbox("Stemming", value=False)
        lemmatization = st.checkbox("Lemmatization", value=False)

    if st.button("Apply Preprocessing", type="primary"):
        with st.spinner("Cleaning text..."):
            cleaner = TextCleaner(
                lowercase=lowercase,
                remove_urls=remove_urls,
                remove_html=remove_html,
                remove_punctuation=remove_punct,
                remove_stopwords=remove_stop,
                stemming=stemming,
                lemmatization=lemmatization,
            )

            if title_col != "None" and main_text_col != "None":
                combined = df[title_col].fillna("") + " " + df[main_text_col].fillna("")
            elif main_text_col != "None":
                combined = df[main_text_col].fillna("")
            else:
                combined = df[title_col].fillna("")

            st.session_state.cleaned_texts = cleaner.clean_series(combined)
            st.success("✅ Preprocessing complete!")

    if st.session_state.cleaned_texts is not None:
        st.subheader("Before / After Comparison")
        comparison = pd.DataFrame({
            "Original": df[st.session_state.text_column].fillna("").head(10),
            "Cleaned": st.session_state.cleaned_texts.head(10),
        })
        st.dataframe(comparison, use_container_width=True)

        stats = TextStatistics.compute_stats(st.session_state.cleaned_texts)
        st.subheader("Text Statistics")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Avg Words", stats["avg_word_count"])
        col2.metric("Vocabulary Size", stats["vocabulary_size"])
        col3.metric("Unique Token Ratio", stats["unique_token_ratio"])
        col4.metric("Empty Docs", stats["empty_documents"])


def show_features():
    st.markdown('<div class="main-header">🧠 Feature Extraction</div>', unsafe_allow_html=True)

    if st.session_state.cleaned_texts is None:
        st.info("Please preprocess your data first.")
        return

    texts = st.session_state.cleaned_texts

    st.subheader("TF-IDF Configuration")
    col1, col2, col3 = st.columns(3)
    with col1:
        max_features = st.number_input("Max Features", 100, 50000, 10000)
    with col2:
        min_df = st.number_input("Min DF", 1, 20, 2)
    with col3:
        max_df = st.slider("Max DF", 0.5, 1.0, 0.95)

    ngram_range = st.selectbox("N-gram Range", [(1, 1), (1, 2), (1, 3)], format_func=lambda x: f"{x[0]}-{x[1]}")
    sublinear_tf = st.checkbox("Sublinear TF", value=True)
    include_stats = st.checkbox("Include Statistical Features", value=False)

    if st.button("Extract Features", type="primary"):
        with st.spinner("Extracting features..."):
            extractor = FeatureExtractor(
                max_features=max_features,
                ngram_range=ngram_range,
                min_df=min_df,
                max_df=max_df,
                sublinear_tf=sublinear_tf,
            )
            X_tfidf = extractor.fit_transform_tfidf(texts)
            st.session_state.feature_extractor = extractor

            if include_stats:
                stat_features = FeatureExtractor.extract_statistical_features(texts)
                from scipy.sparse import hstack, csr_matrix
                X_combined = hstack([X_tfidf, csr_matrix(stat_features.values)])
                st.session_state.X_features = X_combined
            else:
                st.session_state.X_features = X_tfidf

            st.success(f"✅ Features extracted: {X_tfidf.shape[1]} dimensions")

    if st.session_state.feature_extractor is not None:
        extractor = st.session_state.feature_extractor
        st.subheader("Top N-grams")
        col1, col2 = st.columns(2)
        with col1:
            top_unigrams = extractor.get_top_ngrams(texts, n=15, gram_size=1)
            if top_unigrams:
                words, counts = zip(*top_unigrams)
                fig = px.bar(x=list(counts), y=list(words), orientation="h", title="Top Unigrams")
                st.plotly_chart(fig, use_container_width=True)
        with col2:
            top_bigrams = extractor.get_top_ngrams(texts, n=15, gram_size=2)
            if top_bigrams:
                words, counts = zip(*top_bigrams)
                fig = px.bar(x=list(counts), y=list(words), orientation="h", title="Top Bigrams")
                st.plotly_chart(fig, use_container_width=True)


def show_training():
    st.markdown('<div class="main-header">🤖 Model Training</div>', unsafe_allow_html=True)

    if st.session_state.X_features is None:
        st.info("Please extract features first.")
        return

    df = st.session_state.df
    target_cols = ColumnDetector.detect_target_columns(df)

    st.subheader("Target Selection")
    target_col = st.selectbox("Target Column", target_cols if target_cols else df.columns.tolist())

    if target_col:
        st.session_state.target_column = target_col
        unique_labels = df[target_col].dropna().unique()
        st.write(f"**Unique labels:** {unique_labels}")

        if len(unique_labels) < 2:
            st.error("Need at least 2 classes for classification.")
            return

        st.subheader("Label Normalization")
        label_mapping = {}
        default_fake = ["fake", "false", "misinformation", "misleading", "unreliable", "conspiracy", "satire", "0"]
        default_real = ["real", "true", "reliable", "verified", "factual", "1"]

        for label in unique_labels:
            label_str = str(label).lower()
            if label_str in default_fake:
                default = "Fake"
            elif label_str in default_real:
                default = "Real"
            else:
                default = str(label)

            mapped = st.text_input(f"Map '{label}' to:", value=default, key=f"label_{label}")
            label_mapping[label] = mapped

        st.session_state.label_mapping = label_mapping

        y = df[target_col].map(label_mapping).to_numpy(dtype=object)

        st.subheader("Model Selection")
        col1, col2 = st.columns(2)
        with col1:
            use_lr = st.checkbox("Logistic Regression", value=True)
            use_svm = st.checkbox("Linear SVM", value=True)
            use_nb = st.checkbox("Naive Bayes", value=True)
        with col2:
            use_rf = st.checkbox("Random Forest", value=True)
            use_tf = st.checkbox("TensorFlow", value=False)
            use_pt = st.checkbox("PyTorch", value=False)

        st.subheader("Training Configuration")
        col1, col2, col3 = st.columns(3)
        with col1:
            test_size = st.slider("Test Size", 0.1, 0.4, 0.2)
        with col2:
            cv_folds = st.number_input("CV Folds", 2, 10, 5)
        with col3:
            optimize = st.selectbox("Optimization", ["None", "Grid Search", "Randomized Search"])

        if st.button("🚀 Train Selected Models", type="primary"):
            X_train, X_test, y_train, y_test = train_test_split(
                st.session_state.X_features, y, test_size=test_size,
                random_state=42, stratify=y if len(np.unique(y)) >= 2 else None,
            )
            st.session_state.X_train = X_train
            st.session_state.X_test = X_test
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test

            trainer = Trainer()
            evaluator = Evaluator()
            sklearn_models = SklearnModels()

            models_to_train = []
            if use_lr:
                models_to_train.append("logistic_regression")
            if use_svm:
                models_to_train.append("linear_svm")
            if use_nb:
                models_to_train.append("naive_bayes")
            if use_rf:
                models_to_train.append("random_forest")

            progress = st.progress(0)
            for i, model_key in enumerate(models_to_train):
                with st.spinner(f"Training {model_key}..."):
                    try:
                        result = trainer.train_sklearn(
                            model_key, X_train, y_train, X_test, y_test,
                            optimize=None if optimize == "None" else optimize.split()[0].lower(),
                        )
                        st.session_state.trained_models[model_key] = result["model"]
                        evaluator.add_result(
                            model_key, result["metrics"],
                            result["predictions"], result["probabilities"],
                        )
                        st.success(f"✅ {model_key}: F1={result['metrics'].get('f1', 0):.4f}")
                    except Exception as e:
                        st.error(f"❌ {model_key} failed: {e}")
                progress.progress((i + 1) / max(len(models_to_train), 1))

            if use_tf:
                try:
                    from src.models.tensorflow_model import TensorFlowClassifier, TF_AVAILABLE
                    if TF_AVAILABLE:
                        with st.spinner("Training TensorFlow model..."):
                            tf_clf = TensorFlowClassifier(epochs=10)
                            num_classes = len(np.unique(y_train))
                            tf_clf.build(num_classes, pd.Series([str(l) for l in st.session_state.cleaned_texts.iloc[y_train.shape[0]:]]))
                            tf_clf.train(
                                pd.Series([str(l) for l in st.session_state.cleaned_texts.iloc[:y_train.shape[0]]]),
                                y_train,
                                pd.Series([str(l) for l in st.session_state.cleaned_texts.iloc[y_train.shape[0]:]]),
                                y_test,
                            )
                            st.session_state.trained_models["tensorflow"] = tf_clf
                            st.success("✅ TensorFlow model trained")
                    else:
                        st.warning("TensorFlow not available")
                except Exception as e:
                    st.error(f"❌ TensorFlow failed: {e}")

            if use_pt:
                try:
                    from src.models.pytorch_model import PyTorchClassifier, TORCH_AVAILABLE
                    if TORCH_AVAILABLE:
                        with st.spinner("Training PyTorch model..."):
                            pt_clf = PyTorchClassifier(epochs=10)
                            num_classes = len(np.unique(y_train))
                            pt_clf.build(num_classes, pd.Series([str(l) for l in st.session_state.cleaned_texts.iloc[:y_train.shape[0]]]))
                            pt_clf.train(
                                pd.Series([str(l) for l in st.session_state.cleaned_texts.iloc[:y_train.shape[0]]]),
                                y_train,
                                pd.Series([str(l) for l in st.session_state.cleaned_texts.iloc[y_train.shape[0]:]]),
                                y_test,
                            )
                            st.session_state.trained_models["pytorch"] = pt_clf
                            st.success("✅ PyTorch model trained")
                    else:
                        st.warning("PyTorch not available")
                except Exception as e:
                    st.error(f"❌ PyTorch failed: {e}")

            st.session_state.evaluator = evaluator
            st.session_state.predictor = Predictor()
            for name, model in st.session_state.trained_models.items():
                st.session_state.predictor.register_model(name, model)

            st.success("🎉 Training complete! Go to Evaluation to see results.")


def show_evaluation():
    st.markdown('<div class="main-header">📊 Model Evaluation</div>', unsafe_allow_html=True)

    evaluator = st.session_state.evaluator
    if not evaluator.results:
        st.info("No trained models to evaluate. Please train models first.")
        return

    st.subheader("Model Comparison")
    comparison = evaluator.get_comparison_table()
    st.dataframe(comparison, use_container_width=True)

    st.subheader("Primary Metric")
    metric = st.selectbox("Select primary metric", ["f1", "accuracy", "precision", "recall", "roc_auc"])
    best = evaluator.get_best_model(metric)
    if best:
        st.success(f"🏆 Best model by {metric}: **{best}**")

    st.subheader("Detailed Analysis")
    selected_model = st.selectbox("Select model for detailed view", list(evaluator.results.keys()))

    if selected_model:
        y_test = st.session_state.y_test
        col1, col2 = st.columns(2)

        with col1:
            cm = evaluator.get_confusion_matrix(selected_model, y_test)
            if cm is not None:
                fig = px.imshow(cm, text_auto=True, title=f"Confusion Matrix - {selected_model}",
                               labels=dict(x="Predicted", y="Actual"))
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            fpr, tpr, _ = evaluator.get_roc_data(selected_model, y_test)
            if fpr is not None:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC"))
                fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line_dash="dash", name="Random"))
                fig.update_layout(title=f"ROC Curve - {selected_model}", xaxis_title="FPR", yaxis_title="TPR")
                st.plotly_chart(fig, use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            precision, recall, _ = evaluator.get_pr_data(selected_model, y_test)
            if precision is not None:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines", name="PR"))
                fig.update_layout(title=f"Precision-Recall Curve - {selected_model}",
                                 xaxis_title="Recall", yaxis_title="Precision")
                st.plotly_chart(fig, use_container_width=True)

        with col4:
            if st.session_state.feature_extractor and selected_model in ["logistic_regression", "naive_bayes", "random_forest"]:
                sklearn_models = SklearnModels()
                features = sklearn_models.get_top_features(
                    st.session_state.trained_models[selected_model],
                    st.session_state.feature_extractor.feature_names or [],
                    selected_model,
                )
                if features["positive"]:
                    st.write("**Top Positive Features:**")
                    for feat, weight in features["positive"][:10]:
                        st.write(f"- {feat}: {weight}")

    st.markdown("---")
    st.markdown(
        '<div class="disclaimer">'
        "<strong>⚠️ Note:</strong> These metrics reflect model performance on the test set. "
        "They do not establish factual truth. A high F1 score means the model distinguishes "
        "between classes well, not that its predictions are factually correct.</div>",
        unsafe_allow_html=True,
    )


def show_analyzer():
    st.markdown('<div class="main-header">🔎 News Analyzer</div>', unsafe_allow_html=True)

    if not st.session_state.trained_models:
        st.info("Please train models first.")
        return

    st.subheader("Analyze a Single Article")
    title = st.text_input("Title (optional)")
    text = st.text_area("Paste News Article", height=200)

    model_choice = st.selectbox("Model", list(st.session_state.trained_models.keys()))

    if st.button("Analyze Article", type="primary") and text:
        with st.spinner("Analyzing..."):
            cleaner = TextCleaner()
            cleaned = cleaner.clean(text)

            if st.session_state.feature_extractor:
                X = st.session_state.feature_extractor.transform_tfidf(pd.Series([cleaned]))
                model = st.session_state.trained_models[model_choice]
                pred = model.predict(X)[0]

                proba = None
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(X)
                    if proba.ndim > 1 and proba.shape[1] == 2:
                        proba = proba[0][1]
                    else:
                        proba = proba[0]

                if str(pred).lower() in ["fake", "false", "0", "misleading"]:
                    st.markdown(
                        f'<div class="prediction-fake">'
                        f"<h3>⚠️ Model Prediction: Likely Misleading/Fake</h3>"
                        f"<p><strong>Confidence:</strong> {proba:.1%}" if proba else ""
                        f"<br><strong>Model:</strong> {model_choice}"
                        f"</p></div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<div class="prediction-real">'
                        f"<h3>✅ Model Prediction: Likely Real/Reliable</h3>"
                        f"<p><strong>Confidence:</strong> {proba:.1%}" if proba else ""
                        f"<br><strong>Model:</strong> {model_choice}"
                        f"</p></div>",
                        unsafe_allow_html=True,
                    )

                st.markdown(
                    '<div class="disclaimer">'
                    "<strong>⚠️ Important:</strong> This is a model prediction, not a fact-check. "
                    "The model classifies based on patterns in training data, not factual accuracy.</div>",
                    unsafe_allow_html=True,
                )

    st.markdown("---")
    st.subheader("Batch Prediction")
    batch_file = st.file_uploader("Upload CSV/JSON for batch prediction", type=["csv", "json"])
    if batch_file:
        if batch_file.name.endswith(".csv"):
            batch_df, status = DataLoader.load_csv(batch_file)
        else:
            batch_df, status = DataLoader.load_json(batch_file)

        if batch_df is not None:
            text_cols = ColumnDetector.detect_text_columns(batch_df)
            batch_text_col = st.selectbox("Text column in batch file", text_cols if text_cols else batch_df.columns.tolist())

            if st.button("Run Batch Prediction"):
                cleaner = TextCleaner()
                cleaned = cleaner.clean_series(batch_df[batch_text_col].fillna(""))
                X_batch = st.session_state.feature_extractor.transform_tfidf(cleaned)
                model = st.session_state.trained_models[model_choice]
                preds = model.predict(X_batch)

                batch_df["prediction"] = preds
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(X_batch)
                    if proba.ndim > 1 and proba.shape[1] == 2:
                        batch_df["probability"] = proba[:, 1]
                    else:
                        batch_df["probability"] = np.max(proba, axis=1)
                batch_df["model_name"] = model_choice

                st.dataframe(batch_df.head(20), use_container_width=True)
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button("⬇ Download CSV", df_to_csv_bytes(batch_df), "predictions.csv", "text/csv")
                with col2:
                    st.download_button("⬇ Download JSON", df_to_json_bytes(batch_df), "predictions.json", "application/json")


def show_verification():
    st.markdown('<div class="main-header">🧾 Fact Verification</div>', unsafe_allow_html=True)

    if st.session_state.verifier is None:
        st.session_state.verifier = Verifier()

    st.markdown(
        '<div class="disclaimer">'
        "⚠️ LLM verification sends selected article text to the configured external AI provider. "
        "Enable in Settings.</div>",
        unsafe_allow_html=True,
    )

    text = st.text_area("Article to verify", height=200)
    if st.button("Verify Article", type="primary") and text:
        with st.spinner("Verifying..."):
            result = st.session_state.verifier.verify(text)

        st.subheader("Verification Result")
        col1, col2, col3 = st.columns(3)
        col1.metric("Assessment", result["assessment"])
        col2.metric("Confidence", f"{result['confidence']:.0%}")
        col3.metric("Claims Found", len(result["claims"]))

        if result["claims"]:
            st.subheader("Extracted Claims")
            for claim in result["claims"]:
                st.write(f"- {claim.get('claim', claim)}")

        if result["red_flags"]:
            st.subheader("🚩 Red Flags")
            for flag in result["red_flags"]:
                st.warning(flag)

        if result["evidence_needed"]:
            st.subheader("📋 Evidence Needed")
            for ev in result["evidence_needed"]:
                st.info(ev)

        st.subheader("Reasoning")
        st.write(result["reasoning_summary"])

        if result["evidence"].get("status") == "no_source":
            st.info("ℹ️ " + result["evidence"]["message"])

    st.markdown("---")
    st.markdown(
        '<div class="disclaimer">'
        "<strong>⚠️ Important:</strong> ML Classification and Fact Verification are different. "
        "A classifier predicting an article resembles previously labeled fake articles is NOT proof "
        "that the article is factually false. These outputs are probabilistic assessments.</div>",
        unsafe_allow_html=True,
    )


def show_analytics():
    st.markdown('<div class="main-header">📈 Analytics Dashboard</div>', unsafe_allow_html=True)

    if st.session_state.df is None:
        st.info("Please upload a dataset first.")
        return

    df = st.session_state.df

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("📰 Articles", len(df))
    col2.metric("🤖 Predictions", len(st.session_state.trained_models))

    if st.session_state.target_column:
        dist = df[st.session_state.target_column].value_counts()
        fake_count = sum(v for k, v in dist.items() if str(k).lower() in ["fake", "false", "0"])
        real_count = sum(v for k, v in dist.items() if str(k).lower() in ["real", "true", "1"])
        col3.metric("⚠️ Likely Fake", fake_count)
        col4.metric("✅ Likely Real", real_count)

    col5.metric("🔎 Verified", 0)
    col6.metric("❓ Unverified", 0)

    st.markdown("---")

    if st.session_state.evaluator and st.session_state.evaluator.results:
        st.subheader("Model Performance")
        comparison = st.session_state.evaluator.get_comparison_table()
        if not comparison.empty and "f1" in comparison.columns:
            fig = px.bar(comparison, x="Model", y="f1", title="F1 Score by Model",
                        color="f1", color_continuous_scale="Viridis")
            st.plotly_chart(fig, use_container_width=True)

    if st.session_state.target_column:
        st.subheader("Class Distribution")
        dist = df[st.session_state.target_column].value_counts()
        fig = px.pie(values=dist.values, names=dist.index, title="Label Distribution")
        st.plotly_chart(fig, use_container_width=True)

    if st.session_state.feature_extractor and st.session_state.cleaned_texts is not None:
        st.subheader("Top Terms")
        top = st.session_state.feature_extractor.get_top_ngrams(st.session_state.cleaned_texts, n=20, gram_size=1)
        if top:
            words, counts = zip(*top)
            fig = px.bar(x=list(counts), y=list(words), orientation="h", title="Most Common Terms")
            st.plotly_chart(fig, use_container_width=True)


def show_database():
    st.markdown('<div class="main-header">🗄️ PostgreSQL Dashboard</div>', unsafe_allow_html=True)

    repo = st.session_state.db_repository

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Test Connection"):
            success, msg = db.test_connection()
            if success:
                st.success(f"✅ {msg}")
            else:
                st.error(f"❌ {msg}")

    with col2:
        if st.button("Initialize Tables"):
            if db.connect():
                db.create_tables()
                st.success("✅ Tables created")
            else:
                st.error("❌ Could not connect to database")

    st.markdown("---")

    if db.connected:
        counts = repo.get_counts()
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Articles", counts["articles"])
        col2.metric("Predictions", counts["predictions"])
        col3.metric("Verifications", counts["verifications"])
        col4.metric("Model Runs", counts["model_runs"])
        col5.metric("Datasets", counts["datasets"])

        st.subheader("Actions")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Save Current Dataset") and st.session_state.df is not None:
                repo.save_dataset(
                    "uploaded_dataset", len(st.session_state.df),
                    len(st.session_state.df.columns),
                    st.session_state.text_column, st.session_state.target_column,
                )
                st.success("Dataset saved")
        with col2:
            if st.button("Save Predictions") and st.session_state.evaluator.results:
                for model_name, result in st.session_state.evaluator.results.items():
                    repo.save_model_run(model_name, "uploaded_dataset", metrics=result["metrics"])
                st.success("Predictions saved")
    else:
        st.info("ℹ️ PostgreSQL not connected. Continue in local mode.")


def show_settings():
    st.markdown('<div class="main-header">⚙️ Settings</div>', unsafe_allow_html=True)

    st.subheader("LLM Configuration")
    llm_enabled = st.checkbox("Enable LLM Verification", value=False)
    llm_provider = st.selectbox("LLM Provider", ["mock", "openai", "anthropic"])
    llm_model = st.text_input("LLM Model", value=config.llm_model or "gpt-3.5-turbo")
    max_articles = st.number_input("Max articles per batch", 1, 100, 10)

    if llm_provider != "mock":
        st.text_input("API Key", type="password", value="Configured" if config.openai_api_key or config.anthropic_api_key else "")

    if st.button("Save Settings"):
        st.session_state.verifier = Verifier(provider=llm_provider)
        st.success("✅ Settings saved")

    st.markdown("---")
    st.subheader("Generate Sample Data")
    if st.button("Generate Sample Dataset"):
        from src.utils.generate_sample_data import generate_synthetic_dataset
        path = generate_synthetic_dataset()
        st.success(f"✅ Sample data generated: {path}")

    st.markdown("---")
    st.subheader("Application Info")
    st.json({
        "Config Path": str(config.base_dir / "config" / "config.yaml"),
        "Random State": config.random_state,
        "Test Size": config.test_size,
        "CV Folds": config.cv_folds,
        "LLM Provider": config.llm_provider,
    })


if __name__ == "__main__":
    main()
