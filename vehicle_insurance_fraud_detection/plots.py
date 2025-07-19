# vehicle_insurance_fraud_detection/plots.py

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


def plot_class_distribution(y: pd.Series, title="Class Distribution"):
    """Bar plot of fraud vs non-fraud counts."""
    counts = y.value_counts().rename({0: "Non-Fraud", 1: "Fraud"})
    fig = px.bar(
        x=counts.index,
        y=counts.values,
        labels={"x": "Class", "y": "Count"},
        title=title,
        color=counts.index,
        color_discrete_map={"Non-Fraud": "green", "Fraud": "red"},
    )
    return fig


def plot_feature_importance(importance_df: pd.DataFrame, top_n=20):
    """
    Plots top N features by importance.
    Expects a DataFrame with ['Feature', 'Importance'].
    """
    df = importance_df.sort_values(by="Importance", ascending=False).head(top_n)

    fig = px.bar(
        df,
        x="Importance",
        y="Feature",
        orientation="h",
        title=f"Top {top_n} Important Features",
        color="Importance",
        color_continuous_scale="Bluered"
    )
    fig.update_layout(yaxis=dict(autorange="reversed"))
    return fig


def plot_confusion_matrix(cm, labels=["Non-Fraud", "Fraud"]):
    """Plot a confusion matrix heatmap using Plotly."""
    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=labels,
            y=labels,
            hoverongaps=False,
            colorscale="Blues",
            text=cm,
            texttemplate="%{text}",
        )
    )
    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted Label",
        yaxis_title="True Label"
    )
    return fig


def plot_roc_curve(fpr, tpr, auc_score):
    """Plot ROC curve."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC Curve"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(dash="dash")))

    fig.update_layout(
        title=f"ROC Curve (AUC = {auc_score:.2f})",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        showlegend=True
    )
    return fig
