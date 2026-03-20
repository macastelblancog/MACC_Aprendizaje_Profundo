
import pandas as pd
import numpy as np
import plotly.express as px

def pixels_report(X):

    rows = []

    for key in X.keys():
        values = X[key].flatten()
        
        df_tmp = pd.DataFrame({
            "pixel_value": values,
            "dataset": key
        })
        
        rows.append(df_tmp)

    df_plot = pd.concat(rows, ignore_index=True)

    fig = px.histogram(
        df_plot,
        x="pixel_value",
        color="dataset",
        histnorm="probability density",  # comparable across datasets
        opacity=0.5,
        barmode="overlay",
        nbins=100,
        title="Pixel Value Distribution Across Datasets"
    )

    fig.update_layout(
        xaxis_title="Pixel Value",
        yaxis_title="Density"
    )
    return fig

def labels_report(Y):
    rows = []

    for key in Y.keys():
        values = np.ravel(Y[key])
        class_counts = pd.Series(values).value_counts(normalize=True).sort_index() * 100

        for cls, pct in class_counts.items():
            rows.append({
                "dataset": key,
                "class": str(cls),
                "percentage": pct
            })

    df_plot = pd.DataFrame(rows)

    fig = px.bar(
        df_plot,
        x="class",
        y="percentage",
        color="dataset",
        barmode="group",
        text="percentage",
        title="Class Distribution by Dataset",
        labels={
            "class": "Class",
            "percentage": "Percentage (%)",
            "dataset": "Dataset"
        }
    )

    fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    fig.update_layout(yaxis_title="Percentage (%)")
    return fig

def imbalance_report(Y, key="train"):
    y = np.ravel(Y[key])

    # Conteo por clase
    counts = pd.Series(y).value_counts().sort_index()

    # Asegurar que existan todas las clases (0 a n-1)
    all_classes = sorted(np.unique(y))
    counts = counts.reindex(all_classes, fill_value=0)

    # Estadísticas básicas
    total = counts.sum()
    percentages = counts / total * 100

    max_count = counts.max()
    min_count = counts.min()

    imbalance_ratio = max_count / max(min_count, 1)  # evitar división por cero

    # Ratio relativo por clase (vs clase mayoritaria)
    relative_ratio = counts / max_count

    # Construir reporte
    df = pd.DataFrame({
        "count": counts,
        "percentage": percentages.round(2),
        "ratio_vs_max": relative_ratio.round(3)
    })

    # Identificar clases críticas
    df["is_minority"] = df["ratio_vs_max"] < 0.3
    df["is_rare"] = df["ratio_vs_max"] < 0.1

    summary = {
        "total_samples": int(total),
        "n_classes": len(counts),
        "max_count": int(max_count),
        "min_count": int(min_count),
        "imbalance_ratio_max_min": round(imbalance_ratio, 2)
    }

    return df, summary