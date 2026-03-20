
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