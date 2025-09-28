# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import plotly.express as px

st.title("Auto Data Analyzer & Visualizer (demo)")

uploaded = st.file_uploader("Upload CSV", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)
    st.write("Preview:", df.head())
    # Basic summary
    st.write("Shape:", df.shape)
    st.write(df.describe(include='all').T)

    # Auto-detect numeric columns
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        st.warning("No numeric columns detected — visualizations limited.")
    else:
        st.header("Quick EDA")
        col = st.selectbox("Choose numeric column for histogram", num_cols)
        st.plotly_chart(px.histogram(df, x=col, nbins=30, title=f"Histogram of {col}"))

        # PCA + clustering demo
        if st.button("Run PCA + KMeans (auto)"):
            # simple preprocessing
            X = df[num_cols].copy()
            X = X.dropna()  # keep it simple for demo
            scaler = StandardScaler()
            Xs = scaler.fit_transform(X)
            pca = PCA(n_components=2)
            Z = pca.fit_transform(Xs)
            k = st.slider("Number of clusters", 2, 8, 3)
            km = KMeans(n_clusters=k, random_state=0).fit(Z)
            df_pca = pd.DataFrame(Z, columns=["PC1", "PC2"])
            df_pca["cluster"] = km.labels_
            fig = px.scatter(df_pca, x="PC1", y="PC2", color="cluster", title="PCA projection with KMeans")
            st.plotly_chart(fig)
            st.write("Explained variance (PC1, PC2):", pca.explained_variance_ratio_)


