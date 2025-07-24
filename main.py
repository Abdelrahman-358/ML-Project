import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, silhouette_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import joblib

# Neural network imports 
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

st.title("Mini ML Explorer")

# 1) Data upload
uploaded = st.file_uploader("Upload a CSV file", type="csv")
if not uploaded:
    st.info("Please upload a CSV file to proceed.")
    st.stop()

df = pd.read_csv(uploaded)
st.subheader("Data Preview")
st.write(df.head())
st.write("**Columns:**", list(df.columns))

# 2) Target selection
cols = list(df.columns)
target = st.selectbox("Select target column (or None for clustering):", [None] + cols)

# Determine task
def infer_task(df, target_col):
    if target_col is None:
        return 'Clustering'
    return 'Regression' if pd.api.types.is_numeric_dtype(df[target_col]) else 'Classification'

task = infer_task(df, target)
st.write(f"**Auto-detected task:** {task}")

# 3) Feature/target setup
def get_features_targets(df, target_col):
    X = df.select_dtypes(include=[np.number])
    if target_col is None:
        return X, None
    y = df[target_col]
    X = X.drop(columns=[target_col], errors='ignore')
    return X, y

X, y = get_features_targets(df, target)
st.write(f"Number of samples: {X.shape[0]}")

# 4) Train/test split
if task != 'Clustering':
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
else:
    X_train = X

# 5) Model training & evaluation
st.subheader("Training & Evaluation")
fig = None

if task == 'Classification':
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    st.write(f"Accuracy: {acc:.4f}")
    cm = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots()
    ax.imshow(cm, cmap='Blues')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
elif task == 'Regression':
    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    st.write(f"RMSE: {rmse:.4f}")
    fig, ax = plt.subplots()
    ax.scatter(y_test, preds)
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title('Actual vs Predicted')
else:  # Clustering
    max_clusters = min(10, X_train.shape[0])
    n_clusters = st.slider("Number of clusters", min_value=2, max_value=max_clusters, value=min(3, max_clusters))
    model = KMeans(n_clusters=n_clusters, random_state=42)
    model.fit(X_train)
    labels = model.labels_
    sil = silhouette_score(X_train, labels) if X_train.shape[0] >= n_clusters else None
    st.write(f"Silhouette Score: {sil:.4f}" if sil is not None else "Not enough samples for silhouette score.")
    pca = PCA(n_components=2)
    coords = pca.fit_transform(X_train)
    fig, ax = plt.subplots()
    ax.scatter(coords[:,0], coords[:,1], c=labels)
    ax.set_title('PCA of clusters')

# Show plot
if fig:
    st.pyplot(fig)

# 6) Save model
save_btn = st.button("Save trained model")
if save_btn:
    ext = 'pkl'
    joblib.dump(model, f"model_{task.lower()}.{ext}")
    st.success(f"Model saved to model_{task.lower()}.{ext}")
    st.stop()
