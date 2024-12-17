import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv('celeba_baldvsnonbald_normalised.csv')
    return df

# Preprocess Data
def preprocess_data(df):
    features = [col for col in df.columns if col.startswith('A')]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])
    return X_scaled, features

# Build Autoencoder
def build_autoencoder(X_scaled, encoding_dim=10):
    input_dim = X_scaled.shape[1]
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(encoding_dim, activation='relu')(input_layer)
    decoded = Dense(input_dim, activation='sigmoid')(encoded)
    autoencoder = Model(inputs=input_layer, outputs=decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder

# Initialize
st.title("Exploratory analysis of dataset")
st.sidebar.header("Select Options")
df = load_data()
X_scaled, features = preprocess_data(df)

# Dropdown Toggles
tab = st.sidebar.radio("Select Analysis Type", [
    "Class Distribution", 
    "Feature Distribution",
    "Anomaly Detection", 
    "Feature Importance",
    "Correlation Heatmap"
])

# 1. Class Distribution
if tab == "Class Distribution":
    st.subheader("Class Distribution")
    class_counts = df['class'].value_counts()
    st.bar_chart(class_counts)
    st.write("Class 0 count: ", class_counts[0])
    st.write("Class 1 count: ", class_counts[1])

# 2. Feature Distribution
elif tab == "Feature Distribution":
    st.subheader("Feature Distribution")
    feature = st.sidebar.selectbox("Select a Feature", features)
    fig, ax = plt.subplots()
    sns.countplot(x=df[feature], hue=df['class'], ax=ax)
    plt.title(f"Distribution of {feature}")
    st.pyplot(fig)


# 3. Anomaly Detection
elif tab == "Anomaly Detection":
    st.subheader("Anomaly Detection with Autoencoders")
    threshold = st.slider("Select Reconstruction Error Threshold Percentile", 
                      min_value=50, max_value=100, value=99, step=1)

    autoencoder = build_autoencoder(X_scaled)
    history = autoencoder.fit(X_scaled, X_scaled, epochs=10, batch_size=32, verbose=0)
    reconstructed = autoencoder.predict(X_scaled)
    reconstruction_errors = np.mean((X_scaled - reconstructed) ** 2, axis=1)
    threshold_value = np.percentile(reconstruction_errors, threshold)

    fig, ax = plt.subplots()
    sns.histplot(reconstruction_errors, bins=50, kde=True, ax=ax)
    plt.axvline(threshold_value, color='red', linestyle='--', label='Threshold')
    plt.title("Reconstruction Error Distribution")
    plt.legend()
    st.pyplot(fig)

    st.write(f"Threshold: {threshold_value:.4f}")
    st.write(f"Number of anomalies: {(reconstruction_errors > threshold_value).sum()}")

# 4. Feature Importance
elif tab == "Feature Importance":
    st.subheader("Feature Importance using Random Forest Classification")
    X = df[features]
    y = df['class']
    model = RandomForestClassifier()
    model.fit(X, y)
    importances = pd.Series(model.feature_importances_, index=features)
    top_n = st.slider("Number of Top Features to Display", 5, len(features), 10)
    fig, ax = plt.subplots()
    importances.nlargest(top_n).plot(kind='bar', ax=ax)
    plt.title("Top Feature Importances")
    st.pyplot(fig)

# 5. Pairwise Correlation Heatmap
elif tab == "Correlation Heatmap":
    st.subheader("Correlation Heatmap")
    selected_features = st.multiselect("Select Features for Correlation", features, default=features[:10])
    corr_matrix = df[selected_features].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax)
    plt.title("Correlation Heatmap")
    st.pyplot(fig)

# Footer
st.write("**Dashboard built by Uday Grover**")
