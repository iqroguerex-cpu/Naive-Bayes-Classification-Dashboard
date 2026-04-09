import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from matplotlib.colors import ListedColormap

# Page Configuration
st.set_page_config(page_title="Naive Bayes Classifier", layout="wide")

st.title("📊 Naive Bayes Classification Dashboard")
st.markdown("""
This dashboard visualizes the **Naive Bayes** algorithm's performance on the Social Network Ads dataset. 
Adjust parameters in the sidebar to see how the decision boundary changes.
""")

# --- Sidebar: Configuration ---
st.sidebar.header("Model Configuration")

# Upload or use default dataset
uploaded_file = st.sidebar.file_uploader("Upload your CSV", type=["csv"])
test_size = st.sidebar.slider("Test Set Size (%)", 10, 50, 25) / 100
random_state = st.sidebar.number_input("Random State", value=0)


# --- Data Processing ---
@st.cache_data
def load_data(file=None):
    if file is not None:
        df = pd.read_csv(file)
    else:
        # Default fallback (ensure the file is in the same directory)
        df = pd.read_csv('Social_Network_Ads.csv')
    return df


try:
    dataset = load_data(uploaded_file)
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    # Splitting
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Feature Scaling
    sc = StandardScaler()
    X_train_scaled = sc.fit_transform(X_train)
    X_test_scaled = sc.transform(X_test)

    # Training
    classifier = GaussianNB()
    classifier.fit(X_train_scaled, y_train)

    # --- Main Panel: Prediction and Metrics ---
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("🎯 Real-time Prediction")
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        salary = st.number_input("Estimated Salary", min_value=10000, max_value=200000, value=87000)

        prediction = classifier.predict(sc.transform([[age, salary]]))
        result = "Purchased" if prediction[0] == 1 else "Not Purchased"
        st.success(f"The model predicts: **{result}**")

        st.divider()

        st.subheader("📈 Model Metrics")
        y_pred = classifier.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        st.metric("Accuracy Score", f"{acc:.2%}")

        # Confusion Matrix Plot
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax_cm = plt.subplots(figsize=(4, 3))
        ConfusionMatrixDisplay(cm).plot(ax=ax_cm, cmap='Blues', colorbar=False)
        st.pyplot(fig_cm)

    with col2:
        st.subheader("🗺️ Decision Boundary")

        plot_type = st.radio("Select Dataset to Visualize:", ("Training Set", "Test Set"), horizontal=True)


        # Plotting Function
        def plot_boundary(X_data, y_data, title):
            X_set, y_set = sc.inverse_transform(X_data), y_data

            # Larger step size for performance in Streamlit
            X1, X2 = np.meshgrid(
                np.arange(start=X_set[:, 0].min() - 5, stop=X_set[:, 0].max() + 5, step=1),
                np.arange(start=X_set[:, 1].min() - 500, stop=X_set[:, 1].max() + 500, step=500)
            )

            Z = classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape)

            fig, ax = plt.subplots()
            ax.contourf(X1, X2, Z, alpha=0.75, cmap=ListedColormap(['#FA8072', '#1E90FF']))
            ax.set_xlim(X1.min(), X1.max())
            ax.set_ylim(X2.min(), X2.max())

            for i, j in enumerate(np.unique(y_set)):
                ax.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                           c=ListedColormap(['#FA8072', '#1E90FF'])(i), label=j, edgecolors='black')

            ax.set_title(title)
            ax.set_xlabel('Age')
            ax.set_ylabel('Estimated Salary')
            ax.legend()
            return fig


        if plot_type == "Training Set":
            st.pyplot(plot_boundary(X_train_scaled, y_train, "Naive Bayes (Training Set)"))
        else:
            st.pyplot(plot_boundary(X_test_scaled, y_test, "Naive Bayes (Test Set)"))

    # Data Preview
    with st.expander("👀 View Raw Dataset"):
        st.dataframe(dataset)

except FileNotFoundError:
    st.error("Please place the 'Social_Network_Ads.csv' file in the same directory or upload it via the sidebar.")
