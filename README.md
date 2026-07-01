# Global Weather to Country Predictor 🌍🌦️

A Streamlit web application that predicts a geographic location (country) based on real-time weather measurements. The prediction engine is powered by a custom PyTorch Multi-Layer Perceptron (MLP) neural network.

## 📌 Project Overview

This project utilises a deep learning model to classify a country using 10 specific weather features[cite: 5]. The data is standardised before being processed by the PyTorch model, and the resulting predictions are decoded back into readable country names using a scikit-learn label encoder[cite: 5].

## Live Link
[![Live App](https://img.shields.io/badge/Live%20App-Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://global-weather-and-location-prediction-engine-74vjtdcxzmqfr39a.streamlit.app/)

## Dataset
[![Use Dataset](https://img.shields.io/badge/Use%20Dataset-Kaggle-2962FF?style=flat-square&logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/nelgiriyewithana/global-weather-repository)

### The 10 Weather Features Analysed:
* Temperature (°C)[cite: 5]
* Wind Speed (kph)[cite: 5]
* Wind Degree[cite: 5]
* Pressure (mb)[cite: 5]
* Precipitation (mm)[cite: 5]
* Humidity (%)[cite: 5]
* Cloud Cover[cite: 5]
* Visibility (km)[cite: 5]
* UV Index[cite: 5]
* Gust Speed (kph)[cite: 5]

## 📂 Repository Structure

*   `app.py`: The main Streamlit web application script.
*   `Location_Prediction_2.py`: The PyTorch training script[cite: 5].
*   `location_classifier.pth`: Saved PyTorch model weights.
*   `scaler.joblib`: Saved scikit-learn StandardScaler.
*   `label_encoder.joblib`: Saved scikit-learn LabelEncoder.
*   `requirements.txt`: Python dependencies for the server environment.

## 🧠 Model Architecture

The core of this application is a PyTorch neural network built with `torch.nn.Sequential`[cite: 5]. The architecture consists of fully connected linear layers with ReLU activations[cite: 5]:
*   **Input Layer:** 10 features[cite: 5]
*   **Hidden Layer 1:** 256 neurons[cite: 5]
*   **Hidden Layer 2:** 128 neurons[cite: 5]
*   **Hidden Layer 3:** 128 neurons[cite: 5]
*   **Hidden Layer 4:** 64 neurons[cite: 5]
*   **Output Layer:** Matches the number of unique countries in the dataset[cite: 5]

The model was optimised using Stochastic Gradient Descent (SGD) with a learning rate of 0.001 and a momentum of 0.9[cite: 5]. Cross-Entropy Loss was used as the criterion[cite: 5].

## 💻 Local Installation and Usage

To run this application locally, ensure Python 3.8+ is installed.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Launch the Streamlit application:**
    ```bash
    streamlit run app.py
    ```

## ☁️ Deployment on Streamlit Cloud

This application is configured for seamless deployment on Streamlit Community Cloud.

1.  Push this repository to a public GitHub account.
2.  Log in to Streamlit Cloud.
3.  Click **New app**.
4.  Select this repository, set the branch to `main`, and set the main file path to `app.py`.
5.  Click **Deploy!**

---
*Created by Shanta Majumder*
