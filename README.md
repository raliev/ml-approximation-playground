# ML Approximation Playground

**An interactive laboratory to visualize how Machine Learning models "learn" functions, how they fail, and how they generalize.**


## Overview

This Streamlit application serves as a visual playground for understanding fundamental Machine Learning concepts. It allows users to generate synthetic datasets and attempt to fit them using three distinct classes of algorithms: **Neural Networks**, **Polynomial Regression**, and **Decision Trees**.

The goal is not just to get a good prediction, but to visualize **what happens under the hood**. The app highlights critical concepts like the *Universal Approximation Theorem*, *Overfitting* vs. *Underfitting*, and the dangers of *Extrapolation*.

## Key Features

* **Interactive Data Generation:** Create datasets based on Parabolas, Sines, Absolute values, or Step functions with adjustable noise.
* **Real-time Training:** Watch the Loss Curve update dynamically as the Neural Network learns.
* **Train/Test Split:** Visualizes the gap between "memorization" (Train MSE) and "generalization" (Test MSE).
* **Extrapolation Zone:** The plot extends beyond the training data range to reveal how models behave in unknown territory.
* **Neural Decomposition:** (Unique Feature) See the "Basis Functions" — visualize exactly how individual neurons in a hidden layer sum up to create the final prediction.

## Installation & Usage

### Prerequisites
* Python 3.8+

### Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/raliev/ml-approximation-playground.git](https://github.com/raliev/ml-approximation-playground.git)
    cd ml-approximation-playground
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the app:**
    ```bash
    streamlit run app.py
    ```

## Educational Guide: What to Look For

Use this app to demonstrate the following phenomena:

### 1. The Geometry of Learning
* **Neural Networks (ReLU):** Observe that the prediction line is "piecewise linear." If you enable **"Show Basis Functions"**, you can see that the complex curve is actually just a sum of simple "ramps" (neurons).
* **Decision Trees:** Notice that the prediction is always a series of steps (horizontal lines). Trees cannot draw diagonal lines; they can only split data based on thresholds.
* **Polynomials:** Notice the smooth, flowing curves. They have "infinite derivatives," unlike the jagged edges of Trees or ReLUs.

### 2. Extrapolation (The "Green Zone" Limit)
The application highlights a **Training Zone (Green)** and an **Extrapolation Zone (White/Gray)**.
* **Polynomials:** Set the degree high (>10). Watch the line shoot off to infinity outside the green zone. This is the *Runge phenomenon*.
* **Decision Trees:** Outside the green zone, the prediction becomes a flat line. Trees cannot predict trends (e.g., "values are increasing") outside of the range they observed during training.
* **Neural Networks:** Typically continue linearly in the direction of the last active neuron (for ReLU).

### 3. Overfitting (The Variance Error)
* Select **Polynomial Regression** and set **Degree** to **20**.
* Select **Decision Tree** and set **Max Depth** to **15**.
* **Observation:** The model will wiggle frantically to hit every single blue dot (Train Data), driving `Train MSE` to near zero. However, the `Test MSE` will skyrocket, and the red circles (Test Data) will be far from the line.