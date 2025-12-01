import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from nn_visualizer import draw_neural_network

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Universal Approximator: Ultimate Edition", layout="wide")

st.sidebar.header("1. Data Generation")
function_type = st.sidebar.selectbox("Target Function",
                                     ("Parabola (x^2)", "Sine (sin x)", "Absolute (|x|)", "Step Function", "Wavy (sin x + x)"))

data_points = st.sidebar.slider("Data Points", 50, 500, 200)
noise_level = st.sidebar.slider("Noise Level", 0.0, 1.0, 0.2)

# --- 2. DATA PREP ---
# Training Range: [-5, 5]
# Visualization Range: [-8, 8] (To show extrapolation)
TRAIN_RANGE = (-5, 5)
VIS_RANGE = (-8, 8)

# Generate raw data in training range
x_raw = np.linspace(TRAIN_RANGE[0], TRAIN_RANGE[1], data_points).reshape(-1, 1)

def get_true_y(x):
    if function_type == "Parabola (x^2)":
        return 0.5 * x**2
    elif function_type == "Sine (sin x)":
        return np.sin(x) * 2
    elif function_type == "Absolute (|x|)":
        return np.abs(x)
    elif function_type == "Step Function":
        return np.where(x > 0, 2, -2).astype(float)
    elif function_type == "Wavy (sin x + x)":
        return np.sin(x) + x * 0.5
    return x

y_true_raw = get_true_y(x_raw)

# Add Noise
np.random.seed(42)
y_noisy = y_true_raw + np.random.normal(0, noise_level, y_true_raw.shape)

# Split Train/Test
X_train, X_test, y_train, y_test = train_test_split(x_raw, y_noisy, test_size=0.2, random_state=42)

# High-res X for smooth plotting (Wide range)
x_dense = np.linspace(VIS_RANGE[0], VIS_RANGE[1], 1000).reshape(-1, 1)
y_true_dense = get_true_y(x_dense)

# --- 3. MODEL SELECTION & TRAINING ---
st.sidebar.header("2. Model Configuration")
model_type = st.sidebar.radio("Algorithm",
                              ("Neural Network", "Polynomial Regression", "Decision Tree"))

# Variables for results
y_pred_dense = np.zeros_like(x_dense)
train_mse = 0.0
test_mse = 0.0
loss_history = []
model_desc = ""
nn_components = [] # To store individual neuron outputs

# Helper placeholders for the new graph
y_pred_train = None
y_pred_test = None

# ==========================================
# MODEL A: NEURAL NETWORK
# ==========================================
if model_type == "Neural Network":
    st.sidebar.subheader("NN Architecture")
    activation_name = st.sidebar.selectbox("Activation Function",
                                           ("ReLU", "LeakyReLU", "Tanh", "Sigmoid"))

    hidden_layers = st.sidebar.slider("Hidden Layers", 1, 3, 1)
    neurons = st.sidebar.slider("Neurons per Layer", 2, 50, 10)
    epochs = st.sidebar.slider("Epochs", 500, 5000, 2000)
    lr = st.sidebar.slider("Learning Rate", 0.001, 0.1, 0.01, format="%.3f")

    show_neurons = st.sidebar.checkbox("Show Basis Functions (Internal Neurons)", value=False,
                                       help="Only works for 1 hidden layer. Shows what individual neurons are doing.")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Architecture Preview")

    vis_neurons = min(neurons, 20)
    if neurons > 20:
        st.sidebar.caption("Visualizing first 20 neurons only to keep diagram clean.")

    fig_arch = draw_neural_network(hidden_layers, vis_neurons, activation_name)
    st.sidebar.pyplot(fig_arch)
    st.sidebar.markdown("---")

    # 1. Build Model
    act_map = {
        "ReLU": nn.ReLU(), "Tanh": nn.Tanh(),
        "Sigmoid": nn.Sigmoid(), "LeakyReLU": nn.LeakyReLU()
    }

    layers = []
    in_dim = 1
    for _ in range(hidden_layers):
        layers.append(nn.Linear(in_dim, neurons))
        layers.append(act_map[activation_name])
        in_dim = neurons
    layers.append(nn.Linear(in_dim, 1))

    model = nn.Sequential(*layers)

    # 2. Train
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)

    progress_bar = st.sidebar.progress(0)

    for i in range(epochs):
        optimizer.zero_grad()
        y_p = model(X_train_t)
        loss = criterion(y_p, y_train_t)
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            loss_history.append(loss.item())
        if i % (epochs // 10) == 0:
            progress_bar.progress((i + 1) / epochs)

    progress_bar.progress(100)

    # 3. Metrics
    with torch.no_grad():
        y_pred_dense = model(torch.tensor(x_dense, dtype=torch.float32)).numpy()

        # Capture raw predictions for error plotting
        train_pred_t = model(X_train_t)
        test_pred_t = model(X_test_t)

        y_pred_train = train_pred_t.numpy()
        y_pred_test = test_pred_t.numpy()

        train_mse = criterion(train_pred_t, y_train_t).item()
        test_mse = criterion(test_pred_t, y_test_t).item()

    # 4. Extract Basis Functions (If 1 hidden layer)
    if show_neurons and hidden_layers == 1:
        layer1 = model[0]
        layer2 = model[2]
        W1 = layer1.weight.detach().numpy()
        b1 = layer1.bias.detach().numpy()
        W2 = layer2.weight.detach().numpy()
        b2 = layer2.bias.item() # Output bias

        X_d = x_dense
        for i in range(neurons):
            # Calculate output of specific neuron: w_out * Act(w_in * x + b_in)
            if activation_name == "ReLU":
                z = np.maximum(0, X_d * W1[i] + b1[i])
            elif activation_name == "Tanh":
                z = np.tanh(X_d * W1[i] + b1[i])
            elif activation_name == "Sigmoid":
                z = 1 / (1 + np.exp(-(X_d * W1[i] + b1[i])))
            elif activation_name == "LeakyReLU":
                z = np.where(X_d * W1[i] + b1[i] > 0, X_d * W1[i] + b1[i], 0.01 * (X_d * W1[i] + b1[i]))

            # Scale by output weight
            comp = z * W2[0][i]
            nn_components.append(comp)

    model_desc = f"Neural Network ({hidden_layers} hidden layers, {activation_name})"

# ==========================================
# MODEL B: POLYNOMIAL REGRESSION
# ==========================================
elif model_type == "Polynomial Regression":
    st.sidebar.subheader("Poly Hyperparameters")
    degree = st.sidebar.slider("Polynomial Degree", 1, 25, 2)

    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(X_train, y_train)

    y_pred_dense = model.predict(x_dense)

    # Capture raw predictions for error plotting (reshape to match y shape)
    y_pred_train = model.predict(X_train).reshape(-1, 1)
    y_pred_test = model.predict(X_test).reshape(-1, 1)

    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)

    model_desc = f"Polynomial Regression (Degree {degree})"
    show_neurons = False

# ==========================================
# MODEL C: DECISION TREE
# ==========================================
elif model_type == "Decision Tree":
    st.sidebar.subheader("Tree Hyperparameters")
    max_depth = st.sidebar.slider("Max Depth", 1, 20, 5)

    model = DecisionTreeRegressor(max_depth=max_depth)
    model.fit(X_train, y_train)

    y_pred_dense = model.predict(x_dense)

    # Capture raw predictions for error plotting (reshape to match y shape)
    y_pred_train = model.predict(X_train).reshape(-1, 1)
    y_pred_test = model.predict(X_test).reshape(-1, 1)

    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)

    model_desc = f"Decision Tree (Depth {max_depth})"
    show_neurons = False

# --- 4. VISUALIZATION ---
st.title("ML Model Viewer: Overfitting & Extrapolation")
st.markdown(f"**Current Model:** {model_desc}")

col1, col2 = st.columns([3, 1])

with col1:
    fig, ax = plt.subplots(figsize=(10, 6))

    # 1. Background / Zones
    ax.axvspan(TRAIN_RANGE[0], TRAIN_RANGE[1], color='green', alpha=0.05, label="Training Zone")
    ax.text(-7, np.max(y_true_dense), "Extrapolation Zone", fontsize=9, color='gray', style='italic')

    # 2. Data Points
    ax.scatter(X_train, y_train, color='blue', s=20, alpha=0.6, label='Train Data')
    ax.scatter(X_test, y_test, color='red', s=30, facecolors='none', edgecolors='red', linewidth=1.5, label='Test Data')

    # 3. Functions
    ax.plot(x_dense, y_true_dense, color='green', linestyle='--', alpha=0.4, label='True Function')
    ax.plot(x_dense, y_pred_dense, color='black', linewidth=2.5, alpha=0.8, label='Model Prediction')

    # 4. NN Basis Functions (Decomposition)
    if show_neurons and nn_components:
        for idx, comp in enumerate(nn_components):
            ax.plot(x_dense, comp, linestyle=':', linewidth=1, alpha=0.5)
        # Add a dummy plot for legend
        ax.plot([], [], color='gray', linestyle=':', label='Internal Neurons')

    ax.set_ylim(np.min(y_true_dense) - 2, np.max(y_true_dense) + 2)
    ax.set_xlim(VIS_RANGE[0], VIS_RANGE[1])
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_title("Model Fit & Generalization")
    st.pyplot(fig)

with col2:
    st.markdown("### Performance")

    # MSE Metrics
    st.metric("Train MSE", f"{train_mse:.4f}")

    delta_val = train_mse - test_mse
    st.metric("Test MSE", f"{test_mse:.4f}", delta=f"{delta_val:.4f}", delta_color="normal")

    if test_mse > train_mse * 2:
        st.warning("High Overfitting detected!")

    st.markdown("---")

# --- 5. DETAILED LOSS ANALYSIS (NEW SECTION) ---
st.markdown("---")
st.subheader("Point-wise Loss Analysis (Sorted by Severity)")


if y_pred_train is not None and y_pred_test is not None:
    # Calculate squared errors (y_true - y_pred)^2
    sq_errors_train = (y_train.flatten() - y_pred_train.flatten()) ** 2
    sq_errors_test = (y_test.flatten() - y_pred_test.flatten()) ** 2

    # --- SORTING STEP (NEW) ---
    # Sort errors from Highest to Lowest
    sq_errors_train = np.sort(sq_errors_train)[::-1]
    sq_errors_test = np.sort(sq_errors_test)[::-1]

    # Combine for plotting logic
    n_train = len(sq_errors_train)
    n_test = len(sq_errors_test)
    total_points = n_train + n_test

    # Generate indices: Train first (0 to N-1), then Test (N to Total-1)
    indices_train = np.arange(n_train)
    indices_test = np.arange(n_train, total_points)

    fig_err, ax_err = plt.subplots(figsize=(12, 4))

    # Plot Squared Errors
    ax_err.bar(indices_train, sq_errors_train, color='blue', alpha=0.6, width=1.0, label='Train Loss')
    ax_err.bar(indices_test, sq_errors_test, color='red', alpha=0.6, width=1.0, label='Test Loss')

    # Vertical Separation Line
    ax_err.axvline(x=n_train - 0.5, color='black', linestyle='-', linewidth=2, label='Dataset Split')

    # Horizontal Mean Lines
    mean_loss_train = np.mean(sq_errors_train)
    mean_loss_test = np.mean(sq_errors_test)

    ax_err.axhline(y=mean_loss_train, color='blue', linestyle='--', linewidth=2, label=f'Train Mean: {mean_loss_train:.3f}')
    ax_err.axhline(y=mean_loss_test, color='red', linestyle='--', linewidth=2, label=f'Test Mean: {mean_loss_test:.3f}')

    # Styling
    ax_err.set_title("Individual Squared Errors (Sorted High -> Low)")
    ax_err.set_xlabel("Data Points (Sorted by Error Magnitude within Set)")
    ax_err.set_ylabel("Squared Error (y - y_pred)^2")
    ax_err.legend(loc='upper right')
    ax_err.grid(True, alpha=0.3)

    st.pyplot(fig_err)


# Loss Curve (Only for NN)
if model_type == "Neural Network" and len(loss_history) > 0:
    st.markdown("### Learning Curve")
    fig_loss, ax_loss = plt.subplots(figsize=(12, 4))
    ax_loss.plot(loss_history, color='orange', linewidth=1.5)

    # Safe check for log scale with Plain Text formatting
    if min(loss_history) > 0:
        ax_loss.set_yscale('log')
        ax_loss.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.2g}'.format(y)))
        ax_loss.set_ylabel("Log MSE Loss")
    else:
        ax_loss.set_ylabel("MSE Loss")

    ax_loss.set_xlabel("Steps (x10)")
    ax_loss.grid(True, which="both", ls="-", alpha=0.2)
    st.pyplot(fig_loss)

# Insights based on model
st.markdown("### Insights")
if model_type == "Polynomial Regression":
    if degree > 10:
        st.info("Notice how the line goes crazy outside the Green Zone? Polynomials represent global trends and fail at extrapolation.")
elif model_type == "Decision Tree":
    st.info("Outside the Green Zone, the prediction is flat. Trees cannot predict trends they haven't seen.")
elif model_type == "Neural Network":
    if activation_name == "ReLU":
        st.info("With ReLU, the extrapolation is always a straight line going to infinity.")
