import streamlit as st
import numpy as np
import plotly.graph_objs as go
from loss_model import predict_efficiency

st.set_page_config(page_title="Compressor Loss Explorer", layout="wide")
st.title("💨 Physically Decomposed Loss Model Viewer")

# Design variable bounds
bounds = {
    "ΦD": (0.3, 1.0),
    "ΨD": (0.25, 1.0),
    "HTE": (1.5, 2.2),
    "tmax/c": (0.02, 0.12),
    "tTE/tmax": (0.15, 0.9),
    "h/c": (0.5, 3.3),
    "e/c": (0.001, 0.05),
    "θlean": (-20.0, 25.0),
    "θsweep": (-15.0, 20.0)
}

# Sidebar sliders
st.sidebar.header("🛠️ Fixed Design Parameters")
fixed_params = {}
for k, (low, high) in bounds.items():
    fixed_params[k] = st.sidebar.slider(k, min_value=low, max_value=high, value=(low + high) / 2.0)

# Axis selectors
st.sidebar.header("📊 Contour Axes")
x_var = st.sidebar.selectbox("X axis", list(bounds.keys()), index=0)
y_var = st.sidebar.selectbox("Y axis", list(bounds.keys()), index=1)

# Contour grid
x_vals = np.linspace(*bounds[x_var], 40)
y_vals = np.linspace(*bounds[y_var], 40)
Z = np.zeros((40, 40))

for i, x in enumerate(x_vals):
    for j, y in enumerate(y_vals):
        params = fixed_params.copy()
        params[x_var] = x
        params[y_var] = y
        Z[j][i] = predict_efficiency(params)

# Plot efficiency contours
fig = go.Figure(data=go.Contour(
    z=Z,
    x=x_vals,
    y=y_vals,
    contours_coloring='heatmap',
    colorbar_title="Efficiency η",
    line_smoothing=0.85
))
fig.update_layout(title=f"Efficiency vs {x_var} and {y_var}", xaxis_title=x_var, yaxis_title=y_var)
st.plotly_chart(fig, use_container_width=True)

