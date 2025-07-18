import streamlit as st
import numpy as np
import plotly.graph_objs as go
from loss_model import pred_loss, calc_params
import pickle
import pandas as pd
import math


st.set_page_config(page_title="Compressor Loss Explorer", layout="wide")
st.title("💨 Physically Decomposed Loss Model Viewer")

# Design variable bounds


with open('loss_model.pkl', 'rb') as fp:
    models = pickle.load(fp)
# Sidebar sliders
st.sidebar.header("Plot type")
loss_component = st.sidebar.selectbox(
    "Loss Component",
    ["Total Efficiency", "Blade Loss", "Hub Loss", "Casing Loss", "Tip Loss", "Wake Loss"]
)
show_dh_lines = st.sidebar.checkbox("Show Constant DH Lines")
dh_levels = st.sidebar.multiselect("DH Values", [ 0.55, 0.6, 0.65,0.7, 0.75, 0.8,0.85,0.9], default=[0.6, 0.65, 0.7, 0.75, 0.8])

st.sidebar.header("🎯 Sampling Options")
sample_size = st.sidebar.slider("Grid Resolution", min_value=10, max_value=100, value=40, step=10)

st.sidebar.header("🛠️ Fixed Design Parameters")
pitch_mode = st.sidebar.selectbox("Pitch-to-Chord Mode", ["DF", "3D"])
design_mode = st.sidebar.checkbox("Design mode")
if design_mode==False:
    if pitch_mode == 'DF':
        bounds = {
            "Φ": (0.3, 1.0,0.6),
            "Ψ": (0.25, 1.0,0.4),
            "DF": (0.35, 0.55,0.45),
            "tₘₐₓ/c": (0.02, 0.12,0.04),
            "tᴛᴇ⁄tₘₐₓ": (0.15, 0.9,0.33),
            "AR": (0.5, 3.3,1.0),
            "e/c": (0.001, 0.05,0.01),
            "θₗₑₐₙ": (-20.0, 25.0,0.0),
            "θₛᵥₑₑₚ": (-15.0, 20.0,0.0),
            "Re": (3e5, 10e6,6.33e5)
        }
        fixed_params = {}
        for k, (low, high,value) in bounds.items():
            fixed_params[k] = st.sidebar.slider(k, min_value=low, max_value=high, value=value)
    
    elif pitch_mode == '3D':
        bounds = {
            "Φ": (0.3, 1.0,0.6),
            "Ψ": (0.25, 1.0,0.4),
            "mf": (0.8, 0.95,0.85),
            "tₘₐₓ/c": (0.02, 0.12,0.04),
            "tᴛᴇ⁄tₘₐₓ": (0.15, 0.9,0.33),
            "AR": (0.5, 3.3,1.0),
            "e/c": (0.001, 0.05,0.01),
            "θₗₑₐₙ": (-20.0, 25.0,0.0),
            "θₛᵥₑₑₚ": (-15.0, 20.0,0.0),
            "Re": (3e5, 10e6,6.33e5)
        }
        fixed_params = {}
        for k, (low, high,value) in bounds.items():
            fixed_params[k] = st.sidebar.slider(k, min_value=low, max_value=high, value=value)
else:
    if pitch_mode == 'DF':
        bounds = {
            "Φ": (0.3, 1.0,0.6),
            "Ψ": (0.25, 1.0,0.4),
            "DF": (0.35, 0.55,0.45),
          "tₘᵢₙ mm": (0.3, 2., 0.5),
          "eₘᵢₙ mm": (0.3, 2.,0.5),
          "massflow kg/s": (5., 32.,9.1),
          "rpm": (1000., 7000.,3500.),
          "dho kJ/kg": (1000., 7000.,6278.9),
          "Density kg/m^3": (1., 12.,1.2),
          "AR": (0.5, 3.3,1.0),
          "tᴛᴇ⁄tₘₐₓ": (0.15, 0.9,0.33),
          "θₗₑₐₙ": (-20.0, 25.0,0.0),
          "θₛᵥₑₑₚ": (-15.0, 20.0,0.)
        }
        fixed_params = {}
        for k, (low, high,value) in bounds.items():
            fixed_params[k] = st.sidebar.slider(k, min_value=low, max_value=high, value=value)
 
    elif pitch_mode == '3D':
        bounds = {
            "Φ": (0.3, 1.0,0.6),
            "Ψ": (0.25, 1.0,0.4),
            "mf": (0.8, 0.95,0.85),
            "tₘᵢₙ mm": (0.3, 2., 0.5),
            "eₘᵢₙ mm": (0.3, 2.,0.5),
            "massflow kg/s": (5., 32.,9.1),
            "rpm": (1000., 7000.,3500.),
            "dho kJ/kg": (1000., 7000.,6278.9),
            "Density kg/m^3": (1., 12.,1.2),
            "AR": (0.5, 3.3,1.0),
            "tᴛᴇ⁄tₘₐₓ": (0.15, 0.9,0.33),
            "θₗₑₐₙ": (-20.0, 25.0,0.0),
            "θₛᵥₑₑₚ": (-15.0, 20.0,0.)

        }
        fixed_params = {}
        for k, (low, high,value) in bounds.items():
            fixed_params[k] = st.sidebar.slider(k, min_value=low, max_value=high, value=value)

    

# Axis selectors
st.sidebar.header("📊 Contour Axes")
x_var = st.sidebar.selectbox("X axis", list(bounds.keys()), index=0)
y_var = st.sidebar.selectbox("Y axis", list(bounds.keys()), index=1)

# Contour grid
x_vals = np.linspace(*bounds[x_var][:2], sample_size)
y_vals = np.linspace(*bounds[y_var][:2],sample_size)
Z = np.zeros((sample_size, sample_size))


# Create a meshgrid of input values
X, Y = np.meshgrid(x_vals, y_vals)
flat_X = X.ravel()
flat_Y = Y.ravel()

# Create a DataFrame of all grid points
params = pd.DataFrame({
    x_var: flat_X,
    y_var: flat_Y,
})
for k, v in fixed_params.items():
    if k not in [x_var, y_var]:
        params[k] = v
        
params=calc_params(models,params,pitch_mode)        
if design_mode==True:
    U = (params["dho kJ/kg"]/params["Ψ"])**0.5
    r = U/(2*math.pi*params["rpm"]/60)
    Vx = params["Φ"]*U
    V1 = params["V1_U"]*U
    span = params["massflow kg/s"]/(params["Density kg/m^3"]*Vx*2*math.pi*r)

    c = span/params["AR"]
    params["Re"] = V1*c*params["Density kg/m^3"]/(1.8*10**-5)
        #Re = Re*span/span
    params["e/c"] = params["eₘᵢₙ mm"]*1e-3/c
    tec = params["tₘᵢₙ mm"]*1e-3/c
    params["tₘₐₓ/c"] = tec/params["tᴛᴇ⁄tₘₐₓ"]
     

# Apply vectorized loss model
out = pred_loss(models,params)

if loss_component == "Total Efficiency":
    Z_flat = 100-out['lost_eff_tot']
elif loss_component == "Blade Loss":
    Z_flat = out['loss_blade']
elif loss_component == "Hub Loss":
    Z_flat = out['loss_hub']
elif loss_component == "Casing Loss":
    Z_flat = out['loss_cas']
elif loss_component == "Tip Loss":
    Z_flat = out['loss_tip']
elif loss_component == "Wake Loss":
    Z_flat = out['loss_wake']


DH_z = params["DH"].to_numpy().reshape(sample_size, sample_size)

Z_flat[out['lost_eff_tot'] <1] = np.nan
Z_flat[out['lost_eff_tot'] > 20] = np.nan
Z_flat[params["sc"] < 0.2] = np.nan
Z_flat[params["DH"] < 0.55] = np.nan
Z = Z_flat.reshape(sample_size, sample_size)

    

if loss_component == 'Total Efficiency':
    lab = "η %"
else:
    lab = "ω"
# Plot efficiency contours
fig = go.Figure(data=go.Contour(
    z=Z,
    x=x_vals,
    y=y_vals,
    contours_coloring='heatmap',
    colorbar_title=lab,
    colorscale = 'Turbo' , 
    zmin=np.min(Z),             # Scale matches data range
    zmax=np.max(Z),
    line_smoothing=0.85
))
fig.update_layout(title=f"{loss_component} vs {x_var} and {y_var}", xaxis_title=x_var, yaxis_title=y_var,autosize=False,
    width=600,
    height=600
)
if show_dh_lines:
    for dh_val in dh_levels:
        fig.add_trace(go.Contour(
            z=DH_z,
            x=x_vals,
            y=y_vals,
            contours=dict(
                coloring='lines',
                showlabels=True,
                value=[dh_val]  # single level
            ),
            line=dict(width=2, color='black', dash='dash'),
            showscale=False,
            name=f"DH = {dh_val:.2f}",
            opacity=1
        ))


st.plotly_chart(fig, use_container_width=False)

