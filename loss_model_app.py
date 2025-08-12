import streamlit as st
import numpy as np
import plotly.graph_objs as go
from loss_model import pred_loss, calc_params, plot_blade
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
    ["Total Efficiency", "Total Loss","Blade Loss", "Hub Loss", "Casing Loss", "Tip Loss", "Wake Loss"]
)
show_dh_lines = st.sidebar.checkbox("Show Constant DH Lines")
dh_levels = st.sidebar.multiselect("DH Values", [ 0.55, 0.6, 0.65,0.7, 0.75, 0.8,0.85,0.9], default=[0.6, 0.65, 0.7, 0.75, 0.8])

st.sidebar.header("🎯 Display Options")
sample_size = st.sidebar.slider("Grid Resolution", min_value=10, max_value=100, value=40, step=10)
if loss_component == "Total Efficiency":
    
    zmin = st.sidebar.slider("η % min", min_value=60., max_value=90., value=80., step=1.0)
    zmax = st.sidebar.slider("η % max", min_value=80., max_value=100., value=95., step=1.0)
else:
    zmin = st.sidebar.slider("ω min", min_value=0., max_value=0.1, value=0., step=0.005)
    zmax = st.sidebar.slider("ω max", min_value=0.01, max_value=0.7, value=0.1, step=0.005)    
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
    params["r"] = r
    params["c"] = c
    
fixed_params=calc_params(models,fixed_params,pitch_mode)
       
if design_mode==True:
    U = (fixed_params["dho kJ/kg"]/fixed_params["Ψ"])**0.5
    r = U/(2*math.pi*fixed_params["rpm"]/60)
    Vx = fixed_params["Φ"]*U
    V1 = fixed_params["V1_U"]*U
    span = fixed_params["massflow kg/s"]/(fixed_params["Density kg/m^3"]*Vx*2*math.pi*r)

    chord = span/fixed_params["AR"]
    fixed_params["Re"] = V1*c*fixed_params["Density kg/m^3"]/(1.8*10**-5)
        #Re = Re*span/span
    fixed_params["e/c"] = fixed_params["eₘᵢₙ mm"]*1e-3/chord
    tec = fixed_params["tₘᵢₙ mm"]*1e-3/chord
    fixed_params["tₘₐₓ/c"] = tec/fixed_params["tᴛᴇ⁄tₘₐₓ"]
    fixed_params["r"] = r
    fixed_params["c"] = chord
    fixed_params["pitch"] = fixed_params["sc"]*fixed_params["c"]
else:  
    fixed_params["c"] = 0.028 
    fixed_params["pitch"] = fixed_params["sc"]*fixed_params["c"]



# Apply vectorized loss model
out = pred_loss(models,params)

if loss_component == "Total Efficiency":
    Z_flat = 100-out['lost_eff_tot']
elif loss_component == "Total Loss":
    Z_flat = out['loss']
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
fig_contour = go.Figure(data=go.Contour(
    z=Z,
    x=x_vals,
    y=y_vals,
    contours_coloring='heatmap',
    colorbar_title=lab,
    colorscale = 'Turbo' , 
    zmin=zmin,             # Scale matches data range
    zmax=zmax,
    line_smoothing=0.85
))
fig_contour.update_layout(title=f"{loss_component} vs {x_var} and {y_var}", xaxis_title=x_var, yaxis_title=y_var,autosize=False,
    width=600,
    height=500
)
if show_dh_lines:
    for dh_val in dh_levels:
        fig_contour.add_trace(go.Contour(
            z=DH_z,
            x=x_vals,
            y=y_vals,
            contours=dict(
                coloring='lines',
                showlabels=True,    
                start=dh_val,
                end=dh_val,
                size=1e-6  # effectively a single contour line

            ),
            line=dict(width=2, color='black', dash='dash'),
            showscale=False,
            name=f"DH = {dh_val:.2f}",
            opacity=1
        ))
        

if design_mode == "True":
    xrrt, xrrt_hub, xrrt_cas = plot_blade(models, fixed_params, design_mode)
else:
    xrrt, xrrt_hub, xrrt_cas = plot_blade(models, fixed_params, design_mode)
n_r = len(np.unique(xrrt[:, 1]))       # radial divisions
n_chord = xrrt.shape[0] // n_r         # profile points per slice

X = xrrt[:, 0].reshape(n_r, n_chord)
R = xrrt[:, 1].reshape(n_r, n_chord)
RT = xrrt[:, 2].reshape(n_r, n_chord)

fig_blade = go.Figure(data=[go.Surface(
    x=X, y=R, z=RT,
    surfacecolor=np.ones_like(R),  # uniform color layer
    colorscale=[[0, 'grey'], [1, 'grey']],
    cmin=0,
    cmax=1,
    showscale=False,
    opacity=1.0,
    lighting=dict(ambient=0.6, diffuse=0.8, specular=0.3),

)])

fig_blade.add_trace(go.Surface(
    x=X, y=R, z=RT+fixed_params["pitch"],
    surfacecolor=np.ones_like(R),  # uniform color layer
    colorscale=[[0, 'grey'], [1, 'grey']],
    cmin=0,
    cmax=1,
    showscale=False,
    opacity=1.0,
    lighting=dict(ambient=0.6, diffuse=0.8, specular=0.3),

))

fig_blade.add_trace(go.Surface(
    x=X, y=R, z=RT-fixed_params["pitch"],
    surfacecolor=np.ones_like(R),  # uniform color layer
    colorscale=[[0, 'grey'], [1, 'grey']],
    cmin=0,
    cmax=1,
    showscale=False,
    opacity=1.0,
    lighting=dict(ambient=0.6, diffuse=0.8, specular=0.3),

))

n_r = len(np.unique(xrrt_hub[:, 1]))       # radial divisions
n_chord = xrrt_hub.shape[0] // n_r         # profile points per slice

X_hub = xrrt_hub[:, 0].reshape(n_r, n_chord)
R_hub = xrrt_hub[:, 1].reshape(n_r, n_chord)
RT_hub = xrrt_hub[:, 2].reshape(n_r, n_chord)


n_r = len(np.unique(xrrt_cas[:, 1]))       # radial divisions
n_chord = xrrt_cas.shape[0] // n_r         # profile points per slice

X_cas= xrrt_cas[:, 0].reshape(n_r, n_chord)
R_cas = xrrt_cas[:, 1].reshape(n_r, n_chord)
RT_cas = xrrt_cas[:, 2].reshape(n_r, n_chord)
scale  = 3.5*(fixed_params["c"])
xrange = [np.min(X), np.min(X)+scale]
rrange = [np.min(R),  np.min(R)+scale]
rtrange = [np.min(RT)-2*scale/2,  np.min(RT)+2*scale/2]
fig_blade.update_layout(
    title="🌀 Blade Surface Geometry",
    scene=dict(
        xaxis=dict(title='Axial (x)', range=xrange),
       yaxis=dict(title='Radial (r)', range=rrange),
       zaxis=dict(title='Tangential (rt)', range=rtrange),

       camera=dict(
           eye=dict(x=-2, y=0.5, z=2),     # view from front-right
           up=dict(x=0, y=1, z=0)         # sets radial 'up' direction
       ),


        aspectmode='manual',   # Forces fixed scaling
        aspectratio=dict(x=1, y=1, z=2)  # Equal units per axis



    ),
    margin=dict(l=0, r=0, b=0, t=50),
    height=500,
    width=600

)


col1, col2 = st.columns(2)
with col1:
    st.subheader("Loss Contours")
    st.plotly_chart(fig_contour, use_container_width=False)

with col2:
    st.subheader("Blade Geometry")
    st.plotly_chart(fig_blade, use_container_width=False)




