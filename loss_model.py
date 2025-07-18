# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 11:47:57 2025

@author: as2375
"""
import numpy as np # linear algebra
from equadratures import Parameter, Poly, Basis
import operator as op
from functools import reduce
from copy import deepcopy
from scipy.interpolate import CubicSpline, PchipInterpolator, splrep,splev, interp1d

def plot_blade(models, params, design_mode=False):
    phi = params["Φ"]
    psi = params["Ψ"]
    sc = params["sc"]
    tc = params["tₘₐₓ/c"]
    te = params["tᴛᴇ⁄tₘₐₓ"]
    AR = params["AR"]
    ec = params["e/c"]
    lean = params["θₗₑₐₙ"]
    sweep = params["θₛᵥₑₑₚ"]
    
    X = np.reshape(np.array([float(phi),float(psi),float(tc),float(te),float(sc)]),(1,5))   
    xcl = np.linspace(0,0.2,31)
    xct = np.linspace(0.2,1,20)

    xc2 = np.concatenate([xcl,xct[1:]]).reshape(-1,1)
        
    dcam_le,_ = evaluate(models['dcam_le'], X) 
    dcam_te,_ = evaluate(models['dcam_te'], X) 
    chi_le,_ = evaluate(models['chi_le'], X) 
    chi_te,_ = evaluate(models['chi_te'], X) 
    if design_mode==True:
        c = params["c"]
        r_mid = params["r"]
    else:
        c = 0.028
        r_mid = 0.3374
        
    xu,yu,xl,yl,xcam,ycam,thick,chi, schord= gen_bladec(dcam_le[0],dcam_te[0],chi_le[0], chi_te[0],tc,te,xc2,c)
       
    pitch = sc*c
    span = AR*c
    r_hub = r_mid - span/2
    r_cas = r_mid+ span/2
    
    sweep_profile = c*tand(sweep)*np.array([0, AR/3, AR/3 ,0])
    lean_profile = -c*tand(lean)*np.array([0, AR/3, AR/3 ,0])
    sweep_fit=bl_spline_fit(sweep_profile)
    lean_fit=bl_spline_fit(lean_profile)
    
    r_nondim = np.linspace(0.,1.-ec/AR,20)
    r_span = span*r_nondim + r_hub
    sweep_displacement = bl_spline_eval(sweep_fit, r_nondim)
    lean_displacement = bl_spline_eval(lean_fit, r_nondim)

    xrt_chord = np.column_stack((xcam, ycam))
    xrt_upper = np.column_stack((xu, yu))
    xrt_lower = np.column_stack((xl, yl))
    xrt_blade = np.vstack([xrt_upper, np.flipud(xrt_lower)])
    
    X, R, RT = generate_blade_coordinates(xrt_blade, r_span, return_grid=True)
    
    # Compute sweep/lean displacement vectors for each span slice
    p_vec = xrt_chord[-1] - xrt_chord[0]
    p_unit = p_vec / np.linalg.norm(p_vec)
    
    # shape: (n_span, 2)
    displacements = sweep_displacement[:, np.newaxis] * p_unit + \
                    lean_displacement[:, np.newaxis] * np.array([-p_unit[1], p_unit[0]])
    
    # Apply to entire mesh
    X += displacements[:, 0][:, np.newaxis]
    RT += displacements[:, 1][:, np.newaxis]
    
    # Return flat [x, r, rt] array
    xrrt = np.stack([X, R, RT], axis=2).reshape(-1, 3)
    
    x_min = np.min(X)
    x_max = np.max(X)
    x_vals = np.linspace(x_min, x_max, 50)
    rt_vals = np.linspace(-pitch, pitch, 50)
    
    X, RT = np.meshgrid(x_vals, rt_vals)
    R = np.full_like(X, r_hub)

    xrrt_hub = np.stack([X, R, RT], axis=2).reshape(-1, 3)
    R = np.full_like(X, r_cas)

    xrrt_cas = np.stack([X, R, RT], axis=2).reshape(-1, 3)
    return xrrt, xrrt_hub, xrrt_cas




def generate_blade_coordinates(xy_profile, r_span, return_grid=False):
    xy_profile = np.array(xy_profile)
    r_span = np.array(r_span)

    x_profile, y_profile = xy_profile[:, 0], xy_profile[:, 1]
    n_span, n_pts = len(r_span), len(x_profile)

    X = np.tile(x_profile, (n_span, 1))
    RT = np.tile(y_profile, (n_span, 1))
    R = r_span[:, np.newaxis].repeat(n_pts, axis=1)

    if return_grid:
        return X, R, RT

    coords = np.stack([X, R, RT], axis=2).reshape(-1, 3)
    return coords

def bl_spline_fit(param):
    """
    Fit splines through numeric values of blade parameters in dictionary `b`.
    Replaces each parameter with a spline representation.
    """


    if isinstance(param, np.ndarray) or isinstance(param, list):
        param = np.array(param)
        nj = len(param)

        # Case 1: 2D array with explicit coordinates (x, y)
        if param.ndim == 2 and param.shape[1] == 2:
            x_raw, y_raw = param[:, 0], param[:, 1]
            x_interp = np.linspace(0, 1, 100)
            y_interp = PchipInterpolator(x_raw, y_raw)(x_interp)
            # Fit spline (cubic B-spline representation)
            tck = splrep(x_interp, y_interp, s=0, k=3)
            
        # Case 2: 1D specification only
        elif param.ndim == 1:
            x = np.linspace(0, 1, nj)
            tck = splrep(x, param, s=0, k=3)


        else:
            print(f"⚠️ Unexpected format in {key}, skipping spline fit.")
    
    return tck 



def bl_spline_eval(param, r_nondim):
    """
    Evaluate a blade definition `b` across specified normalized radius `r_nondim`.
    Returns a new dictionary `c` with evaluated spline or interpolated values.
    """
    
    # Case 1: spline representation (assumed as tuple from splrep)
    if isinstance(param, tuple):
        # Extrapolation if needed
        xmin, xmax = param[0][0], param[0][-1]
        if np.min(r_nondim) < xmin or np.max(r_nondim) > xmax:
            # Splev extrapolates by default
            val = splev(r_nondim, param)
        else:
            val = splev(r_nondim, param)
    
    # Case 2: raw 1D vector
    elif param.ndim == 1:
        if param.size == 1:
            param = np.repeat(param, 2)
        x = np.linspace(0, 1, len(param))
        spline = interp1d(x, param, kind='cubic', fill_value="extrapolate")
        val = spline(r_nondim)
    
    # Case 3: tabulated 2D (x,y) data
    elif param.ndim == 2 and param.shape[1] == 2:
        x, y = param[:, 0], param[:, 1]
        spline = interp1d(x, y, kind='cubic', fill_value="extrapolate")
        val = spline(r_nondim)

    

    return val


def calc_params(models,params,type='DF'):
    phi = params["Φ"]
    psi = params["Ψ"]
    
    
    if type == 'DF':
        DF = params["DF"]
        params["cos_in"],params["cos_out"], params["DH"], params["V1_U"],params["sc"] =  parameters_pitch(phi,psi,DF)
    elif type == '3D':
        params["cos_in"],params["cos_out"], params["DH"], params["V1_U"],_ =  parameters_pitch(phi,psi,0.45)
        mf = params["mf"]
        X = np.concatenate((np.reshape(params["DH"],(-1,1)),np.reshape(params["Φ"],(-1,1)),np.reshape(mf,(-1,1)) ),axis=1)
        params["sc"] , _ = evaluate(models['rsc'], X)
        
        
    return params

def predict_efficiency(models,params):
    out = predict_loss(models,params)
    lost_eff_tot = out['lost_eff_tot']
    return 1 - lost_eff_tot

def pred_loss(models,params,area_mod =0,area_in=0,area_pass=0,area_out=0,ad=1):
    output ={}
    dh = params["DH"].to_numpy()
    cos_in = params["cos_in"].to_numpy()
    AR = params["AR"].to_numpy()
    ec = params["e/c"].to_numpy()
    sc = params["sc"].to_numpy()
    tc = params["tₘₐₓ/c"].to_numpy()
    te = params["tᴛᴇ⁄tₘₐₓ"].to_numpy()
    lean = params["θₗₑₐₙ"].to_numpy()
    sweep = params["θₛᵥₑₑₚ"].to_numpy()
    Re = params["Re"].to_numpy()
    # Find out inlet and exit velcoities 
    cos_out = cos_in.flatten()/dh.flatten()
    phi, psi, DH, V1_U = angles2(cos_in,cos_out)
    V2_U = V1_U*DH
    #V2 = 68.0525*V2_U
    #V1 = 68.0525*V1_U
    load = sc*psi*cos_in/phi
    X = np.concatenate((np.reshape(AR,(-1,1)),np.reshape(ec,(-1,1)),np.reshape(V1_U,(-1,1)),np.reshape(V2_U,(-1,1)),
                             np.reshape(sc,(-1,1)),np.reshape(tc,(-1,1)),np.reshape(te,(-1,1)),np.reshape(lean,(-1,1)),np.reshape(sweep,(-1,1))),axis=1)
    
 
    tec = te*tc
    theta_total = np.array([0.029328861815324, 0.023795887768032, 0.019053797904466, 0.016509195361694, 0.01453884850265, 0.012498575166737, 0.011262807009850])
    Rec = np.array([1, 2, 5, 10, 20, 50, 100])
    spl = CubicSpline(Rec, theta_total/theta_total[3])
    cd_rat = (spl(Re*10**-5))/spl(6.33)
    #cd_rat = ((Re*10**-5)/6.33)**(-1/6)
    cd_ew = 0.0029*cd_rat
    cd_tip = 0.0029*cd_rat
    cd = 0.0028*cd_rat
    c = 0.028

    
    #cos_in,cos_outd, DHd, V1_U,scr =  parameters_pitch(phi,psi,0.5)
    stagger = 0.5*(np.arccos(cos_in)+np.arccos(cos_out))
    caxc =  np.cos(stagger)
    # use area based on fixed axial distance
    
    
    if area_mod ==0:
        area_in = 0.0*sc.flatten()*caxc.flatten()
        area_pass = sc.flatten()*caxc.flatten()
        area_out = 0.4*sc.flatten()*c
    elif area_mod ==1:
        area_in, _ = evaluate(models['a_in'], X)
        area_pass, _ = evaluate(models['a_pass'], X)
        area_out, _ = evaluate(models['a_out'], X)
    else:
        area_in = area_in.flatten()
        area_pass = area_pass.flatten()
        area_out = area_out.flatten()
 
    Vss,unc = evaluate(models['V_ss'], X) 
    Vps,_ = evaluate(models['V_ps'], X)

    Vss_mid,_ = evaluate(models['V_mid_ss'], X) 
    Vps_mid,_ = evaluate(models['V_mid_ps'], X)
    Vss_ew,_ = evaluate(models['V_ew_ss'], X) 
    Vps_ew,_ = evaluate(models['V_ew_ps'], X)
    Vss_tip,_ = evaluate(models['V_tip_ss'], X) 
    Vps_tip,_ = evaluate(models['V_tip_ps'], X)
    Vew_i,_ = evaluate(models['V_ew_in'], X) 
    Vew_p,_ = evaluate(models['V_ew_pass'], X)
    Vew_o,_ = evaluate(models['V_ew_out'], X) 
    Vtip_i,_ = evaluate(models['V_tip_in'], X) 
    Vtip_p,_ = evaluate(models['V_tip_pass'], X)
    Vtip_o,_ = evaluate(models['V_tip_out'], X) 
    Vtipv, _ = evaluate(models['tipv'], X)
    V2tip, _ = evaluate(models['V2_tip'], X) 
    Vr, _ = evaluate(models['Vr'], X) 
    Vr = np.exp(Vr)
    DHfs, _ = evaluate(models['DHfs'], X)  
    afs, _ = evaluate(models['afs'], X) 
    #rho, _ = evaluate(models['rho'], X)
    Vl, _ = evaluate(models['mass_leak'], X) 
    sinw, _ = evaluate(models['wedge'], X) 
    #cd_ss, _ = evaluate(models['cd_ss'], X) 
    #cd_ss = np.exp(cd_ss)
    cd_ss = 0.0013 + 0.0026*(1-DH+load) +0.125*tc*ec/sc
    cd_ps, _ = evaluate(models['cd_ss'], X) 
  
    #cd_ps = 0.8581*sc*ec**2  +0.01826*load**3/AR   -1.953e-5*sweep + 0.002676
                            
    #cd_ew_ss, _ = evaluate(models['cd_ew_ss'], X) 
    cd_ew_ss= 0.00325 +0.09*ec -0.00075*ec*lean/AR -0.0176*ec/load
    
    #cd_ew_ps, _ = evaluate(models['cd_ew_ps'], X)
    #cd_ew_ps= 0.002677 +0.006873*load*cos_out**3 +0.0001676*AR - 9.325e-6*sc/ec
    #cd_ew, _ = evaluate(models['cd_ew_pass'], X)
    #cd_ew = 0.0013 +0.0019*load*psi/phi
    cd_ss = cd_ss*cd_rat
    cd_ps = cd_ps*cd_rat
    cd_ew_ss = cd_ew_ss*cd_rat
    #cd_ew_ps = cd_ew_ps*cd_rat
    #cd_ew = cd_ew*cd_rat
    lim1 = 0.50
    lim2 = 0.4
    
    if ad ==1:
        if len(cd_ss) >1:
            ind1 = (psi<lim1) & (psi>lim2)
            ind2 = (psi <= lim2)
            cd_ss[ind2] = cd_ss[ind2]*(2.9265 - 10.3737*psi[ind2] +18.5579*psi[ind2]**2-10.98*psi[ind2]**3)
            cd_ew_ss[ind2] = cd_ew_ss[ind2]*(2.9265 - 10.3737*psi[ind2] +18.5579*psi[ind2]**2-10.98*psi[ind2]**3)
            wt = (psi-lim2)/(lim1-lim2)
            cd_ss[ind1] = wt[ind1]*cd_ss[ind1] +(1-wt[ind1])*(cd_ss[ind1]*(2.9265 - 10.3737*psi[ind1] +18.5579*psi[ind1]**2-10.98*psi[ind1]**3))
            cd_ew_ss[ind1] = wt[ind1]*cd_ew_ss[ind1] +(1-wt[ind1])*cd_ew_ss[ind1]*(2.9265 - 10.3737*psi[ind1] +18.5579*psi[ind1]**2-10.98*psi[ind1]**3)
        else:
            if psi <0.45:
                cd_ss = cd_ss*(2.9265 - 10.3737*psi +18.5579*psi**2-10.98*psi**3)
                cd_ew_ss = cd_ew_ss*(2.9265 - 10.3737*psi +18.5579*psi**2-10.98*psi**3)
            
    cd_ew = cd
    #cd_ew_ss =cd
    V3_V2ss = np.reshape((Vss/V2_U)**3,(-1,1))
    V3_V2ps = np.reshape((Vps/V2_U)**3,(-1,1))

    cd_ps = cd_ss
    cd_ew_ps = cd_ss
    #Xte = np.concatenate((X,V3_V2ps, V3_V2ss), axis=1)
    Xte = np.concatenate((np.reshape(AR,(-1,1)),np.reshape(ec,(-1,1)),np.reshape(V3_V2ps,(-1,1)),np.reshape(V3_V2ss,(-1,1)),
                             np.reshape(sc,(-1,1)),np.reshape(tc,(-1,1)),np.reshape(te,(-1,1)),np.reshape(lean,(-1,1)),np.reshape(sweep,(-1,1))),axis=1)
    Vte_V2, _ = evaluate(models['VTE_ss'], X) 
    Vtess_V2, _ = evaluate(models['VTE_sst'], X) 
    Vte_fs, _ = evaluate(models['VTE_fs'], X)
    btot, _ = evaluate(models['block'], Xte) 
    btot2, _ = evaluate(models['block2'], X) 
    H23, _ = evaluate(models['H23V'], X) 
    V2fs_VTE_ss, _ = evaluate(models['V2fs_VTE_ss'], X,0)
    integral, _ = evaluate(models['int'], X,0)
        #th, _ = evaluate(models['th'], X,0)
    

    w = sc.flatten()*cos_out
    
    mlr = Vl.flatten()*ec.flatten()/(V2_U.flatten()*AR.flatten()*w.flatten())
  
    
    loss_blade_ew = 2*(0.2)*(cd_ew_ss*Vss_ew.flatten()**3 + cd_ew_ps*Vps_ew.flatten()**3)/(AR.flatten()*w.flatten()*(V2_U.flatten()**3))
    loss_blade_tip = 2*(0.2-ec.flatten())*(cd_ss*Vss_tip.flatten()**3 + cd_ps*Vps_tip.flatten()**3)/(AR.flatten()*w.flatten()*(V2_U.flatten()**3))
    loss_blade_mid = 2*(AR.flatten()-0.4)*(cd_ss*Vss_mid.flatten()**3 + cd_ps*Vps_mid.flatten()**3)/(AR.flatten()*w.flatten()*(V2_U.flatten()**3))
    loss_blade = loss_blade_ew+loss_blade_mid+loss_blade_tip
    loss_hub_i = 2*area_in.flatten()*cd_ew*(Vew_i.flatten()**3)/(AR.flatten()*w.flatten()*(V2_U.flatten()**3))
    loss_hub_p = 2*area_pass.flatten()*cd_ew*(Vew_p.flatten()**3)/(AR.flatten()*w.flatten()*(V2_U.flatten()**3))
    loss_hub_o = 2*area_out.flatten()*cd_ew*(Vew_o.flatten()**3)/(AR.flatten()*w.flatten()*(V2_U.flatten()**3))
    loss_cas_i = 2*area_in.flatten()*cd_ew*(Vtip_i.flatten()**3)/(AR.flatten()*w.flatten()*(V2_U.flatten()**3))
    loss_cas_p = 2*area_pass.flatten()*cd_ss*(Vtip_p.flatten()**3)/(AR.flatten()*w.flatten()*(V2_U.flatten()**3))
    loss_cas_o = 2*area_out.flatten()*cd_ew*(Vtip_o.flatten()**3)/(AR.flatten()*w.flatten()*(V2_U.flatten()**3))
    #loss_tip_o = 2*area_out.flatten()*cd_tip*(Vtip_o.flatten()**3)/(AR.flatten()*w.flatten()*(V2.flatten()**3))
    loss_tip = 2*(mlr.flatten())*Vtipv.flatten()/(V2_U.flatten()**2)
    loss_hub = loss_hub_i.flatten() + loss_hub_p.flatten() + loss_hub_o.flatten()
    loss_cas = loss_cas_i.flatten() + loss_cas_p.flatten() + loss_cas_o.flatten()
    
    # blockage model
     
    loss_bl  = loss_blade*(2*0.6*V2fs_VTE_ss-1)
    loss_block = (btot.flatten() + tec.flatten()/w.flatten())*Vte_V2.flatten()*Vte_fs.flatten()/(V2_U.flatten()**2)
    loss_block_wedge = sinw*btot*Vtess_V2.flatten()*Vte_fs.flatten()/(V2_U.flatten())

    loss_rad = Vr.flatten()/V2_U.flatten()
    # ske model
    #term2 = DHfs.flatten()*sind(afs.flatten())/cos_out.flatten() -np.tan(np.arccos(cos_out.flatten()))
    #loss_ske = 0.13*(th.flatten()/AR)*2*(term2.flatten()+integral.flatten())/(DHfs.flatten()**2)
    
    
    loss_wake  = loss_bl + loss_block + loss_rad + loss_block_wedge
    loss_tot = loss_hub.flatten() + loss_cas.flatten() +loss_blade.flatten() +loss_wake.flatten() +loss_tip.flatten()


    phi_fs, _ = evaluate(models['phi_fs'], X)

    


    # Get equivalent efficiences
    output['lost_eff_tot'] = 100*((V1_U.flatten()*DH.flatten())**2)*loss_tot.flatten()/psi.flatten()   
    output['lost_eff_blade'] = 100*((V1_U.flatten()*DH.flatten())**2)*loss_blade.flatten()/psi.flatten() 
    output['lost_eff_tip'] = 100*((V1_U.flatten()*DH.flatten())**2)*loss_tip.flatten()/psi.flatten() 
    
    output['lost_eff_cas'] = 100*((V1_U.flatten()*DH.flatten())**2)*loss_cas.flatten()/psi.flatten() 
    output['lost_eff_hub'] = 100*((V1_U.flatten()*DH.flatten())**2)*loss_hub.flatten()/psi.flatten() 
    output['lost_eff_bl'] = 100*((V1_U.flatten()*DH.flatten())**2)*loss_bl.flatten()/psi.flatten()
    output['lost_eff_block'] = 100*((V1_U.flatten()*DH.flatten())**2)*loss_block.flatten()/psi.flatten()
    output['lost_eff_wake'] = 100*((V1_U.flatten()*DH.flatten())**2)*loss_wake.flatten()/psi.flatten()
    output['loss_hub'] = loss_hub
    output['loss_cas'] = loss_cas
    output['loss_tip'] = loss_tip
    output['loss_blade'] = loss_blade
    output['loss_block'] = loss_block
    output['loss_blh'] = loss_bl 
    output['loss_rad'] = loss_rad
    output['loss_wedge'] = loss_block_wedge
    output['loss_wake'] = loss_wake
    output['loss'] = loss_tot
    output['bte'] =btot
    output['Vte_V2'] =Vte_V2
    output['Vte_fs'] =Vte_fs
    output['V2fs_VTE'] =V2fs_VTE_ss
    output['Vss']=Vss_mid/V2_U.flatten()
    output['Vps']=Vps_mid/V2_U.flatten()
    output['cdss']=cd_ss.flatten()
    output['cdps']=cd_ps.flatten()
    output['Vhub'] = Vew_p/V2_U.flatten()
    output['Vcas'] = Vtip_p/V2_U.flatten()
    output['Vleak'] = Vtipv.flatten()/V2_U.flatten()
    output['mleak'] = mlr.flatten()
    output['Vl'] = Vl.flatten() 
    output['V2_U'] = V2_U.flatten()
    output['V1_U'] = V1_U.flatten()
    output['psi'] = psi
    output['phi'] = phi
    output['phi_fs'] = phi_fs
    output['block'] = 1-phi/phi_fs                             
    output['DH'] = DH
    output['cos_out'] = cos_out
    output['cd_ss'] = cd_ss
    output['cd_ew'] = cd_ew
    
    return output


def r2_score(y_true, y_pred):
    correlation_matrix = np.corrcoef(y_true,y_pred)
    
    correlation_xy = correlation_matrix[0,1]
    if np.isnan(correlation_xy):
        correlation_xy = 0
        
    r_squared = correlation_xy**2
    

    return r_squared


def uncertainy_model(model,N,RMSE):
    noise_variance = RMSE**2 * np.ones(N)
    Sigma = np.diag(noise_variance) # a diagonal matrix need not be assumed.


    # On the training data! 
    P = model.get_poly(model._quadrature_points)
    
    W = np.diag(np.sqrt(model._quadrature_weights))
    A = np.dot(W, P.T)
    Q = np.dot( inv( np.dot(A.T, A) ), A.T)
    return Q, Sigma       

def inv(M):
    ll, mm = M.shape
    M2 = deepcopy(M) + 1e-10 * np.eye(ll)
    L = np.linalg.cholesky(M2)
    inv_L = np.linalg.inv(L)
    inv_M = inv_L.T @ inv_L
    return inv_M    

def ncr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer // denom  # or / in Python 2

def normalise(inp):
    in_min = np.nanmin(inp,axis=0)
    in_max = np.nanmax(inp,axis=0)
    out = (inp-in_min)/(in_max-in_min)
    return out

def unnormalise(inp,new):
    in_min = np.nanmin(inp,axis=0)
    in_max = np.nanmax(inp,axis=0)
    out = in_min + new*(in_max-in_min)
    return out


def standardise_orig(X_orig,X):
    M,d=X.shape
    X_stnd=np.zeros((M,d))
    for j in range(0,d):
        max_value = np.max(X_orig[:,j])
        min_value = np.min(X_orig[:,j])
        for i in range(0,M):
            X_stnd[i,j]=2.0 * ( (X[i,j]-min_value)/(max_value - min_value) ) -1
    return X_stnd

def unstandardise_orig(X_orig,X):
    d=X.shape[1]
    X_unstnd=np.zeros_like(X)
    for j in range(0,d):
        max_value = np.max(X_orig[:,j])
        min_value = np.min(X_orig[:,j])
        X_unstnd[:,j] = (X[:,j] +1)*(max_value - min_value)/2 + min_value
    return X_unstnd


def input_check(x,n):
    # Function which takes the input data and polynomial order and checks that a reasonable polynomial can be learned based on those parameters. It finds the number of unique data points and the desired polynomial order and checks that the number of coeficients for the model is less than the number of unique points. If this is not the case the maximum order of polynomial is returned that is possible for the data. 
    nrow, dim = np.shape(x)
    # Calculate the number of unique data points in the data
    unique_rows = np.unique(x, axis=0)
    n_data = len(unique_rows) # number of unique data points
    #for i in range(0,dim):
        
    unique_rows = np.unique(x,axis=0)
    n_row = len(unique_rows) # number of unique data points
    if n_row < n_data:
        n_data = n_row
    
    # calculate the number of coefficients required for the model
    
    k = ncr(dim+n,dim)
    
    if k >= n_data:
        while k >= n_data:
            n = n-1
            k = ncr(dim+n,dim)
            if n == 0:
                break
            
    return n


# Need extra input to define the quantity to be predicted, 'varables' can be independant of n it doesn't matter what the inouts are the equadrature packages automatically generates the required polynomials
def fit(x,y,n, method = 'least-squares',unc=0):
  # FIT performs a `n'-degree polynomial (degree `n') surface regression, and returns the polynomial coefficients as the result.
  #
  # Inputs:
  #	  - n : desired polynomial degree. Must be =< number of data dimensions
  #	  - x: array of points, which are arrays containing values.
  #   - y : array of target values
  #
  #
  # Example: Use polynomial regression based on sepal_length, sepal_width, 
  # and petal_width to model petal_length
  #
  # data = [
  #   sepal_length, sepal_width, petal_length, petal_width
  # [      5.1,          3.5,         1.4,          0.2   ],
  # [      4.9,          3,           1.4,          0.2   ],
  # [      4.7,          3.2,         1.3,          0.2   ],
  # [      4.6,          3.1,         1.5,          0.2   ],
  # [      5,            3.6,         1.4,          0.2   ]
  # ]
  #
  # x = [
  #	  data[0][ sepal_length, sepal_width, petal_width ],
  #	  data[1][ sepal_length, sepal_width, petal_width ],
  #	  data[2][ sepal_length, sepal_width, petal_width ],
  #	  data[3][ sepal_length, sepal_width, petal_width ],
  #   data[4][ sepal_length, sepal_width, petal_width ]
  # ]
  #
  # y = [
  #   data[0][ petal_length ]
  #   data[1][ petal_length ]
  #   data[2][ petal_length ]
  #   data[3][ petal_length ]
  #   data[4][ petal_length ]
  # ]
  #
  # n = 1 # nmax = data.length - 1
  #
  # coefficients = fit(x, y, n);
  #
  # ############################################################
  
  
  # I'm already passing the order in as 'n'. Where should that one go?
    x = np.array( x )
    y = np.array( y )
  
    y_norm = normalise(y)
    nrow, dim = np.shape(x)
    X_range = np.vstack((np.max(x, axis=0), 
              np.min(x,axis=0)))
    Y_range = np.vstack((np.max(y, axis=0), 
              np.min(y,axis=0)))
    X_norm = standardise_orig(X_range,x)
    n = input_check(x,n)
    if n == 0:
        coef = [np.mean(y_norm)]
        Y_norm_pred = np.full(np.shape(y_norm),np.mean(y_norm))
        s1 = Parameter(distribution='uniform', lower=-1., upper=1., order=n)
        myparameters1 = [s1 for _ in range(0, dim)]

        basis = Basis('total-order')
        model = Poly(parameters=myparameters1, basis=basis, method=method, \
                              sampling_args= {'mesh': 'user-defined', 'sample-points': X_norm, 'sample-outputs': y_norm})
      #model._quadrature_weights = np.reshape(Y_norm_pred,(-1,1)) 
    
    else:
      # Create a polynomial as before
        s1 = Parameter(distribution='uniform', lower=-1., upper=1., order=n)
        myparameters1 = [s1 for _ in range(0, dim)]

        basis = Basis('total-order')
        model = Poly(parameters=myparameters1, basis=basis, method=method, \
                                  sampling_args= {'mesh': 'user-defined', 'sample-points': X_norm, 'sample-outputs': y_norm})
        model.set_model()
        coef = model.get_coefficients().tolist()
        N = len(coef)   
        Y_norm_pred = np.squeeze(np.dot(model.get_poly(X_norm).T ,model.get_coefficients().reshape(N, 1))) 
        
    metrics = model_metrics(y_norm, Y_norm_pred,coef)  
    if unc ==1:
        Q , Sigma =  uncertainy_model(model,len(y_norm),metrics['RMSE']) 
        rmse = metrics['RMSE']
        r2 = metrics['Adjusted_R2'] 
        var = metrics['Variance']
    else:
        Q=np.array([0, 0])
        Sigma =np.array([0, 0])
        rmse =0
        var =0
        r2 =0
            
  # Whatever is returned must be JSON serializable. ndarrays are not.
    return {
    "coefficients": coef,
    "X_range": X_range.tolist(),
	"Y_range": Y_range.tolist(),
    "Order" : n,
      "RMSE" : rmse ,
      "Adjusted_R2" : r2,
      "Variance" : var,
      "Q" : Q.tolist(),
      "Sigma" : Sigma.tolist(),
      "EQ_mod": model 
  }
 



def evaluate(model, points,unc =0):
  # Use the model coefficients to evaluate each incoming point. The evaluation results should have the same number of dimensions as `points'.
  
  # Unpack inputs
    coefficients = np.array( model['coefficients'] )
    Y_range = np.array( model['Y_range'] )
    X_range = np.array( model['X_range'] )
    n = model['Order'] 
    if unc ==1:
        Q = np.array(model['Q'])
        Sigma = np.array(model['Sigma'])
    points = np.array(points) 
    nrow, dim = np.shape(points)
  
  
    # Create a polynomial as before
    s1 = Parameter(distribution='uniform', lower=-1., upper=1., order=n)
    myparameters1 = [s1 for _ in range(0, dim)]
 
    X  = standardise_orig(X_range,X_range) 
    Y  = normalise(Y_range)

    basis = Basis('total-order')
    model_temp = Poly(parameters=myparameters1, basis=basis, method='least-squares', \
              sampling_args= {'mesh': 'user-defined', 'sample-points': X, 'sample-outputs': Y})  
    
    N = len(coefficients)  
    X_norm  = standardise_orig(X_range,points)  
    Y_norm = np.squeeze(np.dot(model_temp.get_poly(X_norm).T ,coefficients.reshape(N, 1))) 
    y = unnormalise(Y_range,Y_norm)
    
  # Calculate uncertainity
    # Construct A matrix for test points, but omit weights
    Po = model_temp.get_poly(X_norm)
    Ao = Po.T

  # Propagate the uncertainties
    if unc ==1:
        Sigma_X = np.dot( np.dot(Q, Sigma), Q.T)
        Sigma_F = np.dot( np.dot(Ao, Sigma_X), Ao.T)
        std_F = np.sqrt( np.diag(Sigma_F) )
        uncertainty = std_F.reshape(-1,1)
    else:
        uncertainty = y
  # Just return the list.
    return  y,  uncertainty

def model_metrics(real_data, evaluated_data, coefficients): 
    k = len(coefficients)  
    n = len(real_data) 
    rmse = ((((np.array(real_data) - np.array(evaluated_data))**2).mean())**0.5)/np.std(real_data)
    rmse = (((( 1- np.array(evaluated_data)/np.array(real_data))**2).mean())**0.5)
    R2 = r2_score(np.array(real_data),np.array(evaluated_data))
    Adjusted_R2 = 1 - (1-R2)*(n-1)/(n-k-1)
    Variance = np.var(real_data)
    

    return {"Adjusted_R2" : Adjusted_R2,
           "RMSE": rmse,
           "Variance": Variance}

def meshgrid(X1_ind,X1_min,X1_max, X2_ind,  X2_min, X2_max, Xpoint, N):
    # Creat subset of data for thin aerofoil analysis
    # Xpoint is a lict of the point to evaluate the samples the first 2 columns must correspond to the data to be contoured
    nsamp = N*N
    
    X = np.array( Xpoint )
    dim = len(Xpoint)
 
    # overwirting for structured smith chart
    X1_samples = np.linspace(X1_min, X1_max, N)
    X2_samples = np.linspace(X2_min, X2_max, N)
    [X1, X2] = np.meshgrid(X1_samples, X2_samples)
    X1_vec = np.reshape(X1, (N*N, 1))
    X2_vec = np.reshape(X2, (N*N, 1)) 
    Xsamples = np.ones((N*N,dim ))*X
    
    Xsamples[:,X1_ind] = X1_vec.flatten()
    Xsamples[:,X2_ind] = X2_vec.flatten()
    
    
    return Xsamples

def evaluateMeshgridWithPolynomialModel(model,ind,ranges, N):
    
    ranges = np.array(ranges)
    Xpoint = ranges[:,0]
    
    X1_ind = ind[0]
    X2_ind = ind[1]
    X1_min = ranges[X1_ind,0]
    X1_max = ranges[X1_ind,1]
    X2_min = ranges[X2_ind,0]
    X2_max = ranges[X2_ind,1]
    input_points = meshgrid(X1_ind,X1_min,X1_max, X2_ind,  X2_min, X2_max, Xpoint, N)
    #print(model)
    contour_values2, uncertainty2 = evaluate(model,input_points)
   
    X = np.array( input_points )
    
    X_points = X[:,X1_ind]
    Y_points =X[:,X2_ind]
    contour_values = np.array(contour_values2)
    uncertainty = np.array(uncertainty2)
    #print(np.shape(contour_values))
 
    return X_points,Y_points,   contour_values,uncertainty


def parameters(phi,psi,gc,tc):
    # Calculate flow angles 50% reaction repeating stage
    tan_alpha1 = (1-0.5-(psi/2))/phi
    tan_beta1 = tan_alpha1 - 1/phi
    beta1 = np.arctan(tan_beta1)
    tan_beta2 = tan_beta1 + psi/phi
    beta2 = np.arctan(tan_beta2)
    cos_in = np.cos(beta1)
    cos_out = np.cos(beta2)
    DH = cos_in/cos_out
    V1_U = phi/cos_in
    sc = gc/cos_in
    DF = 1 - DH + 0.5*sc*psi/V1_U
    return cos_in, cos_out, DH, V1_U, sc,DF

def angles(phi,psi):
    # Calculate flow angles 50% reaction repeating stage
    tan_alpha1 = (1-0.5-(psi/2))/phi
    tan_beta1 = tan_alpha1 - 1/phi
    beta1 = np.arctan(tan_beta1)
    tan_beta2 = tan_beta1 + psi/phi
    beta2 = np.arctan(tan_beta2)
    cos_in = np.cos(beta1)
    cos_out = np.cos(beta2)
    DH = cos_in/cos_out
    V1_U = phi/cos_in

    return cos_in, cos_out, DH, V1_U

def angles2(cos_in,cos_out):
    # Calculate flow angles 50% reaction repeating stage
    beta1 = np.arccos(cos_in)
    beta2 = np.arccos(cos_out)
    tan_beta1 = np.tan(beta1)
    tan_beta2 =np.tan(beta2)
    phi = 1/(tan_beta1 + tan_beta2)
    DH = cos_in/cos_out
    V1_U = phi/cos_in
    psi = phi*(tan_beta1 - tan_beta2)
    return phi, psi, DH, V1_U


def diffusion_factor(DF,DH,psi,V1_U):
    sc = 2*V1_U*(DF+DH -1)/psi
    return sc


def ew_loss(DF,DH,AR,ec, mod='WM'):
    if mod == 'WM':
        tclc = np.max([0.005*AR,ec])
        DFst = 0.4134 +0.7093*(0.16-tclc)
        DFcr = DFst - 0.015
        P = np.log(0.01176/AR)/np.log(1.3-DFst)
        G = 0.001*P/((1.3-DFcr)**(P+1))

        Glim = 0.06*AR/(DH**2)
        DFlim = 1.3 - (0.001*P/Glim)**(1/(P+1))
        thlim = 0.001/((1.3-DFlim)**P)
        A = (0.001 - thlim - Glim*(0.3-DFlim))/((DFlim**2)-0.6*DFlim + 0.09)
        B = Glim - 2*A*DFlim
        C = 0.001 - 0.09*A - 0.3*B

        loss = DF
        n = np.size(DF)
        th = 0.001/((1.3-DFcr)**P) + G*(DF-DFcr)
        i =  (DF < 0.3)
        th[i] = 0.001
        i =  (DF < DFlim) & (DF >=0.3)
        th[i] = A[i]*DF[i]**2 + B[i]*DF[i] +C[i]
        i =  (DF >= DFlim) & (DF < DFcr)
        th[i] = 0.001/((1.3-DF[i])**P[i])


        loss_ew = (0.035 + 1.98*((ec)**1.55)+th)/AR
    else:
        beta = np.array([0.06925694,
                        -0.78955089,
                4.79552054,
                -11.28834548,
                9.36594389,
                63.42547887,
                -614.56718287,
                2204.62301995,
                -3470.72680381,
                2034.16092926,
                -1479.23330237,
                12818.31717717,
                -38223.81950502,
                43618.99593009,
                -12516.48005119,
                4165.78162138,
                -20645.77804574,
                -18322.84464532,
                216305.10857349,
                -266162.66625881])
        EC = ec.reshape((-1,1))
        DF2 = DF.reshape((-1,1))
        D = np.concatenate((np.ones(np.shape(EC)), DF2, DF2**2, DF2**3, DF2**4, EC, EC*DF2, EC*DF2**2, EC*DF2**3, EC*DF2**4,EC**2,(EC**2)*DF2,
         (EC**2)*DF2**2, (EC**2)*DF2**3, (EC**2)*DF2**4, (EC**3), (EC**3)*DF2, (EC**3)*DF2**2, (EC**3)*DF2**3, (EC**3)*DF2**4), axis = 1)
        ELP = D@beta.reshape((-1,1))
        ELP = D@beta.reshape((-1,1))
        loss_ew = ((ELP.reshape((-1,)))/AR.reshape((-1,))).reshape((-1,))
        
    return loss_ew
 
    
def profile_loss(DF,DH,tc,w):
   
    scDV_V1  = 2*(DF -1 +DH)
    DF_eq = (1 - DH +(0.1+tc*(10.116-34.15*tc))*scDV_V1)/DH +1
    loss_p = 2*0.0076/w
    loss_p[DF_eq>1.2] = 2*(0.0076 + 0.034*(DF_eq[DF_eq>1.2]-1.2)**2)/w[DF_eq>1.2]    
    return loss_p   

def loss_mod_exist(DF,DH,tc,w,AR,ec,mod='WM'):   
    loss_p = profile_loss(DF,DH,tc,w)
    loss_ew=ew_loss(DF,DH,AR,ec, mod=mod)
    return loss_p, loss_ew

def camber_line(dcam_le,dcam_te):
    A = np.array([[0, 0, 0, 0, 1],
         [1, 1, 1, 1, 1],
         [1, -2, 1, 0, 0],
         [0, 0, 0, 1, 0],
         [4, 3, 2, 1, 0]])
    y = np.array([1, 0, 4, dcam_le, dcam_te])
    p = np.linalg.solve(A,y)
    return p

def cosd(ang):
    out = np.cos(np.radians(ang))
    return out

def sind(ang):
    out = np.sin(np.radians(ang))
    return out

def tand(ang):
    out = np.tan(np.radians(ang))
    return out


def camber(dcam_le,dcam_te,chi_le,chi_te,s_cl):
    p = camber_line(dcam_le,dcam_te)
    cam = np.polyval(p,s_cl)
    chi = cam*(chi_le-chi_te) + chi_te
    #print(np.shape(chi),np.shape(s_cl))
    x = np.cumsum(0.5*(cosd(chi[1:,0]) + cosd(chi[0:-1,0])) * np.diff(np.reshape(s_cl,(-1,)))) 
    rt = np.cumsum(0.5*(sind(chi[1:,0]) + sind(chi[0:-1,0])) * np.diff(np.reshape(s_cl,(-1,)))) 
    
    sf = 0.028 / np.sum((np.diff(x,1,0)**2 + np.diff(rt,1,0)**2)**0.5)
    x_cam = (x - x[0]) * sf 
    y_cam = (rt - rt[0])*sf

    # Coordinates on straight chord line
    x_chord = (s_cl * (x[-1] - x[0]) + x[0])*sf 
    y_chord = (s_cl * (rt[-1] - rt[0]) + rt[0]) * sf
    return chi,x,rt, x_cam, y_cam, x_chord, y_chord


def calc_thick(s_cl,x,x_f,x_r,s_j,s_split,thick_te):
    # Evaluate thickness from stretched polynomials

    # Stretch chordwise spacing of camber line points
    s_interp = s_cl;
    #s_interp[s_interp < s_split] = s_interp[s_interp < s_split] - np.polyval(x,s_interp[s_interp < s_split])

    # Evaluate polynomials at stretched spacing
    S = np.zeros(np.size(s_cl)).reshape(-1,1)
    S[s_interp <= s_j] = np.polyval(x_f,s_interp[s_interp < s_j])
    S[s_interp >= s_j] = np.polyval(x_r,s_interp[s_interp >= s_j])

    # Calculate real thickness
    thick = S * (s_cl**0.5 * (1 - s_cl)) + s_cl * thick_te
    return thick, S
    
def thickness(tc,te, s_cl,ote=1):
    
    if ote == 1:
        ni_te = np.round(0.2 * len(s_cl)) 

    thick_te = te 
    s_thick_max = 0.38 
    rad_thick_max = 0.18 
    rad_le = 5 
    wedge_te = (68  - 10*(thick_te-0.01)/0.32)
    thick_max = tc*0.028 
    tchord = 0.028
    S1 = (2*rad_le)**0.5
    s2 = s_thick_max
    S2 = (1 - s2 * thick_te) / ((s2**0.5) * (1 - s2))
    a = -thick_te / ((s2**0.5) - (s2**1.5))
    b = (1 - s2 * thick_te) *  ((1/(2*(s2**0.5))) - 1.5*(s2**0.5)) / ( ((s2**0.5) -(s2**1.5))**2 )
    dSds_2 = a - b

    # Shape space curvature at max thickness point
    d2tds2_2 = -1 / rad_thick_max
    a = 2 * ((1/(2*(s2**0.5))) - 1.5*(s2**0.5)) * (-thick_te) / ( ((s2**0.5) -(s2**1.5))**2 )
    b = 2 * (((1/(2*(s2**0.5))) - 1.5*(s2**0.5))**2) / ( (s2**0.5 -s2**1.5)**3 )
    d = ( (-1 / (4*s2**1.5)) - (0.75 / (s2**0.5)) ) / ( (s2**0.5 -s2**1.5)**2 )
    d2Sds2_2 = -a + (b-d) * (1 - s2* thick_te) + d2tds2_2 / (s2**0.5 -s2**1.5)

    # Shape space trailing edge point
    S3 = tand(wedge_te) + thick_te

    # Construct shape space spline from two cubics, rear section first
    b = [[S2] , [dSds_2 ], [d2Sds2_2], [S3]]
    A = [[s2**3, s2**2, s2, 1 ], [3*s2**2, 2*s2, 1, 0 ], [6*s2, 2, 0, 0 ], [1, 1, 1, 1]]
    x_r = np.poly1d(np.linalg.solve(A,b).flatten())
    # Calculate value, gradient and curvature at join
    s_j = 0.3
    S_j = np.polyval(x_r,s_j); 
    dSds_j = np.polyval(np.polyder(x_r),s_j)
    d2Sds2_j = np.polyval(np.polyder(np.polyder(x_r)),s_j)

    # Construct cubic for front section
    s_split = 0.11 
    s_stretch = 0.08
    b = [[S1] , [S_j],  [dSds_j], [d2Sds2_j]]
    A = [[(-s_stretch)**3, (-s_stretch)**2, -s_stretch, 1 ], [s_j**3, s_j**2, s_j, 1 ], [3*s_j**2, 2*s_j, 1, 0], [6*s_j, 2, 0, 0]]
    x_f =np.poly1d(np.linalg.solve(A,b).flatten())

    # Construct stretching cubic
    A = [[0, 0, 0, 1], [s_split**3, s_split**2, s_split, 1 ], [3*s_split**2, 2*s_split, 1, 0 ], [6*s_split, 2, 0, 0]]
    b = [[s_stretch ], [0] ,[0 ], [0]]
    x = np.poly1d(np.linalg.solve(A,b).flatten())
    thick, S = calc_thick(s_cl,x,x_f,x_r,s_j,s_split,thick_te)
    
    if ote == 0:
        thick_dim = 0.5 * thick * thick_max 
        s_dim = s_cl * tchord
    
        psi = -np.arctan(grad_mg(s_dim,thick_dim))
    
   
        ro = thick_dim / np.cos(psi);
    
    # Camberwise location of circle tip
        s_cen = s_dim - ro * np.sin(psi)
        s_end = s_cen + ro
    
    # Interpolate location of circle centre and correct radius
        q = s_cl > 0.5;
        s_cen = np.interp(s_end(q),s_cen(q),s_dim(end))  
        ro = np.interp(s_end(q),ro(q),s_dim(end))
        psi = np.interp(s_end(q),psi(q),s_dim(end))
    
        # Evaluate circle geometry
        psi_circ = np.linspace(psi,math.pi()/2,ni_te).T

        xy_circ = np.tile(np.array([s_cen, 0]),(ni_te, 1)) + ro * np.array([np.sin(psi_circ), np.cos(psi_circ)])

        #% Plot trailing edge circle


        # Insert into thickness distribution
        q = s_dim < xy_circ[0,0]
        s_dim = [[s_dim(q) ] [xy_circ[:,0]]]
        thick_dim = [[thick_dim(q)] [ xy_circ[:,1]]];

        #% Non-dimensionalise thickness
        thick = 2 * thick_dim / thick_max 
        s_cl = s_dim / tchord
        
    return thick, S

def grad_mg(x,y):



    dydx = np.zeros(np.size(x));
    d2ydx2 = np.zeros(np.size(x));

    d1 = x[2:] - x[1:-1];
    d2 = x[1:-1] - x[0:-2];
    s = d1 / d2;


    dydx[1:-1] = ((y[2:] - y[0:-2] * s**2 - y[1:-1] * (1-s**2)) / (d1 * (1 + s))).flatten()
    dydx[0] = (y[1] - y[0]) / (x[1] - x[0])
    dydx[-1] = (y[-1] - y[-2]) / (x[-1] - x[-2])


    d2ydx2[:-1] = ((y[2:] + y[0:-2] * s**3 - (y[1:-1] * (1 + s**3)) - dydx[1:-1] * d1 * (1 - s**2)) / (0.5 * d1**2 * (1 + s))).flatten()
    d2ydx2[0] = d2ydx2[1]
    d2ydx2[-1] = d2ydx2[-2]

    return dydx, d2ydx2



def gen_blade(dcam_le,dcam_te,chi_le,chi_te,tc,te,xc,ote = 1):
    chi,x,rt,xcam,ycam, xchord, ychord = camber(dcam_le,dcam_te,chi_le,chi_te,xc)
    schord = (xchord**2 +ychord**2)**0.5
    thick, S = thickness(tc,te,xc,ote)
    nx = -np.diff(ycam)/(np.diff(xcam)**2 +np.diff(ycam)**2)**0.5
    ny = np.diff(xcam)/(np.diff(xcam)**2 +np.diff(ycam)**2)**0.5
    thick = thick*tc*0.028
    xu = np.zeros(np.size(xcam))
    yu = np.zeros(np.size(xcam))
    xl = np.zeros(np.size(xcam))
    yl = np.zeros(np.size(xcam))
    # Construct both sides of blade from thickness and camber surface
    xu[1:] = xcam[1:] + 0.5 * nx *thick[1:-1].reshape(-1,) 
    yu[1:] = ycam[1:] + 0.5 * ny *thick[1:-1].reshape(-1,) 
    xl[1:] = xcam[1:] - 0.5 * nx *thick[1:-1].reshape(-1,) 
    yl[1:] = ycam[1:] - 0.5 * ny *thick[1:-1].reshape(-1,)
    

    return xu, yu, xl, yl, xcam,ycam, thick, chi, schord



def parameters_pitch(phi,psi,DF):
    # Calculate flow angles 50% reaction repeating stage
    tan_alpha1 = (1-0.5-(psi/2))/phi
    tan_beta1 = tan_alpha1 - 1/phi
    beta1 = np.arctan(tan_beta1)
    tan_beta2 = tan_beta1 + psi/phi
    beta2 = np.arctan(tan_beta2)
    cos_in = np.cos(beta1)
    cos_out = np.cos(beta2)
    DH = cos_in/cos_out
    V1_U = phi/cos_in
    sc = 2*(DF -1 +DH)*V1_U/psi
    return cos_in, cos_out, DH, V1_U, sc

def camberc(dcam_le,dcam_te,chi_le,chi_te,s_cl,c):
    p = camber_line(dcam_le,dcam_te)
    cam = np.polyval(p,s_cl)
    chi = cam*(chi_le-chi_te) + chi_te
    #print(np.shape(chi),np.shape(s_cl))
    x = np.cumsum(0.5*(cosd(chi[1:,0]) + cosd(chi[0:-1,0])) * np.diff(np.reshape(s_cl,(-1,)))) 
    rt = np.cumsum(0.5*(sind(chi[1:,0]) + sind(chi[0:-1,0])) * np.diff(np.reshape(s_cl,(-1,)))) 
    
    sf = 0.028 / np.sum((np.diff(x,1,0)**2 + np.diff(rt,1,0)**2)**0.5)
    x_cam = (x - x[0]) * sf 
    y_cam = (rt - rt[0])*sf

    # Coordinates on straight chord line
    x_chord = (s_cl * (x[-1] - x[0]) + x[0])*sf 
    y_chord = (s_cl * (rt[-1] - rt[0]) + rt[0]) * sf
    return chi,x,rt, x_cam, y_cam, x_chord, y_chord

def gen_bladec(dcam_le,dcam_te,chi_le,chi_te,tc,te,xc,c):
    chi,x,rt,xcam,ycam, xchord, ychord = camber(dcam_le,dcam_te,chi_le,chi_te,xc)
    schord = (xchord**2 +ychord**2)**0.5
    thick, S = thicknessc(tc,te,xc,c)
    nx = -np.diff(ycam)/(np.diff(xcam)**2 +np.diff(ycam)**2)**0.5
    ny = np.diff(xcam)/(np.diff(xcam)**2 +np.diff(ycam)**2)**0.5
    thick = thick*tc*c
    xu = np.zeros(np.size(xcam))
    yu = np.zeros(np.size(xcam))
    xl = np.zeros(np.size(xcam))
    yl = np.zeros(np.size(xcam))
    # Construct both sides of blade from thickness and camber surface
    xu[1:] = xcam[1:] + 0.5 * nx *thick[1:-1].reshape(-1,) 
    yu[1:] = ycam[1:] + 0.5 * ny *thick[1:-1].reshape(-1,) 
    xl[1:] = xcam[1:] - 0.5 * nx *thick[1:-1].reshape(-1,) 
    yl[1:] = ycam[1:] - 0.5 * ny *thick[1:-1].reshape(-1,)
    return xu, yu, xl, yl, xcam,ycam, thick, chi, schord

def thicknessc(tc,te, s_cl,c):
    thick_te = te 
    s_thick_max = 0.38 
    rad_thick_max = 0.18 
    rad_le = 5 
    wedge_te = (68  - 10*(thick_te-0.01)/0.32)
    thick_max = tc*0.028
    
    S1 = (2*rad_le)**0.5
    s2 = s_thick_max
    S2 = (1 - s2 * thick_te) / ((s2**0.5) * (1 - s2))
    a = -thick_te / ((s2**0.5) - (s2**1.5))
    b = (1 - s2 * thick_te) *  ((1/(2*(s2**0.5))) - 1.5*(s2**0.5)) / ( ((s2**0.5) -(s2**1.5))**2 )
    dSds_2 = a - b

    # Shape space curvature at max thickness point
    d2tds2_2 = -1 / rad_thick_max
    a = 2 * ((1/(2*(s2**0.5))) - 1.5*(s2**0.5)) * (-thick_te) / ( ((s2**0.5) -(s2**1.5))**2 )
    b = 2 * (((1/(2*(s2**0.5))) - 1.5*(s2**0.5))**2) / ( (s2**0.5 -s2**1.5)**3 )
    d = ( (-1 / (4*s2**1.5)) - (0.75 / (s2**0.5)) ) / ( (s2**0.5 -s2**1.5)**2 )
    d2Sds2_2 = -a + (b-d) * (1 - s2* thick_te) + d2tds2_2 / (s2**0.5 -s2**1.5)

    # Shape space trailing edge point
    S3 = tand(wedge_te) + thick_te

    # Construct shape space spline from two cubics, rear section first
    b = [[S2] , [dSds_2 ], [d2Sds2_2], [S3]]
    A = [[s2**3, s2**2, s2, 1 ], [3*s2**2, 2*s2, 1, 0 ], [6*s2, 2, 0, 0 ], [1, 1, 1, 1]]
    x_r = np.poly1d(np.linalg.solve(A,b).flatten())
    # Calculate value, gradient and curvature at join
    s_j = 0.3
    S_j = np.polyval(x_r,s_j); 
    dSds_j = np.polyval(np.polyder(x_r),s_j)
    d2Sds2_j = np.polyval(np.polyder(np.polyder(x_r)),s_j)

    # Construct cubic for front section
    s_split = 0.11 
    s_stretch = 0.08
    b = [[S1] , [S_j],  [dSds_j], [d2Sds2_j]]
    A = [[(-s_stretch)**3, (-s_stretch)**2, -s_stretch, 1 ], [s_j**3, s_j**2, s_j, 1 ], [3*s_j**2, 2*s_j, 1, 0], [6*s_j, 2, 0, 0]]
    x_f =np.poly1d(np.linalg.solve(A,b).flatten())

    # Construct stretching cubic
    A = [[0, 0, 0, 1], [s_split**3, s_split**2, s_split, 1 ], [3*s_split**2, 2*s_split, 1, 0 ], [6*s_split, 2, 0, 0]]
    b = [[s_stretch ], [0] ,[0 ], [0]]
    x = np.poly1d(np.linalg.solve(A,b).flatten())
    thick, S = calc_thick(s_cl,x,x_f,x_r,s_j,s_split,thick_te)
    return thick, S





