import dill
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

# Estimation helpers (O-base) from GMM.py
from GMM_r import utilities_from_beta, alpha_of_beta, build_all_moments

# ---------------------------------------------------------------------
# 0) Load the dataset
# ---------------------------------------------------------------------
with open("mnl.pkl", "rb") as f:
    data = dill.load(f)

od = data.mnl_data_used.index
W = len(od)
df = data.mnl_data_used.loc[od]  # shorthand

"""
xc/xd already collect a rich set of choice-specific regressors 
carefully zero-filled the “other” mode
"""
imputed_aadt = ((data.all_edge.aadt_MR - data.all_edge.aadt_combi_MR - data.all_edge.aadt_singl_MR))
imputed_aadt = imputed_aadt*0.410480349*0.488  
edge_traffic = imputed_aadt.loc[data.edge_ref.edge_id]
edge_traffic.loc[edge_traffic.isnull()] = 0 
edge_has_traffic_distance = np.multiply((edge_traffic > 0) * 1, data.distance)
car_per_meter = (data.P_bike.T @ edge_traffic)/(data.P_bike.T @ edge_has_traffic_distance)

zero = np.zeros(len(data.mnl_data_used))

od = data.mnl_data_used.index
xc = pd.DataFrame(index=od)
xd = pd.DataFrame(index=od)

xc['c_time'] = list(data.google_bike_time.loc[od])
xd['c_time'] = list(zero)

xc['d_drive_time'] = zero
xd['d_drive_time'] = data.mnl_data_used['form5_2500']

xc['bus_time'] = data.google_bus_time.loc[od]
xd['bus_time'] = data.google_bus_time.loc[od]

xc['bus_distance'] = list(data.google_bus_distance.loc[od])
xd['bus_distance'] = list(data.google_bus_distance.loc[od])

xc['c_bl_proportion'] = data.mnl_data_used["bl_proportion"]
xd['c_bl_proportion'] = zero

xc['c_uppercent'] = data.mnl_data_used.uppercent
xd['c_uppercent'] = zero

xc['c_avg_tan_abs'] = data.mnl_data_used.avg_tan_abs * 100
xd['c_avg_tan_abs'] = zero

xc['c_bicycle_store'] = data.mnl_data_used.bicycle_store
xd['c_bicycle_store'] = zero

xc['c_car_per_meter'] = car_per_meter
xd['c_car_per_meter'] = zero

xc['avg_tan_abs'] = list(data.elevations.loc[od, 'uppercent']) # percent of uphill biking
xd['avg_tan_abs'] = list(zero)

xc['median_hh_income'] =  data.mnl_data_used.median_hh_income / 10000
xc['to_median_hh_income'] =  data.mnl_data_used.to_median_hh_income / 10000
xc['sum_median_hh_income'] = xc['median_hh_income'] + xc['to_median_hh_income']
xd['median_hh_income'] =  data.mnl_data_used.median_hh_income / 10000
xd['to_median_hh_income'] =  data.mnl_data_used.to_median_hh_income / 10000
xd['sum_median_hh_income'] = xd['median_hh_income'] + xd['to_median_hh_income']

xc['percapita_income'] =  data.mnl_data_used.percapita_income / 10000
xc['to_percapita_income'] =  data.mnl_data_used.to_percapita_income / 10000
xc['sum_percapita_income'] = xc['percapita_income'] + xc['to_percapita_income']
xd['percapita_income'] =  data.mnl_data_used.percapita_income / 10000
xd['to_percapita_income'] =  data.mnl_data_used.to_percapita_income / 10000
xd['sum_percapita_income'] = xd['percapita_income'] + xd['to_percapita_income']

xc['male_percent'] = data.mnl_data_used.male_percent * 100
xc['to_male_percent'] = data.mnl_data_used.to_male_percent * 100
xc['sum_male_percent'] = xc['male_percent'] + xc['to_male_percent']
xd['male_percent'] =  data.mnl_data_used.male_percent * 100
xd['to_male_percent'] = data.mnl_data_used.to_male_percent * 100
xd['sum_male_percent'] = xd['male_percent'] + xd['to_male_percent']

xc['c_obtained_undergrad'] = data.mnl_data_used.obtained_undergrad
xd['c_obtained_undergrad'] = zero

xc['c_from_less_fifty'] = data.mnl_data.loc[od, 'from_less_fifty']
xd['c_from_less_fifty'] = zero

xc['c_to_less_fifty'] = data.mnl_data.loc[od, 'to_less_fifty']
xd['c_to_less_fifty'] = zero

xc['c_avg_less_fifty'] = data.mnl_data.loc[od, 'c_avg_less_fifty']
xd['c_avg_less_fifty'] = zero

xc['c_from_less_fourty'] = data.mnl_data.loc[od, 'from_less_fourty']
xd['c_from_less_fourty'] = zero

xc['c_to_less_fourty'] = data.mnl_data.loc[od, 'to_less_fourty']
xd['c_to_less_fourty'] = zero

xc['c_avg_less_fourty'] = data.mnl_data.loc[od, 'c_avg_less_fourty']
xd['c_avg_less_fourty'] = zero

xd['from_population'] = data.mnl_data.loc[od,'from_population'] / 1000
xd['to_population'] = data.mnl_data.loc[od,'to_population'] / 1000

xc['c_const'] = np.ones(len(data.mnl_data_used))
xd['c_const'] = zero

xc['d_const'] = zero
xd['d_const'] = np.ones(len(data.mnl_data_used))

xc['d_with_vehicle_hh'] = zero
xd['d_with_vehicle_hh'] = (data.mnl_data_used['with_vehicle_hh'])

xc['d_bl_proportion_drive'] = zero
xd['d_bl_proportion_drive'] = data.mnl_data_used.bl_proportion

xc['nl'] = data.mnl_data_used.nl / 1000
xc['bl'] = data.mnl_data_used.bl / 1000
xd['nl'] = zero
xd['bl'] = zero

xc['c_bike_length'] = data.mnl_data_used.google_length
xd['c_bike_length'] = zero


ctrls = list(data.control_types)
to_ctrls = list(data.to_control_types)
# sum columns: sum_<ctrl> = ctrl + corresponding to_ctrl
sum_cols = {f'sum_{c}': data.mnl_data_used[c] + data.mnl_data_used[to_c]
            for c, to_c in zip(ctrls, to_ctrls)}
# original control columns (c1, c2, ...)
ctrl_cols = {c: data.mnl_data_used[c] for c in ctrls}
# to-control columns (to_c1, to_c2, ...)
to_ctrl_cols = {t: data.mnl_data_used[t] for t in to_ctrls}
# Build DataFrames and concat once
sum_df = pd.DataFrame(sum_cols, index=od)
ctrl_df = pd.DataFrame(ctrl_cols, index=od)
to_ctrl_df = pd.DataFrame(to_ctrl_cols, index=od)
xc = pd.concat([xc, sum_df, ctrl_df, to_ctrl_df], axis=1)
xd = pd.concat([xd, sum_df, ctrl_df, to_ctrl_df], axis=1)

controls = ["movie_theater", 
            "park", 
            'restaurant', 
            "shopping_mall", 
            "supermarket", 
            "bakery", 
            "bar",
            "university", 
            "hospital", 
            "library",
            "percapita_income",
            "median_hh_income",
            "male_percent"]

sum_controls = ["sum_" + i for i in controls]

# indicator version of the control variables
one_sum_controls = []
c_one_sum_controls = []
d_one_sum_controls = []
for i in sum_controls[:-3]:
    xc['one_' + i], xd['one_' + i] = (xc[i] != 0) * 1, (xd[i] != 0) * 1
    one_sum_controls.append('one_' + i)
    xc['c_one_' + i], xd['c_one_' + i] = xc['one_' + i], zero
    c_one_sum_controls.append('c_one_' + i)
    xc['d_one_' + i], xd['d_one_' + i] = zero, xd['one_' + i]
    d_one_sum_controls.append('d_one_' + i)

c_sum_controls = []
for i in sum_controls:
    xc['c_' + i], xd['d_' + i] = xc[i], zero
    c_sum_controls.append('c_' + i)
    
d_sum_controls = []
for i in sum_controls:
    xc['d_' + i], xd['d_' + i] = zero, xd[i]
    d_sum_controls.append('d_' + i)


xc['shares'] = data.mnl_data_used.bike_demand / data.mnl_data_used.total_demand
#xc['shares'] = mnl_data_used.acs_bike_demand/mnl_data_used.total_demand
xd['shares'] = data.mnl_data_used.drive_demand / data.mnl_data_used.total_demand
o_share = np.ones(len(data.mnl_data_used)) - xc['shares'] - xd['shares']
xc['y'] = np.log(xc['shares']) - np.log(o_share)
xd['y'] = np.log(xd['shares']) - np.log(o_share)


# ---------------------------------------------------------------------
# 1) Observed shares (O as baseline)
# ---------------------------------------------------------------------
share_C = (df.bike_demand / df.total_demand).clip(1e-8, 1.0)
share_D = (df.drive_demand / df.total_demand).clip(1e-8, 1.0)
share_O = (1.0 - share_C - share_D).clip(1e-8, 1.0)
sC, sD, sO = share_C.values, share_D.values, share_O.values

# ---------------------------------------------------------------------
# 2) Core times and mode-specific scalars used in utilities_from_beta
# ---------------------------------------------------------------------
tC  = xc["c_time"].values
tD  = xd["d_drive_time"].values
tO  = xc["bus_time"].values      # same as xd['bus_time']; use one source
rho = xc["c_bl_proportion"].values

# ---------------------------------------------------------------------
# 3) XD / XC / XO: choice-specific control blocks (exclude time & rho)
# ---------------------------------------------------------------------
def pick_prefixed(df_, prefix, *, drop_cols):
    cols = [c for c in df_.columns if c.startswith(prefix) and c not in drop_cols]
    return df_[cols].values if cols else np.zeros((len(df_), 0))


xd_candidate = [
    'median_hh_income',
    'to_median_hh_income',
    'sum_median_hh_income',
    'percapita_income',
    'to_percapita_income',
    'sum_percapita_income',
    'male_percent',
    'to_male_percent',
    'sum_male_percent',

    'from_population',
    'to_population',
]

xc_candidate = [
    'median_hh_income',
    'to_median_hh_income',
    'sum_median_hh_income',
    'percapita_income',
    'to_percapita_income',
    'sum_percapita_income',
    'male_percent',
    'to_male_percent',
    'sum_male_percent',

    'c_obtained_undergrad',
    'c_from_less_fifty',
    'c_to_less_fifty',
    'c_avg_less_fifty',
    'c_from_less_fourty',
    'c_to_less_fourty',
    'c_avg_less_fourty',
]

xo_candidate = ['sum_transit_stations']

# time & intercept handled separately
XD = xd[xd_candidate]
# time & intercept & proportion handled separately
XC = xc[xc_candidate]
# time handled separtely
XO = xc[xo_candidate] # the same as in xd


# ---------------------------------------------------------------------
# 4) Instruments
#    We follow regressors.py patterns:
#      - share equations: Z_D for drivetime; Z_C for bike-lane coverage
#      - demand equation: Q_dem collects all exogenous nearby_ shifters
# ---------------------------------------------------------------------
# Share-equation IVs (one or several columns each). Common names in your scripts:
connected_census = pd.DataFrame(index = data.census_select.index, 
                                columns = data.census_select.index)
print(f'connected_census shape: {connected_census.shape}')   # (95, 95)
for i in data.census_select.index:
    connected_census[i] = 0
for w in od:
    origin = w[0]
    dest = w[1]
    connected_census.loc[origin, dest] = 1 
    connected_census.loc[dest, origin] = 1

# FIRST FOR LOOP: CREATES spatial instruments
# This loop CALCULATES the instrument values by averaging characteristics from distant OD pairs
for far_threshold in [6500, 7000, 7500, 8000, 9000]:
    name = 'fy' + str(far_threshold)
    print(f"Processing far_threshold: {far_threshold}")
    for w in od:  # XN: w stands for a specific OD pair
        origin = w[0]
        dest = w[1] 
        # focus_nod = data.od_ref.loc[w,'ind']
        # find origins connected to the current origin
        far_origin = connected_census.loc[connected_census.loc[origin] == 1].index
        # print(max(distance_mat[dest]))
        # find destinations far from the current destination:
        far_dest = data.distance_mat.loc[(data.distance_mat[dest] >= far_threshold)].index
        # far_origin = distance_mat.loc[(distance_mat[origin] <= near_threshold)].index
        # far_dest = distance_mat.loc[(distance_mat[dest] >= far_threshold)].index
        temp = data.od_incidence.loc[far_origin, far_dest]
        # select OD pairs with positive incidence
        temp_ind = temp.where(temp > 0).stack().index
        far_ind = temp_ind
        
        if len(far_ind) == 0:
            print(f"No far_ind found for OD pair {w} - setting instruments to 0")
            data.mnl_data_used.loc[w, 'nearby_proportion' + name] = 0
            data.mnl_data_used.loc[w, 'from_nearby_population' + name] = 0
            data.mnl_data_used.loc[w, 'to_nearby_population' + name] = 0
            for control in controls:
                data.mnl_data_used.loc[w, 'from_nearby_' + control + name] = 0
                data.mnl_data_used.loc[w, 'to_nearby_' + control + name] = 0
        else:
            print(f"Calculating instruments from {len(far_ind)} far_ind pairs for OD {w}")
            data.mnl_data_used.loc[w, 'nearby_proportion' + name] = (sum(data.mnl_data_used.loc[far_ind,'bl_proportion'])
                                                                    /len(far_ind))
            data.mnl_data_used.loc[w, 'from_nearby_population' + name] = (sum(data.mnl_data_used.loc[far_ind,'from_population'])
                                                                          /len(far_ind))
            data.mnl_data_used.loc[w, 'to_nearby_population' + name] = (sum(data.mnl_data_used.loc[far_ind,'to_population'])
                                                                        /len(far_ind))
            for control in controls:
                data.mnl_data_used.loc[w, 'from_nearby_' + control + name] = (sum(data.xc.loc[far_ind,control])
                                                                              /len(far_ind))
                data.mnl_data_used.loc[w, 'to_nearby_' + control + name] = (sum(data.xc.loc[far_ind,'to_' + control])
                                                                            /len(far_ind))
        # get current trip length
        temp_len = data.mnl_data_used.loc[w, 'length']
        temp_mnl_data_used = data.mnl_data_used.loc[temp_ind]

        if len(far_ind) == 0:
            print(f"No far_ind for drive time calculation - setting nearby_drivetime0 to 0")
            data.mnl_data_used.loc[w, 'nearby_drivetime0' + name] = 0
        else:
            print(f"Calculating nearby_drivetime0 from {len(far_ind)} pairs")
            data.mnl_data_used.loc[w, 'nearby_drivetime0' + name] = (sum(data.mnl_data_used.loc[far_ind,'form5_2500'])
                                                                     /len(far_ind))
                  
        # Only uses distant OD pairs whose trip length is within ±25% of current trip
        far_ind_25 = temp_mnl_data_used[(temp_mnl_data_used['length'] >= temp_len * 0.75) 
                                        & (temp_mnl_data_used['length'] <= temp_len * 1.25)].index 
        
        # find no intersection 
        # non_overlap_od = non_overlap_drive.loc[non_overlap_drive.loc[:,ind]>0].index
        # far_ind = list(set(temp_ind).intersection(set(non_overlap_od)))
        if len(far_ind_25) == 0:
            print(f"No far_ind_25 pairs found (±25% length filter) - setting nearby_drivetime25 to 0")
            data.mnl_data_used.loc[w, 'nearby_drivetime25' + name] = 0
        else:
            print(f"Calculating nearby_drivetime25 from {len(far_ind_25)} pairs (±25% length filter)")
            data.mnl_data_used.loc[w, 'nearby_drivetime25' + name] = (sum(data.mnl_data_used.loc[far_ind_25, 'form5_2500'])
                                                                      /len(far_ind_25))
        # Only uses distant OD pairs whose trip length is within ±50% of current trip
        far_ind_50 = temp_mnl_data_used[(temp_mnl_data_used['length'] >= temp_len * 0.5) 
                                        & (temp_mnl_data_used['length'] <= temp_len * 1.5)].index 
        # find no intersection 
        # non_overlap_od = non_overlap_drive.loc[non_overlap_drive.loc[:,ind]>0].index
        # far_ind = list(set(temp_ind).intersection(set(non_overlap_od)))
        if len(far_ind_50) == 0:
            print(f"No far_ind_50 pairs found (±50% length filter) - setting nearby_drivetime50 to 0")
            data.mnl_data_used.loc[w, 'nearby_drivetime50' + name] = 0
        else:
            print(f"Calculating nearby_drivetime50 from {len(far_ind_50)} pairs (±50% length filter)")
            data.mnl_data_used.loc[w,'nearby_drivetime50' + name] = (sum(data.mnl_data_used.loc[far_ind_50,'form5_2500'])
                                                                     /len(far_ind_50))



# SECOND FOR LOOP: ASSIGNS instruments to choice datasets  
# This loop ORGANIZES the previously created instruments into xc/xd matrices for estimation
for far_threshold in [6500, 7000, 7500, 8000, 9000]:  
    #fanyin instrument
    name = 'fy' + str(far_threshold)
    for control in controls:
        # create sum variables for controls
        data.mnl_data_used['sum_nearby_' + control + name] = (data.mnl_data_used['from_nearby_' + control + name] 
                                                              + data.mnl_data_used['to_nearby_' + control + name])

    # build comprehensive instrument names list:
    inst_names = ['nearby_drivetime' + i + name for i in ["0", "25", "50"]]
    inst_names += ['nearby_proportion' + name] 
    inst_names += ['from_nearby_population' + name, 'to_nearby_population' + name]
    inst_names += ['from_nearby_' + control + name for control in controls]
    inst_names += ['to_nearby_' + control + name for control in controls]
    inst_names += ['sum_nearby_' + control + name for control in controls]

    # assign instruments to choice-specific datasets:
    xc[inst_names[0]] = zero
    xd[inst_names[:3]] = data.mnl_data_used[['nearby_drivetime' + i + name for i in ["0", "25", "50"]]] 

    xc[inst_names[3]] = data.mnl_data_used['nearby_proportion' + name]
    xd[inst_names[3]] = zero

    xc[inst_names[4:]] = data.mnl_data_used[inst_names[4:]]
    xd[inst_names[4:]] = data.mnl_data_used[inst_names[4:]]



Z_D_names = [
    "nearby_drivetime0fy6500",  # drivetime shifter for D eq
    "nearby_drivetime25fy6500",
    "nearby_drivetime50fy6500",
]
Z_C_names = [
    "nearby_proportionfy6500",   # bike-lane shifter for C eq
]

Z_D = xd[Z_D_names]
Z_C = xc[Z_C_names]

# Demand-side excluded IVs:
# Start with the canonical ones you showed before...
Q_list = [
    "nearby_drivetime0fy7000",  # drivetime shifter for D eq
    "nearby_drivetime25fy7000",
    "nearby_drivetime50fy7000", 
]


""" # ...then append ANY other “nearby_*” shifters that regressors.py may have created
Q_extra = stack_any_nearby(df, prefixes=["nearby_", "from_poi", "to_poi", "sum_poi"])
Q_dem = np.column_stack([cols_if_exist(df, Q_list), Q_extra]) if Q_extra.shape[1] else cols_if_exist(df, Q_list) """
Q_dem = xd[Q_list]

# ---------------------------------------------------------------------
# 5) Demand LHS (d_obs) and Z_dem (none for now)
# ---------------------------------------------------------------------
d_obs = df["total_demand"].astype(float).values
Z_dem = np.empty((W, 0))  # no OD-level controls in demand equation for now

# ---------------------------------------------------------------------
# 6) Residual builder for least_squares: uses GMM.build_all_moments (O-base)
# ---------------------------------------------------------------------
def residuals(beta):
    return build_all_moments(
        beta,
        sD=sD, sC=sC, sO=sO,
        XD=XD, XC=XC, XO=XO,
        tD=tD, tC=tC, tO=tO, rho=rho,
        utilities_from_beta=utilities_from_beta,
        Z_D=Z_D, Z_C=Z_C,
        d_obs=d_obs, Z_dem=Z_dem, Q_dem=Q_dem,
        alpha_solver=alpha_of_beta,
        return_per_obs=False
    )

# ---------------------------------------------------------------------
# 7) Optimize over β (profiling α(β) internally)
#     β layout (must match utilities_from_beta):
#       [β0D, βtD, βXD(kD),  β0C, βtC, βrhoC, βXC(kC),  β0O, βtO, βXO(kO)]
# ---------------------------------------------------------------------
kD, kC, kO = XD.shape[1], XC.shape[1], XO.shape[1]
beta_dim = (2 + kD) + (3 + kC) + (2 + kO)
beta0 = np.zeros(beta_dim)

# helpful negative starts for time coefficients
idx_tD = 1
idx_tC = 2 + kD + 1
idx_tO = 2 + kD + 3 + kC + 1
beta0[idx_tD] = -0.01
beta0[idx_tC] = -0.01
beta0[idx_tO] = -0.01

fit = least_squares(
    residuals, beta0,
    method="trf",
    xtol=1e-9, ftol=1e-9, gtol=1e-6,
    max_nfev=800,
)
beta_hat = fit.x
print("Converged:", fit.success, "||g||:", np.linalg.norm(fit.fun))
print("beta_hat (len={}):".format(beta_hat.size), beta_hat)

# ---------------------------------------------------------------------
# 8) Inner demand step at β̂ to report α̂
# ---------------------------------------------------------------------
alpha_hat, ivgmm_res, IV_vec, d_hat, resid_dem = alpha_of_beta(
    beta_hat, d_obs, Z_dem, Q_dem, XD, XC, XO, tD, tC, tO, rho
)
print("alpha_hat:", alpha_hat)
print(ivgmm_res.summary)
