import dill
from gurobipy import *
import pandas as pd

import os
os.chdir(r'/Users/jz2293/Library/CloudStorage/Box-Box/Bike_lane_emission/code')

#dill.load_session('/Users/jingweizhang/Box/Bike_lane/clean_code/UE-reduction_bi/selected_census/obj_bike_adoption/threshold/small_area20/bike_lane_correction/workspace/R5N5opt_mac.pkl')

#dill.load_session(r'D:\Box\Bike_lane_emission\code\workspaces\R5N5opt_win.pkl') #windows
dill.load_session('workspaces/R5N5opt_win.pkl') #mac

xc = pd.read_csv("estimation_results/xc.csv",dtype={"from":str,"to":str})
xc = xc.set_index(['from','to'])
xd = pd.read_csv("estimation_results/xd.csv",dtype={"from":str,"to":str})
xd = xd.set_index(['from','to'])
valid_estimate = pd.read_csv("estimation_results/valid_estimaten500f8000.csv",index_col = 0)
valid_estimate['estimate'] = -valid_estimate['estimate']

c_notchange = ['avg_tan_abs'] + [i for i in valid_estimate.index if i[0] == 'c'][:-1]
d_notchange = [i for i in valid_estimate.index if i[0] == 'd'][:-1]
o_notchange = [i for i in valid_estimate.index if ((i[0] != 'c')&(i[0] != 'd')&(i!='avg_tan_abs'))]
c_change = ['c_bl_proportion']
d_change = ['d_drive_time']


xo = -xc.loc[od_ref.index,o_notchange]
xc = xc.loc[od_ref.index,c_notchange+c_change]
xd = xd.loc[od_ref.index,d_notchange+d_change]
uo = np.transpose(list(xo@valid_estimate.loc[o_notchange,'estimate']))
bl_coef = valid_estimate.loc[c_change,'estimate'].tolist()[0]
dt_coef = valid_estimate.loc[d_change,'estimate'].tolist()[0]
uc_unchanged = np.transpose(list(xc[c_notchange]@valid_estimate.loc[c_notchange,'estimate']))
ud_unchanged = np.transpose(list(xd[d_notchange]@valid_estimate.loc[d_notchange,'estimate']))

########################## linear approximation ###############################
#d_od = od_d['acs_S000']
od_d = od_d.set_index(['from','to'])

d_od = od_d['S000']
d_od = np.transpose(d_od.to_list())

N = 15
R = 15

a_ff = []
b_ff = []
for i in range(Nod):
    log_drange = np.linspace(np.log(0.01), np.log(d_od[i]), N)
    drange = np.exp(log_drange)
    a_temp = log_drange + 1
    b_temp = -drange
    a_ff.append(a_temp)
    b_ff.append(b_temp)
a_ff = np.transpose(a_ff)
b_ff = np.transpose(b_ff)
for i in range(N):
    a_ff[i][np.abs(a_ff[i])<1e-03] = 0
    b_ff[i][np.abs(b_ff[i])<1e-03] = 0



########################### Linear approximation ##############################
import math

a_f = []
b_f = []
a_f_lane = []
b_f_lane = []
dub = P@U.T@d_od
M_x_lane = np.zeros((Nedge, R))
M_x = np.zeros((Nedge, R))
for i in range(Nedge):
    if dub[i] == 0:
        a_f.append(np.zeros(R))
        b_f.append(np.zeros(R))
        a_f_lane.append(np.zeros(R))
        b_f_lane.append(np.zeros(R))
    else:
        xrange = np.linspace(0.01, dub[i], R)
        a_f.append(weight_cost_wo_lane[i]*xrange + weight_cost_wo_lane[i]*v0[i])
        b_f.append( - 1/2*weight_cost_wo_lane[i]*(xrange)**2)
        if weight_cost_w_lane[i] == math.inf:
            a_f_lane.append(weight_cost_wo_lane[i]*xrange + weight_cost_w_lane[i]*v0[i])
            b_f_lane.append( - 1/2*weight_cost_wo_lane[i]*(xrange)**2)
            weight_cost_w_lane[i] = weight_cost_wo_lane[i]
        else:
            a_f_lane.append(weight_cost_w_lane[i]*xrange + weight_cost_w_lane[i]*v0[i])
            b_f_lane.append( - 1/2*weight_cost_w_lane[i]*(xrange)**2)
    for r in range(R):
        M_x_lane[i,r] = a_f_lane[i][r] * dub[i] + (b_f_lane[i][r]-b_f[i][r])
        M_x[i,r] = a_f[i][r] * dub[i] + (b_f[i][r]-b_f_lane[i][r])
a_f = np.transpose(a_f)
b_f = np.transpose(b_f)
a_f_lane = np.transpose(a_f_lane)
b_f_lane = np.transpose(b_f_lane)
for i in range(R):
    a_f[i][np.abs(a_f[i])<1e-03] = 0
    b_f[i][np.abs(b_f[i])<1e-03] = 0
    a_f_lane[i][np.abs(a_f_lane[i])<1e-03] = 0
    b_f_lane[i][np.abs(b_f_lane[i])<1e-03] = 0

B = distance@np.ones(Nedge)

P_bike_temp = P_bike.copy()
exist = [s for s in range(Nedge) if bike_lane_loc[s] == 1]
notallow = [s for s in range(Nedge) if allow_lane[s] == 0]
locations = list(set(exist+notallow))
#locations = np.concatenate((np.where(bike_lane_loc == 1), np.where(allow_lane == 0)),axis=1)[0]
P_bike_temp[locations,:] = 0

bike_edges = sparse.find(P_bike_temp.T)[1]
bike_ods = sparse.find(P_bike_temp.T)[0]


###############################################################
# lower level - UE
###############################################################

from gurobipy import *

def UE(lane=bike_lane_loc):
    print(N,R,"=======================================================================")
    #uc = np.transpose([P_bike.T@distance/1000,P_bike.T@(distance*lane)/(P_bike.T @distance),np.ones(Nod)])@c_coef
    uc = uc_unchanged+P_bike.T@(distance*lane)/(P_bike.T @distance)*bl_coef

    uc = np.transpose(uc.tolist())
    a_f_s = []
    b_f_s = []
    for i in range(Nedge):
        if lane[i] == 0:
            a_f_s.append(a_f[:,i])
            b_f_s.append(b_f[:,i])
        else:
            a_f_s.append(a_f_lane[:,i])
            b_f_s.append(b_f_lane[:,i])
    a_f_s = np.transpose(a_f_s)
    b_f_s = np.transpose(b_f_s)

    W = np.diag(np.multiply(weight_cost_w_lane, lane) + np.multiply(weight_cost_wo_lane, 1-lane))

    model = Model("estimate1")
    #demand
    φ = model.addMVar(Npath)
    v = model.addMVar(Nedge)
    dd = model.addMVar(Nod)
    do = model.addMVar(Nod)
    dc = model.addMVar(Nod)
    #linear approximation
    ωc = model.addMVar(Nod,lb=-GRB.INFINITY)
    ωo = model.addMVar(Nod,lb=-GRB.INFINITY)
    ωd = model.addMVar(Nod,lb=-GRB.INFINITY)
    us = model.addMVar(Nedge,lb=-GRB.INFINITY)

    model.addConstr(do + dc + dd == d_od)
    model.addConstr(U @ φ == dd)
    model.addConstr(P @ φ == v)
    for i in range(N):
        model.addConstr(ωc >= np.diag(a_ff[i,:]) @ dc + b_ff[i,:])
        model.addConstr(ωo >= np.diag(a_ff[i,:]) @ do + b_ff[i,:])
        model.addConstr(ωd >= np.diag(a_ff[i,:]) @ dd + b_ff[i,:])
    for i in range(R):
        model.addConstr(us >= np.diag(a_f_s[i,:])@v + b_f_s[i,:])

    model.setObjective(dc@uc + do@uo + dd@ud_unchanged + us@np.ones(Nedge)*dt_coef + c0@φ*dt_coef + ωc@np.ones(Nod) + ωo@np.ones(Nod) + ωd@np.ones(Nod))
    model.modelSense = GRB.MINIMIZE
    #model.params.OutputFlag = 0
    model.update()
    model.optimize()

    demand = {}
    demand['φ_opt'] = φ.x
    demand['v_opt'] = v.x
    demand['do_opt'] = do.x
    demand['dc_opt'] = dc.x
    demand['dd_opt'] = dd.x
    demand['us_opt'] = us.x
    #print(demand, c0@φ.x + 2*sum(us.x), model.getObjective().getValue())

#    return sum(dd.x)*d_coef[0] + c0@φ.x*d_coef[1] + 2*sum(us.x)*d_coef[1]
    return demand, c0@φ.x + 2*sum(us.x), model.getObjective().getValue()

demands, drive_time, obj = UE(bike_lane_loc)

import matplotlib.pyplot as plt
plt.hist(demands['v_opt'])
sum(demands['v_opt'])
sum(demands['dc_opt'])/sum(d_od)


######################################################
# Optimization
######################################################

from datetime import datetime
from datetime import date


os.chdir(r'optimization_results')

thresholds = [0]
Bs = [10,25,50,75]
result = pd.DataFrame(index=Bs,columns=['MIPGap','Obj','ObjBound','Runtime'])
prev = thresholds[0]
bike_lane_loc_start = bike_lane_loc.copy()
bike_phi_loc_start = np.zeros(Nod)

threshold = prev



#od_d = od_d.set_index([od_d['from'],od_d['to']])
demand_census = od_d.loc[od_d.S000>threshold].index.tolist()
used_cod = np.array(od_ref.loc[demand_census].ind.tolist())                       #variable
notused_cod = np.array(od_ref.loc[~od_ref.index.isin(demand_census)].ind.tolist())#lane_phi = 0
#notused_cedge = np.unique(sparse.find(P_bike[:,notused_cod])[0])
used_cedge = np.unique(sparse.find(P_bike[:,used_cod])[0])
#notused_cedge = list(set(notused_cedge)-set(used_cedge))
notused_cedge = np.array(list(set(range(Nedge))-set(used_cedge)))
notused_exist = np.array([i for i in notused_cedge if (exist_lane[i] == 1)])#lane = 1
notused_noexist = np.array([i for i in notused_cedge if (exist_lane[i] == 0)])#lane = 0
used_allow_noexist = np.array([i for i in used_cedge if (allow_lane[i] == 1)&(exist_lane[i] == 0)]) #variable
used_notallow = np.array([i for i in used_cedge if (allow_lane[i] == 0)]) #lane = 0
used_exist = np.array([i for i in used_cedge if (exist_lane[i] == 1)]) #lane = 1


start = Npath+Nedge+3*Nod
start1 = Npath+Nedge+3*Nod+len(used_allow_noexist)
end = start1 + len(used_cod)


bike_lane_loc_full = bike_lane_loc.copy()
bike_lane_loc_full[used_allow_noexist] = 1
demands_ub, drive_time_ub, obj_ub = UE(bike_lane_loc_full)
#dc_ub = sum(demands_ub['dc_opt'])
#print(dc_ub)
print(sum(demands_ub['dc_opt']))

weights = np.multiply(weight_cost_w_lane, bike_lane_loc) + np.multiply(weight_cost_wo_lane, 1-bike_lane_loc)
weights1 = np.multiply(weight_cost_w_lane, bike_lane_loc_full) + np.multiply(weight_cost_wo_lane, 1-bike_lane_loc_full)

time = P.T@(np.multiply(weights,(demands['v_opt']+v0))) + c0
time1 = P.T@(np.multiply(weights1,(demands_ub['v_opt']+v0))) + c0
time_od = np.zeros(Nod)
time_od1 = np.zeros(Nod)
for i in range(Nod):
    time_od[i] = min(time[sparse.find(U[i,:])[1]])
    time_od1[i] = min(time1[sparse.find(U[i,:])[1]])
plt.hist((time_od1-time_od)/time_od)
print("+++++++++++++++++++++++++++++ max time",max((time1-time)/time),'+++++++++++++++++++++++++++++')
print("+++++++++++++++++++++++++++++ max od time",max((time_od1-time_od)/time_od),'+++++++++++++++++++++++++++++')
print((sum(demands_ub['dc_opt'])-sum(demands['dc_opt']))/sum(demands['dc_opt']))
print((sum(demands_ub['dd_opt'])-sum(demands['dd_opt']))/sum(demands['dd_opt']))
max((time1-time))

print(sum(demands['dc_opt'])/sum(d_od))
print(sum(demands_ub['dc_opt'])/sum(d_od) )

f, ax = plt.subplots(figsize=(6, 6))
ax.scatter(demands['v_opt'],demands_ub['v_opt'] , c=".3")
ax.plot([0, 1e4], [0, 1e4], ls="--", c=".3")

##########################################################################################################
#warm start
##########################################################################################################
def pre_solve(lane_sol, tol):
    model1 = Model("estimate1")
    #demand
    φ  = model1.addMVar(Npath, name = 'phi')
    v  = model1.addMVar(Nedge, name = 'v')
    dd = model1.addMVar(Nod, name = 'dd')
    do = model1.addMVar(Nod, name = 'do')
    dc = model1.addMVar(Nod, name = 'dc')
    #dnd = model1.addMVar(Nod) #demand of not driving
    #lane design
    #lane = model1.addMVar(len(used_allow_noexist), vtype = GRB.BINARY)
    lane = np.array(lane_sol)
    #linear approximation
    start = Npath+Nedge+3*Nod
    start1 = Npath+Nedge+3*Nod+len(used_allow_noexist)
    end = start1 + len(used_cod)

    ωc = model1.addMVar(Nod,lb=-GRB.INFINITY, name = 'wc')
    ωo = model1.addMVar(Nod,lb=-GRB.INFINITY, name = 'wo')
    ωd = model1.addMVar(Nod,lb=-GRB.INFINITY, name = 'wd')
    us = model1.addMVar(Nedge,lb=-GRB.INFINITY, name = 'us')
    #utility
    uc_s = model1.addMVar(Nedge,lb=-GRB.INFINITY, name = 'uc_s')
    uc = model1.addMVar(Nod,lb=-GRB.INFINITY, name = 'uc')
    d_uc = model1.addMVar(Nedge,lb=-GRB.INFINITY, name = 'd_uc')
    #duc_proportion = model1.addMVar(1,lb=-GRB.INFINITY)

    #dual
    μ  = model1.addMVar(Nod,lb=-GRB.INFINITY, name = 'mu')
    γ  = model1.addMVar(Nod,lb=-GRB.INFINITY, name = 'gamma')
    λ  = model1.addMVar(Nedge,lb=-GRB.INFINITY, name = 'lambda')
    πo = model1.addMVar((Nod, N), name = 'pio')
    πc = model1.addMVar((Nod, N), name = 'pic')
    πd = model1.addMVar((Nod, N), name = 'pic')
    σ  = model1.addMVar((Nedge, R), name = 'sigma')
    σb = model1.addMVar(Nedge,lb=-GRB.INFINITY, name = 'sigmab')

    dc_prop = model1.addMVar(Nedge, name = 'dc_prop')
    duc_prop = model1.addMVar(len(used_allow_noexist), name = 'duc_prop')
    σaf = model1.addMVar(len(used_allow_noexist), lb =-GRB.INFINITY, name = 'sigmaaf')
    σaf_lane = model1.addMVar(len(used_allow_noexist), lb =-GRB.INFINITY, name = 'sigmaaf_lane')
    σbf = model1.addMVar(len(used_allow_noexist), lb =-GRB.INFINITY, name = 'sigmabf')
    σbf_lane = model1.addMVar(len(used_allow_noexist), lb =-GRB.INFINITY, name = 'sigmabf_lane')


    bike_length = P_bike.T@distance
    length_coeff = P_bike@np.diag(1/bike_length)
    uc_const = uc_unchanged#bike_length/1000*c_coef[0] + np.ones(Nod)*c_coef[2]

    model1.addConstr(dc_prop == P_bike@np.diag(1/bike_length)@dc)

    M_lane = P_bike@d_od
    ones_lane = np.concatenate((notused_exist, used_exist))
    zeros_lane = np.concatenate((notused_noexist, used_notallow))

    model1.addConstr(uc_s[ones_lane] == distance[ones_lane]*bl_coef)
    model1.addConstr(d_uc[ones_lane] == np.diag(distance[ones_lane])@dc_prop[ones_lane])
    model1.addConstr(uc_s[zeros_lane] == 0)
    model1.addConstr(d_uc[zeros_lane] == 0)
    model1.addConstr(uc_s[used_allow_noexist] == np.diag(distance[used_allow_noexist]*bl_coef) @ lane)

    model1.addConstr(duc_prop == np.diag(distance[used_allow_noexist])@dc_prop[used_allow_noexist])
    for s_ind in range(len(used_allow_noexist)):
        s = used_allow_noexist[s_ind]
        model1.addConstr(d_uc[s] == duc_prop[s_ind]*lane[s_ind])

    model1.addConstr(uc == uc_const + length_coeff.T@uc_s)

    model1.addConstr(do + dc + dd == d_od)
    model1.addConstr(U @ φ == dd)
    model1.addConstr(P @ φ == v)


    for i in range(N):
        model1.addConstr(ωc >= np.diag(a_ff[i,:])@dc+b_ff[i,:])
        model1.addConstr(ωo >= np.diag(a_ff[i,:])@do+b_ff[i,:])
        model1.addConstr(ωd >= np.diag(a_ff[i,:])@dd+b_ff[i,:])
    for i in range(R):
        #notused_exist & used_exist
        model1.addConstr(us[ones_lane] >= np.diag(a_f_lane[i,ones_lane])@v[ones_lane] + b_f_lane[i,ones_lane])

        #notused_noexist & used_notallow
        model1.addConstr(us[zeros_lane] >= np.diag(a_f[i,zeros_lane])@v[zeros_lane] + b_f[i,zeros_lane])

        #used_allow_noexist
        inds = range(len(used_allow_noexist))
        model1.addConstr(us[used_allow_noexist] >= np.diag(a_f[i,used_allow_noexist])@v[used_allow_noexist] + b_f[i,used_allow_noexist] - np.diag(M_x[used_allow_noexist,i])@lane)
        model1.addConstr(us[used_allow_noexist] >= np.diag(a_f_lane[i,used_allow_noexist])@v[used_allow_noexist] + b_f_lane[i,used_allow_noexist] - (M_x_lane[used_allow_noexist,i]-np.diag(M_x_lane[used_allow_noexist,i])@lane))


    for w in range(Nod):
        model1.addConstr(μ[w] - πo[w,:]@a_ff[:,w] <= uo[w])
        model1.addConstr(μ[w] - πc[w,:]@a_ff[:,w] <= uc[w])
        model1.addConstr(μ[w] + γ[w] - πd[w,:]@a_ff[:,w] <= ud_unchanged[w])
    model1.addConstr(- U.T@γ - P.T@λ <= c0*dt_coef)



    a_f_diff = np.multiply([sum(a_f_lane[:,s]) for s in range(Nedge)],dt_coef)
    for s in notused_exist:
        model1.addConstr(λ[s] <= σ[s,:]@a_f_lane[:,s])
        model1.addConstr(sum(σ[s,:]) == dt_coef)
    for s in used_exist:
        model1.addConstr(λ[s] <= σ[s,:]@a_f_lane[:,s])
        model1.addConstr(sum(σ[s,:]) == dt_coef)
    for s in notused_noexist:
        model1.addConstr(λ[s] <= σ[s,:]@a_f[:,s])
        model1.addConstr(sum(σ[s,:]) == dt_coef)
    for s in used_notallow:
        model1.addConstr(λ[s] <= σ[s,:]@a_f[:,s])
        model1.addConstr(sum(σ[s,:]) == dt_coef)
    for s_ind in range(len(used_allow_noexist)):
        s = used_allow_noexist[s_ind]
        model1.addConstr(σaf[s_ind] == σ[s,:]@a_f[:,s])
        model1.addConstr(σaf_lane[s_ind] == σ[s,:]@a_f_lane[:,s])
        #model1.addConstr(λ[s] <= σ[s,:]@a_f[:,s] +  a_f_diff[s]*lane[s_ind])
        #model1.addConstr(λ[s] <= σ[s,:]@a_f_lane[:,s])
        model1.addConstr(λ[s] <= σaf[s_ind]*(1-lane[s_ind]) + σaf_lane[s_ind]*lane[s_ind])
        model1.addConstr(sum(σ[s,:]) == dt_coef)

    for w in range(Nod):
        model1.addConstr(πo[w,:]@np.ones(N) == 1)
        model1.addConstr(πc[w,:]@np.ones(N) == 1)
        model1.addConstr(πd[w,:]@np.ones(N) == 1)


    b_f_diff = np.multiply([(-sum(b_f_lane[:,s])) for s in range(Nedge)],dt_coef)
    for s in notused_exist:
        model1.addConstr(σb[s] == σ[s,:]@b_f_lane[:,s])
    for s in used_exist:
        model1.addConstr(σb[s] == σ[s,:]@b_f_lane[:,s])
    for s in notused_noexist:
        model1.addConstr(σb[s] == σ[s,:]@b_f[:,s])
    for s in used_notallow:
        model1.addConstr(σb[s] == σ[s,:]@b_f[:,s])
    for s_ind in range(len(used_allow_noexist)):
        s = used_allow_noexist[s_ind]
        model1.addConstr(σbf[s_ind] == σ[s,:]@b_f[:,s])
        model1.addConstr(σbf_lane[s_ind] == σ[s,:]@b_f_lane[:,s])
        model1.addConstr(σb[s] == σbf[s_ind]*(1-lane[s_ind]) + σbf_lane[s_ind]*lane[s_ind])


    model1.addConstr(do@uo + dc@uc_const + d_uc@np.ones(Nedge)*bl_coef + dd@ud_unchanged + us@np.ones(Nedge)*dt_coef + c0@φ*dt_coef + ωc@np.ones(Nod) + ωo@np.ones(Nod) + ωd@np.ones(Nod) <= (μ@d_od + sum(πo[:,n]@b_ff[n,:] + πc[:,n]@b_ff[n,:] + πd[:,n]@b_ff[n,:] for n in range(N)) + σb@np.ones(Nedge))+1e-06)


    model1.setObjective(dc@np.ones(Nod))
    model1.modelSense = GRB.MAXIMIZE
    model1.update()
    model1.optimize()

    path = "starts/" + "R"+str(R)+"N"+str(N)+"tol"+str(tol)+","+str(round(lane_sol@distance[used_allow_noexist]/sum(distance[used_allow_noexist]),4))
    if not os.path.exists(path):
        os.makedirs(path)


    np.savetxt(path+"/φ_start.csv", φ.x, delimiter=",")
    np.savetxt(path+"/v_start.csv", v.x, delimiter=",")
    np.savetxt(path+"/dd_start.csv", dd.x, delimiter=",")
    np.savetxt(path+"/do_start.csv", do.x, delimiter=",")
    np.savetxt(path+"/dc_start.csv", dc.x, delimiter=",")
    np.savetxt(path+"/ωc_start.csv", ωc.x, delimiter=",")
    np.savetxt(path+"/ωo_start.csv", ωo.x, delimiter=",")
    np.savetxt(path+"/ωd_start.csv", ωd.x, delimiter=",")
    np.savetxt(path+"/us_start.csv", us.x, delimiter=",")
    np.savetxt(path+"/uc_s_start.csv", uc_s.x, delimiter=",")
    np.savetxt(path+"/uc_start.csv", uc.x, delimiter=",")
    np.savetxt(path+"/d_uc_start.csv", d_uc.x, delimiter=",")
    np.savetxt(path+"/μ_start.csv", μ.x, delimiter=",")
    np.savetxt(path+"/γ_start.csv", γ.x, delimiter=",")
    np.savetxt(path+"/λ_start.csv", λ.x, delimiter=",")
    np.savetxt(path+"/πo_start.csv", πo.x, delimiter=",")
    np.savetxt(path+"/πc_start.csv", πc.x, delimiter=",")
    np.savetxt(path+"/πd_start.csv", πd.x, delimiter=",")
    np.savetxt(path+"/σ_start.csv", σ.x, delimiter=",")
    np.savetxt(path+"/σb_start.csv", σb.x, delimiter=",")

    φ_start  = φ.x
    v_start  = v.x
    dd_start = dd.x
    do_start = do.x
    dc_start = dc.x
    ωc_start = ωc.x
    ωo_start = ωo.x
    ωd_start = ωd.x
    us_start = us.x
    uc_s_start = uc_s.x
    uc_start = uc.x
    d_uc_start = d_uc.x
    μ_start = μ.x
    γ_start = γ.x
    λ_start = λ.x
    πo_start = πo.x
    πc_start = πc.x
    πd_start = πd.x
    σ_start = σ.x
    σb_start = σb.x
    return φ_start , v_start , dd_start, do_start, dc_start, ωc_start, ωo_start, ωd_start, us_start, uc_s_start, uc_start, d_uc_start, μ_start , γ_start , λ_start , πo_start, πc_start, πd_start, σ_start, σb_start
 
lane_start = bike_lane_loc[used_allow_noexist] 
φ_start , v_start , dd_start, do_start, dc_start, ωc_start, ωo_start, ωd_start, us_start, uc_s_start, uc_start, d_uc_start, μ_start , γ_start , λ_start , πo_start, πc_start, πd_start, σ_start, σb_start = pre_solve(lane_start, 1.05)
#edges_start = pd.read_csv(r'D:\Box\Bike_lane\clean_code\UE-reduction_bi\selected_census\obj_bike_adoption\threshold\small_area20\bike_lane_correction\heuristic_results\greedy\RN10\warm_starts\tol1.05bud50_edges_.csv',header = None)[0]
#edges_start = pd.read_csv(r'D:\Box\Bike_lane\clean_code\UE-reduction_bi\selected_census\obj_bike_adoption\threshold\small_area20\bike_lane_correction\heuristic_results\greedy\RN14\RN14tol1.05B10_edges_.csv',header = None)[0]

################################################################
# start by .sol
################################################################ 
solutions = pd.read_csv("/Users/jz2293/Library/CloudStorage/Box-Box/Bike_lane/major/code/optimization/n500f8000/R15N15B10/thre0tol1.1/sol_16.sol")
lane_sol = solutions.loc[solutions.iloc[:,0].apply(lambda x: x[:5] == 'lane['),'# Solution for model estimate1'].apply(lambda x: x[-1]) 

#lane_start = bike_lane_loc.copy() 
#lane_start = lane_start[used_allow_noexist]
lane_start = np.transpose(list(lane_sol.apply(int)))
φ_start , v_start , dd_start, do_start, dc_start, ωc_start, ωo_start, ωd_start, us_start, uc_s_start, uc_start, d_uc_start, μ_start , γ_start , λ_start , πo_start, πc_start, πd_start, σ_start, σb_start = pre_solve(lane_start, 1.05)



#φ_start , v_start , dd_start, do_start, dc_start, ωc_start, ωo_start, ωd_start, us_start, uc_s_start, uc_start, d_uc_start, μ_start , γ_start , λ_start , πo_start, πc_start, πd_start, σ_start, σb_start = pre_solve(bike_lane_loc[used_allow_noexist], 1.05)

#lane_start = bike_lane_loc[used_allow_noexist]

'''
def retrive_start(path):
    lane_start  = np.loadtxt(path+"/lane_start.csv", delimiter=",")
    φ_start  = np.loadtxt(path+"/φ_start.csv", delimiter=",")
    v_start  = np.loadtxt(path+"/v_start.csv", delimiter=",")
    dd_start = np.loadtxt(path+"/dd_start.csv", delimiter=",")
    do_start = np.loadtxt(path+"/do_start.csv", delimiter=",")
    dc_start = np.loadtxt(path+"/dc_start.csv", delimiter=",")
    ωc_start = np.loadtxt(path+"/ωc_start.csv", delimiter=",")
    ωo_start = np.loadtxt(path+"/ωo_start.csv", delimiter=",")
    ωd_start = np.loadtxt(path+"/ωd_start.csv", delimiter=",")
    us_start = np.loadtxt(path+"/us_start.csv", delimiter=",")
    uc_s_start = np.loadtxt(path+"/uc_s_start.csv", delimiter=",")
    uc_start = np.loadtxt(path+"/uc_start.csv", delimiter=",")
    d_uc_start = np.loadtxt(path+"/d_uc_start.csv", delimiter=",")
    μ_start =  np.loadtxt(path+"/μ_start.csv", delimiter=",")
    γ_start =  np.loadtxt(path+"/γ_start.csv", delimiter=",")
    λ_start =  np.loadtxt(path+"/λ_start.csv", delimiter=",")
    πo_start = np.loadtxt(path+"/πo_start.csv", delimiter=",")
    πc_start = np.loadtxt(path+"/πc_start.csv", delimiter=",")
    πd_start = np.loadtxt(path+"/πd_start.csv", delimiter=",")
    σ_start =  np.loadtxt(path+"/σ_start.csv", delimiter=",")
    σb_start = np.loadtxt(path+"/σb_start.csv", delimiter=",")
    return lane_start, φ_start , v_start , dd_start, do_start, dc_start, ωc_start, ωo_start, ωd_start, us_start, uc_s_start, uc_start, d_uc_start, μ_start , γ_start , λ_start , πo_start, πc_start, πd_start, σ_start, σb_start

#path = "starts/" + "R"+str(R)+"N"+str(N)+","+str(round(lane_sol@distance[used_allow_noexist]/sum(distance[used_allow_noexist]),4))
#path = "starts/" + "R"+str(R)+"N"+str(N)+"tol"+str(tol)+","+str(round(lane_sol@distance[used_allow_noexist]/sum(distance[used_allow_noexist]),4))
path = r"D:\Box\Bike_lane\clean_code\UE-reduction_bi\selected_census\obj_bike_adoption\threshold\small_area20\bike_lane_correction\R10N10B10\thre0tol1.1\sol"
lane_start, φ_start , v_start , dd_start, do_start, dc_start, ωc_start, ωo_start, ωd_start, us_start, uc_s_start, uc_start, d_uc_start, μ_start , γ_start , λ_start , πo_start, πc_start, πd_start, σ_start, σb_start = retrive_start(path)

'''

#os.chdir(r'D:\Box\Bike_lane\major\code\optimization')



#######################################
# define emission
#################################
speeds = np.array([0.1]+[2.5+i*5 for i in range(15)]+[120])
speeds=speeds.T
es = np.array([1.781301904, 1.007831021, 0.621095826, 0.492184219, 0.425514679, 0.381057519, 0.341038929, 0.323903576, 0.313793007, 0.30626232, 0.300800763, 0.298487502, 0.299128484, 0.302748302, 0.313656613, 0.331026158])
es = es.T



##########################################################################################################
#optimization
##########################################################################################################
B,tol = 10,1.15 
for tol in [1.1, 1.15]:
    model1 = Model("estimate1")
    #demand
    φ  = model1.addMVar(Npath, name = 'phi')
    v  = model1.addMVar(Nedge, name = 'v')
    dd = model1.addMVar(Nod, name = 'dd')
    do = model1.addMVar(Nod, name = 'do')
    dc = model1.addMVar(Nod, name = 'dc')
    lane = model1.addMVar(len(used_allow_noexist), vtype = GRB.BINARY, name = 'lane')
    lane_phi = model1.addMVar(len(used_cod), vtype = GRB.BINARY, name = 'lane_phi')
    #linear approximation
    start = Npath+Nedge+3*Nod
    start1 = Npath+Nedge+3*Nod+len(used_allow_noexist)
    end = start1 + len(used_cod)

    ωc = model1.addMVar(Nod,lb=-GRB.INFINITY, name = 'wc')
    ωo = model1.addMVar(Nod,lb=-GRB.INFINITY, name = 'wo')
    ωd = model1.addMVar(Nod,lb=-GRB.INFINITY, name = 'wd')
    us = model1.addMVar(Nedge,lb=-GRB.INFINITY, name = 'us')
    #utility
    uc_s = model1.addMVar(Nedge,lb=-GRB.INFINITY, name = 'uc_s')
    uc = model1.addMVar(Nod,lb=-GRB.INFINITY, name = 'uc')
    d_uc = model1.addMVar(Nedge,lb=-GRB.INFINITY, name = 'd_uc')
    #duc_proportion = model1.addMVar(1,lb=-GRB.INFINITY)

    #dual
    μ  = model1.addMVar(Nod,lb=-GRB.INFINITY, name = 'mu')
    γ  = model1.addMVar(Nod,lb=-GRB.INFINITY, name = 'gamma')
    λ  = model1.addMVar(Nedge,lb=-GRB.INFINITY, name = 'lambda')
    πo = model1.addMVar((Nod, N), name = 'pio')
    πc = model1.addMVar((Nod, N), name = 'pic')
    πd = model1.addMVar((Nod, N), name = 'pic')
    σ  = model1.addMVar((Nedge, R), name = 'sigma')
    σb = model1.addMVar(Nedge,lb=-GRB.INFINITY, name = 'sigmab')

    dc_prop = model1.addMVar(Nedge, name = 'dc_prop')
    duc_prop = model1.addMVar(len(used_allow_noexist), name = 'duc_prop')
    σaf = model1.addMVar(len(used_allow_noexist), lb =-GRB.INFINITY, name = 'sigmaaf')
    σaf_lane = model1.addMVar(len(used_allow_noexist), lb =-GRB.INFINITY, name = 'sigmaaf_lane')
    σbf = model1.addMVar(len(used_allow_noexist), lb =-GRB.INFINITY, name = 'sigmabf')
    σbf_lane = model1.addMVar(len(used_allow_noexist), lb =-GRB.INFINITY, name = 'sigmabf_lane')
    cost_s = model1.addMVar(Nedge, name = 'cost_s')
    cost_p = model1.addMVar(Npath, name = 'cost_p')


    for s in range(len(used_allow_noexist)):
        ods = bike_ods[np.where(bike_edges==used_allow_noexist[s])]
        ods = np.array(list(set(used_cod).intersection(set(ods))))
        phi_vec = np.zeros(len(used_cod))
        phis = [np.where(used_cod == w) for w in ods]
        for phi in phis:
            phi_vec[phi] = 1
        #model1.addConstr(lane[s] <= sum(lane_phi[phi] for phi in phis))
        model1.addConstr(lane[s] <= lane_phi@phi_vec)
        #model1.addConstr(lane[s]*len(phis) >= sum(lane_phi[phi] for phi in phis))
        model1.addConstr(lane[s]*len(phis) >= lane_phi@phi_vec) 

    try:
        lane.start = lane_start#bike_lane_loc_start[used_allow_noexist]
        φ.start = φ_start
        v.start = v_start
        dd.start = dd_start
        do.start = do_start
        dc.start = dc_start
        ωc.start = ωc_start
        ωo.start = ωo_start
        ωd.start = ωd_start
        us.start = us_start
        uc_s.start = uc_s_start
        uc.start = uc_start
        d_uc.start = d_uc_start
        μ.start = μ_start
        γ.start = γ_start
        λ.start = λ_start
        πo.start = πo_start
        πc.start = πc_start
        πd.start = πd_start
        σ.start = σ_start
        σb.start = σb_start
    except:
        print('unable to start at tol', tol,'=============')
        pass

    bike_length = P_bike.T@distance
    length_coeff = P_bike@np.diag(1/bike_length)
    uc_const = uc_unchanged

    model1.addConstr(dc_prop == P_bike@np.diag(1/bike_length)@dc)

    M_lane = P_bike@d_od
    ones_lane = np.concatenate((notused_exist, used_exist))
    zeros_lane = np.concatenate((notused_noexist, used_notallow))

    model1.addConstr(uc_s[ones_lane] == distance[ones_lane]*bl_coef)
    model1.addConstr(d_uc[ones_lane] == np.diag(distance[ones_lane])@dc_prop[ones_lane])
    model1.addConstr(uc_s[zeros_lane] == 0)
    model1.addConstr(d_uc[zeros_lane] == 0)
    model1.addConstr(uc_s[used_allow_noexist] == np.diag(distance[used_allow_noexist]*bl_coef) @ lane)

    model1.addConstr(duc_prop == np.diag(distance[used_allow_noexist])@dc_prop[used_allow_noexist])
    for s_ind in range(len(used_allow_noexist)):
        s = used_allow_noexist[s_ind]
        model1.addConstr(d_uc[s] == duc_prop[s_ind]*lane[s_ind])
    #for s_ind in range(len(used_allow_noexist)):
    #    s = used_allow_noexist[s_ind]
    #    model1.addConstr(uc_s[s] == lane[s_ind]*distance[s]*c_coef[1])
    #    ods = sparse.find(P_bike[s,:])[1]
    #    model1.addConstr(d_uc[s] >= distance[s]*sum(dc[w]/bike_length[w] for w in ods) - (1-lane[s_ind])*M_lane[s])
    #    model1.addConstr(d_uc[s] <= distance[s]*sum(dc[w]/bike_length[w] for w in ods))
    #    model1.addConstr(d_uc[s] >= 0)
    #    model1.addConstr(d_uc[s] <= lane[s_ind]*M_lane[s])



    model1.addConstr(uc == uc_const + length_coeff.T@uc_s)

    model1.addConstr(do + dc + dd == d_od)
    model1.addConstr(U @ φ == dd)
    model1.addConstr(P @ φ == v)


    for i in range(N):
        model1.addConstr(ωc >= np.diag(a_ff[i,:])@dc+b_ff[i,:])
        model1.addConstr(ωo >= np.diag(a_ff[i,:])@do+b_ff[i,:])
        model1.addConstr(ωd >= np.diag(a_ff[i,:])@dd+b_ff[i,:])
    for i in range(R):
        #notused_exist & used_exist
        model1.addConstr(us[ones_lane] >= np.diag(a_f_lane[i,ones_lane])@v[ones_lane] + b_f_lane[i,ones_lane])

        #notused_noexist & used_notallow
        model1.addConstr(us[zeros_lane] >= np.diag(a_f[i,zeros_lane])@v[zeros_lane] + b_f[i,zeros_lane])

        #used_allow_noexist
        inds = range(len(used_allow_noexist))
        model1.addConstr(us[used_allow_noexist] >= np.diag(a_f[i,used_allow_noexist])@v[used_allow_noexist] + b_f[i,used_allow_noexist] - np.diag(M_x[used_allow_noexist,i])@lane)
        model1.addConstr(us[used_allow_noexist] >= np.diag(a_f_lane[i,used_allow_noexist])@v[used_allow_noexist] + b_f_lane[i,used_allow_noexist] - (M_x_lane[used_allow_noexist,i]-np.diag(M_x_lane[used_allow_noexist,i])@lane))


    for w in range(Nod):
        model1.addConstr(μ[w] - πo[w,:]@a_ff[:,w] <= uo[w])
        model1.addConstr(μ[w] - πc[w,:]@a_ff[:,w] <= uc[w])
        model1.addConstr(μ[w] + γ[w] - πd[w,:]@a_ff[:,w] <= ud_unchanged[w])
    model1.addConstr(- U.T@γ - P.T@λ <= c0*dt_coef)


    a_f_diff = np.multiply([sum(a_f_lane[:,s]) for s in range(Nedge)],dt_coef)
    for s in notused_exist:
        model1.addConstr(λ[s] <= σ[s,:]@a_f_lane[:,s])
        model1.addConstr(sum(σ[s,:]) == dt_coef)
    for s in used_exist:
        model1.addConstr(λ[s] <= σ[s,:]@a_f_lane[:,s])
        model1.addConstr(sum(σ[s,:]) == dt_coef)
    for s in notused_noexist:
        model1.addConstr(λ[s] <= σ[s,:]@a_f[:,s])
        model1.addConstr(sum(σ[s,:]) == dt_coef)
    for s in used_notallow:
        model1.addConstr(λ[s] <= σ[s,:]@a_f[:,s])
        model1.addConstr(sum(σ[s,:]) == dt_coef)

    for s_ind in range(len(used_allow_noexist)):
        s = used_allow_noexist[s_ind]
        model1.addConstr(σaf[s_ind] == σ[s,:]@a_f[:,s])
        model1.addConstr(σaf_lane[s_ind] == σ[s,:]@a_f_lane[:,s])
        #model1.addConstr(λ[s] <= σ[s,:]@a_f[:,s] +  a_f_diff[s]*lane[s_ind])
        #model1.addConstr(λ[s] <= σ[s,:]@a_f_lane[:,s])
        model1.addConstr(λ[s] <= σaf[s_ind]*(1-lane[s_ind]) + σaf_lane[s_ind]*lane[s_ind])
        model1.addConstr(sum(σ[s,:]) == dt_coef)

    for w in range(Nod):
        model1.addConstr(πo[w,:]@np.ones(N) == 1)
        model1.addConstr(πc[w,:]@np.ones(N) == 1)
        model1.addConstr(πd[w,:]@np.ones(N) == 1)



    b_f_diff = np.multiply([(-sum(b_f_lane[:,s])) for s in range(Nedge)],dt_coef)
    for s in notused_exist:
        model1.addConstr(σb[s] == σ[s,:]@b_f_lane[:,s])
    for s in used_exist:
        model1.addConstr(σb[s] == σ[s,:]@b_f_lane[:,s])
    for s in notused_noexist:
        model1.addConstr(σb[s] == σ[s,:]@b_f[:,s])
    for s in used_notallow:
        model1.addConstr(σb[s] == σ[s,:]@b_f[:,s])
    for s_ind in range(len(used_allow_noexist)):
        s = used_allow_noexist[s_ind]
        model1.addConstr(σbf[s_ind] == σ[s,:]@b_f[:,s])
        model1.addConstr(σbf_lane[s_ind] == σ[s,:]@b_f_lane[:,s])
        model1.addConstr(σb[s] == σbf[s_ind]*(1-lane[s_ind]) + σbf_lane[s_ind]*lane[s_ind])
        #model1.addConstr(σb[s] >= σ[s,:]@b_f[:,s] - b_f_diff[s]*lane[s_ind])
        #model1.addConstr(σb[s] <= σ[s,:]@b_f[:,s])
        #model1.addConstr(σb[s] >= σ[s,:]@b_f_lane[:,s])
        #model1.addConstr(σb[s] <= σ[s,:]@b_f_lane[:,s] +  b_f_diff[s]*(1-lane[s_ind]))


    model1.addConstr(do@uo + dc@uc_const + d_uc@np.ones(Nedge)*bl_coef + dd@ud_unchanged + us@np.ones(Nedge)*dt_coef + c0@φ*dt_coef + ωc@np.ones(Nod) + ωo@np.ones(Nod) + ωd@np.ones(Nod) <= (μ@d_od + sum(πo[:,n]@b_ff[n,:] + πc[:,n]@b_ff[n,:] + πd[:,n]@b_ff[n,:] for n in range(N)) + σb@np.ones(Nedge))+1e-06)


    ###########################
    #add time constraint
    ###########################
    
    v_origin = demands['v_opt']
    ws_l = np.array(weight_cost_w_lane)
    ws_n = np.array(weight_cost_wo_lane)
    ws = np.diag(np.multiply(weight_cost_w_lane, bike_lane_loc) + np.multiply(weight_cost_wo_lane, 1-bike_lane_loc))
    #w0l = np.diag(ws_l)@v0
    #w0n = np.diag(ws_n)@v0
    w_diff = np.multiply(ws_l, dub)# + w0l
    φs_total = c0 + P.T@ws@(v_origin + v0)
    #notused_exist & used_exist
    model1.addConstr(cost_s[ones_lane] == np.diag(ws_l[ones_lane])@v[ones_lane])
    #notused_noexist & used_notallow
    model1.addConstr(cost_s[zeros_lane] == np.diag(ws_n[zeros_lane])@v[zeros_lane])
    #used_allow_noexist
    ws_n_var = np.diag(ws_n[used_allow_noexist])
    w_diff_var = np.diag(w_diff[used_allow_noexist])
    ws_l_var = np.diag(ws_l[used_allow_noexist])
    model1.addConstr(cost_s[used_allow_noexist] >= ws_n_var@v[used_allow_noexist])
    model1.addConstr(cost_s[used_allow_noexist] <= ws_n_var@v[used_allow_noexist] + w_diff_var@lane)
    model1.addConstr(cost_s[used_allow_noexist] <= ws_l_var@v[used_allow_noexist])
    model1.addConstr(cost_s[used_allow_noexist] >= ws_l_var@v[used_allow_noexist] - (np.diag(w_diff_var)-w_diff_var@lane))
    model1.addConstr(cost_p == c0 + P.T@cost_s)
    #lazy = model1.addConstr(cost_p <= tol*φs_total)
    #for i in range(Npath):
    #    lazy[i].Lazy = 1 
    
    ###########################
    #add emission constraint
    ###########################
    #emission binary

    
    #model1 = Model("estimate1")

    z = model1.addMVar((Npath, K), vtype = GRB.BINARY, name = 'z') 
    emission = model1.addMVar(Npath, name = 'path_emission')

    pathdistance = P.T@distance/1609.34 #pathdistance in miles
    pathdistance_diag = np.diag(pathdistance)
    pathdistance = pathdistance.T
    time_bucket = np.zeros((Npath,K+1))
    M = max(pathdistance)/min(speeds)
    for k in range(K+1):
        time_bucket[:,k] = pathdistance/speeds[k]  #time_bucket in hours

    for k in range(K):
        model1.addConstr(np.diag(time_bucket[:,k+1])@z[:,k]*60 <= cost_p)
        model1.addConstr(np.diag(time_bucket[:,k])@z[:,k] + M*(np.ones(Npath)-z[:,k]) >= cost_p/60)

    
    model1.addConstr(z@np.ones(K)==np.ones(Npath))
 

    #model1.setObjective(sum(dd)*d_coef[0] + c0@φ*d_coef[1] + 2*sum(us)*d_coef[1] + dc@uc_const + sum(d_uc)*c_coef[1] + do@uo)

    #model1.addConstr(emission == z@es)
    #model1.setObjective(emission @ pathdistance_diag @ φ) 

    model1.setObjective(sum(φ[p]*pathdistance[p]*sum(z[p,k]*es[k] for k in range(K)) for p in range(Npath))) 



    #model1.setObjective(sum(dd)*d_coef[0] + c0@φ*d_coef[1] + 2*sum(us)*d_coef[1] + dc@uc_const + sum(d_uc)*c_coef[1] + do@uo)

    #model1.setObjective(dc@np.ones(Nod))
    #model1.setObjective(0)

    result = pd.DataFrame(columns=['MIPGap','Obj','ObjBound','Runtime'])
    all_dist = sum(distance[used_allow_noexist])

    model1.params.MIPGap=0.05
    #model1.params.NonConvex = 2

    model1.params.MIPFocus = 1
    #model1.params.FeasibilityTol = 1e-06
    #model1.params.DegenMoves = -1
    model1.modelSense = GRB.MINIMIZE
    #model1.update()
    #model0 = model1.copy()
 

    model1.addConstr(distance[used_allow_noexist]@lane <= B*1609.34)
    model1.Params.lazyConstraints = 1
  
    model1.params.TimeLimit = 3600

    model1.update()
    model1.optimize()






















    result.loc[tol] = model1.MIPGap, model1.getObjective().getValue(), model1.ObjBound, model1.Runtime# + result.loc[B,'Runtime']

    result.to_csv("R"+str(R)+"N"+str(N)+"B"+str(B)+"/tol"+str(tol) + "result.csv")

    path = sol_loc
    if not os.path.exists(path):
        os.makedirs(path)
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)
    try:
        print("++++++++++++++++++++++++++++++,max_time", max((cost_p.x -φs_total)/φs_total),"+++++++++++++++++++++++++++")
    except:
        pass
    try:
        lane_start = lane.x
        φ_start  = φ.x
        v_start  = v.x
        dd_start = dd.x
        do_start = do.x
        dc_start = dc.x
        ωc_start = ωc.x
        ωo_start = ωo.x
        ωd_start = ωd.x
        us_start = us.x
        uc_s_start = uc_s.x
        uc_start = uc.x
        d_uc_start = d_uc.x
        μ_start = μ.x
        γ_start = γ.x
        λ_start = λ.x
        πo_start = πo.x
        πc_start = πc.x
        πd_start = πd.x
        σ_start = σ.x
        σb_start = σb.x

        np.savetxt(path+"/lane_start.csv", lane.x, delimiter=",")
        np.savetxt(path+"/φ_start.csv", φ.x, delimiter=",")
        np.savetxt(path+"/v_start.csv", v.x, delimiter=",")
        np.savetxt(path+"/dd_start.csv", dd.x, delimiter=",")
        np.savetxt(path+"/do_start.csv", do.x, delimiter=",")
        np.savetxt(path+"/dc_start.csv", dc.x, delimiter=",")
        np.savetxt(path+"/ωc_start.csv", ωc.x, delimiter=",")
        np.savetxt(path+"/ωo_start.csv", ωo.x, delimiter=",")
        np.savetxt(path+"/ωd_start.csv", ωd.x, delimiter=",")
        np.savetxt(path+"/us_start.csv", us.x, delimiter=",")
        np.savetxt(path+"/uc_s_start.csv", uc_s.x, delimiter=",")
        np.savetxt(path+"/uc_start.csv", uc.x, delimiter=",")
        np.savetxt(path+"/d_uc_start.csv", d_uc.x, delimiter=",")
        np.savetxt(path+"/μ_start.csv", μ.x, delimiter=",")
        np.savetxt(path+"/γ_start.csv", γ.x, delimiter=",")
        np.savetxt(path+"/λ_start.csv", λ.x, delimiter=",")
        np.savetxt(path+"/πo_start.csv", πo.x, delimiter=",")
        np.savetxt(path+"/πc_start.csv", πc.x, delimiter=",")
        np.savetxt(path+"/πd_start.csv", πd.x, delimiter=",")
        np.savetxt(path+"/σ_start.csv", σ.x, delimiter=",")
        np.savetxt(path+"/σb_start.csv", σb.x, delimiter=",")
    except:
        print('unable to save at tol', tol,'=============')
        break



path = r"D:\Box\Bike_lane\clean_code\UE-reduction_bi\selected_census\obj_bike_adoption\threshold\small_area20\bike_lane_correction\R5N5B10\thre0tol1.05\sol\lane_start.csv"
lanes_start = pd.read_csv(path, header = None)
lane_start = lanes_start[0].apply(int).tolist()

def get_arcgis(lanes_start):
    selected_lanes = edge_ref.iloc[used_allow_noexist[[i for i in range(len(used_allow_noexist)) if lane_start[i]>0]]].index
    return np.array(all_edge_bi.loc[all_edge_bi.bi_ind.isin(selected_lanes)].index)

get_arcgis(lanes_start)

len(all_edge_bi.loc[all_edge_bi.bi_ind.isin(edges_used)].index)
sum(edges_used)
sum(all_edge_bi.lane)

#    #sanity check
#    #----________________________________________________________________
#    model1.setObjective(0)
#    model1.update()
#    model1.optimize()
#    φ = demands['φ_opt']
#    (z.x@es) @ pathdistance_diag @ φ   
#
#    def em(avg_speed):
        if  avg_speed < 2.5:
            return 1.781301904
        elif 2.5 <= avg_speed < 7.5:
            return 1.007831021
        elif 7.5 <= avg_speed < 12.5:
            return 0.621095826
        elif 12.5 <= avg_speed < 17.5:
            return 0.492184219
        elif 17.5 <= avg_speed < 22.5:
            return 0.425514679
        elif 22.5 <= avg_speed < 27.5:
            return 0.381057519
        elif 27.5 <= avg_speed < 32.5:
            return 0.341038929
        elif 32.5 <= avg_speed < 37.5:
            return 0.323903576
        elif 37.5 <= avg_speed < 42.5:
            return 0.313793007
        elif 42.5 <= avg_speed < 47.5:
            return 0.30626232
        elif 47.5 <= avg_speed < 52.5:
            return 0.300800763
        elif 52.5 <= avg_speed < 57.5:
            return 0.298487502
        elif 57.5 <= avg_speed < 62.5:
            return 0.299128484
        elif 62.5 <= avg_speed < 67.5:
            return 0.302748302
        elif 67.5 <= avg_speed < 72.5:
            return 0.313656613
        else: 
            return 0.331026158
#    #E = sum(length_path * phi_path * em(length_path/time_phi_optimal))
#    length_path = P.T@distance/1609.34
#
#    def total_emission_fun(time_phi_optimal, φ_optimal):
        speed = P.T@distance/(time_phi_optimal*60)*2.23694 #Npath size vector, speed in mph
        #print(speed)
        total_emission = 0
        for i in range(Npath):
            # emission of optimal solution 
            total_emission += length_path[i] * φ_optimal[i] * em(length_path[i]/(time_phi_optimal[i]/60))
        return total_emission
#    total_emission_fun(time, φ)
#    #----________________________________________________________________

