import dill
from importlib import import_module
import pandas as pd

dill.load_session(r'D:\bike_lane_emission\code\workspaces\R5N5opt_win.pkl')

#R5N5 = import_module('3-R5N5.UE')
 

#os.chdir('/Users/jingweizhang/Box/Bike_lane/clean_code/UE-reduction_bi/selected_census/obj_bike_adoption/threshold/small_area20/bike_lane_correction')
os.chdir(r'D:\bike_lane_emission\code\optimization_results')

cdemands = pd.DataFrame(index = ['d0','dfull'])
#for near_threshold in [500,750,1000,1250, 1500,1750]:
    #for far_threshold in [6000,6500, 7000, 7500, 8000,8500,9000]:
        #for near_threshold in [500]:
        #    for far_threshold in [8000]:
name = "n"+str(500)+'f' + str(8000) 
print(name)
xc = pd.read_csv(r"D:\bike_lane_emission\code\estimation_results\xc.csv",dtype={"from":str,"to":str})
xc = xc.set_index(['from','to'])
xd = pd.read_csv(r"D:\bike_lane_emission\code\estimation_results\xd.csv",dtype={"from":str,"to":str})
xd = xd.set_index(['from','to'])
valid_estimate = pd.read_csv(r"D:\bike_lane_emission\code\estimation_results\valid_estimate"+name+".csv",index_col = 0)
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


######################### linear approximation ###############################
#d_od = od_d['acs_S000']
d_od = od_d['S000']
d_od = np.transpose(d_od.to_list())
N = 150
R = 150
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
    model.params.OutputFlag = 0
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
    return demand, c0@φ.x + 2*sum(us.x), model.getObjective().getValue()
demands, drive_time, obj = UE(bike_lane_loc)
import matplotlib.pyplot as plt
plt.hist(demands['v_opt'])
sum(demands['v_opt'])
 

######################################################
# Optimization
######################################################

from datetime import datetime
from datetime import date
 
Bs = [10,25,50,75]
tols = [1.05, 1.1, 1.15]
result = pd.DataFrame(index=Bs,columns=['MIPGap','Obj','ObjBound','Runtime'])
prev = 0
bike_lane_loc_start = bike_lane_loc.copy()
bike_phi_loc_start = np.zeros(Nod)

threshold = prev



od_d = od_d.set_index([od_d['from'],od_d['to']])
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
#bike_lane_loc_full[notallow] = 1


demands_ub, drive_time_ub, obj_ub = UE(bike_lane_loc_full)
dc_ub = sum(demands_ub['dc_opt'])
print(dc_ub/sum(d_od))

weights = np.multiply(weight_cost_w_lane, bike_lane_loc) + np.multiply(weight_cost_wo_lane, 1-bike_lane_loc)
weights1 = np.multiply(weight_cost_w_lane, bike_lane_loc_full) + np.multiply(weight_cost_wo_lane, 1-bike_lane_loc_full)

time = P.T@(np.multiply(weights,(demands['v_opt']+v0))) + c0
time1 = P.T@(np.multiply(weights1,(demands_ub['v_opt']+v0))) + c0
time_od = np.zeros(Nod)
time_od1 = np.zeros(Nod)
time_od_empty = np.zeros(Nod)
c0s = []
c0s1 = []
for i in range(Nod):
    temp = time[sparse.find(U[i,:])[1]]
    time_od[i] = min(temp)
    c0s.append(c0[(sparse.find(U[i,:])[1])[list(temp).index(min(temp))]])
    temp1 = time1[sparse.find(U[i,:])[1]]
    time_od1[i] = min(temp1)
    c0s1.append(c0[(sparse.find(U[i,:])[1])[list(temp1).index(min(temp1))]])
    time_od_empty[i] = min(c0[sparse.find(U[i,:])[1]])
plt.hist((time_od1-time_od)/time_od)
plt.hist((time1-time)/time)
print("+++++++++++++++++++++++++++++ max time",max((time1-time)/time),'+++++++++++++++++++++++++++++')
print("+++++++++++++++++++++++++++++ max od time",max((time_od1-time_od)/time_od),'+++++++++++++++++++++++++++++')



################################################################
# compare by R=N=150
################################################################ 
path_base = 'Julia_optimal'
def get_data(path=path_base+'/build_0'):
    dd = pd.read_csv(path+'/dd_opt.csv',header=None)[0]
    dc = pd.read_csv(path+'/dc_opt.csv',header=None)[0]
    do = pd.read_csv(path+'/do_opt.csv',header=None)[0]
    time_phi = pd.read_csv(path+'/t_phi.csv',header=None)[0]
    v = pd.read_csv(path+'/v_opt.csv',header=None)[0]
    φ = pd.read_csv(path+'/φ_opt.csv',header=None)[0]
    time_od = np.zeros(Nod)
    for i in range(Nod):
        temp = time_phi[sparse.find(U[i,:])[1]]
        time_od[i] = min(temp)
    return dd, dc, do, time_phi, v, φ, time_od
#origin 
dd_0, dc_0, do_0, time_phi_0, v_0, φ_0, time_od_0 = get_data(path_base+'/build_0')
#everywhere 
dd_1, dc_1, do_1, time_phi_1, v_1, φ_1, time_od_1 = get_data(path_base+'/build_all')


import itertools
indices = [i for i in itertools.product(Bs, tols)]

sort_od = pd.DataFrame(index=Bs, columns = ['bike_demand','drive_demand','weighted_time','w_avg_time','max_time','max_time_phi', 'rel_total'])
sort_bike = pd.DataFrame(index=Bs, columns = ['bike_demand','drive_demand','weighted_time','w_avg_time','max_time','max_time_phi', 'rel_total'])
fixed = pd.DataFrame(index=Bs, columns = ['bike_demand','drive_demand','weighted_time','w_avg_time','max_time','max_time_phi', 'rel_total'])
optimal = pd.DataFrame(index=indices, columns = ['bike_demand','drive_demand','weighted_time','w_avg_time','max_time','max_time_phi', 'rel_total'])
greedy = pd.DataFrame(index=indices, columns = ['bike_demand','drive_demand','weighted_time','w_avg_time','max_time','max_time_phi', 'rel_total'])

def compareR15N15(path, option = 1):        
    bike_lane_loc_start = bike_lane_loc.copy()
    if option == 'full':
        bike_lane_loc_start[used_allow_noexist] = 1
    elif option == 'empty':
        bike_lane_loc_start[used_allow_noexist] = 0
    else:        
        lanes_start = pd.read_csv(path, header = None)
        lane_start = lanes_start[0].apply(int).tolist()
        bike_lane_loc_start[used_allow_noexist] = lane_start

    demands_ub, drive_time_ub, obj_ub = UE(bike_lane_loc_start)
    dc_ub = sum(demands_ub['dc_opt'])
    print(dc_ub)

    weights1 = np.multiply(weight_cost_w_lane, bike_lane_loc_start) + np.multiply(weight_cost_wo_lane, 1-bike_lane_loc_start)

    time1 = P.T@(np.multiply(weights1,(demands_ub['v_opt']+v0))) + c0
    time_od1 = np.zeros(Nod)
    time_od_empty = np.zeros(Nod)
    for i in range(Nod):
        temp1 = time1[sparse.find(U[i,:])[1]]
        time_od1[i] = min(temp1)
    return demands_ub['dd_opt'], demands_ub['dc_opt'], demands_ub['do_opt'], time1, demands_ub['v_opt'], demands_ub['φ_opt'], time_od1

#origin #no bike lane at all
dd_origin, dc_origin, do_origin, time_phi_origin, v_origin, φ_origin, time_od_origin = compareR15N15(path_base, 'empty')
#everywhere
dd_full, dc_full, do_full, time_phi_full, v_full, φ_full, time_od_full = compareR15N15(path_base, 'full')

 

### total emission ###
def em(avg_speed):
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

#E = sum(length_path * phi_path * em(length_path/time_phi_optimal))
length_path = P.T@distance
def total_emission_fun(time_phi_optimal, φ_optimal):
    speed = P.T@distance/(time_phi_optimal*60)*2.23694 #Npath size vector, speed in mph
    #print(speed)
    total_emission = 0
    for i in range(Npath):
        # emission of optimal solution 
        total_emission += length_path[i]/1609.34 * φ_optimal[i] * em(length_path[i]/time_phi_optimal[i])
    return total_emission

Bs = [10, 25, 50, 75]
tols = [1.05, 1.1, 1.15]

emission = pd.DataFrame(columns = ['emission'])
emission.loc['origin','emission'] = total_emission_fun(time_phi_origin, φ_origin)
emission.loc['all','emission'] = total_emission_fun(time_phi_full, φ_full)

for B in Bs:
    #od_demand sort
    dd_god, dc_god, do_god, time_phi_god, v_god, φ_god, time_od_god = compareR15N15('heuristic_results/od_demand_sort/tol1.15B'+str(B)+"_lane_start.csv")
    emission.loc['od_sort'+str(B),'emission'] = total_emission_fun(time_phi_god, φ_god)
    #bike_demand sort
    dd_bike, dc_bike, do_bike, time_phi_bike, v_bike, φ_bike, time_od_bike = compareR15N15('heuristic_results/bike_demand_sort/tol1.15B'+str(B)+"_lane_start.csv") 
    emission.loc['bike_sort'+str(B),'emission'] = total_emission_fun(time_phi_bike, φ_bike)
    
    #fixed_time
    dd_fixed, dc_fixed, do_fixed, time_phi_fixed, v_fixed, φ_fixed, time_od_fixed = compareR15N15('heuristic_results/fixed_time/R15N15B'+str(B)+"/sol/lane_start.csv") 
    emission.loc['fixed'+str(B),'emission'] = total_emission_fun(time_phi_fixed, φ_fixed)
     


for B in Bs:
    for tol in tols:
        #greedy
        dd_greedy, dc_greedy, do_greedy, time_phi_greedy, v_greedy, φ_greedy, time_od_greedy = compareR15N15('heuristic_results/greedy/RN15/RN15tol'+str(tol)+"B"+str(B)+"_lane_start.csv") 
        emission.loc['greedy'+str(B)+'_'+str(tol),'emission'] = total_emission_fun(time_phi_greedy, φ_greedy)
 

for B in Bs:
    for tol in tols:
        #optimal
        dd_optimal, dc_optimal, do_optimal, time_phi_optimal, v_optimal, φ_optimal, time_od_optimal = compareR15N15('n500f8000/R15N15B'+str(B)+'/thre0tol'+str(tol)+"/sol/lane_start.csv") 
        do_optimal = d_od - dc_optimal - dd_optimal
        emission.loc['optimal'+str(B)+'_'+str(tol),'emission'] = total_emission_fun(time_phi_optimal, φ_optimal)

        #optimal.loc[[(B,tol)],['bike_demand','drive_demand','weighted_time','w_avg_time','max_time','max_time_phi', 'rel_total']] = sum(dc_optimal), sum(dd_optimal), sum(dd_optimal*time_od_optimal), sum(dd_optimal*(time_od_optimal-time_od_origin)/time_od_origin)/sum(dd_optimal), max((time_od_optimal-time_od_origin)/time_od_origin), max((time_phi_optimal-time_phi_origin)/time_phi_origin), (dd_optimal@time_od_optimal -dd_origin@time_od_origin)/(dd_origin@time_od_origin)

emission.to_csv(r"D:\bike_lane_emission\code\emission.csv")
total_emission_fun(time_phi_optimal, φ_optimal)
#benchmark: no bike lane 
#use this time_phi_origin and φ_origin
total_emission_fun(time_phi_origin,φ_origin)


print(sum(time_od_optimal<time_od_origin)/len(time_od_origin))
print(sum(time_od_optimal>time_od_origin)/len(time_od_origin))
np.mean((time_od_optimal-time_od_origin)/time_od_origin)
np.mean((time_od_optimal-time_od_origin)/time_od_origin)
np.min((time_od_optimal-time_od_origin)/time_od_origin)
'''
import seaborn as sns

def label_point(x, y, val, ax):
        a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
        for i, point in a.iterrows():
            ax.text(point['x']+5, point['y']+10, str(point['val']))


optimal['specification'] = list(optimal.index)
optimal['B'] = optimal['specification'].apply(lambda x:x[0])
optimal['tol'] = optimal['specification'].apply(lambda x:x[1])
greedy['specification'] = list(greedy.index)
optimal['type'] = 'Optimal'
greedy['type'] = 'Greedy'
greedy['B'] = greedy['specification'].apply(lambda x:x[0])
greedy['tol'] = greedy['specification'].apply(lambda x:x[1])

sort_od['specification'] = list(sort_od.index)
sort_bike['specification'] = list(sort_bike.index)
fixed['specification'] = list(fixed.index)
sort_od['B'] = list(sort_od.index)
sort_bike['B'] = list(sort_bike.index)
fixed['B'] = list(fixed.index)
sort_od['type'] = 'GOD'
sort_bike['type'] = 'Gbike'
fixed['type'] = 'Fixed-time'



#results = pd.concat([optimal, greedy, sort_od, sort_bike, fixed], axis=0).reindex()
results = pd.concat([optimal, greedy, fixed, sort_od, sort_bike], axis=0).reindex()
results['Bike Share'] = results.bike_demand/sum(d_od)
results['Relative increase in bike share'] = (results.bike_demand-sum(dc_0))/sum(dc_0)
results['Drive Share'] = results.drive_demand/sum(d_od)




################################################################
# Pareto front
################################################################
import pandas as pd
import seaborn as sns
sns.set_palette(sns.color_palette())
import matplotlib.pyplot as plt
from  matplotlib.ticker import PercentFormatter
from matplotlib.legend import Legend
import matplotlib.font_manager as font_manager
#sns.set_style("whitegrid",{'font.family':'serif', 'font_scale':3, 'font.serif':'Times New Roman'})

font_path = font_manager.findfont(font_manager.FontProperties(family='Times New Roman'))

# Set the font properties for the plot
font = {'family': 'serif',
        'serif': ['Times New Roman'],
        'weight': 'normal',
        'size': 16}

# Set the font properties in matplotlib rcParams
plt.rcParams.update({'font.family': font['family'],
                     'font.serif': font['serif'],
                     'font.weight': font['weight'],
                     'font.size': font['size']})

#sns.set(style="whitegrid", font_scale=1.5)

base = sum(dc_0)/sum(d_od)
results['Increase in bike share'] = results['Bike Share'] - base
results['Resulting bike share'] = results['Increase in bike share'] + 0.036
#results = results.loc[results['B']<70]
results.loc[results.type == 'GOD','type'] = 'Demand-heuristic'
#results.loc[results.type == 'Gbike','type'] = 'G-Bike'
results = results.loc[results.type != 'Gbike']
try:
    results['w_avg_time'] = results['Weighted average relative increase in driving time']
    results['max_time'] = results['Worst relative increase in driving time']
except:
    pass


#results.to_csv("4-compare.csv")

results = pd.read_csv(r"4-compare.csv",index_col = 0)

def change_tol(x):
    try:
        y = int((x-1)*100)
        if y == 14:
            y = 15
        return str(y)+"%"
    except:
        pass
results.tol = results.tol.apply(change_tol)
results.loc[(results.tol).isna(),'tol'] = ''
results.loc[results.type == 'Optimal','type'] = 'BLPS-A'



def label_point(x, y, val, ax):
        a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
        diff_y = (max(y)-min(y))/20*0
        diff_x = (max(x)-min(x))/30
        for i, point in a.iterrows():
            ax.text(point['x']+diff_x, point['y']+diff_y, str((point['val'])), fontsize= 14)


def label_point_3(x, y, val, ax, B):
        a = pd.concat({'x': x, 'y': y, 'val': val, 'B':B}, axis=1)
        diff_y = (max(y)-min(y))/20*0
        diff_x = (max(x)-min(x))/30
        for i, point in a.iterrows():
            if (point['B'] == 'B = 25' and point['val'] == '10%') or (point['B'] == 'B = 10' and point['val'] == '10%') or (point['B'] == 'B = 50' and point['val'] == '5%') or (point['B'] == 'B = 50' and point['val'] == '15%'):
                ax.text(point['x']-diff_x*3, point['y']-(max(y)-min(y))/40, str((point['val'])), fontsize= 14)
            elif (point['B'] == 'B = 75' and point['val'] == '10%'):
                ax.text(point['x'], point['y']-(max(y)-min(y))/20, str((point['val'])), fontsize= 14)
            else:                
                ax.text(point['x']+diff_x, point['y']+diff_y, str((point['val'])), fontsize= 14)


def label_point_2(x, y, val, ax, B):
        a = pd.concat({'x': x, 'y': y, 'val': val, 'B':B}, axis=1)
        diff_y = (max(y)-min(y))/20*0
        diff_x = (max(x)-min(x))/30
        for i, point in a.iterrows():
            if (point['B'] == 'B = 25' and point['val'] == '10%') or point['B'] == 'B = 50':
                ax.text(point['x']-diff_x*3, point['y']+diff_y, str((point['val'])), fontsize= 14)
            else:                
                ax.text(point['x']+diff_x, point['y']+diff_y, str((point['val'])), fontsize= 14)


results['Algorithm'] = results['type']
results['τ'] = results['tol']
 
#########################################################################################
#start from here

def label_point_1(x, y, val, ax, B):
    a = pd.concat({'x': x, 'y': y, 'val': val, 'B':B}, axis=1)
    diff_y = (max(y)-min(y))/20
    diff_x = (max(x)-min(x))/30
    for i, point in a.iterrows():
        if (point['B'] == 'B = 10' and point['val'] == '5%'):
            ax.text(point['x'], point['y']+diff_y*2, str((point['val'])), fontsize= 14)
        elif (point['B'] == 'B = 25' and point['val'] == '10%') or (point['B'] == 'B = 50' and point['val'] == '10%'):
            ax.text(point['x']-diff_x*3, point['y'], str((point['val'])), fontsize= 14)
        elif (point['B'] == 'B = 50' and point['val'] == '15%'):
            ax.text(point['x'], point['y']-diff_y*1.5, str((point['val'])), fontsize= 14)
        else:                
            ax.text(point['x']+diff_x, point['y'], str((point['val'])), fontsize= 14)
data=results.loc[results['type']=='BLPS-A']
data['B'] = data['B'].apply(lambda x: 'B = '+str(x))
g1 = sns.scatterplot(data=data, x="Increase in bike share", y="max_time_phi", hue = 'B', style = 'B',palette=sns.color_palette()[:4], s=80)
#plt.title('Weighted average driving time vs Bike Share')
plt.ylabel('Worst-case increase in driving time')
plt.xlabel('Increase in cycling share')
g1.xaxis.set_major_formatter(PercentFormatter(1))
g1.yaxis.set_major_formatter(PercentFormatter(1)) 
#label_point_1(data['Increase in bike share'], data['max_time_phi'], data['tol'], plt.gca(), data['B'])
plt.xlim([0.01, max(data["Increase in bike share"]+0.005)])
#plt.legend(loc='upper left', bbox_to_anchor=(1.04,1))
plt.legend(loc='lower right')
plt.grid(linewidth=0.3)
plt.savefig(r"D:\Box\Bike_lane\major\code\Figures\optimal2_nolabel.pdf", bbox_inches='tight')
plt.show()
 


def label_point_right(x, y, val, ax, B, types):
        a = pd.concat({'x': x, 'y': y, 'val': val, 'B':B, 'type': types}, axis=1)
        diff_y = 0
        diff_x = (max(x)-min(x))/30
        for i, point in a.iterrows():
            ax.text(point['x']+diff_x, point['y'], str((point['val'])), fontsize= 14)
            
for B in [10,25,50,75]:
    data=results.loc[results['B']==B]    
    #if B == 50:
    #    data = data.loc[~((data.type == 'Greedy') & (data.tol == '10%'))]
    #    data.loc[((data.type == 'Greedy') & (data.tol == '5%')),'tol'] = '5%(10%)'
    g1 = sns.scatterplot(data=data, x="Increase in bike share", y="max_time_phi", hue = 'type', style = 'type', s=80)
    plt.title('B = '+str(B))
    plt.ylabel('Worst-case increase in driving time')
    plt.xlabel('Increase in cycling share')
    g1.xaxis.set_major_formatter(PercentFormatter(1))
    g1.yaxis.set_major_formatter(PercentFormatter(1)) 
    #label_point_right(data['Increase in bike share'], data['max_time_phi'], data['tol'], plt.gca(), data['B'], data['type'])     
    if B ==10:
        plt.xlim([0,0.025])
    if B ==25:
        plt.xlim([0.005,0.03])
    if B ==50:
        plt.xlim([0.015,0.035])
    if B ==75:
        plt.xlim([0.015,0.035])
    plt.ylim([max(0,min(data["max_time_phi"])-0.01), 0.15])
    plt.legend([],[], frameon=False)
    #plt.legend(loc='lower right')
    #plt.legend().get_texts()[0].set_fontsize('10')
    #plt.legend(loc='lower right')
    plt.grid(linewidth=0.3)
    plt.savefig(r"D:\Box\Bike_lane\major\code\Figures\B"+str(B)+"_2_nolabel.pdf", bbox_inches='tight')
    plt.show()
  

for B in [10,25,50,75]:
    data=results.loc[results['B']==B]    
    #if B == 50:
    #    data = data.loc[~((data.type == 'Greedy') & (data.tol == '10%'))]
    #    data.loc[((data.type == 'Greedy') & (data.tol == '5%')),'tol'] = '5%(10%)'
    g1 = sns.scatterplot(data=data, x="Increase in bike share", y="rel_total", hue = 'type', style = 'type', s=80)
    plt.title('B = '+str(B))
    plt.ylabel('Total driving time change')
    plt.xlabel('Increase in cycling share')
    g1.xaxis.set_major_formatter(PercentFormatter(1))
    g1.yaxis.set_major_formatter(PercentFormatter(1)) 
    #label_point_right(data['Increase in bike share'], data['rel_total'], data['tol'], plt.gca(), data['B'], data['type'])   
    if B ==10:
        plt.xlim([0,0.025])
    if B ==25:
        plt.xlim([0.005,0.03])
    if B ==50:
        plt.xlim([0.015,0.035])
    if B ==75:
        plt.xlim([0.015,0.035])
    #plt.ylim([0,0.15])
    #plt.ylim([0.03, 0.16])
    #plt.legend(loc='upper left', bbox_to_anchor=(1.04,1))
    plt.legend([],[], frameon=False)
    #plt.legend(loc='upper right')
    plt.grid(linewidth=0.3)
    plt.savefig(r"D:\Box\Bike_lane\major\code\Figures\B"+str(B)+"_3_nolabel.pdf", bbox_inches='tight')
    plt.show() 


################################################################
# where does the driving time increase come from?
################################################################

dd_origin, dc_origin, do_origin, time_phi_origin, v_origin, φ_origin, time_od_origin = compareR15N15(path_base, 'empty')

dd_optimal, dc_optimal, do_optimal, time_phi_optimal, v_optimal, φ_optimal, time_od_optimal = compareR15N15('n500f8000/R15N15B'+str(B)+'/thre0tol'+str(tol)+"/sol/lane_start.csv") 
do_optimal = d_od - dc_optimal - dd_optimal

path = 'n500f8000/R15N15B'+str(B)+'/thre0tol'+str(tol)+"/sol/lane_start.csv"
bike_lane_loc_optimal = bike_lane_loc.copy()
lanes_start = pd.read_csv(path, header = None)
lane_start = lanes_start[0].apply(int).tolist()
bike_lane_loc_optimal[used_allow_noexist] = lane_start


weights_optimal = np.multiply(weight_cost_w_lane, bike_lane_loc_optimal) + np.multiply(weight_cost_wo_lane, 1-bike_lane_loc_optimal)
time_s_optimal = np.multiply(weights_optimal,(v_optimal+v0))

#fix capacity
weights_fix_capacity = weight_cost_wo_lane
time_fix_capacity = P.T@(np.multiply(weights_fix_capacity,(v_optimal+v0))) #+ c0
time_s_fix_capacity = np.multiply(weights_fix_capacity,(v_optimal+v0))

#fix flow
weights_fix_flow = np.multiply(weight_cost_w_lane, bike_lane_loc_optimal) + np.multiply(weight_cost_wo_lane, 1-bike_lane_loc_optimal)
time_fix_flow = P.T@(np.multiply(weights_fix_flow,(v_origin+v0)))# + c0
time_s_fix_flow = np.multiply(weights_fix_flow,(v_origin+v0))

percent_fix_capacity = (time_fix_capacity-(time_phi_optimal-c0))/time_phi_optimal
percent_fix_flow = (time_fix_flow-(time_phi_optimal-c0))/time_phi_optimal
 


import pandas as pd
import seaborn as sns
sns.set_palette(sns.color_palette()) 
import matplotlib.pyplot as plt
from  matplotlib.ticker import *
from webcolors import rgb_to_name


import matplotlib.font_manager as font_manager
#sns.set_style("whitegrid",{'font.family':'serif', 'font_scale':3, 'font.serif':'Times New Roman'})

font_path = font_manager.findfont(font_manager.FontProperties(family='Times New Roman'))

# Set the font properties for the plot
font = {'family': 'serif',
        'serif': ['Times New Roman'],
        'weight': 'normal',
        'size': 16}

# Set the font properties in matplotlib rcParams
plt.rcParams.update({'font.family': font['family'],
                     'font.serif': font['serif'],
                     'font.weight': font['weight'],
                     'font.size': font['size']})

x = -percent_fix_capacity
y = -percent_fix_flow 
np.mean(x)
np.mean(y)

from scipy.stats import linregress
linregress(x, y)

g1 = sns.scatterplot(x, y)
g1.xaxis.set_major_formatter(PercentFormatter(1))
g1.yaxis.set_major_formatter(PercentFormatter(1))
plt.xlabel('Driving time change due to capacity reduction')
plt.ylabel('Driving time change due to mode shift') 
g1.plot([0,0.15], [0,0], '--', color = 'orange')
g1.plot([0,0], [min(y),max(y)], '--', color = 'orange')
plt.title('Relative path driving time change')
plt.savefig(r'D:\Box\Bike_lane\major\code\Figures\drive_time_decomposition.pdf', bbox_inches='tight')
'''