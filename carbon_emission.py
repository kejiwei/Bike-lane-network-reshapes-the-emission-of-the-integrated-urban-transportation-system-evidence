import dill
from gurobipy import *
import pandas as pd

import os
# os.chdir(r'/Users/jz2293/Library/CloudStorage/Box-Box/Bike_lane_emission/code')
#os.chdir(r'D:/Box/Bike_lane_emission/code')


#dill.load_session('/Users/jingweizhang/Box/Bike_lane/clean_code/UE-reduction_bi/selected_census/obj_bike_adoption/threshold/small_area20/bike_lane_correction/workspace/R5N5opt_mac.pkl')

dill.load_session(r'E:\vs.code\MyPapers\Bike_lane_planning\workspaces\R5N5opt_win.pkl') #windows
# dill.load_session('workspaces/R5N5opt_win.pkl') #mac 


# xc和xd是什么参数，表达的含义
xc = pd.read_csv("E:/vs.code/MyPapers/Bike_lane_planning/estimation_results/xc.csv",dtype={"from":str,"to":str}) #骑车的attibutes
xc = xc.set_index(['from','to'])
xd = pd.read_csv("E:/vs.code/MyPapers/Bike_lane_planning/estimation_results/xd.csv",dtype={"from":str,"to":str}) # 驾车的属性
xd = xd.set_index(['from','to'])
valid_estimate = pd.read_csv("E:/vs.code/MyPapers/Bike_lane_planning/estimation_results/valid_estimaten500f8000.csv",index_col = 0) #系数估计值,
print(valid_estimate)
valid_estimate['estimate'] = -valid_estimate['estimate'] # 取相反数

#mode choice 参数
c_notchange = ['avg_tan_abs'] + [i for i in valid_estimate.index if i[0] == 'c'][:-1]
d_notchange = [i for i in valid_estimate.index if i[0] == 'd'][:-1]
o_notchange = [i for i in valid_estimate.index if ((i[0] != 'c')&(i[0] != 'd')&(i!='avg_tan_abs'))]
c_change = ['c_bl_proportion'] #变量，atti名字 beta2*rho
d_change = ['d_drive_time'] #c = b0 + b1X1


# od_ref是od pair？以下每项的参数表达的含义？
xo = -xc.loc[od_ref.index,o_notchange] # 含义是？
xc = xc.loc[od_ref.index,c_notchange+c_change] # 含义是？
xd = xd.loc[od_ref.index,d_notchange+d_change] # 含义是？
uo = np.transpose(list(xo@valid_estimate.loc[o_notchange,'estimate']))
bl_coef = valid_estimate.loc[c_change,'estimate'].tolist()[0] # bike lane coverage系数
dt_coef = valid_estimate.loc[d_change,'estimate'].tolist()[0] # driving time系数
uc_unchanged = np.transpose(list(xc[c_notchange]@valid_estimate.loc[c_notchange,'estimate'])) # xiao  
ud_unchanged = np.transpose(list(xd[d_notchange]@valid_estimate.loc[d_notchange,'estimate'])) # 含义是？

########################## linear approximation of xxx log(dw) ###############################
#d_od = od_d['acs_S000']
od_d = od_d.set_index(['from','to']) 

d_od = od_d['S000']
d_od = np.transpose(d_od.to_list())

N = 15 #number of linear segments used to approximate xxx
R = 15 #number of linear segments used to approximate xxx

a_ff = []
b_ff = []
for i in range(Nod):
    log_drange = np.linspace(np.log(0.01), np.log(d_od[i]), N) # 生成N个在np.log(0.01)和np.log(d_od[i])之间等间隔分布的数值
    drange = np.exp(log_drange)
    a_temp = log_drange + 1
    b_temp = -drange
    a_ff.append(a_temp)
    b_ff.append(b_temp)
a_ff = np.transpose(a_ff)
b_ff = np.transpose(b_ff)
for i in range(N):
    a_ff[i][np.abs(a_ff[i])<1e-03] = 0  #np.abs是为了numerical issue，让其范围波动过大
    b_ff[i][np.abs(b_ff[i])<1e-03] = 0



########################### Linear approximation of xxx congestion function ##############################
import math

a_f = []
b_f = []
a_f_lane = []
b_f_lane = []
dub = P@U.T@d_od #？？？？？？dub是什么？P和U代表0-1matrix？
M_x_lane = np.zeros((Nedge, R)) # M_x_lane代表，创建0矩阵是为了后续矩阵操作
M_x = np.zeros((Nedge, R)) # M_x代表什么，为什么创建0矩阵
for i in range(Nedge):
    if dub[i] == 0: #有些路段需求为0
        a_f.append(np.zeros(R))
        b_f.append(np.zeros(R))
        a_f_lane.append(np.zeros(R))
        b_f_lane.append(np.zeros(R))
    else:
        xrange = np.linspace(0.01, dub[i], R) # xrange是什么？为什么等间隔生成？为了证明方便 斜率等间隔
        a_f.append(weight_cost_wo_lane[i]*xrange) # weight_cost_wo_lane[i]是什么？     
        b_f.append( - 1/2*weight_cost_wo_lane[i]*(xrange)**2)
        if weight_cost_w_lane[i] == math.inf: # 为什么有两种情况？weight_cost_wo_lane和weight_cost_w_lane区别是？without bikelane和with bikelane
            a_f_lane.append(weight_cost_wo_lane[i]*xrange + weight_cost_w_lane[i]*v0[i])
            b_f_lane.append( - 1/2*weight_cost_wo_lane[i]*(xrange)**2)
            weight_cost_w_lane[i] = weight_cost_wo_lane[i]
        else:
            a_f_lane.append(weight_cost_w_lane[i]*xrange + weight_cost_w_lane[i]*v0[i])
            b_f_lane.append( - 1/2*weight_cost_w_lane[i]*(xrange)**2)
    for r in range(R):
        M_x_lane[i,r] = a_f_lane[i][r] * dub[i] + (b_f_lane[i][r]-b_f[i][r]) #取最紧的大M数
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
exist = [s for s in range(Nedge) if bike_lane_loc[s] == 1] # z在已有的车道建bikelae.bike_lane_loc是向量，长度是segment？内嵌在dill.load_session里面吗？因为在此之前没看到bike_lane_loc[s]的定义
notallow = [s for s in range(Nedge) if allow_lane[s] == 0] # 同上。allow_lane是向量？
locations = list(set(exist+notallow))
#locations = np.concatenate((np.where(bike_lane_loc == 1), np.where(allow_lane == 0)),axis=1)[0]
P_bike_temp[locations,:] = 0

bike_edges = sparse.find(P_bike_temp.T)[1] #从 P_bike_temp 的转置中找到非零元素的列坐标数组
bike_ods = sparse.find(P_bike_temp.T)[0]


###############################################################
# lower level - UE 
###############################################################

from gurobipy import *

def UE(lane=bike_lane_loc):  #solve for approximated equilibrium
    print(N,R,"=======================================================================")
    #uc = np.transpose([P_bike.T@distance/1000,P_bike.T@(distance*lane)/(P_bike.T @distance),np.ones(Nod)])@c_coef
    uc = uc_unchanged+P_bike.T@(distance*lane)/(P_bike.T @distance)*bl_coef # P_bike是0-1matrix

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

    W = np.diag(np.multiply(weight_cost_w_lane, lane) + np.multiply(weight_cost_wo_lane, 1-lane)) # weight_cost

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
    us = model.addMVar(Nedge,lb=-GRB.INFINITY) #v_s

    model.addConstr(do + dc + dd == d_od)
    model.addConstr(U @ φ == dd) # U是OD对w电动车驾驶path集合
    model.addConstr(P @ φ == v) # P是所有电动车驾驶路径path集合
    for i in range(N):
        model.addConstr(ωc >= np.diag(a_ff[i,:]) @ dc + b_ff[i,:])
        model.addConstr(ωo >= np.diag(a_ff[i,:]) @ do + b_ff[i,:])
        model.addConstr(ωd >= np.diag(a_ff[i,:]) @ dd + b_ff[i,:])
    for i in range(R):
        model.addConstr(us >= np.diag(a_f_s[i,:])@v + b_f_s[i,:])

    model.setObjective(dc@uc + do@uo + dd@ud_unchanged + us@np.ones(Nedge)*dt_coef + c0@φ*dt_coef + ωc@np.ones(Nod) + ωo@np.ones(Nod) + ωd@np.ones(Nod)) #为什么没有关于路段流v的目标项
    model.modelSense = GRB.MINIMIZE
    model.params.OutputFlag = 0
    model.update()
    model.optimize()

    demand = {}
    demand['φ_opt'] = φ.x  # 路径流 path flows
    demand['v_opt'] = v.x  # 路段流 segment flows
    demand['do_opt'] = do.x # 在OD对w上其他出行方式的需求
    demand['dc_opt'] = dc.x # 在OD对w上自行车需求
    demand['dd_opt'] = dd.x # 在OD对w上驾驶需求
    demand['us_opt'] = us.x # 非线性项\xi_s
    #print(demand, c0@φ.x + 2*sum(us.x), model.getObjective().getValue())

#    return sum(dd.x)*d_coef[0] + c0@φ.x*d_coef[1] + 2*sum(us.x)*d_coef[1]
    return demand, c0@φ.x + 2*sum(us.x), model.getObjective().getValue()

demands, drive_time, obj = UE(bike_lane_loc)

# import matplotlib.pyplot as plt
# plt.hist(demands['v_opt'])
# sum(demands['v_opt'])
# sum(demands['dc_opt'])/sum(d_od)


######################################################
# Optimization
######################################################

from datetime import datetime
from datetime import date

# os.chdir(r'\vs.code\MyPapers\Bike_lane_planning\optimization_results') #将当前工作目录更改为 optimization_results 目录

thresholds = [0]
Bs = [10,25,50,75]
result = pd.DataFrame(index=Bs,columns=['MIPGap','Obj','ObjBound','Runtime'])
prev = thresholds[0]
bike_lane_loc_start = bike_lane_loc.copy()
bike_phi_loc_start = np.zeros(Nod)

threshold = prev


# 以下各项参数代表的含义？
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
dc_ub = sum(demands_ub['dc_opt'])
print(dc_ub)
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
# plt.hist((time_od1-time_od)/time_od)
print("+++++++++++++++++++++++++++++ max time",max((time1-time)/time),'+++++++++++++++++++++++++++++')
print("+++++++++++++++++++++++++++++ max od time",max((time_od1-time_od)/time_od),'+++++++++++++++++++++++++++++')
print((sum(demands_ub['dc_opt'])-sum(demands['dc_opt']))/sum(demands['dc_opt']))
print((sum(demands_ub['dd_opt'])-sum(demands['dd_opt']))/sum(demands['dd_opt']))
max((time1-time))

print(sum(demands['dc_opt'])/sum(d_od))
print(sum(demands_ub['dc_opt'])/sum(d_od) )

# f, ax = plt.subplots(figsize=(6, 6))
# ax.scatter(demands['v_opt'],demands_ub['v_opt'] , c=".3")
# ax.plot([0, 1e4], [0, 1e4], ls="--", c=".3")

B,tol = 100,1.15

#######################################
# define emission
#################################
speeds1 = np.array([0.1]+[2.5+i*5 for i in range(1)]+[60]) # reduce number of intervals
speeds1=speeds1.T
es1 = np.array([1.781301904, 1.007831021, 0.621095826, 0.492184219, 0.425514679, 0.381057519, 0.341038929, 0.323903576, 0.313793007, 0.30626232, 0.300800763, 0.298487502, 0.299128484, 0.302748302, 0.313656613, 0.331026158]) #kg/mile # reduce number of emission rates
# 排放率kg/mile es = np.array([1.781301904, 1.007831021]) #kg/mile # reduce number of emission rates
es1 = es1[:(len(speeds1)-1)]
es1 = es1.T
  
K1 = len(es1) 
##########################################################################################################
#warm start not necessary  lane = lane_sol
##########################################################################################################
def pre_solve1(K1):  #lane_sol is the bike lane location binary vector
    model1 = Model("CarbonEmission")
    φ  = model1.addMVar(Npath, name = 'phi')
    v  = model1.addMVar(Nedge, name = 'v')
    dd = model1.addMVar(Nod, name = 'dd')
    do = model1.addMVar(Nod, name = 'do')
    dc = model1.addMVar(Nod, name = 'dc')
    lane = model1.addMVar(len(used_allow_noexist), vtype = GRB.BINARY, name = 'lane')
    lane_phi = model1.addMVar(len(used_cod), vtype = GRB.BINARY, name = 'lane_phi')
    
    z = model1.addMVar((Npath, K1), vtype=GRB.BINARY, name="z")
    emission = model1.addMVar(Npath, name = 'path_emission')
    
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

    pathdistance = P.T@distance/1609.34 #pathdistance in miles
    pathdistance_diag = np.diag(pathdistance)
    pathdistance = pathdistance.T
    time_bucket = np.zeros((Npath,K1+1))
    M = max(pathdistance)/min(speeds1)
    for k in range(K1+1):
        time_bucket[:,k] = pathdistance/speeds1[k]  #time_bucket in hours
    for k in range(K1):
        model1.addConstr(np.diag(time_bucket[:,k+1]*60)@z[:,k] <= cost_p)
        model1.addConstr(np.diag(time_bucket[:,k]*60)@z[:,k] + M*60*(np.ones(Npath)-z[:,k]) >= cost_p) # equation 5 in overleaf

    model1.addConstr(z@np.ones(K1)==np.ones(Npath))

    ###########################
    #add time constraint
    ###########################

    # 690-710行code是求什么？每行的含义呢？
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

    model1.addConstr(distance[used_allow_noexist]@lane <= B*1609.34)


    model1.setObjective(φ@pathdistance_diag@(z@es1))
    result = pd.DataFrame(columns=['MIPGap','Obj','ObjBound','Runtime'])
    all_dist = sum(distance[used_allow_noexist])
    model1.params.MIPGap=0.05 
    model1.params.NonConvex = 2
    model1.params.MIPFocus = 1
    # model1.params.FeasibilityTol = 1e-06
    #model1.params.DegenMoves = -1
    model1.modelSense = GRB.MINIMIZE
    model1.Params.lazyConstraints = 1
    model1.params.TimeLimit = 3600*6   # running time in seconds. 

    sol_loc = "em_B"+str(B)+"/thre" + str(threshold)+"tol"+str(tol) + "/sol"
    if not os.path.exists(sol_loc):
        os.makedirs(sol_loc) # save .sol file
    model1.setParam("SolFiles", sol_loc)
    model1.update()
    model1.optimize()

    lane_sol = lane.x
    lane_phi_sol = lane_phi.x

    return lane_sol, lane_phi_sol
 
lane_sol, lane_phi_sol = pre_solve1(K1)


#===========================第二次presolve,得到

speeds2 = np.array([0.1]+[2.5+i*5 for i in range(3)]+[60]) # reduce number of intervals
speeds2=speeds2.T
es2 = np.array([1.781301904, 1.007831021, 0.621095826, 0.492184219, 0.425514679, 0.381057519, 0.341038929, 0.323903576, 0.313793007, 0.30626232, 0.300800763, 0.298487502, 0.299128484, 0.302748302, 0.313656613, 0.331026158]) #kg/mile # reduce number of emission rates
# 排放率kg/mile es = np.array([1.781301904, 1.007831021]) #kg/mile # reduce number of emission rates
es2 = es2[:(len(speeds2)-1)]
es2 = es2.T
  
K2 = len(es2) 
##########################################################################################################
#warm start not necessary  lane = lane_sol
##########################################################################################################
def pre_solve2(lane_sol, lane_phi_sol, K2):  #lane_sol is the bike lane location binary vector
    model1 = Model("CarbonEmission")
    φ  = model1.addMVar(Npath, name = 'phi')
    v  = model1.addMVar(Nedge, name = 'v')
    dd = model1.addMVar(Nod, name = 'dd')
    do = model1.addMVar(Nod, name = 'do')
    dc = model1.addMVar(Nod, name = 'dc')
    # lane = model1.addMVar(len(used_allow_noexist), vtype = GRB.BINARY, name = 'lane')
    # lane_phi = model1.addMVar(len(used_cod), vtype = GRB.BINARY, name = 'lane_phi')
    lane = np.array(lane_sol)
    lane_phi = np.array(lane_phi_sol)


    z = model1.addMVar((Npath, K2), vtype=GRB.BINARY, name="z")
    emission = model1.addMVar(Npath, name = 'path_emission')
    
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
        
        # model1.addConstr(lane[s] <= lane_phi@phi_vec)
        # model1.addConstr(lane[s]*len(phis) >= lane_phi@phi_vec) 
   
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

    pathdistance = P.T@distance/1609.34 #pathdistance in miles
    pathdistance_diag = np.diag(pathdistance)
    pathdistance = pathdistance.T
    time_bucket = np.zeros((Npath,K2+1))
    M = max(pathdistance)/min(speeds2)
    for k in range(K2+1):
        time_bucket[:,k] = pathdistance/speeds2[k]  #time_bucket in hours
    for k in range(K2):
        model1.addConstr(np.diag(time_bucket[:,k+1]*60)@z[:,k] <= cost_p)
        model1.addConstr(np.diag(time_bucket[:,k]*60)@z[:,k] + M*60*(np.ones(Npath)-z[:,k]) >= cost_p) # equation 5 in overleaf

    model1.addConstr(z@np.ones(K2)==np.ones(Npath))

    ###########################
    #add time constraint
    ###########################

    # 690-710行code是求什么？每行的含义呢？
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
    
    # model1.addConstr(distance[used_allow_noexist]@lane <= B*1609.34)


    model1.setObjective(φ@pathdistance_diag@(z@es2))
    result = pd.DataFrame(columns=['MIPGap','Obj','ObjBound','Runtime'])
    all_dist = sum(distance[used_allow_noexist])
    model1.params.MIPGap=0.05 
    model1.params.NonConvex = 2
    model1.params.MIPFocus = 1
    # model1.params.FeasibilityTol = 1e-06
    #model1.params.DegenMoves = -1
    model1.modelSense = GRB.MINIMIZE
    model1.Params.lazyConstraints = 1
    model1.params.TimeLimit = 3600*6   # running time in seconds. 

    sol_loc = "em_B"+str(B)+"/thre" + str(threshold)+"tol"+str(tol) + "/sol"
    if not os.path.exists(sol_loc):
        os.makedirs(sol_loc) # save .sol file
    model1.setParam("SolFiles", sol_loc)
    model1.update()
    model1.optimize()

    z_start = z.x
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

    return z_start, φ_start , v_start , dd_start, do_start, dc_start, ωc_start, ωo_start, ωd_start, us_start, uc_s_start, uc_start, d_uc_start, μ_start , γ_start , λ_start , πo_start, πc_start, πd_start, σ_start, σb_start
 
z_start, φ_start , v_start , dd_start, do_start, dc_start, ωc_start, ωo_start, ωd_start, us_start, uc_s_start, uc_start, d_uc_start, μ_start , γ_start , λ_start , πo_start, πc_start, πd_start, σ_start, σb_start = pre_solve2(lane_sol, lane_phi_sol, K2) 


#######################################
# define emission
#################################
speeds = np.array([0.1]+[2.5+i*5 for i in range(3)]+[60]) # reduce number of intervals
speeds=speeds.T
es = np.array([1.781301904, 1.007831021, 0.621095826, 0.492184219, 0.425514679, 0.381057519, 0.341038929, 0.323903576, 0.313793007, 0.30626232, 0.300800763, 0.298487502, 0.299128484, 0.302748302, 0.313656613, 0.331026158]) #kg/mile # reduce number of emission rates
# 排放率kg/mile es = np.array([1.781301904, 1.007831021]) #kg/mile # reduce number of emission rates
es = es[:(len(speeds)-1)]
es = es.T


##########################################################################################################
#optimization of linearly approximated bi-level programing BL-A
##########################################################################################################
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

# 690-710行code是求什么？每行的含义呢？
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
 #add emission constraint  ### save for Mengling
 ###########################
 #emission binary

# 碳排放模型
#model1 = Model("estimate1")
K = len(es)  #sanity check: should be reduced to ?? 3?4?
# z = model1.addMVar((Npath, K), lb=0,ub = 1, name = 'z') #change to binary
z = model1.addMVar((Npath, K), vtype=GRB.BINARY, name="z")
emission = model1.addMVar(Npath, name = 'path_emission')
pathdistance = P.T@distance/1609.34 #pathdistance in miles
pathdistance_diag = np.diag(pathdistance)
pathdistance = pathdistance.T
time_bucket = np.zeros((Npath,K+1))
M = max(pathdistance)/min(speeds)
for k in range(K+1):
    time_bucket[:,k] = pathdistance/speeds[k]  #time_bucket in hours
for k in range(K):
    model1.addConstr(np.diag(time_bucket[:,k+1]*60)@z[:,k] <= cost_p)
    model1.addConstr(np.diag(time_bucket[:,k]*60)@z[:,k] + M*60*(np.ones(Npath)-z[:,k]) >= cost_p) # equation 5 in overleaf

model1.addConstr(z@np.ones(K)==np.ones(Npath))

# 给定lane_start,看是否有解
try:
   lane.start = lane_sol#bike_lane_loc_start[used_allow_noexist]
   lane_phi.start = lane_phi_sol
   φ.start = φ_start  # xxx.start,gurobi中初始解
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
   z.start = z_start
   pass
except:
    print('unable to start at tol', tol,'=============')
    pass

# try linear approx --------------------------------
#emm = model1.addMVar(Npath,lb=-GRB.INFINITY, name = 'emm')
#for p in range(Npath):
#    model1.addConstr(emm[p] * cost_p[p]>=0.14+0.01* cost_p[p] )
    

#model1.setObjective(sum(dd)*d_coef[0] + c0@φ*d_coef[1] + 2*sum(us)*d_coef[1] + dc@uc_const + sum(d_uc)*c_coef[1] + do@uo)
#model1.addConstr(emission == z@es)
#model1.setObjective(emission @ pathdistance_diag @ φ) 

#model1.setObjective(sum(φ[p]*pathdistance[p]*sum(z[p,k]*es[k] for k in range(K)) for p in range(Npath)))
model1.setObjective(φ@pathdistance_diag@(z@es))

#model1.setObjective(sum(dd)*d_coef[0] + c0@φ*d_coef[1] + 2*sum(us)*d_coef[1] + dc@uc_const + sum(d_uc)*c_coef[1] + do@uo)
#model1.setObjective(dc@np.ones(Nod))
#model1.setObjective(0)
result = pd.DataFrame(columns=['MIPGap','Obj','ObjBound','Runtime'])
all_dist = sum(distance[used_allow_noexist])
model1.params.MIPGap=0.05 
model1.params.NonConvex = 2
model1.params.MIPFocus = 1
# model1.params.FeasibilityTol = 1e-06
#model1.params.DegenMoves = -1
model1.modelSense = GRB.MINIMIZE
#model1.update()
#model0 = model1.copy()

model1.addConstr(distance[used_allow_noexist]@lane <= B*1609.34)
model1.Params.lazyConstraints = 1

model1.params.TimeLimit = 3600*8   # running time in seconds. 

sol_loc = "em_B"+str(B)+"/thre" + str(threshold)+"tol"+str(tol) + "/sol"
if not os.path.exists(sol_loc):
    os.makedirs(sol_loc) # save .sol file
model1.setParam("SolFiles", sol_loc)
model1.update()
model1.optimize()
# model1.computeIIS()
# model1.write("carbon model1_add numerical error.lp")


