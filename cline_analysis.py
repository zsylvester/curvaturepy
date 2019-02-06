import numpy as np
import matplotlib.pyplot as plt
from dp_python_master.dpcore import dp
import os
import datetime
import pandas as pd
from scipy.signal import correlate

def load_data(dirname):
    fnames = os.listdir(dirname)
    for fname in fnames:
        if fname == '.DS_Store':
            os.remove(dirname+fname) 
    fnames = os.listdir(dirname)
    fnames.sort()
    clxs = []
    clys = []
    ages = []
    widths = []
    rbxs = []
    lbxs = []
    rbys = []
    lbys = []
    curvatures = []
    for i in range(0,len(fnames)):
        df = pd.read_csv(dirname+fnames[i])
        x = np.array(df['centerline_x'])
        y = np.array(df['centerline_y'])
        lbx = np.array(df['right_bank_x']) # left and right are switched in the CSV file! - fix this
        rbx = np.array(df['left_bank_x'])
        lby = np.array(df['right_bank_y'])
        rby = np.array(df['left_bank_y'])
        clxs.append(x)
        clys.append(y)
        rbxs.append(rbx)
        lbxs.append(lbx)
        rbys.append(rby)
        lbys.append(lby)
        curvatures.append(np.array(df['curvature']))
        ages.append(int(fnames[i][-12:-4]))
        widths.append(df['width (m)'])
    plt.figure()
    for i in range(len(clxs)):
        plt.plot(clxs[i],clys[i],'k.-')
        plt.plot(rbxs[i],rbys[i],'b')
        plt.plot(lbxs[i],lbys[i],'r')
    plt.axis('equal')
    dates = []
    for i in range(len(ages)):
        year = int(str(ages[i])[:4])
        month = int(str(ages[i])[4:6])
        day = int(str(ages[i])[6:])
        date = datetime.datetime(year, month, day)
        dates.append(date)
    return fnames,clxs,clys,rbxs,lbxs,rbys,lbys,curvatures,ages,widths,dates

def correlate_clines(x1,x2,y1,y2,penalty):
    """use dynamic time warping to correlate centerlines
    inputs:
    x1, y1 - coordinates of first centerline
    x2, y2 - coordinates of second centerline
    penalty - parameter that forces more parallel correlation (or not)
    outputs:
    p - indices of correlation in second centerline
    q - indices of correlation in first centerline
    sm - distance matrix"""
    c = len(x1)
    r = len(x2)
    sm = np.zeros((r,c))
    for i in range(0,r):
        sm[i,:] = ((x1-x2[i])**2 + (y1-y2[i])**2)**0.5
    p,q,C,phi = dp(sm,penalty=penalty,gutter=0.0)
    return p,q,sm
  
def get_migr_rate(x1,y1,x2,y2,years,penalty):
    """use dynamic time warping to correlate centerlines
    inputs:
    x1, y1 - coordinates of first centerline
    x2, y2 - coordinates of second centerline
    years - time between the two centerlines, in years
    penalty - parameter that forces more parallel correlation (or not)
    outputs:
    migr_rate - migration rate (in m/years)
    migr_sign - migration sign
    p - indices of correlation in second centerline
    q - indices of correlation in first centerline"""
    p,q,sm = correlate_clines(x1,x2,y1,y2,penalty)
    qn = np.delete(np.array(q),np.where(np.diff(q)==0)[0]+1)
    pn = np.delete(np.array(p),np.where(np.diff(q)==0)[0]+1)
    xa = x1[:-1]
    xb = x1[1:]
    ya = y1[:-1]
    yb = y1[1:]
    x = x2[pn][1:]
    y = y2[pn][1:]
    migr_sign = np.sign((x-xa)*(yb-ya) - (y-ya)*(xb-xa))
    migr_rate = migr_sign*sm[pn,qn][1:]/years
    migr_rate = np.hstack((0,migr_rate))
    return migr_rate, migr_sign, p, q
  
def compute_curvature(x,y):
    """compute curvature of curve defined by coordinates x and y
    curvature is returned in units of 1/(unit of x and y)
    s - distance along curve"""
    dx = np.gradient(x); dy = np.gradient(y)  # first derivatives    
    ds = np.sqrt(dx**2+dy**2)
    ddx = np.gradient(dx); ddy = np.gradient(dy) # second derivatives 
    curvature = (dx*ddy - dy*ddx) / ((dx**2 + dy**2)**1.5) # curvature
    s = np.cumsum(ds) # along-channel distance
    return curvature, s
  
def compute_derivatives(x,y):
    dx = np.gradient(x) # first derivatives
    dy = np.gradient(y)      
    ds = np.sqrt(dx**2+dy**2)
    s = np.cumsum(ds)
    return dx, dy, ds, s
  
def find_zero_crossings(curve):
    """find zero crossings of a curve
    input: 
    a one-dimensional array that describes the curve
    outputs: 
    loc_zero_curv - indices of zero crossings
    loc_max_curv - indices of maximum values"""
    n_curv = abs(np.diff(np.sign(curve)))
    n_curv[plt.mlab.find(n_curv==2)] = 1
    loc_zero_curv = plt.mlab.find(n_curv)
    loc_zero_curv = loc_zero_curv +1
    loc_zero_curv = np.hstack((0,loc_zero_curv,len(curve)-1))
    n_infl = len(loc_zero_curv)
    max_curv = np.zeros(n_infl-1)
    loc_max_curv = np.zeros(n_infl-1, dtype=int)
    for i in range(1, n_infl):
        if np.mean(curve[loc_zero_curv[i-1]:loc_zero_curv[i]])>0:
            max_curv[i-1] = np.max(curve[loc_zero_curv[i-1]:loc_zero_curv[i]])
        if np.mean(curve[loc_zero_curv[i-1]:loc_zero_curv[i]])<0:
            max_curv[i-1] = np.min(curve[loc_zero_curv[i-1]:loc_zero_curv[i]])
        max_local_ind = plt.mlab.find(curve[loc_zero_curv[i-1]:loc_zero_curv[i]]==max_curv[i-1])
        if len(max_local_ind)>1:
            loc_max_curv[i-1] = loc_zero_curv[i-1] + max_local_ind[0]
        elif len(max_local_ind)==1:
            loc_max_curv[i-1] = loc_zero_curv[i-1] + max_local_ind
        else:
            loc_max_curv[i-1] = 0
    return loc_zero_curv, loc_max_curv

def get_predicted_migr_rate(curvature,W,k,Cf,D,kl,s):
    """function for calculating predicted migration rate
    using the simplified Howard-Knutson model
    inputs:
    W - channel width (m)
    k - constant (=1)
    Cf - friction factor
    D - channel depth (m)
    kl - migration constant (m/year)
    s - along-channel distance (m)
    output:
    R1 - predicted migration rate"""
    ds = np.diff(s)
    alpha = k*2*Cf/D
    ns = len(s)
    R0 = kl*W*curvature # preallocate vector for nominal channel migration rate
    R1 = np.zeros(ns) # preallocate adjusted channel migration rate
    for i in range(0,len(R1)):
        si2 = np.hstack((0,np.cumsum(ds[i-1::-1])))  # distance along centerline, backwards from current point 
        G = np.exp(-alpha*si2) # weighting function   
        R1[i] = -1*R0[i] + 2.5*np.sum(R0[i::-1]*G)/np.sum(G) # actual migration rate (m/year)
    return R1

def get_time_shifts(migr_rate,curv,window_length):
    delta_t = np.arange(-len(migr_rate[:window_length]), len(migr_rate[:window_length]))
    time_shifts = []
    i = 0
    while i+window_length < len(curv):
        if np.sum(np.isnan(migr_rate[i:i+window_length]))>0: # get rid of windows with NaNs
            time_shifts.append(np.NaN)
        else:
            corr = correlate(curv[i:i+window_length], migr_rate[i:i+window_length])  
            time_shift = delta_t[corr.argmax()]
            time_shifts.append(time_shift)
        i = i+window_length
    time_shifts = np.array(time_shifts)
    time_shifts = time_shifts[np.isnan(time_shifts)==0] # get rid of NaNs
    return time_shifts

# function for optimizing for Cf:
def get_friction_factor(Cf,curvature,migr_rate,kl,W,k,D,s):
    R1 = get_predicted_migr_rate(curvature,W,k,Cf,D,kl,s)
    corr = correlate(R1, migr_rate)
    # delta time array to match xcorr:
    delta_t = np.arange(1-len(R1), len(R1))
    time_shift = delta_t[corr.argmax()]
    return time_shift # goal is to minimize the time shift