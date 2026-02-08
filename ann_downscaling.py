import numpy as np 
from scipy.io import loadmat
import matplotlib.pyplot as plt
import os
import re
import numpy as np
from datetime import datetime
import netCDF4 as nc
import torch.nn as nn
import torch
import tqdm
import copy
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split
import scipy 
import glob
import cartopy.crs as ccrs
import pandas as pd
import torch.nn as nn
import torch.optim as optim


# Starter data
ms = loadmat('MatlabStarter (1).mat')

# How many days do you want to run (starts Jan 1, 2013)
nday = 10

# Runs three-hourly because that is the timing of the AWS data 



def restructuring_filtered_array(filtered_values,boolean_mask,rows,cols):
    reconstructed_array = np.full((rows, cols), np.nan)
    flat_array = reconstructed_array.flatten()
    flat_array[boolean_mask] = filtered_values
    reconstructed_array = flat_array.reshape((rows, cols))
    return reconstructed_array



alt = np.tile(ms['alt'].T,(24,1,1)).reshape(-1,1)
frocean = np.tile(ms['frocean'].T,(24,1,1)).reshape(-1,1)
pLat = np.tile(ms['LAT'].T,(24,1,1)).reshape(-1,1)
pLon = np.tile(ms['LON'].T,(24,1,1)).reshape(-1,1)
hlats = ms['hlat'].T.reshape(-1,1)
hlons = ms['hlon'].T.reshape(-1,1)
hs = ms['h'].T.reshape(-1,1)

i1 = ms['i1'][0][0]
i2 = ms['i2'][0][0]
i3 = ms['i3'][0][0]
i4 = ms['i4'][0][0]


# Loading in flux data from MERRA-2 
fnames = glob.glob('/scratch/satellite/MERRA2/M2T1NXFLX_2013/MERRA2_400.tavg*flx*n*')
nm = len(fnames) # Number of days for output
dtf = []
for i in range(nm):
    dtf.append(int(fnames[i][-12:-4].strip('.')))
ib = sorted(range(nm), key=lambda k: dtf[k])
fnames = [fnames[i] for i in ib]


# Loading in slv data from MERRA-2 (temp, etc)
fnamesS = glob.glob('/scratch/satellite/MERRA2/M2T1NXSLV_2013/MERRA2_400.tavg*slv*n*')
nm = len(fnamesS)
dtf = []
for i in range(nm):
    dtf.append(int(fnamesS[i][-12:-4].strip('.')))
ib = np.argsort(dtf)
fnamesS = [fnamesS[i] for i in ib]




# Constants
mns_in_day = 24 * 60   # Number of minutes in a day
hday = 24              # number of hours in the day
scYr = 60 * 60 * 24 * 365   # seconds per year


# Grabbing lat/lon/evap/snow precip/total precip
for j in range(nday):
    nme = fnames[j]
    # retrieve date information from the filename
    yr = int(nme[-12:-8])
    mo = int(nme[-8:-6])
    dy = int(nme[-6:-4])
    t = datetime(yr, mo, dy)
    doy = t.timetuple().tm_yday  # day of year for that file
    nt = datetime(yr, 12, 31)
    ndoy = nt.timetuple().tm_yday  # total days of year in that year
    if j == 0:  # read in common variables once
        with nc.Dataset(nme) as ds:
            lat = ds.variables['lat'][:]
            lon = ds.variables['lon'][:]
            mn = ds.variables['time'][:].astype(float)
        ev = np.full((len(lon), len(lat), nm // nday), np.nan)

    with nc.Dataset(nme) as ds:
        evap = ds.variables['EVAP'][:,i2:i1+1,i3:i4+1]
        sno = ds.variables['PRECSNO'][:, i2:i1+1,i3:i4+1]
        tot = ds.variables['PRECTOT'][:,i2:i1+1,i3:i4+1]
        

    d_y = yr + (doy - 1 + mn / mns_in_day) / ndoy





# Projection
projection = ccrs.Stereographic(
    central_latitude=-90,
    false_easting=0.0,
    false_northing=0.0,
    true_scale_latitude=-71.0,
    globe=ccrs.Globe('WGS84'))

# AWS data
lat_locs = [-67.02, -67.57, -65.93, -65.75, -70.73, -64.8,-65.25 ]
lon_locs = [-61.5, -62.12, -61.85, -62.88, -63.82, -62.82,-59.44 ]



# *** OUTPUT ***
results = np.zeros((8*nday,len(lat_locs))) # Modeled temperatures using downscaling
weknows = np.zeros((8*nday,len(lat_locs))) # 




threehourlyindx = 0 # index of 3 hourly output 

for j in range(nday):
    nme = fnamesS[j]
    ds = nc.Dataset(nme)
    t2m = ds.variables['T2M'][:,i2:i1+1, i3:i4+1]
    oY = t2m.reshape(-1,1)

    # t2m = t2m.reshape((-1,1))
    hrs = np.arange(0,24)
    hrs = hrs.reshape(24, 1, 1)
    HR = np.tile(hrs,(t2m[0,:,:].shape))
    hrs = HR.reshape(-1,1)
            
    sfts = [[0, 1, 1], [0,1, 0], [0, 1, -1], [0, 1, 0], [0, 0, -1], [0, -1, 1 ], [0, -1, 0], [0, -1, -1]]
    neightT = []
    neightA = []

    alt_3d =np.tile(ms['alt'].T,(24,1,1))

    for g in range(len(sfts)):
        temp = np.roll(t2m,sfts[g])
        neightT.append(temp.reshape(-1,1))
        temp = np.roll(alt_3d,sfts[g])
        neightA.append(temp.reshape(-1,1))

    main_array = np.hstack((pLat, pLon, alt, hrs, np.squeeze(np.asarray(neightA)).T, np.squeeze(np.asarray(neightT)).T,oY))
    tq = ((np.squeeze(pLat)>=-76) & (np.squeeze(pLat)<=-61) & (np.squeeze(pLon)>-80.625) & (np.squeeze(pLon)<=-47.5) & (np.squeeze(frocean) < 1) & np.sum(~np.isnan(main_array),axis=1))
               
    main_arr_filtered = main_array[tq==True,:]


    X_train_raw, X_test_raw, y_train, y_test = train_test_split(main_arr_filtered[:,:-1], main_arr_filtered[:,-1], test_size=0.2, random_state=42)

    
    # Neural Network with two hidden layers 

    model = nn.Sequential(
        nn.Linear(20, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
    )

 
    
    # loss function and optimizer
    loss_fn = nn.MSELoss()  # mean square error
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    scaler = StandardScaler()
    scaler.fit(X_train_raw)
    X_train = scaler.transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)
    
    # training parameters
    n_epochs = 100   # number of epochs to run
    batch_size = 100 # size of each batch
    batch_start = torch.arange(0, len(X_train), batch_size)
    
    # Hold the best model
    best_mse = np.inf   # init to infisnity
    best_weights = None
    history = []
    
    # training loop
    for epoch in range(n_epochs):
        model.train()
        with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=False) as bar:
            bar.set_description(f"Epoch {epoch}")
            for start in bar:
                # take a batch
                X_batch = X_train[start:start+batch_size]
                y_batch = y_train[start:start+batch_size]
                # forward pass
                y_pred = model(X_batch)
                loss = loss_fn(y_pred, y_batch)
                # backward pass
                optimizer.zero_grad()
                loss.backward()
                # update weights
                optimizer.step()
                # print progress
                bar.set_postfix(mse=float(loss))
        # evaluate accuracy at end of each epoch
        model.eval()
        y_pred = model(X_test)
        mse = loss_fn(y_pred, y_test)
        mse = float(mse)
        history.append(mse)
        if mse < best_mse:
            best_mse = mse
            best_weights = copy.deepcopy(model.state_dict())
 
    # restore model and return best accuracy
    model.load_state_dict(best_weights) 
 

    t2m = ds.variables['T2M'][:,i2:i1+1, i3:i4+1]
    tqq = ((hlats>=-76) & (hlats <=-61) & (hlons>=-80.625) & (hlons<=-47.5) & (~np.isnan(hs)))
    rows, cols = ms['h'].T.shape
    full_reconstructed_array = np.zeros((24,rows,cols))

    for indx, hridx in enumerate(np.arange(0,24,3)):
        interpolator = scipy.interpolate.RegularGridInterpolator((ms['LAT'][0,:],ms['LON'][:,0]),HR[hridx, :,:])
        hours = interpolator(np.column_stack((hlats[tqq],hlons[tqq])))

        neightT2 = []
        neightA2 = []
        for g in range(len(sfts)):

            temp = np.roll(t2m,sfts[g])
            interpolator = scipy.interpolate.RegularGridInterpolator((ms['LAT'][0,:],ms['LON'][:,0]),temp[hridx, :,:])
            T2M = interpolator(np.column_stack((hlats[tqq],hlons[tqq])))
            neightT2.append(T2M.reshape(-1,1))

            temp = np.roll(alt_3d,sfts[g])
            interpolator = scipy.interpolate.RegularGridInterpolator((ms['LAT'][0,:],ms['LON'][:,0]),temp[hridx, :,:])
            elev = interpolator(np.column_stack((hlats[tqq],hlons[tqq])))        
            neightA2.append(elev.reshape(-1,1))
        
        
        predX = np.hstack((hlats[tqq].reshape(-1,1),hlons[tqq].reshape(-1,1),hs[tqq].reshape(-1,1), hours.reshape(-1,1), np.squeeze(np.asarray(neightA2)).T, np.squeeze(np.asarray(neightT2)).T))
        predX_s = scaler.transform(predX)
        predX_t = torch.tensor(predX_s, dtype=torch.float32)
        ytorchped = model(predX_t)
        filtered_values = np.squeeze(ytorchped.detach().numpy())
        boolean_mask = np.squeeze(tqq)
        rows, cols = ms['h'].T.shape
        reconstructed_array = restructuring_filtered_array(filtered_values,boolean_mask,rows,cols,)
        full_reconstructed_array[hridx,:,:] = reconstructed_array

        # Flask Glacier

        interpolator = scipy.interpolate.RegularGridInterpolator((ms['xs'][0,:],ms['ys'][:,0]),reconstructed_array)
        interpolator2 = scipy.interpolate.RegularGridInterpolator((ms['LAT'][0,:],ms['LON'][:,0]),t2m[hridx,:,:])


        for locx,location in enumerate(lat_locs):
            zeepoint =projection.transform_points(ccrs.PlateCarree(),np.asarray(lon_locs[locx]),np.asarray(lat_locs[locx]))
            results[threehourlyindx,locx] = interpolator((zeepoint[0,0],zeepoint[0,1]))
            weknows[threehourlyindx,locx] =  interpolator2((lat_locs[locx],lon_locs[locx]))
        threehourlyindx += 1 




locations_names = ['aws14_3h.csv','aws15_3h.csv','aws17_3h.csv','Flask Glacier_3h.csv','Duthiers Point_3h.csv','Robertson Island_3h.csv']

for locx,location in enumerate(locations_names): 
    fig = plt.figure()
    fg = pd.read_csv(locations_names[locx])
    plt.plot(np.arange(0,len(results[:,0]))/8.,results[:,locx],'b--')
    plt.plot(np.arange(0,len(results[:,0]))/8.,weknows[:,locx],'b-')
    plt.plot(np.arange(0,len(results[:,0]))/8.,fg['Temperature(C)'][fg['Year'] == 2013][:len(results[:,0])]+273.15,'k-')
    difference1 = results[:,locx]-fg['Temperature(C)'][fg['Year'] == 2013][:len(results[:,0])]+273.15
    difference2 = weknows[:,locx]-fg['Temperature(C)'][fg['Year'] == 2013][:len(results[:,0])]+273.15

    print(np.nanmean(np.sqrt((difference1)**2)))
    print(np.nanmean(np.sqrt((difference2)**2)))

    fig.savefig('NN_run1_'+location[:-4]+'.png')

    plt.close()
   
import scipy


plt.figure()

def restructuring_filtered_array(filtered_values,boolean_mask,rows,cols):
    reconstructed_array = np.full((rows, cols), np.nan)
    flat_array = reconstructed_array.flatten()
    flat_array[boolean_mask] = filtered_values
    reconstructed_array = flat_array.reshape((rows, cols))
    return reconstructed_array

filtered_values = np.squeeze(ytorchped.detach().numpy())
boolean_mask = np.squeeze(tqq)
rows, cols = ms['h'].T.shape
reconstructed_array = restructuring_filtered_array(filtered_values,boolean_mask,rows,cols,)
plt.pcolormesh(reconstructed_array.T,cmap='rainbow')
plt.colorbar(label='Temperature [K]')
plt.xlim([200,1400])
plt.ylim([2400,4600])

plt.savefig('MAP.png')

fig = plt.figure(figsize=(10, 8)) # Create a figure with a specific size
ax = plt.axes(projection=projection) # Create GeoAxes with Plate Carree projection

ax.pcolor(ms['LON'],ms['LAT'],t2m[-1,:,:],transform=ccrs.PlateCarre())



# Saving the Results 
scipy.io.savemat('NN_run1.mat',{'results':results,'weknows':weknows})








