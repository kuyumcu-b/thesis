import warnings

warnings.filterwarnings("ignore")

from parcels import FieldSet, ParticleSet, JITParticle, AdvectionRK4, AdvectionRK4_3D, plotTrajectoriesFile, ErrorCode
from glob import glob
from datetime import timedelta as delta
from os import path
import netCDF4
import xarray as xr
import matplotlib.pyplot as plt
import sys
import numpy as np
import matplotlib.animation as animation
import pandas as pd
import bisect as b
import openpyxl
import os
import datetime


file = '20192022.nc'
filenames = {'U': {'lon': file, 'lat': file, 'depth': file, 'data': file},
             'V': {'lon': file, 'lat': file, 'depth': file, 'data': file}}
variables = {'U': 'uo', 'V': 'vo', }
dimensions = {'U': {'lon': 'lon', 'lat': 'lat', 'depth': 'depth', 'time': 'time'},
              'V': {'lon': 'lon', 'lat': 'lat', 'depth': 'depth', 'time': 'time'}}
# with stuck particle
# all months

month = ["jan19", "feb19", "mar19", "apr19", "may19", "jun19", "jul19", "aug19", "sep19", "oct19", "nov19", "dec19",
         "jan20", "feb20", "mar20", "apr20", "may20", "jun20", "jul20", "aug20", "sep20", "oct20", "nov20", "dec20",
         "jan21", "feb21", "mar21", "apr21", "may21", "jun21", "jul21", "aug21", "sep21", "oct21", "nov21", "dec21"]
day = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335,
       366, 397, 426, 457, 487, 518, 548, 579, 610, 640, 671, 701,
       732, 763, 791, 822, 852, 883, 913, 944, 975, 1005, 1036, 1066]
dir = str(datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S"))
if not os.path.exists("results"):
    os.mkdir("results")
os.mkdir(os.path.join("results" + "/", dir))

for j in range(19, 23, 1):
    df4 = pd.DataFrame([])
    os.mkdir(os.path.join("results" + "/" + dir + "/", month[j]))

    for depth in range(10, 601, 10):

        os.mkdir(os.path.join("results" + "/" + dir + "/" + month[j] + "/", str(depth)))

        fieldset = FieldSet.from_netcdf(filenames, variables, dimensions)
        npart = 1000
        time = np.full(shape=1000,
                       fill_value=1,
                       dtype=float) * delta(days=day[j]).total_seconds()
        pset = ParticleSet.from_list(fieldset=fieldset, pclass=JITParticle,
                                     lon=np.linspace(34.6, 35.66667, 1000),
                                     lat=np.full(
                                         shape=1000,
                                         fill_value=35.72,
                                         dtype=float),
                                     depth=np.full(
                                         shape=1000,
                                         fill_value=depth,
                                         dtype=float),
                                     time=time)


        def DeleteParticle(particle, fieldset, time):
            particle.delete()


        kernels = pset.Kernel(AdvectionRK4)
        output_file = pset.ParticleFile(name='Residence_' + str(depth) + 'm_' + month[j] + '.nc', outputdt=delta(days=1))
        for time in range(1400 - day[j]):
            try:
                # pset.show(savefile='particles'+str(time).zfill(3), field='vector', land=True, vmax=None)
                pset.execute(kernels, runtime=delta(days=1), dt=delta(days=1), output_file=output_file,
                             recovery={ErrorCode.ErrorOutOfBounds: DeleteParticle})
            except ValueError:  # raised if `y` is empty.
                pass                                        
        output_file.export()  # export the trajectory data to a netcdf file

        # anim = plotTrajectoriesFile('Residence_' + str(depth) + 'm_' + month[j] + '.nc', mode='movie2d_notebook')
        # anim.save("results" + "/" + dir + "/" + month[j] + "/" + str(depth) + "/animation.mp4",writer=animation.FFMpegWriter(fps=10))
        data_xarray = xr.open_dataset('Residence_' + str(depth) + 'm_' + month[j] + '.nc')
        df = data_xarray['trajectory'].to_dataframe().unstack()
        df.columns = df.columns.droplevel(0)
        df = df.rename_axis(None, axis=1)
        df.to_excel("results" + "/" + dir + "/" + month[j] + "/" + str(depth) + "/" + 'trajectory.xlsx', index=False,
                 header=False, na_rep='nan')
        df2 = df.isna().sum().to_frame('nan count')
        df2 = df2.rename(columns={'index': 'day'})
        df2["day"] = df2.index + 1

        # Calculating when a particle is stuck
        d = xr.open_dataset('Residence_' + str(depth) + 'm_' + month[j] + '.nc')
        pdx = np.diff(d['lon'], axis=1, prepend=0)
        pdy = np.diff(d['lat'], axis=1, prepend=0)
        distance = np.sqrt(np.square(pdx) + np.square(pdy))  # approximation of the distance travelled
        stuck = distance < 1e-5  # [degrees] 1e-5 degrees ~~ 1 m in dt = 1 hour.

        df2["stuck count"] = sum(stuck)
        df2["cum stuck count"] = df2["stuck count"]
        for i in range(2,len(df2["stuck count"])+1):
            df2.iloc[-i,df2.columns.get_loc("cum stuck count")] = df2[-i:]["stuck count"].min()
        df2.insert(len(df2.columns), 'tot particle', 1000)
        df2["tot particle"] = df2["tot particle"] - df2["cum stuck count"]
        print('depth:' + str(depth) + ' month:' + month[j])
        #print('****************************')
        mlist = []
        rtlist = []
        for i in range(len(df2["cum stuck count"])):
            median = np.median([*range(1, df2["tot particle"].iat[i]+1, 1)])
            if df2['nan count'].isin([median]).any().any() == True:
                if df2[:i+1][df2[:i+1]['nan count'].values>median].any().any() == False:
                    df3 = df2.loc[df2['day'] == len(df2["cum stuck count"])].tail(1)
                    df3.reset_index()
                    df3['median'] = median
                    #print(df3.iloc[-3:-2])
                    #print(df3.iloc[-2:-1])
                    #print('MEAN---------------------------------------------------------------------')
                    #print(df3.iloc[-1:])
                    df3['depth'] = depth
                    rtlist.append(0)
                    mlist.append(median)
                    #print('-------------------------------------------------------------------------')
                else:
                    df3 = df2.loc[df2['nan count'] == median].head(1)
                    df3['median'] = median
                    #print(df3.iloc[-1:])
                    df3['depth'] = depth
                    mlist.append(median)
                    rtlist.append(df3.iloc[-1]["day"])
                    #print('-------------------------------------------------------------------------')
            elif df2['nan count'].isin([median]).any().any() == False:
                if df2[:i+1][df2[:i+1]['nan count'].values>median].any().any() == False:
                    df3 = df2.loc[df2['day'] == len(df2["cum stuck count"])].tail(1)
                    df3.reset_index()
                    df3['median'] = median
                    #print(df3.iloc[-3:-2])
                    #print(df3.iloc[-2:-1])
                    #print('MEAN---------------------------------------------------------------------')
                    #print(df3.iloc[-1:])
                    df3['depth'] = depth
                    rtlist.append(0)
                    mlist.append(median)
                    #print('-------------------------------------------------------------------------')
                else:
                    c = df2[df2['nan count'] < median].tail(1).iloc[0]['nan count']
                    df3 = pd.concat([df2[df2['nan count'] == c].head(1), df2[df2['nan count'] > median].head(1), pd.concat([df2[df2['nan count'] == c].head(1),
                            df2[df2['nan count'] > median].head(1)]).mean().to_frame().transpose()], ignore_index=True)
                    df3.reset_index()
                    df3['median'] = median
                    #print(df3.iloc[-3:-2])
                    #print(df3.iloc[-2:-1])
                    #print('MEAN---------------------------------------------------------------------')
                    #print(df3.iloc[-1:])
                    df3['depth'] = depth
                    rtlist.append(df3.iloc[-1]["day"])
                    mlist.append(median)
                    #print('-------------------------------------------------------------------------')            
        df2.insert(len(df2.columns), 'median',mlist)
        df2.insert(len(df2.columns), 'residence time',rtlist)
        df2.to_excel("results" + "/" + dir + "/" + month[j] + "/" + str(depth) + "/" + 'summary.xlsx', index=False)
        #print('****************************')
        if df2['residence time'].max() == 0:
            df3.insert(len(df3.columns), 'residence time',str(np.nan))
        else:
            df3.insert(len(df3.columns), 'residence time',df2['residence time'].max())
        df4 = df4.append(df3.iloc[-1:], ignore_index=True)
    df4.to_excel("results" + "/" + dir + "/" + month[j] + "/" + 'residence_time.xlsx', index=False)
