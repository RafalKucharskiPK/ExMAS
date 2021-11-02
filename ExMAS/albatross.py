#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from shapely.geometry import Point
import os
import matplotlib
import matplotlib.pyplot as plt
import geopandas as gp
import seaborn as sns
import numpy as np
matplotlib.rcParams['figure.figsize'] = [8, 4]


# In[2]:

def read_postcodes(_params):
    PC4 = gp.read_file(_params.paths.postcodes)
    PC4 = PC4.to_crs(_params.crs)
    PC4.PC4 = PC4.PC4.astype(int)
    #PC4['centroid'] = PC4['geometry'].centroid
    return PC4


# In[ ]:


def nodes_gdf(df, crs):
    geometry = [Point(xy) for xy in zip(df.x, df.y)]
    #df = df.drop(['x', 'y'], axis=1)
    return gp.GeoDataFrame(df, crs, geometry=geometry)


def albatross_import(_inData,_params):
    from dotmap import DotMap
    # read the albratross .xtxt file and merge them into pd.DataFrame
    # read the postcodes and assing PC$ to each of nodes in the graph
    _inData.albatross = DotMap(_dynamic=False)
    for file in os.listdir(_params.paths.albatross):
        if file.endswith('.txt'):
            _inData.albatross[file.split(".")[0]]=pd.read_csv(os.path.join(_params.paths.albatross,file),sep='\t')
    df = pd.concat([_inData.albatross.AlbModel1,_inData.albatross.AlbModel2])
    df.PPC = df.PPC.fillna(0)
    df.PPC = df.PPC.astype(int)
    _inData.albatross.activities = df
    _inData.albatross.PC4 = read_postcodes(_params)[['PC4',"pc4_naam",'geometry']]
    _inData.albatross.gdf_nodes = gp.GeoDataFrame(_inData.nodes, geometry=gp.points_from_xy(_inData.nodes.x, _inData.nodes.y)).drop(['x', 'y'], axis=1)
    _inData.albatross.gdf_nodes = gp.tools.sjoin(_inData.albatross.gdf_nodes, _inData.albatross.PC4[['PC4','geometry']], how="left")
    return _inData


# In[3]:


def albatross_process(_inData,_params):
    # parse albatross activities into trips and filter 
    # pivots the activities data into trips data,
    # filter for the trip within the city,
    df = _inData.albatross.activities.copy()
    df['person_id']=(df.Hhid*10+df.Gend).astype(int) #unique person id
    fs = df.shift(-1)
    cols = df.columns 
    df.columns = [_+"_FROM" for _ in cols]
    cols = fs.columns 
    fs.columns = [_+"_TO" for _ in cols] #merge two subsequent activities into a trip
    trips = pd.concat([df,fs], axis = 1)
    
    codes = _inData.albatross.gdf_nodes.PC4.dropna().unique() # unique postcodes in G
    codes = set([int(_) for _ in codes])   
    print(_params.city,len(codes))   
    trips = trips[trips.PPC_TO.isin(codes) & trips.PPC_FROM.isin(codes)] #within areas
    trips = trips[trips.person_id_FROM == trips.person_id_TO] # the same household
    trips = trips[trips['BT_TO']>trips['ET_FROM']] # positive travel time
    _inData.albatross.trips = trips
    return _inData



# In[4]:


def generate_demand_albatross(_inData,_params, copy = True, sample = None):
    """
    generate demand (or sample of it)
    tweat albatross trips into MaaSSim requests
    populate the _inData.albratross.requests for both the simulation and sblt
    """
    if sample == -1:
        sample = _params.nP
    
    def make_time(df, col):   
        df[col] = df[col].astype(int)
        df = df[df[col].gt(0)] 
        df = df[df[col].lt(2400)]
        df[col] =  pd.to_datetime([str(int(_)).zfill(4) for _ in df[col].values],
                                format = "%H%M")
        return df        
    if sample:
        t = _inData.albatross.trips.sample(sample)
    else:
        t = _inData.albatross.trips
    pcs = _inData.albatross.gdf_nodes.groupby('PC4').osmid.apply(list)

    t['origin']= t.apply(lambda x: np.random.choice(pcs[x.PPC_TO]), axis=1) 
    t['destination']= t.apply(lambda x: np.random.choice(pcs[x.PPC_FROM]), axis=1)
    
    t['dist'] = [_inData.skim[t.origin][t.destination] for _, t in t.iterrows()] 
    
    #for i, request in t[t.dist>=params.dist_threshold].iterrows():
    #    #if request.dist >= params.dist_threshold:         #redraw for disconnected trips
    #    #    #while request.dist >= params.dist_threshold:
    #    #    request.origin = np.random.choice(pcs[t.loc[i].PPC_FROM])
    #    #    request.origin = np.random.choice(pcs[t.loc[i].PPC_TO])
    #    #    request.dist = _inData.skim[request.origin][request.destination]
    #     
    #    #    t.loc[i] = request 
    #t.ttrav = pd.to_timedelta(t.ttrav)
    t = make_time(t,'ET_FROM')
    t = make_time(t,'BT_TO')
    requests = t[['origin','destination','ET_FROM','BT_TO']]
    requests.columns = ['origin','destination','treq','tarr']
    requests['ttrav'] = requests.tarr - requests.treq
    _inData.albatross.requests = requests
    if copy:
        _inData.requests = requests
    _inData.albatross.sample_trips = t
    return _inData


# In[5]:


def save_albatross_to_csv(_inData, _params):
    _inData.albatross.requests.to_csv(os.path.join(_params.paths.albatross,
                                                   _params.city.split(",")[0]+"_requests.csv"))
    _inData.albatross.sample_trips.to_csv(os.path.join(_params.paths.albatross,
                                                       _params.city.split(",")[0]+"_demand.csv"))


# In[6]:


def load_albatross_csv(_inData, _params, sample = True):
    from utils.utils import generic_generator

    from MaaSSimpy.dataStructures import generate_passenger

    # loads the full csv of albatross for a given city
    # changes date for today
    # filters for simulation time (t0 hour + simTime)
    # samples the n 
    df = pd.read_csv(os.path.join(_params.paths.albatross,
                                    _params.city.split(",")[0]+"_requests.csv"),
                       index_col = 'Unnamed: 0')
    df['treq']=pd.to_datetime(df['treq'])
    df.treq = df.treq + (_params.t0.date() -  df.treq.iloc[0].date())
    df['tarr']=pd.to_datetime(df['tarr'])
    df.tarr = df.tarr + (_params.t0.date() -  df.tarr.iloc[0].date())
    #sample within simulation time
    df= df[df.treq.dt.hour>=_params.t0.hour]
    df = df[df.treq.dt.hour<=(_params.t0.hour+_params.simTime)]
 
    df['dist'] = df.apply(lambda request: _inData.skim.loc[request.origin,request.destination],axis=1)     
    df = df[df.dist< _params.dist_threshold]
    
    if sample:
        df = df.sample(_params.nP)
    
    df['ttrav_alb']= pd.to_timedelta(df.ttrav)
    
    df['ttrav'] = df.apply(lambda request: pd.Timedelta(request.dist,'s').floor('s'),axis=1)
    _inData.requests = df
    _inData.passengers = generic_generator(generate_passenger,_params.nP).reindex(_inData.requests.index)
    _inData.passengers.pos = _inData.requests.origin
    
    return _inData


# In[8]:


def full_albatross(_inData, _params):
    from ExMAS.utils import make_paths
    for city in _params.cities:
        _params.city = city
        make_paths()
        full_albatross_one(_inData,_params)

def full_albatross_one(_inData, _params):    
    _inData = load_G(_inData, stats = True) #download_G(inData) # download the graph for the 'params.city' and calc the skim matrices
    print('loaded')
    _inData = albatross_import(_inData,_params)
    print('imported')
    _inData = albatross_process(_inData,_params)
    print('processed')
    _inData = generate_demand_albatross(_inData,_params, sample = None)
    print('generated')
    _inData.requests = _inData.albratross.copy
    save_albatross_to_csv(_inData)






