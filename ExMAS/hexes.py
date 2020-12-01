# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 15:41:45 2020

@author: Marko
"""
#%%
import pandas as pd
import geopandas as gpd
import folium
from shapely.geometry import Polygon, Point, LineString
import pandas as pd
import geopandas as gpd
import folium
from IPython.display import Image
import selenium.webdriver
import shapely.geometry
import osmnx as ox
import pyproj
from geojson.feature import Feature, FeatureCollection
from shapely.geometry import Point, Polygon, LineString
import h3
import json

#%%
import os
cwd = os.getcwd()
os.chdir(os.path.join(cwd,'..'))
import ExMAS.main
import ExMAS.utils

from ExMAS.utils import inData as inData

#%%
params = ExMAS.utils.get_config('ExMAS/data/configs/my_config.json') # load the default 
params.paths.dumps = 'hexes'
params.times.patience = 1200
params.simTime = 4
params.parallel.nThread = 4
params.parallel.nReplications = 20
#%%
inData = ExMAS.utils.load_G(inData, params, stats=True)  # download the graph
inData = ExMAS.utils.load_albatross_csv(inData, params, sample=True)
#%%
inData = ExMAS.main(inData, params, plot = True)
#%%
resop = pd.DataFrame(index =range(len(inData.sblts.requests)))
resop = resop.fillna(0)

def plot_hex_map(inData,res, APERTURE_SIZE = 9, threshold = 1, col = 'change_in_u'):
    res['change_in_u'] = inData.sblts.requests.u-inData.sblts.requests.u_sh
    orig = inData.requests.origin
    res['x'] = orig.reset_index(drop=True)
    res['y'] = orig.reset_index(drop=True)
    res['x'] = res['x'].apply(lambda row: inData.nodes.loc[row].x)
    res['y'] = res['y'].apply(lambda row: inData.nodes.loc[row].y)
    res['hex_o_{}'.format(APERTURE_SIZE)] = res.apply(lambda row: h3.geo_to_h3(row.y,row.x,APERTURE_SIZE),axis = 1) 
    trips = res
    
    col_geom = 'hex_o_{}'.format(APERTURE_SIZE)
    hexes = pd.Series(list(set(list(trips[col_geom].unique())+list(trips[col_geom].unique())))).to_frame(col_geom)
    hexes = hexes.set_index(col_geom)
    hexes[col_geom] = hexes.index.copy()
    hexes['nobs'] = trips.groupby(col_geom).size()
    hexes = hexes[hexes['nobs']>threshold]
    hexes['geom'] = hexes.apply(lambda x: {"type": "Polygon","coordinates": [h3.h3_to_geo_boundary(h = x[col_geom], geo_json = True)]}, axis = 1)
    aggr = trips.groupby(col_geom)[col]
    hexes['nobs'] = aggr.size()
    hexes[col] = aggr.mean()/60
    hexes[col+'_std'] = aggr.std()
    list_features = []
    for i, row in hexes.iterrows():
        feature = Feature(geometry = row["geom"],
                          id = row[col_geom],
                          properties = {"resolution": 9})
        list_features.append(feature)

    feat_collection = FeatureCollection(list_features)
    geojson_hexes = json.dumps(feat_collection)
    CENTER = list(inData.nodes.loc[inData.stats.center][['y','x']].values)
    tile = 'cartodbpositron'
    base_map = folium.Map(location=CENTER, zoom_start=13,tiles=tile, zoomControl =  False)
    bins = [0, 1, 2, 3, 5, 10, 15, 20, 25, 114]
    m = folium.Choropleth(geo_data = geojson_hexes,data = hexes, columns = [col_geom,col],key_on ="feature.id",
            fill_color='Blues', control = False, bins = bins,
            fill_opacity=0.7, line_opacity=0.1).add_to(base_map)
    return base_map
    
plot_hex_map(inData, resop, APERTURE_SIZE = 9)
