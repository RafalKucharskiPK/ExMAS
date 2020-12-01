# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 10:48:04 2020

@author: Marko
"""
#%%
import os
import pprint
cwd = os.getcwd()
import main
import utils
from utils import inData as inData
import albatross as alb
#%%

params = utils.get_config('data/configs/my_config.json')
os.chdir(os.path.join(cwd,'..'))
         
inData = utils.load_G(inData, params)
#%%

#A = alb.read_postcodes(params)

inData = alb.albatross_import(inData, params)

#%%

inData = alb.albatross_process(inData,params)

#%%

inData = alb.generate_demand_albatross(inData, params, copy = True, sample = None)

#%%
alb.save_albatross_to_csv(inData,params)


