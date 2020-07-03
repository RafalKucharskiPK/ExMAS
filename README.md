# ExMAS
> Exact Matching of Attractive Shared rides (ExMAS) for system-wide strategic evaluations
---

ExMAS allows you to match trips into attractive shared rides.

For a given:
* network (`osmnx` graph), 
* demand (microscopic set of trips $q_i = (o_i, d_i, t_i)$)
* parameters (behavioural, like _willingness-to-share_ and system like _discount_ for shared rides)

It computes:
* optimal set of shared rides (results of bipartite matching with a given objective)
* shareability graph
* set of all feasible rides
* KPIs of sharing
* trip sharing attributes 

ExMAS is a `python` based open-source package applicable to general networks and demand patterns.

If you find this code useful in your research, please consider citing: _Kucharski R. , Cats. O 2020. Exact matching of attractive shared rides (ExMAS) for system-wide strategic evaluations, Transportation Research Part B 139 (2020) 285-310_


[Quickstart tutorial](https://github.com/RafalKucharskiPK/ExMAS/blob/master/notebooks/ExMAS.ipynb)

----
Rafał Kucharski, TU Delft, 2020 r.m.kucharski (at) tudelft.nl








