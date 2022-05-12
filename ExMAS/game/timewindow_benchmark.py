"""
Here we modify the core code to run benchamrk against the fixed window solution
"""


from ExMAS.main import *

def ExMAS_windows(_inData, params, plot=False):
    """
    main call
    :param _inData: input (graph, requests, .. )
    :param params: parameters
    :param plot: flag to plot charts for consecutive steps
    :return: inData.sblts.schedule - selecgted shared rides
    inData.sblts.rides - all ride candidates
    inData.sblts.res -  KPIs
    """
    _inData.logger = init_log(params)  # initialize console logger

    _inData = single_rides(_inData, params)  # prepare requests as a potential single rides
    degree = 1

    _inData = sblt_pairs_windows(_inData, params, plot=plot)
    degree = 2

    _inData.logger.info('Degree {} \tCompleted'.format(degree))

    if degree < params.max_degree:
        _inData = make_shareability_graph(_inData)

        while degree < params.max_degree and _inData.sblts.R[degree].shape[0] > 0:
            _inData.logger.info('trips to extend at degree {} : {}'.format(degree,
                                                                           _inData.sblts.R[degree].shape[0]))
            _inData = extend_degree_window(_inData, params, degree)
            degree += 1
            _inData.logger.info('Degree {} \tCompleted'.format(degree))
        if degree == params.max_degree:
            _inData.logger.info('Max degree reached {}'.format(degree))
            _inData.logger.info('Trips still possible to extend at degree {} : {}'.format(degree,
                                                                                          _inData.sblts.R[degree].shape[0]))
        else:
            _inData.logger.info(('No more trips to extend at degree {}'.format(degree)))


    _inData.sblts.rides = _inData.sblts.rides.reset_index(drop=True)  # set index
    _inData.sblts.rides['index'] = _inData.sblts.rides.index  # copy index

    if params.get('without_matching', False):
        return _inData  # quit before matching
    else:
        _inData = matching(_inData, params, plot=plot)
        _inData.logger.info('Calculations  completed')
        _inData = evaluate_shareability(_inData, params, plot=plot)

        return _inData





# ALGORITHM 1
def sblt_pairs_windows(_inData, sp, _print=False, process=True, check=False, plot=False):
    """
    Identifies pair-wise shareable trips S_ij, i.e. for which utility of shared ride is greater than utility of
    non-shared ride for both trips i and j.
    First S_ij.FIFO2 trips are identified, i.e. sequence o_i,o_j,d_i,d_j.
    Subsequently, from FIFO2 trips we identify LIFO2 trips, i.e.
    :param _inData: main data structure, with .skim (node x node dist matrix) , .requests (with origin, dest and treq)
    :param sp: .json populated dictionary of parameters
    :param _print: boolean flag to toggle silent mode
    :param process: boolean flag to calculate measures at the end of calulations
    :param check: run test to make sure results are consistent
    :param plot: plot matrices illustrating the shareability
    :return: _inData with .sblts
    """

    req = _inData.sblts.requests.copy()
    """
    VECTORIZED FUNCTIONS TO QUICKLY COMPUTE FORMULAS ALONG THE DATAFRAME
    """



    def utility_ns_i():
        # utility of non-shared trip i
        return sp.price * r.dist_i / 1000 + r.VoT_i * r.ttrav_i

    def utility_ns_j():
        # utility of non shared trip j
        return sp.price * r.dist_j / 1000 + r.VoT_j * r.ttrav_j

    def utility_sh_i():
        # utility of shared trip i
        return (sp.price * (1 - sp.shared_discount) * r.dist_i / 1000 +
                r.VoT_i * sp.WtS * (r.t_oo + sp.pax_delay + r.t_od + sp.delay_value * abs(r.delay_i)))

    def utility_sh_j():
        # utility of shared trip j
        return (sp.price * (1 - sp.shared_discount) * r.dist_j / 1000 +
                r.VoT_j * sp.WtS * (r.t_od + r.t_dd + sp.pax_delay +
                                   sp.delay_value * abs(r.delay_j)))

    def utility_i():
        # difference u_i - u_ns_i
        return (sp.price * r.dist_i / 1000 * sp.shared_discount
                + r.VoT_i * (r.ttrav_i - sp.WtS * (r.t_oo + r.t_od + sp.pax_delay + sp.delay_value * abs(r.delay_i))))

    def utility_j():
        # difference u_i - u_ns_i
        return (sp.price * r.dist_j / 1000 * sp.shared_discount
                + r.VoT_j * (r.ttrav_j - sp.WtS * (r.t_od + r.t_dd + sp.pax_delay + sp.delay_value * abs(r.delay_i))))

    def utility_i_LIFO():
        return (sp.price * r.dist_i / 1000 * sp.shared_discount
                + r.VoT_i * (r.ttrav_i - sp.WtS * (
                        r.t_oo + r.t_od + 2 * sp.pax_delay + r.t_dd + sp.delay_value * abs(r.delay_i))))

    def utility_j_LIFO():
        # difference
        return (sp.price * r.dist_i / 1000 * sp.shared_discount
                + r.VoT_j * (r.ttrav_i - sp.WtS * (r.t_od + sp.delay_value * abs(r.delay_i))))

    def utility_sh_i_LIFO():
        # utility of shared trip
        return (sp.price * (1 - sp.shared_discount) * r.dist_i / 1000 +
                r.VoT_i * sp.WtS * (r.t_oo + r.t_od + r.t_dd + 2 * sp.pax_delay + sp.delay_value * abs(r.delay_i)))

    def utility_sh_j_LIFO():
        # utility of shared trip
        return (sp.price * (1 - sp.shared_discount) * r.dist_j / 1000 +
                r.VoT_j * sp.WtS * (r.t_od + sp.delay_value * abs(r.delay_j)))

    def query_skim(r, _from, _to, _col, _filter=True):
        """
        returns trip times for given node pair _from, _to and stroes into _col of df
        :param r: current set of queries
        :param _from: column name in r designating origin
        :param _to: column name in r designating destination
        :param _col: name of column in 'r' where matrix entries are stored
        :param _filter: do we filter the skim for faster queries (used always apart from the last query for LIFO2)
        :return:
        """
        #
        if _filter:
            skim = the_skim.loc[
                r[_from].unique(), r[_to].unique()].unstack().to_frame()  # reduce the skim size for faster join
        else:
            skim = the_skim.unstack().to_frame()  # unstack and to_frame for faster column representation of matrix
        skim.index.names = ['o', 'd']  # unify names for join
        skim.index = skim.index.set_names("o", level=0)
        skim.index = skim.index.set_names("d", level=1)

        skim.columns = [_col]
        # requests now has also two indexes
        r = r.set_index([_to, _from], drop=False)
        r.index = r.index.set_names("o", level=0)
        r.index = r.index.set_names("d", level=1)


        return r.join(skim, how='left')  # get the travel times between origins

    def sp_plot(_r, r, nCall, title):
        # function to plot binary shareability matrix at respective stages
        _r[1] = 0  # init boolean column
        if nCall == 0:
            _r.loc[r.index, 1] = 1  # initialize for first call
            sizes['initial'] = sp.nP * sp.nP
        else:
            _r.loc[r.set_index(['i', 'j']).index, 1] = 1
        sizes[title] = r.shape[0]
        print(r.shape[0], '\t', title)
        mtx = _r[1].unstack().values
        axes[nCall].spy(mtx)
        axes[nCall].set_title(title)
        axes[nCall].set_xticks([])

    def check_me_FIFO():
        # does the asserion checks, on one single random row and on the whole dataser
        t = r.sample(1).iloc[0]
        assert the_skim.loc[t.origin_i, t.origin_j] == t.t_oo  # check travel times consistency
        assert the_skim.loc[t.origin_j, t.destination_i] == t.t_od
        assert the_skim.loc[t.destination_i, t.destination_j] == t.t_dd
        assert abs(int(
            nx.shortest_path_length(_inData.G, t.origin_i, t.origin_j,
                                    weight='length') / sp.avg_speed) - t.t_oo) < 2
        assert abs(int(nx.shortest_path_length(_inData.G, t.origin_j, t.destination_i,
                                               weight='length') / sp.avg_speed) - t.t_od) < 2
        assert (r.t_i - r.ttrav_i > -2).all()  # share time is not smaller than direct
        assert (r.t_j - r.ttrav_j > -2).all()
        assert (abs(r.delay_i) <= r.delta_i).all()  # is the delay within bounds
        assert (abs(r.delay_j) <= r.delta_j).all()
        assert (r.u_i <= utility_ns_i() * 1.01).all()  # do we have positive sharing utility
        assert (r.u_j <= utility_ns_j() * 1.01).all()  # do we have positive sharing utility
        # except:
        #     print('cos nie tak')
        #     r['u_ns_i'] = utility_ns_i()
        #     r['u_ns_j'] = utility_ns_j()
        #     return r[r.u_i >= utility_ns_i()], r[r.u_j >= utility_ns_j()]

    def check_me_LIFO():
        # does the asserion checks, on one single random row and on the whole dataser
        t = r.sample(1).iloc[0]
        try:
            assert the_skim.loc[t.origin_i, t.origin_j] == t.t_oo  # check travel times consistency
            assert the_skim.loc[t.origin_j, t.destination_j] == t.t_od
            assert the_skim.loc[t.destination_j, t.destination_i] == t.t_dd
            assert abs(int(
                nx.shortest_path_length(_inData.G, t.origin_i, t.origin_j,
                                        weight='length') / sp.avg_speed) - t.t_oo) < 2
            assert abs(int(nx.shortest_path_length(_inData.G, t.destination_j, t.destination_i,
                                                   weight='length') / sp.avg_speed) - t.t_dd) < 2
            assert (r.t_i + 3 - r.ttrav_i > 0).all()  # share time is not smaller than direct (3 seconds for rounding)
            assert (abs(r.delay_i) <= r.delta_i).all()  # is the delay within bounds
            assert (abs(r.delay_j) <= r.delta_j).all()
            assert (r.u_i <= utility_ns_i() * 1.01).all()  # do we have positive sharing utility
            assert (r.u_j <= utility_ns_j() * 1.01).all()  # do we have positive sharing utility
        except:
            pass
            print('pass')
        return r[r.delay_j >= r.delta_j]

    if plot:
        #matplotlib.rcParams['figure.figsize'] = [16, 4]
        fig, axes = plt.subplots(1, 4)
    else:
        _r = None

    sizes = dict()
    # MAIN CALULATIONS
    print('Initializing pairwise trip shareability between {0} and {0} trips.'.format(sp.nP)) if _print else None
    r = pd.DataFrame(index=pd.MultiIndex.from_product([req.index, req.index]))  # new df with a pariwise index
    print('creating combinations') if _print else None
    cols = ['origin', 'destination', 'ttrav', 'treq', 'delta', 'dist', 'VoT']
    r[[col + "_i" for col in cols]] = req.loc[r.index.get_level_values(0)][cols].set_index(r.index)  # assign columns
    r[[col + "_j" for col in cols]] = req.loc[r.index.get_level_values(1)][cols].set_index(r.index)  # time consuming


    r['i'] = r.index.get_level_values(0)  # assign index to columns
    r['j'] = r.index.get_level_values(1)
    r = r[~(r.i == r.j)]  # remove diagonal
    print(r.shape[0], '\t nR*(nR-1)') if _print else None
    _inData.sblts.log.sizes[2] = {'potential': r.shape[0]}

    # first condition (before querying any skim)
    # TESTED - WORKS!#
    #if sp.horizon > 0:
    #    r = r[abs(r.treq_i - r.treq_j) < sp.horizon]
    q = '(treq_j + delta_j >= treq_i - delta_i)  & (treq_j - delta_j <= (treq_i + ttrav_i + delta_i))'
    #r = r.query(q)  # this reduces the size of matrix quite a lot
    if plot:
        _r = r.copy()
        _r[1] = 0
        sp_plot(_r, r, 0, 'departure compatibility')
    if len(r) == 0:
        _inData.sblts.FIFO2 = r  # early exit with empty result
        return _inData

    # make the skim smaller  (query only between origins and destinations)
    skim_indices = list(set(r.origin_i.unique()).union(r.origin_j.unique()).union(
        r.destination_j.unique()).union(r.destination_j.unique()))  # limit skim to origins and destination only
    the_skim = _inData.skim.loc[skim_indices, skim_indices].div(sp.avg_speed).astype(int)
    _inData.the_skim = the_skim

    r = query_skim(r, 'origin_i', 'origin_j', 't_oo')  # add t_oo to main dataframe (r)
    q = '(treq_i + t_oo + delta_i >= treq_j - delta_j) & (treq_i + t_oo - delta_i <= treq_j + delta_j)'
    #r = r.query(q)  # can we arrive at origin of j within his time window?

    # now we can see if j is reachebale from i with the delay acceptable for both
    # determine delay for i and for j (by def delay/2, otherwise use bound of one delta and remainder for other trip)
    r['delay'] = r.treq_i + r.t_oo - r.treq_j
    r['delay_i'] = r.apply(lambda x: min(abs(x.delay / 2), x.delta_i, x.delta_j) * (1 if x.delay < 0 else -1), axis=1)
    r['delay_j'] = r.delay + r.delay_i

    #r = r[abs(r.delay_j) <= r.delta_j / sp.delay_value]  # filter for acceptable
    #r = r[abs(r.delay_i) <= r.delta_i / sp.delay_value]
    if plot:
        sp_plot(_r, r, 1, 'origins shareability')
    if len(r) == 0:
        _inData.sblts.FIFO2 = r  # early exit with empty result
        return _inData

    r = query_skim(r, 'origin_j', 'destination_i', 't_od')

    def utility_window_i():
        return (abs(r.t_oo+r.t_od - r.ttrav_i) < sp.max_detour) & (abs(r.delay) < sp.max_delay)

    def utility_window_j():
        return (abs(r.t_od+r.t_dd - r.ttrav_j) < sp.max_detour) & (abs(r.delay) < sp.max_delay)

    r = r[utility_window_i() > 0]  # and filter only for positive utility
    if plot:
        sp_plot(_r, r, 2, 'utility for i')
    if len(r) == 0:
        _inData.sblts.FIFO2 = r
        return _inData
    rLIFO = r.copy()

    r = query_skim(r, 'destination_i', 'destination_j', 't_dd')  # and now see if it is attractive also for j
    # now we have times for all segments: # t_oo_i_j # t_od_j_i # dd_i_j
    # let's compute utility for j

    r = r[utility_window_j() > 0]
    if plot:
        sp_plot(_r, r, 3, 'utility for j')
    if len(r) == 0:
        _inData.sblts.FIFO2 = r
        return _inData

    # profitability
    r['ttrav'] = r.t_oo + r.t_od + r.t_dd + 2 * sp.pax_delay

    r = r.set_index(['i', 'j'], drop=False)  # done - final result of pair wise FIFO shareability



    if process:
        # lets compute some more measures
        r['kind'] = SbltType.FIFO2.value
        r['indexes'] = r.apply(lambda x: [int(x.i), int(x.j)], axis=1)
        r['indexes_orig'] = r.indexes
        r['indexes_dest'] = r.apply(lambda x: [int(x.i), int(x.j)], axis=1)

        r['u_i'] = utility_sh_i()
        r['u_j'] = utility_sh_j()

        r['t_i'] = r.t_oo + r.t_od + sp.pax_delay
        r['t_j'] = r.t_od + r.t_dd + sp.pax_delay
        r['delta_ij'] = r.apply(lambda x: x.delta_i - sp.delay_value * abs(x.delay_i) - (x.t_i - x.ttrav_i), axis=1)
        r['delta_ji'] = r.apply(lambda x: x.delta_j - sp.delay_value * abs(x.delay_j) - (x.t_j - x.ttrav_j), axis=1)
        r['delta'] = r[['delta_ji', 'delta_ij']].min(axis=1)
        r['u_pax'] = r['u_i'] + r['u_j']
        check_me_FIFO() if check else None

    _inData.sblts.FIFO2 = r.copy()
    del r

    # LIFO2
    r = rLIFO
    r = query_skim(r, 'destination_j', 'destination_i', 't_dd')  # set different sequence of times
    r.t_od = r.ttrav_j

    def utility_window_i_LIFO():
        return (abs(r.t_oo+r.t_od+r.t_dd - r.ttrav_i) < sp.max_detour) & (abs(r.delay_i) < sp.max_delay)

    def utility_window_j_LIFO():
        return (abs(r.t_od - r.ttrav_j) <sp.max_detour) & (abs(r.delay_j) < sp.max_delay)




    r = r[utility_window_i_LIFO() > 0]
    r = r[utility_window_j_LIFO() > 0]
    r = r.set_index(['i', 'j'], drop=False)
    r['ttrav'] = r.t_oo + r.t_od + r.t_dd + 2 * sp.pax_delay
    # if sp.profitability:
    #    q = '(1 - ttrav/(ttrav_i+ttrav_j)) >{}'.format(sp.shared_discount)
    #    r = r.query(q) #only profitable trips

    if plot:
        print(r.shape[0], '\tLIFO pairs') if _print else None
        sizes['LIFO'] = r.shape[0]

    if r.shape[0] > 0 and process:
        r['kind'] = SbltType.LIFO2.value

        r['indexes'] = r.apply(lambda x: [int(x.i), int(x.j)], axis=1)
        r['indexes_orig'] = r.indexes
        r['indexes_dest'] = r.apply(lambda x: [int(x.j), int(x.i)], axis=1)

        r['u_i'] = utility_sh_i_LIFO()
        r['u_j'] = utility_sh_j_LIFO()

        r['t_i'] = r.t_oo + r.t_od + r.t_dd + 2 * sp.pax_delay
        r['t_j'] = r.t_od
        r['delta_ij'] = r.apply(
            lambda x: x.delta_i - sp.delay_value * abs(x.delay_i) - (x.t_oo + x.t_od + x.t_dd - x.ttrav_i), axis=1)
        r['delta_ji'] = r.apply(lambda x: x.delta_j - sp.delay_value * abs(x.delay_j), axis=1)
        r['delta'] = r[['delta_ji', 'delta_ij']].min(axis=1)
        r['u_pax'] = r['u_i'] + r['u_j']
        check_me_LIFO() if check else None

    _inData.sblts.LIFO2 = r.copy()
    _inData.sblts.pairs = pd.concat([_inData.sblts.FIFO2, _inData.sblts.LIFO2],
                                    sort=False).set_index(['i', 'j', 'kind'],
                                                          drop=False).sort_index()
    _inData.sblts.log.sizes[2]['feasible'] = _inData.sblts.LIFO2.shape[0] + _inData.sblts.FIFO2.shape[0]
    _inData.sblts.log.sizes[2]['feasibleFIFO'] = _inData.sblts.FIFO2.shape[0]
    _inData.sblts.log.sizes[2]['feasibleLIFO'] = _inData.sblts.LIFO2.shape[0]

    for df in [_inData.sblts.FIFO2.copy(), _inData.sblts.LIFO2.copy()]:
        if df.shape[0] > 0:
            df['u_paxes'] = df.apply(lambda x: [x.u_i, x.u_j], axis=1)
            df['u_veh'] = df.ttrav
            df['times'] = df.apply(
                lambda x: [x.treq_i + x.delay_i, x.t_oo + sp.pax_delay, x.t_od, x.t_dd], axis=1)

            df = df[RIDE_COLS]

            _inData.sblts.rides = pd.concat([_inData.sblts.rides, df], sort=False)

    print(float(r.shape[0]) / (sp.nP * (sp.nP - 1)), ' total gain') if _print else None
    if plot:

        if 'figname' in sp.keys():
            plt.savefig(sp.figname)
        #matplotlib.rcParams['figure.figsize'] = [4, 4]
        fig, ax = plt.subplots()
        pd.Series(sizes).plot(kind='barh', ax=ax, color='black') if plot else None
        ax.set_xscale('log')
        ax.invert_yaxis()
        plt.show()

    return _inData

# ALGORITHM 2/3
def extend_degree_window(_inData, sp, degree, _print=False):
    R = _inData.sblts.R

    dist_dict = _inData.sblts.requests.dist.to_dict()  # non-shared travel distances
    ttrav_dict = _inData.sblts.requests.ttrav.to_dict()  # non-shared travel times
    treq_dict = _inData.sblts.requests.treq.to_dict()  # non-shared travel times
    VoT_dict = _inData.sblts.requests.VoT.to_dict()  # non-shared travel distances

    _inData.sblts.log.sizes[degree + 1] = dict()
    nPotential = 0
    retR = list()  # output

    for _, r in R[degree].iterrows():  # iterate through all rides to extend
        newtrips, nSearched = extend_window(r,_inData.sblts.S, R, sp, degree, dist_dict, ttrav_dict, treq_dict, VoT_dict)
        retR.extend(newtrips)
        nPotential += nSearched

    df = pd.DataFrame(retR, columns=['indexes', 'indexes_orig', 'u_pax', 'u_veh', 'kind',
                                     'u_paxes', 'times', 'indexes_dest'])

    df = df[RIDE_COLS]
    df = df.reset_index()
    print(df.shape[0], ' feasible extensions found') if _print else None
    _inData.sblts.log.sizes[degree + 1]['potential'] = nPotential
    _inData.sblts.log.sizes[degree + 1]['feasible'] = df.shape[0]

    _inData.sblts.R[degree + 1] = df
    _inData.sblts.rides = pd.concat([_inData.sblts.rides, df], sort=False)
    print(_inData.sblts.log.sizes[degree + 1]) if _print else None
    if df.shape[0] > 0:
        assert_extension(_inData, sp, degree + 1)

    return _inData




def extend_window(r, S, R, sp, degree, dist_dict, ttrav_dict, treq_dict, VoT_dict):
    # deptimefun = lambda dep: sum([abs(dep + delay) for delay in delays]) #total
    deptimefun = lambda dep: max([abs(dep + delay) for delay in delays])  # minmax
    deptimefun = np.vectorize(deptimefun)
    accuracy = 10
    retR = list()
    potential = 0
    for extension in enumerate_ride_extensions(r, S):  # all possible extensions
        Eplus, Eminus, t, kind = list(), list(), list(), None
        E = dict()  # star extending r with q
        indexes_dest = r.indexes_dest.copy()
        potential+=1

        for trip in extension:  # E = Eplus + Eminus
            t = R[2].loc[trip]
            E[t.i] = t  # trips lookup table to determine times
            if t.kind == 20:  # to determine destination sequence
                Eplus.append(indexes_dest.index(t.i))  # FIFO
            elif t.kind == 21:
                Eminus.append(indexes_dest.index(t.i))  # LIFO

        q = t.j

        if len(Eminus) == 0:
            kind = 0  # pure FIFO
            pos = degree
        elif len(Eplus) == 0:
            kind = 1  # pure LIFO
            pos = 0
        else:
            if min(Eminus) > max(Eplus):
                pos = min(Eminus)
                kind = 2
            else:
                kind = -1

        if kind >= 0:  # feasible ride
            re = DotMap()  # new extended ride
            re.indexes = r.indexes + [q]
            re.indexes_orig = re.indexes
            indexes_dest.insert(pos, q)  # new destination order
            re.indexes_dest = indexes_dest

            # times[1] = oo, times[2] = od, times[3]=dd

            new_time_oo = [E[re.indexes_orig[-2]].times[1]]  # this is always the case

            if pos == degree:  # insert as last destination
                new_time_od = [E[re.indexes_dest[0]].times[2]]
                new_time_dd = [E[re.indexes_dest[-2]].times[3]]
                new_times = [r.times[0:degree] +
                             new_time_oo +
                             new_time_od +
                             r.times[degree + 1:] +
                             new_time_dd]

            elif pos == 0:  # insert as first destination
                new_times = [r.times[0:degree] +
                             E[re.indexes_orig[-2]].times[1:3] +
                             [E[re.indexes_dest[1]].times[3]] +
                             r.times[degree + 1:]]

            else:
                new_time_od = [E[re.indexes_dest[0]].times[2]]
                new_times = r.times[0:degree] + new_time_oo + new_time_od
                if len(r.times[degree + 1:degree + 1 + pos - 1]) > 0:  # not changed sequence before insertion
                    new_times += r.times[degree + 1:degree + 1 + pos - 1]  # only for degree>3

                # insertion
                new_times += [E[re.indexes_dest[pos - 1]].times[-1]]  # dd
                new_times += [E[re.indexes_dest[pos + 1]].times[-1]]  # dd

                if len(r.times[(degree + 1 + pos):]) > 0:  # not changed sequence after insertion
                    new_times += r.times[degree + 1 + pos:]  # only for degree>3

                new_times = [new_times]

            new_times = new_times[0]
            re.times = new_times

            # detrmine utilities
            dists = [dist_dict[_] for _ in re.indexes]  # distances
            ttrav_ns = [ttrav_dict[_] for _ in re.indexes]  # non shared travel times
            VoT = [VoT_dict[_] for _ in re.indexes]  # VoT of sharing travellers
            # shared travel times
            ttrav = [sum(new_times[i + 1:degree + 2 + re.indexes_dest.index(re.indexes[i])]) for i in
                     range(degree + 1)]

            # first assume null delays
            feasible_flag = True
            for i in range(degree + 1):
                if trip_sharing_utility_window(sp, dists[i], 0, ttrav[i], ttrav_ns[i], VoT[i]) < 0:
                    feasible_flag = False
                    break
            if feasible_flag:
                # determine optimal departure time (if feasible with null delay)
                treq = [treq_dict[_] for _ in re.indexes]  # distances

                delays = [new_times[0] + sum(new_times[1:i]) - treq[i] for i in range(degree + 1)]

                dep_range = int(max([abs(_) for _ in delays]))

                if dep_range == 0:
                    x = [0]
                else:
                    x = np.arange(-dep_range, dep_range, min(dep_range, accuracy))
                d = (deptimefun(x))

                delays = [abs(_ + x[np.argmin(d)]) for _ in delays]
                # if _print:
                #    pd.Series(d, index=x).plot()  # option plot d=f(dep)
                u_paxes = list()

                for i in range(degree + 1):

                    u_paxes.append(trip_sharing_utility_window(sp, dists[i], delays[i], ttrav[i], ttrav_ns[i],VoT[i]))
                    if u_paxes[-1] < 0:
                        feasible_flag = False
                        break
                if feasible_flag:
                    re.u_paxes = [shared_trip_utility(sp, dists[i], delays[i], ttrav[i], VoT[i]) for i in range(degree + 1)]
                    re.pos = pos
                    re.times = new_times
                    re.u_pax = sum(re.u_paxes)
                    re.u_veh = sum(re.times[1:])
                    if degree > 4:
                        re.kind = 100
                    else:
                        re.kind = 10 * (degree + 1) + kind
                    retR.append(dict(re))

    return retR, potential

def trip_sharing_utility_window(sp, dist, dep_delay, ttrav, ttrav_ns, VoT):
    # trip sharing utility for a trip, trips are shared only if this is positive.
    # difference
    return (abs(ttrav - ttrav_ns) < sp.max_detour) & (abs(dep_delay) < sp.max_delay)






