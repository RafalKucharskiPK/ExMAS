import itertools

def algo_TNE(inData, price_column ='uniform_split'):
    rm = inData.sblts.rides_multi_index
    rides = inData.sblts.rides
    rm['price_single'] = rm.apply(
        lambda r: rm[(rm.traveller == r.traveller) & (rm.shared == False)][price_column].max(), axis=1)

    rm['surplus'] = rm.price_single - rm[price_column]
    rm['pruned_user'] = (rm[price_column] < rm.price_single) | (rm.shared == False)
    pruned = rm.groupby('ride')['pruned_user'].min().to_frame('pruned')
    pruned['ride'] = pruned.index
    rides['pruned'] = pruned['pruned']
    rm = rm.join(pruned['pruned'], on='ride')

    inData.sblts.rides_multi_index = rm
    inData.sblts.rides = rides

    return inData

def algo_HERMETIC(inData,  price_column ='uniform_split'):
    # checks if the groups are hermetic and returns only hermetic ones ('pruned' == True
    rm = inData.sblts.rides_multi_index
    rides = inData.sblts.rides

    # rides['indexes_set'] = rides.indexes.apply(set)  # moved to prapare_PoA
    #
    # def givemesubsets(row):
    #     # returns list of all the subgroup indiced contained in a ride
    #     return rides[rides.indexes_set.apply(lambda x: x.issubset(row.indexes))].index.values
    #
    # rides['subgroups'] = rides.apply(givemesubsets, axis=1)

    def hermetic(ride):
        #
        G = ride.name
        prices_G = rm.loc[G, :][[price_column]]
        hermetic = True
        for H in ride.subgroups:
            # determine whether there exists at least one traveller who wants to use G rather than H
            df = rm.loc[H, :][[price_column]].join(prices_G, lsuffix='_H', rsuffix='_G')
            df['surplus'] = df.iloc[:, 0] - df.iloc[:, 1]
            if df.surplus.max() < 0:
                hermetic = False
                inData.logger.info('non-hermetic group '+str(G))
                break
        return hermetic

    rides['hermetic'] = rides.apply(hermetic,axis = 1)
    rides['pruned'] = rides['hermetic']

    inData.sblts.rides_multi_index = rm
    inData.sblts.rides = rides

    return inData


def algo_RUE(inData, price_column ='uniform_split'):
    # deterimnes set of mergeable groups
    rm = inData.sblts.rides_multi_index
    rides = inData.sblts.rides
    lsuffix = '_G1G2'
    rsuffix = '_G'
    mergeable = list()
    indexes_set = rides.indexes_set



    for i in rides.index:
        ride = rides.loc[i]
        subgroups = ride.subgroups
        costs_G = rm.loc[ride.name, :][['traveller', price_column]]
        for G1, G2 in itertools.combinations(subgroups, 2):
            # G1, G2 = subgroup_pair[0],subgroup_pair[1]
            if indexes_set[G1].isdisjoint(indexes_set[G2]):  # disjoint
                if indexes_set[G1].union(indexes_set[G2]) == ride.indexes_set:  # sum to G

                    df = rm.loc[[G1, G2], :][['traveller', price_column]]
                    df = df.join(costs_G, on='traveller', lsuffix=lsuffix, rsuffix=rsuffix)
                    df['surplus'] = df[price_column + lsuffix] - df[price_column + rsuffix]
                    if df.surplus.min() >= 0 and df.surplus.max() > 0:
                        mergeable.append([G1, G2])
                        inData.logger.info('Mergeable groups: {}-{}'.format(G1, G2))
    inData.sblts.mutually_exclusives = mergeable
    inData.sblt.rides.pruned = 'True'  # there is no pruning in this algorithm
    return inData


def algo_RSIE(inData, price_column='uniform_split', _print=False):
    rm = inData.sblts.rides_multi_index
    rides = inData.sblts.rides
    lsuffix = '_x'
    rsuffix = '_y'
    unstables = list()
    indexes_set = rides.indexes_set

    def are_unstable(G1, G2):
        for i in indexes_set[G1]:  # examine each i in G1
            G2s_with_i = G2s.union({i})  # move i to G2
            for r in rides[rides.indexes_set == G2s_with_i].index:  # loop over rides where i joining G2
                if rm.loc[r, i].cost_user < rm.loc[G1, i].cost_user:  # condition 1 (i want to join G1)
                    costs_of_G2_with_i = rm.loc[pd.IndexSlice[r, G2s], :][
                        ['traveller', price_column]]  # costs for travellers in G2 with i
                    compare = pd.merge(costs_of_G2, costs_of_G2_with_i, on=['traveller'])
                    # df = costs_of_G2.join(costs_of_G2_with_i[price_column], lsuffix=lsuffix, rsuffix=rsuffix) # merge for comparison
                    compare['surplus'] = compare[price_column + lsuffix] - compare[price_column + rsuffix]
                    if compare.surplus.min() >= 0:
                        if _print:
                            print('Group1:', G1, G1s)
                            print('Group2:', G2, G2s)
                            print('Moving traveller:', i)
                            print('Group2 with i:', r, G2s_with_i)
                            print('Costs for i in G1:', rm.loc[G1, i].cost_user)
                            print('Costs for i in G2:', rm.loc[r, i].cost_user)
                            print('Costs for G2 without i \n ', costs_of_G2[price_column])
                            print('Costs for G2 with i \n ', costs_of_G2_with_i[price_column])
                        return True
        return False

    for G2 in rides.index:  # loop over ride
        G2s = indexes_set[G2]  # travellers in G2
        costs_of_G2 = rm.loc[G2, :][['traveller', price_column]]  # costs of group G2 before joining
        for G1 in rides.index:  # pairs
            G1s = indexes_set[G1]  # travellers in G1
            if indexes_set[G1].isdisjoint(indexes_set[G2]):  # if rides are disjoint
                if are_unstable(G1, G2):
                    unstables.append([G1, G2])
    inData.sblts.mutually_exclusives = unstables

    return inData


