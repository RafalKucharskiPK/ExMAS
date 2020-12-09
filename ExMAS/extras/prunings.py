import itertools

def algo_1(inData, price_column ='uniform_split'):
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

def algo_2(inData,  price_column ='uniform_split'):
    # checks if the groups are hermetic and returns only hermetic ones ('pruned' == True
    rm = inData.sblts.rides_multi_index
    rides = inData.sblts.rides

    rides['indexes_set'] = rides.indexes.apply(set)

    def givemesubsets(row):
        # returns list of all the subgroup indiced contained in a ride
        return rides[rides.indexes_set.apply(lambda x: x.issubset(row.indexes))].index.values

    rides['subgroups'] = rides.apply(givemesubsets, axis=1)

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
                break
        return hermetic

    rides['hermetic'] = rides.apply(hermetic,axis = 1)
    rides['pruned'] = rides['hermetic']

    inData.sblts.rides_multi_index = rm
    inData.sblts.rides = rides

    return inData


def algorithm_3(inData):
    # deterimnes set of mergeable groups
    rm = inData.sblts.rides_multi_index
    rides = inData.sblts.rides
    price_column = 'uniform_split'
    lsuffix = '_G1G2'
    rsuffix = '_G'
    mergeable = list()

    for i in rides.index:
        ride = rides.loc[i]
        subgroups = ride.subgroups
        indexes_set = rides.indexes_set
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
                        print(i, G1, G2)
    return mergeable


