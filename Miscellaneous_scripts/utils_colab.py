import json
import numpy as np
import pandas as pd
import os
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, date


def get_initial_parameters(path):
    with open(path) as json_file:
        data = json.load(json_file)
        config = DotMap(data)
    return config


def rand_node(df):
    # returns a random node of a graph
    return df.loc[random.choice(df.index)].name


def amended_generate_demand(_inData, _params, avg_speed=False):
    duration_time_fixed = _params.duration_in_minutes
    df = pd.DataFrame(index=np.arange(1, _params.nP + 1), columns=inData.passengers.columns)
    df.status = 0
    df.pos = df.apply(lambda x: rand_node(_inData.nodes), axis=1)
    _inData.passengers = df
    requests = pd.DataFrame(index=df.index, columns=_inData.requests.columns)

    distances = _inData.skim[_inData.stats['center']].to_frame().dropna()  # compute distances from center
    distances.columns = ['distance']
    distances = distances[distances['distance'] < _params.dist_threshold]
    distances['p_origin'] = distances['distance'].apply(lambda x:
                                                        math.exp(
                                                            _params.demand_structure.origins_dispertion * x))  # apply negative exponential distributions
    distances['p_destination'] = distances['distance'].apply(
        lambda x: math.exp(_params.demand_structure.destinations_dispertion * x))
    if _params.demand_structure.temporal_distribution == 'uniform' and duration_time_fixed == 0:
        treq = np.random.uniform(0, _params.simTime * 60 * 60,
                                 _params.nP)  # apply uniform distribution on request times
    if _params.demand_structure.temporal_distribution == 'uniform' and duration_time_fixed != 0:
        treq = np.random.uniform(0, duration_time_fixed * 60, _params.nP)  # apply uniform distribution on request times
    elif _params.demand_structure.temporal_distribution == 'normal':
        treq = np.random.normal(_params.simTime * 60 * 60 / 2,
                                _params.demand_structure.temporal_dispertion * _params.simTime * 60 * 60 / 2,
                                _params.nP)  # apply normal distribution on request times
    requests.treq = [_params.t0 + pd.Timedelta(int(_), 's') for _ in treq]
    requests.origin = list(
        distances.sample(_params.nP, weights='p_origin', replace=True).index)  # sample origin nodes from a distribution
    requests.destination = list(distances.sample(_params.nP, weights='p_destination',
                                                 replace=True).index)  # sample destination nodes from a distribution

    requests['dist'] = requests.apply(lambda request: _inData.skim.loc[request.origin, request.destination], axis=1)
    while len(requests[requests.dist >= _params.dist_threshold]) > 0:
        requests.origin = requests.apply(lambda request: (distances.sample(1, weights='p_origin').index[0]
                                                          if request.dist >= _params.dist_threshold else request.origin),
                                         axis=1)
        requests.destination = requests.apply(lambda request: (distances.sample(1, weights='p_destination').index[0]
                                                               if request.dist >= _params.dist_threshold else request.destination),
                                              axis=1)
        requests.dist = requests.apply(lambda request: _inData.skim.loc[request.origin, request.destination], axis=1)
    requests['ttrav'] = requests.apply(lambda request: pd.Timedelta(request.dist, 's').floor('s'), axis=1)
    # requests.ttrav = pd.to_timedelta(requests.ttrav)
    if avg_speed:
        requests.ttrav = (pd.to_timedelta(requests.ttrav) / _params.speeds.ride).dt.floor('1s')
    requests.tarr = [request.treq + request.ttrav for _, request in requests.iterrows()]
    requests = requests.sort_values('treq')
    requests['pax_id'] = requests.index.copy()
    _inData.requests = requests
    _inData.passengers.pos = _inData.requests.origin
    return _inData


def alternate_kpis(dataset):
    dataset['Proportion_singles'] = dataset['SINGLE'] / dataset['nP']
    dataset['Proportion_pairs'] = dataset['PAIRS'] / dataset['nP']
    dataset['Proportion_triples'] = dataset['TRIPLES'] / dataset['nP']
    dataset['Proportion_triples_plus'] = (dataset['nP'] - dataset['SINGLE'] -
                                          dataset['PAIRS']) / dataset['nP']
    dataset['Proportion_quadruples'] = dataset['QUADRIPLES'] / dataset['nP']
    dataset['Proportion_quintets'] = dataset['QUINTETS'] / dataset['nP']
    dataset['Proportion_six_plus'] = dataset['PLUS5'] / dataset['nP']
    dataset['SavedVehHours'] = (dataset['VehHourTrav_ns'] - dataset['VehHourTrav']) / \
                               dataset['VehHourTrav_ns']
    dataset['AddedPasHours'] = (dataset['PassHourTrav'] - dataset['PassHourTrav_ns']) / \
                               dataset['PassHourTrav_ns']
    dataset['UtilityGained'] = (dataset['PassUtility'] - dataset['PassUtility_ns']) / \
                               dataset['PassUtility_ns']
    return dataset


def rename_files(path):
    all_files = glob.glob(path + '*.csv')

    all_files = [re.split("\\\\", file)[-1][:-4] for file in all_files]
    all_files = [x for x in all_files if x[:4] == 'city']

    for num, file in enumerate(all_files):
        os.rename(path + file + '.csv', path + 'demand_' + str(num) + '.csv')
        os.rename(path + 'result_' + file + '.csv', path + 'result_' + str(num) + '.csv')


def split_geo_positions(dataset, inData):
    positional_df = inData.nodes
    # positional_df['pos'] = positional_df.index
    positional_df = positional_df[['osmid', 'x', 'y']]
    dataset = dataset[['origin', 'destination', 'treq']].rename(columns={'treq': 'time_request'}).reset_index()
    origin = dataset[['origin', 'time_request']].copy()
    destination = dataset[['destination', 'time_request']].copy()
    origin[['x_geo', 'y_geo']] = pd.merge(origin, positional_df, left_on='origin', right_on='osmid')[['x', 'y']]
    destination[['x_geo', 'y_geo']] = pd.merge(destination, positional_df,
                                               left_on='destination', right_on='osmid')[['x', 'y']]
    return origin, destination


def create_space_grid(inData, number_of_points):
    x_boundaries = [min(inData.nodes.x), max(inData.nodes.x)]
    y_boundaries = [min(inData.nodes.y), max(inData.nodes.y)]
    x_grid = np.append(np.arange(x_boundaries[0], x_boundaries[1], (x_boundaries[1] - x_boundaries[0]) /
                                 number_of_points), (x_boundaries[1]))
    y_grid = np.append(np.arange(y_boundaries[0], y_boundaries[1], (y_boundaries[1] - y_boundaries[0]) /
                                 number_of_points), (y_boundaries[1]))
    return x_grid, y_grid


def add_time(time_date, timedelta):
    return (time_date + pd.Timedelta(minutes=timedelta)).time()


def create_time_space_grid(inData, number_of_space_points, parameters, interval=5):
    x_grid, y_grid = create_space_grid(inData, number_of_space_points)
    lower_boundary = pd.Timestamp(parameters.t0)
    if parameters.duration_in_minutes == 0:
        duration = 60
    else:
        duration = parameters.duration_in_minutes
    time_grid = [add_time(lower_boundary, t * interval) for t in range(int(duration / interval) + 1)]
    return [x_grid, y_grid, time_grid]


def apply_time_space_grid(dataset, grid):
    x_grid, y_grid, t_grid = grid
    space_map = torch.zeros_like(torch.empty(len(t_grid) - 1, len(y_grid) - 1, len(x_grid) - 1))
    dataset['time_request'] = dataset['time_request'].apply(lambda x: pd.Timestamp(x).time())
    for x_ind in range(len(x_grid) - 1):
        for y_ind in range(len(y_grid) - 1):
            for t_ind in range(len(t_grid) - 1):
                t = dataset.loc[(dataset['x_geo'] >= x_grid[x_ind]) & (dataset['x_geo'] < x_grid[x_ind + 1])
                                & (dataset['y_geo'] >= y_grid[y_ind]) & (dataset['y_geo'] < y_grid[y_ind + 1])
                                & (dataset['time_request'] >= t_grid[t_ind])
                                & (dataset['time_request'] < t_grid[t_ind + 1])].shape[0]
                space_map[t_ind, y_ind, x_ind] += t
    return space_map


def apply_space_grid(dataset, grid, x_name='x_geo', y_name='y_geo'):
    x_grid, y_grid = grid
    space_map = torch.zeros_like(torch.empty(len(y_grid) - 1, len(x_grid) - 1))
    for x_ind in range(len(x_grid) - 1):
        for y_ind in range(len(y_grid) - 1):
            t = dataset.loc[(dataset[x_name] >= x_grid[x_ind]) & (dataset[x_name] < x_grid[x_ind + 1])
                            & (dataset[y_name] >= y_grid[y_ind]) & (dataset[y_name] < y_grid[y_ind + 1])].shape[0]
            space_map[y_ind, x_ind] += t
    return space_map


def dataset_to_grid(list_dataframes, grid, space_time='space'):
    output = []
    if space_time == 'space':
        func = apply_space_grid
        if len(grid) != 2: raise Exception('Wrong grid input. Check arguments')
    elif space_time == 'both':
        func = apply_time_space_grid
        if len(grid) != 3: raise Exception('Wrong grid input. Check arguments')
    else:
        raise Exception('Wrong space_time argument. Choose either "space" or "both"')
    for j in list_dataframes:
        output.append(func(j, grid))
    return output


def merge_origins_destinations(map_origins, map_destinations):
    return [torch.stack([map_origins[i], map_destinations[i]], dim=0) for i in range(len(map_origins))]


def convert_results_to_tensor(results, val=False):
    if val:
        return [torch.FloatTensor(x.values[0]) for x in results]
    else:
        return [torch.FloatTensor(x) for x in results]


""" New approach """


def create_whole_inputs(dataset, inData, params):
    positional_df = inData.nodes
    positional_df = positional_df[['osmid', 'x', 'y']].reset_index(drop=True)
    dataset = dataset[['origin', 'destination', 'treq']].rename(columns={'treq': 'time_request'}).reset_index(drop=True)
    dataset[['x_origin', 'y_origin']] = pd.merge(dataset, positional_df, left_on='origin', right_on='osmid')[['x', 'y']]
    dataset[['x_destination', 'y_destination']] = pd.merge(dataset, positional_df,
                                                           left_on='destination', right_on='osmid')[['x', 'y']]
    dataset['x_trav'] = dataset['x_destination'] - dataset['x_origin']
    dataset['y_trav'] = dataset['y_destination'] - dataset['y_origin']
    start_time = pd.Timestamp(params.t0).time()
    t = [t.time() for t in pd.to_datetime(dataset['time_request'], format='%Y-%m-%d %H:%M:%S').tolist()]
    dataset['time_request'] = [(datetime.combine(date.min, x) - datetime.combine(date.min, start_time)).seconds for x in
                               t]
    return dataset


def whole_inputs_to_tensor(inputs):
    inputs = [t.drop(columns=['origin', 'destination']) for t in inputs]
    return [torch.FloatTensor(x.values) for x in inputs]


class WholeDataset(torch.utils.data.Dataset):
    def __init__(self, demand, labels):
        self.demand = demand
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.demand[index], self.labels[index]


def equalise_sets(demand, results, bins):
    equalised_sets = []
    combined = list(zip(demand, results))
    for i in range(len(bins) - 1):
        equalised_sets.append([x for x in combined if (x[1] >= bins[i]) & (x[1] < bins[i + 1])])
    min_len = min([len(x) for x in equalised_sets])
    equalised_sets = [random.sample(x, min_len) for x in equalised_sets]
    combined = []
    for j in range(len(equalised_sets)):
        combined += equalised_sets[j]
    demand, results = zip(*combined)
    return demand, results


""" Networks """


class FirstTryNet(nn.Module):
    def __init__(self, padding=1):
        super().__init__()
        self.padding = padding
        self.conv1 = nn.Conv2d(2, 32, 3, padding=self.padding)
        self.conv2 = nn.Conv2d(32, 128, 3, padding=self.padding)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=self.padding)
        self.batch1 = nn.BatchNorm2d(32)
        self.batch2 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(25600, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 32)
        self.fc4 = nn.Linear(32, 4)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.batch1(x)
        x = F.relu(self.conv2(x))
        x = self.batch2(x)
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class SimplifiedNet(nn.Module):
    def __init__(self, padding=1):
        super().__init__()
        self.padding = padding
        self.conv1 = nn.Conv2d(2, 8, 3, padding=self.padding, stride=1)
        self.conv2 = nn.Conv2d(8, 32, 3, padding=self.padding, stride=2)
        self.conv2_drop = nn.Dropout2d()
        self.batch1 = nn.BatchNorm2d(8)
        self.fc1 = nn.Linear(800, 256)
        self.fc2 = nn.Linear(256, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.batch1(x)
        x = F.relu(self.conv2_drop(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class FinalNet(nn.Module):
    def __init__(self, padding=0):
        super().__init__()
        self.padding = padding
        self.conv1 = nn.Conv2d(2, 8, 4, padding=self.padding, stride=2)
        self.conv2 = nn.Conv2d(8, 32, 3, padding=self.padding, stride=1)
        self.conv1_drop = nn.Dropout2d()
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(800, 256)
        self.fc2 = nn.Linear(256, 32)
        self.fc3 = nn.Linear(32, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(7)

    def forward(self, x):
        x = F.relu(self.conv2_drop(self.conv1(x)))
        x = self.avg_pool(x)
        x = F.relu(self.conv2_drop(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class NewNet(nn.Module):
    def __init__(self, padding=0):
        super().__init__()
        self.padding = padding
        self.conv1 = nn.Conv2d(1, 8, 4, padding=self.padding, stride=2)
        self.conv2 = nn.Conv2d(8, 32, 3, padding=self.padding, stride=1)
        self.conv1_drop = nn.Dropout2d()
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(800, 256)
        self.fc2 = nn.Linear(256, 32)
        self.fc3 = nn.Linear(32, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(7)

    def forward(self, x):
        x = F.relu(self.conv2_drop(self.conv1(x)))
        x = self.avg_pool(x)
        x = F.relu(self.conv2_drop(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class LinearBenchmarkNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(200, 1)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x


def train_net(net, criterion, optimizer, train_dataloader, test_dataloader,
              number_of_epochs, output_name=None, use_gpu=True, add_channel=False):
    train_loss = []
    test_loss = []
    for epoch in range(number_of_epochs):
        epoch_losses_train = []
        epoch_losses_test = []
        for i, data in enumerate(train_dataloader):
            inputs, kpis = data
            if use_gpu:
                inputs = inputs.cuda()
                kpis = kpis.cuda()
            if add_channel:
                inputs = inputs.unsqueeze(1)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, kpis)
            loss.backward()
            optimizer.step()

            epoch_losses_train.append(loss.item())

            x, y = next(iter(test_dataloader))
            if use_gpu:
                x = x.cuda()
                y = y.cuda()
            if add_channel:
                x = x.unsqueeze(1)
            loss = criterion(y, net(x))
            epoch_losses_test.append(loss.item())

        train_loss.append(np.mean(epoch_losses_train))
        test_loss.append(np.mean(epoch_losses_test))
        if epoch % 10 == 9:
            print(f' ####### Epoch: {epoch} #######')
            print(f'Train loss: {train_loss[epoch]:.4f}')
            print(f'Test loss: {test_loss[epoch]:.4f}')
            print('\n')
    plt.plot(train_loss, label='Train loss')
    plt.plot(test_loss, label='Test loss')
    plt.legend(title='Loss with respect to epoch')
    plt.show()
    if output_name is not None:
        plt.savefig(output_name + '.png')
        pd.DataFrame({'Train_loss': train_loss, 'Test_loss': test_loss}).to_excel(output_name + '.xlsx')


def scatterplot_network(net, dataloader, arg_number=0, use_gpu=True, add_channel=False):
    x, y = next(iter(dataloader))
    if use_gpu:
        x = x.cuda()
    if add_channel:
        x = x.unsqueeze(1)
    predicted = [t[arg_number].detach().numpy().tolist() for t in net(x).cpu()]
    actual = [t[arg_number].numpy().tolist() for t in y]
    plt.scatter(actual, predicted)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.show()


""" Just for the notebook """


def load_data_notebook(inData):
    path = 'C:/Users/szmat/Documents/GitHub/Studies/Projekt/results/temp_mini_notebook/'
    rename_files(path)
    demand, results = load_data(path)

    # Add geographical attributes and separate the data into origins and destinations
    origins = []
    destinations = []
    for item in demand:
        o, d = split_geo_positions(item, inData)
        origins.append(o)
        destinations.append(d)
    return origins, destinations, results


def create_heatmap(org_dest):
    t = org_dest.numpy()
    sns.heatmap(t, annot=True)


def nyc_grid_apply(demand, x_name_org='x_org', x_name_dest='x_dest', y_name_org='y_org', y_name_dest='y_dest',
                   x_num=21, y_num=11):
    x_min = min([min(np.concatenate([d[x_name_org].values, d[x_name_dest].values])) for d in demand])
    x_max = max([max(np.concatenate([d[x_name_org].values, d[x_name_dest].values])) for d in demand])
    y_min = min([min(np.concatenate([d[y_name_org].values, d[y_name_dest].values])) for d in demand])
    y_max = max([max(np.concatenate([d[y_name_org].values, d[y_name_dest].values])) for d in demand])
    x_grid = np.linspace(x_min, x_max, num=x_num)
    y_grid = np.linspace(y_min, y_max, num=y_num)
    o = [np.histogram2d(d[y_name_org], d[x_name_org], bins=[y_grid, x_grid])[0].astype(int) for d in demand]
    d = [np.histogram2d(d[y_name_dest], d[x_name_dest], bins=[y_grid, x_grid])[0].astype(int) for d in demand]
    o = [torch.from_numpy(x) for x in o]
    d = [torch.from_numpy(x) for x in d]
    return [torch.stack([x, y], dim=0) for x, y in zip(o, d)]
