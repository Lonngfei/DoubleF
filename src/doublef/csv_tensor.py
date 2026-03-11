import numpy as np
import torch
import pandas as pd

class CsvTorch(object):
    def __init__(self, ptt_npy, stt_npy, phase_csv, device):
        self.ptt_npy = ptt_npy
        self.stt_npy = stt_npy
        self.phase_csv = phase_csv
        self.device = device

    def load_tt(self):
        p_npy = np.load(self.ptt_npy)
        s_npy = np.load(self.stt_npy)
        self.p_tt = torch.tensor(p_npy, dtype=torch.float32, device=self.device)
        self.s_tt = torch.tensor(s_npy, dtype=torch.float32, device=self.device)
        return self.p_tt, self.s_tt

    def generate_station_data(self):
        self.df = pd.read_csv(self.phase_csv)
        if len(self.df) == 0:
            return False
        self.df['net_sta'] = self.df['network'].astype(str) + '_' + self.df['station'].astype(str)
        self.unique_stations = self.df[['net_sta', 'latitude', 'longitude', 'elevation']].drop_duplicates()
        self.unique_stations.reset_index(drop=True, inplace=True)
        self.unique_stations['id'] = self.unique_stations.index
        self.station_tensor = torch.tensor(self.unique_stations[['id', 'latitude', 'longitude', 'elevation']].to_numpy(), dtype=torch.float32, device=self.device)
        self.sta_dict = {row['id']: row['net_sta'].split('_') for _, row in self.unique_stations.iterrows()}
        self.df = pd.merge(self.df, self.unique_stations[['net_sta', 'id']], on='net_sta', how='left')
        self.num_stations = self.unique_stations['id'].nunique()
        self.max_id = self.station_tensor.shape[0]
        return self.station_tensor, self.num_stations, self.sta_dict

    def generate_initial_point(self):
        df_P = self.df[self.df["phasetype"] == "P"]
        df_P = df_P[df_P['RelativeTime'] < 86400]
        initial_point = torch.tensor(df_P[['latitude', 'longitude', 'RelativeTime']].values, dtype=torch.float32, device=self.device)
        number = initial_point.shape[0]
        constant_depth = torch.full((number, 1), 0, device=self.device, dtype=torch.float32)
        initial_point = torch.hstack((initial_point[:, :2], constant_depth, initial_point[:, 2:]))
        #initial_point = initial_point.repeat(5, 1)
        indices = torch.argsort(initial_point[:, 3])
        initial_point = initial_point[indices]
        zero_matrix = torch.zeros((initial_point.shape[0], 1), dtype=torch.float32, device=self.device)
        score_matrix = torch.hstack((initial_point, zero_matrix))
        return initial_point, score_matrix

    def generate_phase_tensor(self, df_subset):
        df_subset = df_subset[df_subset['RelativeTime'] < 86400].copy()
        df_subset['RelativeTime_Hour'] = (df_subset['RelativeTime'] // 5).astype(int)

        values = df_subset[["id", "RelativeTime_Hour", "RelativeTime", "Probability", "Amplitude"]].to_numpy()
        tensor_data = torch.tensor(values, dtype=torch.float32, device=self.device)

        ids = tensor_data[:, 0].to(torch.long)
        hours = tensor_data[:, 1].to(torch.long)
        rel_time = tensor_data[:, 2]
        prob = tensor_data[:, 3]
        amp = tensor_data[:, 4]

        df_subset['group_key'] = list(zip(ids.cpu().tolist(), hours.cpu().tolist()))
        group_counts = df_subset['group_key'].value_counts()
        max_rows = group_counts.max()

        time_bins = 86400 // 5

        result = torch.full(
            (max_rows, self.max_id, time_bins, 3),
            float('nan'),
            dtype=torch.float32,
            device=self.device
        )

        group_indices = {}
        row_indices = torch.empty(len(df_subset), dtype=torch.long, device=self.device)
        current_positions = {}

        for idx, (i, h) in enumerate(zip(ids.tolist(), hours.tolist())):
            key = (i, h)
            pos = current_positions.get(key, 0)
            row_indices[idx] = pos
            current_positions[key] = pos + 1

        indices = (
            row_indices,
            ids,
            hours,
        )

        result.index_put_(indices + (torch.tensor([0], device=self.device),), rel_time, accumulate=False)
        result.index_put_(indices + (torch.tensor([1], device=self.device),), prob, accumulate=False)
        result.index_put_(indices + (torch.tensor([2], device=self.device),), amp, accumulate=False)

        return result

    def generate_all_data(self):
        df_P = self.df[self.df['phasetype'] == 'P'].copy()
        df_S = self.df[self.df['phasetype'] == 'S'].copy()
        P_tensor = self.generate_phase_tensor(df_P)
        S_tensor = self.generate_phase_tensor(df_S)
        return P_tensor, S_tensor, len(df_P), len(df_S)




