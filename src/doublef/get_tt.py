import numpy as np
import pandas as pd
from pyrocko import cake
import csv
import torch
from scipy.interpolate import griddata
import os


def interpolate_torch_matrix(matrix, block_size=10000, device='cuda'):
    matrix_torch = torch.as_tensor(matrix, dtype=torch.float32, device=device)
    x = matrix_torch[:, 0]
    y = matrix_torch[:, 1]
    z = matrix_torch[:, 2]

    x_max = torch.max(x).item()
    y_max = torch.max(y).item()

    xi = torch.arange(0, x_max + 0.01, 0.01, device=device, dtype=torch.float32)
    yi = torch.arange(0, y_max + 0.01, 0.01, device=device, dtype=torch.float32)

    xi, yi = torch.meshgrid(xi, yi, indexing='ij')

    zi = torch.empty_like(xi, device=device, dtype=torch.float32)

    n_blocks_x = (xi.shape[0] + block_size - 1) // block_size
    n_blocks_y = (xi.shape[1] + block_size - 1) // block_size

    xi_cpu = xi.cpu().numpy()
    yi_cpu = yi.cpu().numpy()

    for i in range(n_blocks_x):
        for j in range(n_blocks_y):
            x_block = xi_cpu[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size]
            y_block = yi_cpu[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size]

            zi_block = griddata(
                (x.cpu().numpy(), y.cpu().numpy()),
                z.cpu().numpy(),
                (x_block, y_block),
                method='linear',
                fill_value=np.nan
            )

            zi[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size] = torch.tensor(zi_block,
                                                                                                        device=device)
    return zi


class TravelTime(object):
    def __init__(self, filename, tt_csv, ptt_npy, stt_npy, logger, sdepth_max=31, depth_step=1, distance_max=2.0,
                 distance_step=0.01):
        self.model = cake.load_model(filename)
        self.filename = filename
        self.sdepth_max = sdepth_max
        self.depth_step = depth_step
        self.distance_max = distance_max
        self.distance_step = distance_step
        self.distances = np.arange(0, distance_max, distance_step)
        self.tt_csv = tt_csv
        self.ptt_npy = ptt_npy
        self.stt_npy = stt_npy
        self.logger = logger
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def get_phasetime(self):
        p = cake.PhaseDef('p')
        s = cake.PhaseDef('s')
        P = cake.PhaseDef('P<(moho)')
        S = cake.PhaseDef('S<(moho)')
        Pn = cake.PhaseDef("Pv_(moho)p")
        Sn = cake.PhaseDef("Sv_(moho)s")
        p_phasetime, s_phasetime, P_phasetime, S_phasetime, Pn_phasetime, Sn_phasetime = [], [], [], [], [], []
        # get all phasetime with different depth and different phase
        self.logger.info(f'Starting calculate 1-D travel time')
        self.logger.info(f'The input model is {self.filename}')
        self.logger.info(f'The calculated theoretical tt-table is {self.tt_csv}')
        self.logger.info(f'The interpolated theoretical p-tt-table is {self.ptt_npy}')
        self.logger.info(f'The interpolated theoretical s-tt-table is {self.stt_npy}')
        self.logger.info(f'The max source depth in tt-table is {self.sdepth_max} km')
        self.logger.info(f'The interval of source depth in tt-table is {self.depth_step} km')
        self.logger.info(f'The max distance in tt-table is {self.distance_max}°')
        self.logger.info(f'The interval of distance in tt-table is {self.distance_step}°')
        for source_depth in np.arange(0, self.sdepth_max, self.depth_step):
            p_cash, s_cash, P_cash, S_cash, Pn_cash, Sn_cash = {}, {}, {}, {}, {}, {}
            for arrivalp in self.model.arrivals(self.distances, phases=p, zstart=source_depth * 1000):
                distance = arrivalp.x * cake.d2m / 1000
                p_cash[f'{distance:.3f}'] = arrivalp.t
            p_phasetime.append(p_cash)
            for arrivals in self.model.arrivals(self.distances, phases=s, zstart=source_depth * 1000):
                distance = arrivals.x * cake.d2m / 1000
                s_cash[f'{distance:.3f}'] = arrivals.t
            s_phasetime.append(s_cash)
            for arrivalP in self.model.arrivals(self.distances, phases=P, zstart=source_depth * 1000):
                distance = arrivalP.x * cake.d2m / 1000
                P_cash[f'{distance:.3f}'] = arrivalP.t
            P_phasetime.append(P_cash)
            for arrivalS in self.model.arrivals(self.distances, phases=S, zstart=source_depth * 1000):
                distance = arrivalS.x * cake.d2m / 1000
                S_cash[f'{distance:.3f}'] = arrivalS.t
            S_phasetime.append(S_cash)
            for arrivalPn in self.model.arrivals(self.distances, phases=Pn, zstart=source_depth * 1000):
                distance = arrivalPn.x * cake.d2m / 1000
                Pn_cash[f'{distance:.3f}'] = arrivalPn.t
            Pn_phasetime.append(Pn_cash)
            for arrivalSn in self.model.arrivals(self.distances, phases=Sn, zstart=source_depth * 1000):
                distance = arrivalSn.x * cake.d2m / 1000
                Sn_cash[f'{distance:.3f}'] = arrivalSn.t
            Sn_phasetime.append(Sn_cash)
        return p_phasetime, s_phasetime, P_phasetime, S_phasetime, Pn_phasetime, Sn_phasetime


    def get_tt(self):
        w = open(self.tt_csv, 'w')
        writer = csv.writer(w)
        writer.writerow(["sourcedepth", "phasetype", "truetype", "distance", "traveltime"])
        p_phasetime, s_phasetime, P_phasetime, S_phasetime, Pn_phasetime, Sn_phasetime = self.get_phasetime()
        for index, depth in enumerate(np.arange(0, self.sdepth_max, self.depth_step)):
            for x in self.distances:
                result = []
                distance = f'{x * cake.d2m / 1000:.3f}'
                present_in = []
                if distance in p_phasetime[index]:
                    present_in.append((p_phasetime[index][distance], 'p'))
                if distance in P_phasetime[index]:
                    present_in.append((P_phasetime[index][distance], 'Pg'))
                if distance in Pn_phasetime[index]:
                    present_in.append((Pn_phasetime[index][distance], 'Pn'))
                if len(present_in) == 1:
                    result = [depth, 'P', present_in[0][1], distance, f'{present_in[0][0]:.15f}']
                elif len(present_in) > 1:
                    min_value = 1000
                    for i in present_in:
                        if i[0] < min_value:
                            min_value = i[0]
                            result = [depth, 'P', i[1], distance, f'{min_value:.15f}']
                else:
                    result = [depth, 'P', np.nan, distance, np.nan]
                writer.writerow(result)

        for index, depth in enumerate(np.arange(0, self.sdepth_max, self.depth_step)):
            for x in self.distances:
                result = []
                distance = f'{x * cake.d2m / 1000:.3f}'
                present_in = []
                if distance in s_phasetime[index]:
                    present_in.append((s_phasetime[index][distance], 's'))
                if distance in S_phasetime[index]:
                    present_in.append((S_phasetime[index][distance], 'Sg'))
                if distance in Sn_phasetime[index]:
                    present_in.append((Sn_phasetime[index][distance], 'Sn'))
                if len(present_in) == 1:
                    result = [depth, 'S', present_in[0][1], distance, f'{present_in[0][0]:.15f}']
                elif len(present_in) > 1:
                    min_value = 1000
                    for i in present_in:
                        if i[0] < min_value:
                            min_value = i[0]
                            result = [depth, 'S', i[1], distance, f'{min_value:.15f}']
                else:
                    result = [depth, 'S', np.nan, distance, np.nan]
                writer.writerow(result)
        w.close()


    def run(self):
        self.get_tt()
        df = pd.read_csv(self.tt_csv)
        p_df = df[df["phasetype"] == "P"]
        s_df = df[df["phasetype"] == "S"]

        self.p_tt = torch.tensor(p_df[["sourcedepth", "distance", "traveltime"]].to_numpy(), dtype=torch.float32,
                                 device=self.device)
        self.s_tt = torch.tensor(s_df[["sourcedepth", "distance", "traveltime"]].to_numpy(), dtype=torch.float32,
                                 device=self.device)
        self.p_tt = interpolate_torch_matrix(self.p_tt, device=self.device)
        self.s_tt = interpolate_torch_matrix(self.s_tt, device=self.device)
        os.remove(self.tt_csv)
        np.save(self.ptt_npy, self.p_tt.cpu().numpy().T)
        np.save(self.stt_npy, self.s_tt.cpu().numpy().T)

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tt = TravelTime('TravelTime/original_model.nd',  'TravelTime/p_tt.csv', 'TravelTime/s_tt.csv', device)
    tt.run()