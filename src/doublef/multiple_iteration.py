from .sampling import Sampler
from .write_results import GetResult
import torch
import gc


class Mutiple_Iteration(object):
    def __init__(self, logger, is_repeat, max_distance, location_matrix, station_matrix, station_dic, p_phase_matrix, s_phase_matrix, p_tt_matrix, s_tt_matrix,
                 P_weight, S_weight, magnitude_weight, time_type, number_type, magnitude_type, dis0, dis1, lat_max, lon_max, dep_max, time_max,
                 sample_index, top_number_index, number_weight_index, time_weight_index, confidence_level_index,
                 p_number, s_number, sum_number, both_number, p_tol_min, p_tol_max, s_tol_min, s_tol_max, only_double, datetime, savename,
                 max_batch_size, is_plot, plot_dir='plot', device='cuda'):
        self.device = device
        self.logger = logger
        self.is_repeat = is_repeat
        self.max_distance = max_distance
        self.location_matrix = location_matrix
        self.station_matrix = station_matrix
        self.station_dic = station_dic
        self.p_phase_matrix = p_phase_matrix
        self.s_phase_matrix = s_phase_matrix
        self.p_tt_matrix = p_tt_matrix
        self.s_tt_matrix = s_tt_matrix
        self.P_weight = P_weight
        self.S_weight = S_weight
        self.time_type = time_type
        self.magnitude_weight = magnitude_weight
        self.number_type = number_type
        self.magnitude_type = magnitude_type
        self.dis0 = dis0
        self.dis1 = dis1
        self.lat_max = lat_max
        self.lon_max = lon_max
        self.dep_max = dep_max
        self.time_max = time_max
        self.sample_index = sample_index
        self.top_number_index = top_number_index
        self.number_weight_index = number_weight_index
        self.time_weight_index = time_weight_index
        self.confidence_level_index = confidence_level_index
        self.p_number = p_number
        self.s_number = s_number
        self.sum_number = sum_number
        self.both_number = both_number
        self.p_tol_min = p_tol_min
        self.s_tol_min = s_tol_min
        self.p_tol_max = p_tol_max
        self.s_tol_max = s_tol_max
        self.only_double = only_double
        self.datetime = datetime
        self.savename = savename
        self.is_plot = is_plot
        self.plot_dir = plot_dir
        self.max_batch_size = max_batch_size

    def sampler(self):
        sa = Sampler(self.logger, self.max_distance, self.location_matrix, self.station_matrix,
                     self.p_tol_min, self.p_tol_max, self.s_tol_min, self.s_tol_max,
                     self.p_phase_matrix, self.s_phase_matrix, self.p_tt_matrix, self.s_tt_matrix,
                     self.P_weight, self.S_weight, self.magnitude_weight,
                     self.time_type, self.number_type, self.magnitude_type,
                     self.dis0, self.dis1, self.lat_max, self.lon_max, self.dep_max, self.time_max,
                     self.sample_index, self.top_number_index, self.number_weight_index, self.time_weight_index,
                     self.confidence_level_index, self.max_batch_size, self.is_plot, self.plot_dir, self.device)
        self.top_samples = sa.run()

    def get_results(self):
        if self.is_plot or (not self.is_repeat):
            i = 1
        else:
            i = 0
            self.logger.info(f'{i + 1}th location and association')
        self.sampler()
        self.write_dict = {}
        self.sum_eve_num, self.sum_p_num, self.sum_s_num, self.sum_both_num = 0, 0, 0, 0
        gr = GetResult(i, self.max_distance, self.location_matrix, self.top_samples.squeeze(-1), self.station_matrix,
                       self.station_dic, self.p_tol_min, self.p_tol_max, self.s_tol_min, self.s_tol_max,
                       self.p_phase_matrix, self.s_phase_matrix, self.p_tt_matrix, self.s_tt_matrix,
                       self.P_weight, self.S_weight, 0.95, 0.05, 0,
                       self.time_type, self.number_type, self.magnitude_type, self.dis0, self.dis1,
                       self.write_dict, self.sum_eve_num, self.sum_p_num, self.sum_s_num, self.sum_both_num,
                       self.p_number, self.s_number, self.sum_number, self.both_number,  self.only_double,
                       self.datetime, self.savename, device=self.device)
        results = gr.write_results()
        while len(results) == 8:
            i += 1
            self.logger.info(f'{i + 1}th location and association')
            del self.location_matrix, self.p_phase_matrix, self.s_phase_matrix
            torch.cuda.empty_cache()
            gc.collect()
            self.location_matrix, self.p_phase_matrix, self.s_phase_matrix, self.write_dict, self.sum_eve_num, self.sum_p_num, self.sum_s_num, self.sum_both_num = results
            self.sampler()
            gr = GetResult(i, self.max_distance, self.location_matrix, self.top_samples.squeeze(-1), self.station_matrix,
                           self.station_dic, self.p_tol_min, self.p_tol_max, self.s_tol_min, self.s_tol_max,
                           self.p_phase_matrix, self.s_phase_matrix, self.p_tt_matrix, self.s_tt_matrix,
                           self.P_weight, self.S_weight, 0.95, 0.05, 0,
                           self.time_type, self.number_type, self.magnitude_type, self.dis0, self.dis1,
                           self.write_dict, self.sum_eve_num, self.sum_p_num, self.sum_s_num, self.sum_both_num,
                           self.p_number, self.s_number, self.sum_number, self.both_number, self.only_double,
                           self.datetime, self.savename, device=self.device)
            results = gr.write_results()
        return results

    def __del__(self):
        del self.location_matrix
        del self.p_phase_matrix
        del self.s_phase_matrix
        del self.p_tt_matrix
        del self.s_tt_matrix
        del self.station_matrix
        del self.station_dic

