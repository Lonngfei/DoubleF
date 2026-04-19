import warnings

import numpy as np
import torch
from torch.quasirandom import SobolEngine

from .batch_cal_score import BatchScore
from .perf import timed

warnings.filterwarnings("ignore")

try:
    import sobol_seq as _sobol_seq_module
except Exception:
    _sobol_seq_module = None


class Sampler(object):
    _GLOBAL_SOBOL_CACHE = {}

    def __init__(self, logger, max_distance, location_matrix, station_matrix,
                 p_tol_min, p_tol_max, s_tol_min, s_tol_max,
                 phase_index, p_tt_matrix, s_tt_matrix, tt_distance_step_km, tt_depth_step_km,
                 P_weight, S_weight, magnitude_weight, time_type, number_type, magnitude_type, dis0, dis1, lat_max, lon_max, dep_max, time_max,
                 sample_index, top_number_index, number_weight_index, time_weight_index, confidence_level_index,
                 sampling_batch_size=32, score_event_batch_size=64, device='cuda'):
        self.device = device
        self.logger = logger
        self.max_distance = max_distance
        self.location_matrix = location_matrix
        self.station_matrix = station_matrix
        self.phase_index = phase_index
        self.p_tt_matrix = p_tt_matrix
        self.s_tt_matrix = s_tt_matrix
        self.tt_distance_step_km = tt_distance_step_km
        self.tt_depth_step_km = tt_depth_step_km
        self.p_tol_min = p_tol_min
        self.p_tol_max = p_tol_max
        self.s_tol_min = s_tol_min
        self.s_tol_max = s_tol_max
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
        self.sampling_batch_size = sampling_batch_size
        self.score_event_batch_size = score_event_batch_size
        self.sobol_backend = "sobol_seq" if _sobol_seq_module is not None else "torch"
        self._sobol_engine = SobolEngine(dimension=4, scramble=False) if self.sobol_backend == "torch" else None

    def get_initial_bounds(self):
        lower_bound = self.location_matrix.clone()
        upper_bound = self.location_matrix.clone()

        lower_bound[:, 0] = self.location_matrix[:, 0] - self.lat_max
        upper_bound[:, 0] = self.location_matrix[:, 0] + self.lat_max
        lower_bound[:, 1] = self.location_matrix[:, 1] - self.lon_max
        upper_bound[:, 1] = self.location_matrix[:, 1] + self.lon_max
        lower_bound[:, 2] = self.location_matrix[:, 2]
        upper_bound[:, 2] = self.location_matrix[:, 2] + self.dep_max
        lower_bound[:, 3] = self.location_matrix[:, 3] - self.time_max
        upper_bound[:, 3] = self.location_matrix[:, 3] + 1
        return lower_bound, upper_bound

    def sobol_sample_batch(self, lower_bound, upper_bound, batch_size):
        cache_key = (self.sobol_backend, str(self.device), int(batch_size))
        sobol_samples = self._GLOBAL_SOBOL_CACHE.get(cache_key)
        if sobol_samples is None:
            with timed(self.logger, "sampling.sobol_generate"):
                if self.sobol_backend == "sobol_seq":
                    sobol_np = _sobol_seq_module.i4_sobol_generate(4, batch_size)
                    sobol_samples = torch.from_numpy(np.asarray(sobol_np, dtype=np.float32)).to(self.device)
                else:
                    sobol_samples = self._sobol_engine.draw(batch_size).to(self.device, dtype=torch.float32)
            self._GLOBAL_SOBOL_CACHE[cache_key] = sobol_samples
        else:
            with timed(self.logger, "sampling.sobol_cache_hit"):
                sobol_samples = sobol_samples
        lower = lower_bound.to(torch.float32).unsqueeze(0)
        upper = upper_bound.to(torch.float32).unsqueeze(0)
        with timed(self.logger, "sampling.sobol_scale"):
            return lower + (upper - lower) * sobol_samples.unsqueeze(1)

    @staticmethod
    def merge_topk(existing_locations, existing_scores, new_locations, new_scores, top_number):
        combined_scores = torch.cat((existing_scores, new_scores), dim=1)
        combined_locations = torch.cat((existing_locations, new_locations), dim=2)
        top_scores, top_indices = torch.topk(combined_scores, top_number, dim=1, largest=True, sorted=False)
        top_locations = torch.gather(
            combined_locations,
            2,
            top_indices.unsqueeze(1).expand(-1, combined_locations.shape[1], -1),
        )
        return top_locations, top_scores

    def batch_get_top_samples(self, lower_bound, upper_bound, number_weight, time_weight, top_number, num_samples):
        with timed(self.logger, "sampling.sobol_draw"):
            sample_batch = self.sobol_sample_batch(lower_bound, upper_bound, num_samples)

        scorer = BatchScore(
            self.max_distance,
            sample_batch,
            self.station_matrix,
            self.p_tol_min,
            self.p_tol_max,
            self.s_tol_min,
            self.s_tol_max,
            self.phase_index,
            self.p_tt_matrix,
            self.s_tt_matrix,
            self.tt_distance_step_km,
            self.tt_depth_step_km,
            self.P_weight,
            self.S_weight,
            time_weight,
            number_weight,
            self.magnitude_weight,
            self.number_type,
            self.time_type,
            self.magnitude_type,
            self.dis0,
            self.dis1,
            self.score_event_batch_size,
            self.device,
            logger=self.logger,
        )

        with timed(self.logger, "sampling.batch_score"):
            batch_scores = scorer.cal_weight_score()
        batch_locations = batch_scores[:, :, :4].permute(1, 2, 0)
        batch_score_values = batch_scores[:, :, 4].permute(1, 0)
        with timed(self.logger, "sampling.topk"):
            top_scores, top_indices = torch.topk(
                batch_score_values,
                min(top_number, batch_score_values.shape[1]),
                dim=1,
                largest=True,
                sorted=False,
            )
            top_locations = torch.gather(
                batch_locations,
                2,
                top_indices.unsqueeze(1).expand(-1, batch_locations.shape[1], -1),
            )

        del scorer, batch_scores, batch_locations, batch_score_values, sample_batch
        self.top_samples = torch.cat((top_locations, top_scores.unsqueeze(1)), dim=1)

    def compute_confidence_interval(self, quantile):
        lower_quantile = (1 - quantile) / 2
        upper_quantile = 1 - lower_quantile

        self.lower_bound = torch.quantile(self.top_samples[:, :4, :], lower_quantile, dim=2)
        self.upper_bound = torch.quantile(self.top_samples[:, :4, :], upper_quantile, dim=2)
        del self.top_samples

    def run(self):
        lower_bound, upper_bound = self.get_initial_bounds()
        for i, para in enumerate(zip(
            self.sample_index,
            self.top_number_index,
            self.number_weight_index,
            self.time_weight_index,
            self.confidence_level_index,
        )):
            num_samples, top_number, number_weight, time_weight, confidence_interval = para
            if i == len(self.sample_index) - 1:
                top_number = 1
                self.final_lower_bound = lower_bound.clone()
                self.final_upper_bound = upper_bound.clone()

            self.batch_get_top_samples(lower_bound, upper_bound, number_weight, time_weight, top_number, num_samples)

            if i != (len(self.sample_index) - 1):
                with timed(self.logger, "sampling.quantile"):
                    self.compute_confidence_interval(confidence_interval)
                lower_bound = self.lower_bound
                upper_bound = self.upper_bound

        return self.top_samples
