import torch
import sobol_seq
from .visual import visualize_progress_and_policy
import warnings
from tqdm import tqdm
from .batch_cal_score import BatchScore
from .cal_score import Score
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")

class LocationDataset(Dataset):
    def __init__(self, location_matrix_sample):
        """
        Custom Dataset for location samples.
        :param location_matrix_sample: Tensor of shape [batch_size, N, 4]
        """
        self.location_matrix_sample = location_matrix_sample
        self.num_samples = location_matrix_sample.shape[2]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.location_matrix_sample[:, :, idx]

class Sampler(object):
    def __init__(self, logger, max_distance, location_matrix, station_matrix,
                 p_tol_min, p_tol_max, s_tol_min, s_tol_max,
                 p_phase_matrix, s_phase_matrix, p_tt_matrix, s_tt_matrix,
                 P_weight, S_weight, magnitude_weight, time_type, number_type, magnitude_type, dis0, dis1, lat_max, lon_max, dep_max, time_max,
                 sample_index, top_number_index, number_weight_index, time_weight_index, confidence_level_index,
                 max_batch_size=700000, is_plot=False, plot_dir='plot', device='cuda'):
        self.device = device
        self.logger = logger
        self.max_distance = max_distance
        self.location_matrix = location_matrix
        self.station_matrix = station_matrix
        self.p_phase_matrix = p_phase_matrix
        self.s_phase_matrix = s_phase_matrix
        self.p_tt_matrix = p_tt_matrix
        self.s_tt_matrix = s_tt_matrix
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
        self.is_plot = is_plot
        self.plot_dir = plot_dir
        self.max_batch_size = max_batch_size

    def sobol_sample(self, min_values, max_values, num_samples):
        sobol_samples = sobol_seq.i4_sobol_generate(4, num_samples)

        sobol_samples = torch.tensor(sobol_samples, dtype=torch.float32).to(self.device)

        min_vals = min_values.unsqueeze(2)
        max_vals = max_values.unsqueeze(2)

        min_vals = min_vals.to(torch.float32)
        max_vals = max_vals.to(torch.float32)

        sampled_values = min_vals + (max_vals - min_vals) * sobol_samples.t().unsqueeze(0)
        return sampled_values

    def get_initial_sample_matrix(self, num_samples):
        location_matrix_min = self.location_matrix.clone()
        location_matrix_max = self.location_matrix.clone()

        location_matrix_min[:, 0] = self.location_matrix[:, 0] - self.lat_max
        location_matrix_max[:, 0] = self.location_matrix[:, 0] + self.lat_max
        location_matrix_min[:, 1] = self.location_matrix[:, 1] - self.lon_max
        location_matrix_max[:, 1] = self.location_matrix[:, 1] + self.lon_max
        location_matrix_min[:, 2] = self.location_matrix[:, 2]
        location_matrix_max[:, 2] = self.location_matrix[:, 2] + self.dep_max
        location_matrix_min[:, 3] = self.location_matrix[:, 3] - self.time_max
        location_matrix_max[:, 3] = self.location_matrix[:, 3] + 1

        self.location_matrix_sample = self.sobol_sample(location_matrix_min, location_matrix_max, num_samples)


        if self.is_plot:
            self.lat0 = location_matrix_min[0, 0]
            self.lat1 = location_matrix_max[0, 0]
            self.lon0 = location_matrix_min[0, 1]
            self.lon1 = location_matrix_max[0, 1]
            self.dep0 = location_matrix_min[0, 2]
            self.dep1 = location_matrix_max[0, 2]
            self.time0 = location_matrix_min[0, 3]
            self.time1 = location_matrix_max[0, 3]

            grid_x1 = torch.linspace(self.lat0, self.lat1, 61)
            grid_x2 = torch.linspace(self.lon0, self.lon1, 61)
            grid_x3 = torch.linspace(self.dep0, self.dep1, 61)
            grid_x4 = torch.linspace(self.time0, self.time1, 81)
            grid_x1, grid_x2, grid_x3, grid_x4 = torch.meshgrid(grid_x1, grid_x2, grid_x3, grid_x4, indexing="ij")
            self.xs = torch.vstack([grid_x1.flatten(), grid_x2.flatten(), grid_x3.flatten(), grid_x4.flatten()]).transpose(-1, -2)
            self.ys = Score(self.max_distance, self.xs, self.station_matrix,
                      self.p_tol_min, self.p_tol_max, self.s_tol_min, self.s_tol_max,
                      self.p_phase_matrix, self.s_phase_matrix, self.p_tt_matrix, self.s_tt_matrix,
                      self.P_weight, self.S_weight, self.time_weight_index[0], self.number_weight_index[0], self.magnitude_weight,
                      self.number_type, self.time_type, self.magnitude_type, self.dis0, self.dis1, device='cpu').cal_weight_score()[:, 4]


    def get_top_samples(self, number_weight, time_weight, top_number):
         N, _, num_samples = self.location_matrix_sample.shape
         S, _ = self.station_matrix.shape
         self.scores = torch.zeros(N, 5, num_samples, dtype=torch.float32).to(self.device)
         self.scores[:, :4, :] = self.location_matrix_sample

    
         for i in tqdm(range(num_samples)):
             sample = self.location_matrix_sample[:, :, i]
             s = Score(self.max_distance, sample, self.station_matrix,
                       self.p_tol_min, self.p_tol_max, self.s_tol_min, self.s_tol_max,
                       self.p_phase_matrix, self.s_phase_matrix, self.p_tt_matrix, self.s_tt_matrix,
                       self.P_weight, self.S_weight, time_weight, number_weight, self.magnitude_weight,
                       self.number_type, self.time_type, self.magnitude_type, self.dis0, self.dis1, self.device)

             self.scores[:, 4, i:i + 1] = s.cal_weight_score()[:, 4].unsqueeze(1)
    
             _, top_k_indices = torch.topk(self.scores[:, 4, :], top_number, dim=1, largest=True, sorted=False)
             self.top_samples = torch.gather(self.scores, 2, top_k_indices.unsqueeze(1).expand(-1, 5, -1))


    def batch_get_top_samples(self, number_weight, time_weight, top_number, batch_size=500):
        """
        Get top samples using DataLoader for batch processing.
        :param number_weight: Weight for number score.
        :param time_weight: Weight for time score.
        :param top_number: Number of top samples to select.
        :param batch_size: Batch size for DataLoader.
        :return:
        top_samples: Tensor of shape [N, 5, top_number]
        """
        N, _, num_samples = self.location_matrix_sample.shape
        S, _ = self.station_matrix.shape
        self.scores = torch.zeros(N, 5, num_samples, dtype=torch.float32).to(self.device)
        self.scores[:, :4, :] = self.location_matrix_sample

        batch_size = min(batch_size, int(self.max_batch_size / N / S))
        
        if batch_size == 0:
            batch_size=1
        
        dataset = LocationDataset(self.location_matrix_sample)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        for batch_idx, sample_batch in enumerate(tqdm(dataloader)):
            sample_batch = sample_batch.to(self.device)  # Shape: [batch_size, N, 4]
            batch_size_current = sample_batch.shape[0]

            s = BatchScore(self.max_distance, sample_batch, self.station_matrix,
                           self.p_tol_min, self.p_tol_max, self.s_tol_min, self.s_tol_max,
                      self.p_phase_matrix, self.s_phase_matrix, self.p_tt_matrix, self.s_tt_matrix,
                      self.P_weight, self.S_weight, time_weight, number_weight, self.magnitude_weight,
                      self.number_type, self.time_type, self.magnitude_type, self.dis0, self.dis1, self.device)

            batch_scores = s.cal_weight_score()
            self.scores[:, 4, batch_idx * batch_size: batch_idx * batch_size + batch_size_current] = batch_scores[:, :,
                                                                                                     4].permute(1, 0)

        # Select top-k samples
        _, top_k_indices = torch.topk(self.scores[:, 4, :], top_number, dim=1, largest=True, sorted=False)
        self.top_samples = torch.gather(self.scores, 2, top_k_indices.unsqueeze(1).expand(-1, 5, -1))
        del self.scores, top_k_indices, batch_scores,  self.location_matrix_sample
        return self.top_samples

    def compute_confidence_interval(self, quantile):
        lower_quantile = (1 - quantile) / 2
        upper_quantile = 1 - lower_quantile

        self.lower_bound = torch.quantile(self.top_samples[:, :4, :], lower_quantile, dim=2)
        self.upper_bound = torch.quantile(self.top_samples[:, :4, :], upper_quantile, dim=2)
        if not self.is_plot:
            del self.top_samples

    def run(self):
        for i, para in enumerate(zip(self.sample_index, self.top_number_index, self.number_weight_index, self.time_weight_index, self.confidence_level_index)):
            num_samples, top_number, number_weight, time_weight, confidence_interval = para
            if i == len(self.sample_index) - 1:
                top_number = 1
            if i == 0:
                self.logger.info(f'Generating {num_samples} Sobol samples and getting {top_number} top samples......')
                self.get_initial_sample_matrix(num_samples)
                if self.is_plot:
                    self.get_top_samples(number_weight, time_weight, top_number)
                else:
                    self.batch_get_top_samples(number_weight, time_weight, top_number)
                self.compute_confidence_interval(confidence_interval)
                if self.is_plot:
                    visualize_progress_and_policy(self.ys, self.lat0, self.lat1, self.lon0, self.lon1, self.dep0,
                                                  self.dep1,
                                                  self.time0, self.time1, self.location_matrix, self.scores,
                                                  self.top_samples[:, :4, :], self.plot_dir + '/iter_1.jpg', 1, top_number)
            else:
                self.logger.info(f'Generating {num_samples} Sobol samples and getting {top_number} top samples......')
                self.location_matrix_sample = self.sobol_sample(self.lower_bound, self.upper_bound, num_samples)
                if self.is_plot:
                    self.get_top_samples(number_weight, time_weight, top_number)
                else:
                    self.batch_get_top_samples(number_weight, time_weight, top_number)
                if i != (len(self.sample_index) - 1):
                    self.compute_confidence_interval(confidence_interval)
                if self.is_plot:
                    visualize_progress_and_policy(self.ys, self.lat0, self.lat1, self.lon0, self.lon1, self.dep0,
                                                  self.dep1,
                                                  self.time0, self.time1, self.location_matrix, self.scores,
                                                  self.top_samples[:, :4, :], self.plot_dir + f'/iter_{i + 1}.jpg',
                                                  i + 1, top_number)
        return self.top_samples




