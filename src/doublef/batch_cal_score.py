import torch
from .weight import NumberScore, TimeScore, SCORE_CHUNK_TARGET_ELEMENTS
from .perf import timed


def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371
    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1

    a = torch.sin(delta_lat / 2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(delta_lon / 2) ** 2
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))
    return R * c


class BatchScore(object):
    """
    input para
    time_offset: maximum travel time deviation
    max_distance: maximum epicenter distance
    location_matrix: N x 4; N: number of events; 4: lat, lon, dep, otime
    station_matrix: S x 4; S: number of stations; 4: id, lat, lon, ele
    p_phase_matrix: M x S x 3; M: max number of pick from one station; S: number of stations; 3: picktime, prob, amp
    s_phase_matrix: M x S x 3; M: max number of pick from one station; S: number of stations; 3: picktime, prob, amp
    p_tt_matrix: H x V; H: number of Horizontal sampling points; V: number of Vertical sampling points
    s_tt_matrix: H x V; H: number of Horizontal sampling points; V: number of Vertical sampling points
    """

    def __init__(self, max_distance, location_matrix, station_matrix,
                 p_tol_min, p_tol_max, s_tol_min, s_tol_max,
                 phase_index, p_tt_matrix, s_tt_matrix, tt_distance_step_km, tt_depth_step_km,
                 P_weight, S_weight, time_weight, number_weight, magnitude_weight,
                 number_type, time_type, magnitude_type, dis0, dis1, event_batch_size=256, device='cuda', logger=None):
        self.max_distance = max_distance
        self.location_matrix = torch.as_tensor(location_matrix, dtype=torch.float32, device=device)
        self.station_matrix = torch.as_tensor(station_matrix, dtype=torch.float32, device=device)
        self.phase_index = phase_index
        self.p_tt_matrix = torch.as_tensor(p_tt_matrix, dtype=torch.float32, device=device)
        self.s_tt_matrix = torch.as_tensor(s_tt_matrix, dtype=torch.float32, device=device)
        self.tt_distance_step_km = tt_distance_step_km
        self.tt_depth_step_km = tt_depth_step_km
        self.P_weight = P_weight
        self.S_weight = S_weight
        self.time_weight = time_weight
        self.number_weight = number_weight
        self.magnitude_weight = magnitude_weight
        self.number_type = number_type
        self.time_type = time_type
        self.magnitude_type = magnitude_type
        self.dis0 = dis0
        self.dis1 = dis1
        self.p_tol_min = p_tol_min
        self.p_tol_max = p_tol_max
        self.s_tol_min = s_tol_min
        self.s_tol_max = s_tol_max
        self.event_batch_size = event_batch_size
        self.device = device
        self.logger = logger

    def _sample_chunk_size(self):
        total_samples, total_events, total_stations = self.location_matrix.shape[:3]
        elements_per_sample = max(1, total_events * total_stations)
        target_elements = SCORE_CHUNK_TARGET_ELEMENTS
        chunk_size = max(1, target_elements // elements_per_sample)
        return min(total_samples, chunk_size)

    def calculate_distances(self):
        """
        Calculate the distance between each station and each event for each batch.
        :return:
        distance_matrix: B x N x S x 3; B: batch size; N: number of events; S: number of stations; 3: epicenter distance, depth, otime
        """
        batch_size = self.location_matrix.shape[0]
        num_events = self.location_matrix.shape[1]
        num_stations = self.station_matrix.shape[0]

        lat1 = torch.deg2rad(self.location_matrix[:, :, 0])  # B x N
        lon1 = torch.deg2rad(self.location_matrix[:, :, 1])  # B x N
        lat2 = torch.deg2rad(self.station_matrix[:, 1])  # S
        lon2 = torch.deg2rad(self.station_matrix[:, 2])  # S

        depth_time = self.location_matrix[:, :, 2:4]  # B x N x 2

        lat1_expanded = lat1.unsqueeze(2)  # B x N x 1
        lon1_expanded = lon1.unsqueeze(2)  # B x N x 1

        lat2_expanded = lat2.unsqueeze(0).unsqueeze(0)  # 1 x 1 x S
        lon2_expanded = lon2.unsqueeze(0).unsqueeze(0)  # 1 x 1 x S

        distances = haversine_distance(lat1_expanded, lon1_expanded, lat2_expanded, lon2_expanded)  # B x N x S

        depth_time_expanded = depth_time.unsqueeze(2).expand(-1, -1, num_stations, -1)  # B x N x S x 2

        self.distances_matrix = torch.cat([distances.unsqueeze(-1), depth_time_expanded], dim=-1)  # B x N x S x 3

        return self.distances_matrix

    def get_theoretical_time(self):
        """
        Get the theoretical time for every earthquake and station for each batch.
        :return:
        p_tt_phases: B x N x S; B: batch size; N: number of events; S: number of stations
        s_tt_phases: B x N x S; B: batch size; N: number of events; S: number of stations
        ps_tt_phases: B x N x S; B: batch size; N: number of events; S: number of stations
        """
        B, N, S, _ = self.distances_matrix.shape
        distances_raw = self.distances_matrix[:, :, :, 0]
        depths_raw = self.distances_matrix[:, :, :, 1]
        times_raw = self.distances_matrix[:, :, :, 2]
        valid_mask = distances_raw <= self.max_distance

        distances = torch.round(torch.clamp_min(distances_raw, 0.0) / self.tt_distance_step_km).long()
        depths = torch.round(torch.clamp_min(depths_raw, 0.0) / self.tt_depth_step_km).long()

        if self.p_tt_matrix.ndim == 2:
            distance_idx = torch.clamp(distances, 0, self.p_tt_matrix.shape[0] - 1)
            depth_idx = torch.clamp(depths, 0, self.p_tt_matrix.shape[1] - 1)
            p_time_values = self.p_tt_matrix[distance_idx, depth_idx]
            s_time_values = self.s_tt_matrix[distance_idx, depth_idx]
        else:
            distance_idx = torch.clamp(distances, 0, self.p_tt_matrix.shape[1] - 1)
            depth_idx = torch.clamp(depths, 0, self.p_tt_matrix.shape[2] - 1)
            station_idx = torch.arange(S, device=self.device).view(1, 1, S).expand(B, N, -1)
            p_time_values = self.p_tt_matrix[station_idx, distance_idx, depth_idx]
            s_time_values = self.s_tt_matrix[station_idx, distance_idx, depth_idx]

        self.p_tt_distance = times_raw + p_time_values
        self.s_tt_distance = times_raw + s_time_values
        self.p_tt_distance = torch.where(valid_mask, self.p_tt_distance, torch.full_like(self.p_tt_distance, float('nan')))
        self.s_tt_distance = torch.where(valid_mask, self.s_tt_distance, torch.full_like(self.s_tt_distance, float('nan')))

        return self.p_tt_distance, self.s_tt_distance

    def cal_score_P(self):
        """
        Get phases that under the time_offset for each eqs and stations
        :return:
        p_err: batch_size x N x S
        p_prob: batch_size x N x S
        p_amp: batch_size x N x S
        p_pick: batch_size x N x S
        """
        p_time_offset = (self.distances_matrix[:, :, :, 0] / self.max_distance) * (self.p_tol_max - self.p_tol_min) + self.p_tol_min
        self.p_err, self.p_prob, self.p_amp, self.p_pick, self.p_pick_uid = self.phase_index.lookup(
            "P",
            self.p_tt_distance,
            p_time_offset,
        )
        return self.p_err, self.p_prob, self.p_amp, self.p_pick

    def cal_score_S(self):
        """
        Get phases that under the time_offset for each eqs and stations
        :return:
        s_err: batch_size x N x S
        s_prob: batch_size x N x S
        s_amp: batch_size x N x S
        s_pick: batch_size x N x S
        """
        s_time_offset = (self.distances_matrix[:, :, :, 0] / self.max_distance) * (self.s_tol_max - self.s_tol_min) + self.s_tol_min
        self.s_err, self.s_prob, self.s_amp, self.s_pick, self.s_pick_uid = self.phase_index.lookup(
            "S",
            self.s_tt_distance,
            s_time_offset,
        )
        return self.s_err, self.s_prob, self.s_amp, self.s_pick

    def _cal_weight_score_impl(self):
        """
        Calculate the weighted score for each earthquake.
        :return:
        score_matrix: B x (N + 1); B: batch size; N: number of events
        """
        with timed(self.logger, "score.calculate_distances"):
            self.calculate_distances()
        with timed(self.logger, "score.get_theoretical_time"):
            self.get_theoretical_time()
        with timed(self.logger, "score.lookup_p"):
            self.cal_score_P()
        with timed(self.logger, "score.lookup_s"):
            self.cal_score_S()
        B, N, S = self.p_prob.shape
        with timed(self.logger, "score.number_score"):
            ns = NumberScore(self.p_prob, self.s_prob, self.P_weight, self.S_weight, S, self.number_type,
                             device=self.device)
            number_score_matrix = ns.cal()  # Shape: [B, N]
            number_score_matrix[torch.isnan(number_score_matrix)] = 0

        with timed(self.logger, "score.time_score"):
            ts = TimeScore(self.p_tol_max, self.s_tol_max, self.p_err, self.s_err, self.p_prob, self.s_prob, self.P_weight,
                           self.S_weight, self.distances_matrix[:, :, :,  0], self.dis0, self.dis1, self.time_type,
                           device=self.device, logger=self.logger)
            time_score_matrix = 1 - ts.cal()  # Shape: [B, N]
            time_score_matrix[torch.isnan(time_score_matrix)] = 0
        self.score_index = self.number_weight * number_score_matrix + self.time_weight * time_score_matrix.squeeze(-1)

        # if self.magnitude_weight > 0:
        #     ms = MagnitudeScore(self.p_amp, self.s_amp, self.P_weight, self.S_weight, self.distances_matrix[:, :, 0],
        #                         self.magnitude_type, device=self.device)
        #     magnitude_score_matrix = ms.nan_std()  # Shape: [B, N]
        #     self.score_index = self.score_index + self.magnitude_weight * (1 - magnitude_score_matrix)

        score_matrix = torch.cat((self.location_matrix, self.score_index.unsqueeze(-1)), dim=-1)  # Shape: [B, N + 1]
        return score_matrix

    def cal_weight_score(self):
        total_samples = self.location_matrix.shape[0]
        chunk_size = self._sample_chunk_size()
        if total_samples <= chunk_size:
            return self._cal_weight_score_impl()

        score_chunks = []
        for start in range(0, total_samples, chunk_size):
            end = min(start + chunk_size, total_samples)
            chunk = BatchScore(
                self.max_distance,
                self.location_matrix[start:end],
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
                self.time_weight,
                self.number_weight,
                self.magnitude_weight,
                self.number_type,
                self.time_type,
                self.magnitude_type,
                self.dis0,
                self.dis1,
                event_batch_size=self.event_batch_size,
                device=self.device,
                logger=self.logger,
            )
            score_chunks.append(chunk._cal_weight_score_impl())
            del chunk

        return torch.cat(score_chunks, dim=0)
