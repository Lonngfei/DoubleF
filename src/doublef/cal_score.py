import torch
from .weight import NumberScore, TimeScore, MagnitudeScore


def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371
    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1

    a = torch.sin(delta_lat / 2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(delta_lon / 2) ** 2
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))
    return R * c


class Score(object):
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
                 p_phase_matrix, s_phase_matrix, p_tt_matrix, s_tt_matrix,
                 P_weight, S_weight, time_weight, number_weight, magnitude_weight,
                 number_type, time_type, magnitude_type, dis0, dis1, device='cuda'):
        self.max_distance = max_distance
        self.location_matrix = torch.as_tensor(location_matrix, dtype=torch.float32, device=device)
        self.station_matrix = torch.as_tensor(station_matrix, dtype=torch.float32, device=device)
        self.p_phase_matrix = torch.as_tensor(p_phase_matrix, dtype=torch.float32, device=device)
        self.s_phase_matrix = torch.as_tensor(s_phase_matrix, dtype=torch.float32, device=device)
        self.p_tt_matrix = torch.as_tensor(p_tt_matrix, dtype=torch.float32, device=device)
        self.s_tt_matrix = torch.as_tensor(s_tt_matrix, dtype=torch.float32, device=device)
        self.P_weight = P_weight
        self.S_weight = S_weight
        self.p_tol_min = p_tol_min
        self.p_tol_max = p_tol_max
        self.s_tol_min = s_tol_min
        self.s_tol_max = s_tol_max
        self.time_weight = time_weight
        self.number_weight = number_weight
        self.magnitude_weight = magnitude_weight
        self.number_type = number_type
        self.time_type = time_type
        self.magnitude_type = magnitude_type
        self.dis0 = dis0
        self.dis1 = dis1
        self.device = device


    def calcualte_distances(self):
        """
        calculate the distance between each station and each event
        :return:
        distance_matrix: N x S x 3; N: number of events; S: number of stations; 3: epicenter distance, depth, otime
        """
        lat1 = torch.deg2rad(self.location_matrix[:, 0])
        lon1 = torch.deg2rad(self.location_matrix[:, 1])
        lat2 = torch.deg2rad(self.station_matrix[:, 1])
        lon2 = torch.deg2rad(self.station_matrix[:, 2])
        depth_time = self.location_matrix[:, 2:4]

        lat1_expanded = lat1.unsqueeze(1)
        lon1_expanded = lon1.unsqueeze(1)

        lat2_expanded = lat2.unsqueeze(0)
        lon2_expanded = lon2.unsqueeze(0)

        distances = haversine_distance(lat1_expanded, lon1_expanded, lat2_expanded, lon2_expanded)
        depth_time_expanded = depth_time.unsqueeze(1).expand(-1, distances.shape[1], -1)

        self.distances_matrix = torch.cat([distances.unsqueeze(-1), depth_time_expanded], dim=-1)
        return self.distances_matrix


    def get_theoretical_time(self):
        """
        get the theoretical time for every earthquake and station
        :return:
        p_tt_phases: N x S; N: number of events; S: number of stations
        s_tt_phases: N x S; N: number of events; S: number of stations
        """
        N, S, _ = self.distances_matrix.shape

        self.p_tt_distance = torch.full((N, S), float('nan'), dtype=torch.float32, device=self.device)
        self.s_tt_distance = torch.full((N, S), float('nan'), dtype=torch.float32, device=self.device)

        mask_above_threshold = self.distances_matrix[:, :, 0] > self.max_distance  # Shape: [N, S]
        valid_mask = ~mask_above_threshold  # Shape: [N, S]

        distances_filtered = torch.full_like(self.distances_matrix[:, :, 0], float('nan'))
        depths_filtered = torch.full_like(self.distances_matrix[:, :, 1], float('nan'))
        times_filtered = torch.full_like(self.distances_matrix[:, :, 2], float('nan'))

        distances_filtered[valid_mask] = self.distances_matrix[:, :, 0][valid_mask]
        depths_filtered[valid_mask] = self.distances_matrix[:, :, 1][valid_mask]
        times_filtered[valid_mask] = self.distances_matrix[:, :, 2][valid_mask]

        distances = torch.round(distances_filtered, decimals=2)
        depths = torch.round(depths_filtered, decimals=2)

        distance_idx = torch.clamp(torch.round(distances * 100).long(), 0, self.p_tt_matrix.shape[0] - 1)
        depth_idx = torch.clamp(torch.round(depths * 100).long(), 0, self.p_tt_matrix.shape[1] - 1)

        p_time_values = self.p_tt_matrix[distance_idx, depth_idx]
        self.p_tt_distance = times_filtered + p_time_values

        s_time_values = self.s_tt_matrix[distance_idx, depth_idx]
        self.s_tt_distance = times_filtered + s_time_values

        self.ps_tt_distance = self.s_tt_distance - self.p_tt_distance

        return self.p_tt_distance, self.s_tt_distance, self.ps_tt_distance


    def cal_score_P(self):
        """
        Get phases that under the time_offset for each eqs and stations
        :return:
        p_err: N x S; N: number of events; S: number of stations
        p_prob: N x S; N: number of events; S: number of stations
        p_amp: N x S; N: number of events; S: number of stations
        """
        N, S = self.p_tt_distance.shape
        M, S, _, _ = self.p_phase_matrix.shape

        p_time_offset = (self.distances_matrix[:, :,  0] / self.max_distance) * (self.p_tol_max - self.p_tol_min) + self.p_tol_min

        valid_mask = ~torch.isnan(self.p_tt_distance)
        p_tt_distance_min = self.p_tt_distance - p_time_offset
        p_tt_distance_max = self.p_tt_distance + p_time_offset
        p_tt_distance_min = torch.floor(p_tt_distance_min / 5)
        p_tt_distance_max = torch.floor(p_tt_distance_max / 5)
        m_min = torch.clamp(p_tt_distance_min, 0, int(86400/5)-1)
        m_max = torch.clamp(p_tt_distance_max, 0, int(86400/5)-1)
        p_tt_distance_min[~valid_mask] = float('nan')
        p_tt_distance_max[~valid_mask] = float('nan')
        m_min[~valid_mask] = float('nan')
        m_max[~valid_mask] = float('nan')
        m_min = torch.nan_to_num_(m_min, nan=0)
        m_max = torch.nan_to_num_(m_max, nan=0)
        m_min = m_min.long().unsqueeze(-1).expand(-1, -1, M)
        m_max = m_max.long().unsqueeze(-1).expand(-1, -1, M)

        p_phase_matrix_expand = self.p_phase_matrix[:, :, :, 0].unsqueeze(3).expand(-1, -1, -1, N).permute(3, 1, 0, 2)
        p_phase_matrix_min = torch.gather(p_phase_matrix_expand, dim=3, index=m_min.unsqueeze(-1))
        p_phase_matrix_max = torch.gather(p_phase_matrix_expand, dim=3, index=m_max.unsqueeze(-1))

        p_tt_difference_min = self.p_tt_distance[:, :, None] - p_phase_matrix_min[:, :, :, 0]
        p_tt_difference_max = self.p_tt_distance[:, :, None] - p_phase_matrix_max[:, :, :, 0]

        p_tt_difference = torch.cat((p_tt_difference_min, p_tt_difference_max), dim=2)
        m = torch.cat((m_min, m_max), dim=2)

        torch.nan_to_num_(p_tt_difference, nan=float('100'), posinf=None, neginf=None)
        min_abs_values, min_abs_indices = torch.min(torch.abs(p_tt_difference), dim=2)

        mask = torch.abs(min_abs_values) > p_time_offset
        min_abs_values.masked_fill_(mask, float('inf'))

        self.p_err = torch.full((N, S), float('nan'), dtype=torch.float32, device=self.p_tt_distance.device)
        self.p_prob = torch.full((N, S), float('nan'), dtype=torch.float32, device=self.p_tt_distance.device)
        self.p_amp = torch.full((N, S), float('nan'), dtype=torch.float32, device=self.p_tt_distance.device)
        self.p_pick = torch.full((N, S), float('nan'), dtype=torch.float32, device=self.p_tt_distance.device)

        valid_mask = torch.isfinite(min_abs_values)
        valid_rows, valid_cols = torch.where(valid_mask)

        selected_indices = min_abs_indices[valid_rows, valid_cols].long()
        m_adjusted = selected_indices.clone()
        m_adjusted[m_adjusted >= M] -= M

        self.p_err[valid_rows, valid_cols] = p_tt_difference[valid_rows, valid_cols, selected_indices]

        d = m[valid_rows, valid_cols, selected_indices]

        self.p_prob[valid_rows, valid_cols] = self.p_phase_matrix[m_adjusted, valid_cols, d, 1]
        self.p_amp[valid_rows, valid_cols] = self.p_phase_matrix[m_adjusted, valid_cols, d, 2]
        self.p_pick[valid_rows, valid_cols] = self.p_phase_matrix[m_adjusted, valid_cols, d,0]
        return self.p_err, self.p_prob, self.p_amp, self.p_pick


    def cal_score_S(self):
        """
        Get phases that under the time_offset for each eqs and stations
        :return:
        s_err: N x S; N: number of events; S: number of stations
        s_prob: N x S; N: number of events; S: number of stations
        s_amp: N x S; N: number of events; S: number of stations
        """
        N, S = self.s_tt_distance.shape
        M, S, _, _ = self.s_phase_matrix.shape

        s_time_offset = (self.distances_matrix[:, :,  0] / self.max_distance) * (self.s_tol_max - self.s_tol_min) + self.s_tol_min

        valid_mask = ~torch.isnan(self.s_tt_distance)
        s_tt_distance_min = self.s_tt_distance - s_time_offset
        s_tt_distance_max = self.s_tt_distance + s_time_offset
        s_tt_distance_min = torch.floor(s_tt_distance_min / 5)
        s_tt_distance_max = torch.floor(s_tt_distance_max / 5)
        m_min = torch.clamp(s_tt_distance_min, 0, int(86400/5)-1)
        m_max = torch.clamp(s_tt_distance_max, 0, int(86400/5)-1)

        s_tt_distance_min[~valid_mask] = float('nan')
        s_tt_distance_max[~valid_mask] = float('nan')
        m_min[~valid_mask] = float('nan')
        m_max[~valid_mask] = float('nan')
        m_min = torch.nan_to_num_(m_min, nan=0)
        m_max = torch.nan_to_num_(m_max, nan=0)
        m_min = m_min.long().unsqueeze(-1).expand(-1, -1, M)
        m_max = m_max.long().unsqueeze(-1).expand(-1, -1, M)

        s_phase_matrix_expand = self.s_phase_matrix[:, :, :, 0].unsqueeze(3).expand(-1, -1, -1, N).permute(3, 1, 0, 2)
        s_phase_matrix_min = torch.gather(s_phase_matrix_expand, dim=3, index=m_min.unsqueeze(-1))
        s_phase_matrix_max = torch.gather(s_phase_matrix_expand, dim=3, index=m_max.unsqueeze(-1))

        s_tt_difference_min = self.s_tt_distance[:, :, None] - s_phase_matrix_min[:, :, :, 0]
        s_tt_difference_max = self.s_tt_distance[:, :, None] - s_phase_matrix_max[:, :, :, 0]

        s_tt_difference = torch.cat((s_tt_difference_min, s_tt_difference_max), dim=2)
        m = torch.cat((m_min, m_max), dim=2)

        torch.nan_to_num_(s_tt_difference, nan=float('100'), posinf=None, neginf=None)
        min_abs_values, min_abs_indices = torch.min(torch.abs(s_tt_difference), dim=2)

        mask = torch.abs(min_abs_values) > s_time_offset
        min_abs_values.masked_fill_(mask, float('inf'))

        self.s_err = torch.full((N, S), float('nan'), dtype=torch.float32, device=self.s_tt_distance.device)
        self.s_prob = torch.full((N, S), float('nan'), dtype=torch.float32, device=self.s_tt_distance.device)
        self.s_amp = torch.full((N, S), float('nan'), dtype=torch.float32, device=self.s_tt_distance.device)
        self.s_pick = torch.full((N, S), float('nan'), dtype=torch.float32, device=self.s_tt_distance.device)

        valid_mask = torch.isfinite(min_abs_values)
        valid_rows, valid_cols = torch.where(valid_mask)

        selected_indices = min_abs_indices[valid_rows, valid_cols].long()
        m_adjusted = selected_indices.clone()
        m_adjusted[m_adjusted >= M] -= M

        self.s_err[valid_rows, valid_cols] = s_tt_difference[valid_rows, valid_cols, selected_indices]

        d = m[valid_rows, valid_cols, selected_indices]

        self.s_prob[valid_rows, valid_cols] = self.s_phase_matrix[m_adjusted, valid_cols, d, 1]
        self.s_amp[valid_rows, valid_cols] = self.s_phase_matrix[m_adjusted, valid_cols, d, 2]
        self.s_pick[valid_rows, valid_cols] = self.s_phase_matrix[m_adjusted, valid_cols, d,0]
        return  self.s_err, self.s_prob, self.s_amp, self.s_pick

    def cal_score_ps(self):
        p_phase_matrix = self.p_tt_distance - self.p_err
        s_phase_matrix = self.s_tt_distance - self.s_err
        ps_phase_matrix = s_phase_matrix - p_phase_matrix
        self.ps_err = self.ps_tt_distance - ps_phase_matrix


    def cal_weight_score(self):
        """
        cal score for every earthquake
        :return:
        score: N x 1; N: Number of events
        """
        self.calcualte_distances()
        self.get_theoretical_time()
        self.cal_score_P()
        self.cal_score_S()
        self.cal_score_ps()

        _, S = self.p_prob.shape

        ns = NumberScore(self.p_prob, self.s_prob, self.P_weight, self.S_weight, S, self.number_type, device=self.device)
        number_score_matrix = ns.cal()
        number_score_matrix[torch.isnan(number_score_matrix)] = 0

        ts = TimeScore(self.p_tol_max, self.s_tol_max, self.p_err, self.s_err, self.p_prob, self.s_prob, self.P_weight,
                           self.S_weight, self.distances_matrix[:, :, 0], self.dis0, self.dis1, self.time_type, device=self.device)

        time_score_matrix = 1 - ts.cal()
        time_score_matrix[torch.isnan(time_score_matrix)] = 0

        self.score_index = self.number_weight * number_score_matrix + self.time_weight * time_score_matrix
        if self.magnitude_weight > 0:
            ms = MagnitudeScore(self.p_amp, self.s_amp, self.P_weight, self.S_weight, self.distances_matrix[:, :, 0],
                                self.magnitude_type, device=self.device)
            magnitude_score_matrix = ms.nan_std()
            self.score_index = self.score_index + self.magnitude_weight * (1 - magnitude_score_matrix)

        score_matrix = torch.hstack((self.location_matrix, self.score_index))
        return score_matrix