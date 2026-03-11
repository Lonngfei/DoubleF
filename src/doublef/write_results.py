import torch
from obspy import UTCDateTime
from .weight import MagnitudeScore, NumberScore, TimeScore
import io

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371
    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1

    a = torch.sin(delta_lat / 2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(delta_lon / 2) ** 2
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))
    return R * c


class GetResult:
    def __init__(self, i, max_distance, initial_location_matrix, score_matrix, station_matrix, station_dic,
                 p_tol_min, p_tol_max, s_tol_min, s_tol_max,
                 p_phase_matrix, s_phase_matrix, p_tt_matrix, s_tt_matrix,
                 P_weight, S_weight, number_weight, time_weight, magnitude_weight,
                 time_type, number_type, magnitude_type, dis0, dis1,
                 write_dict, sum_eve_num, sum_p_num, sum_s_num, sum_both_num,
                 p_number, s_number, sum_number, both_number, only_double, datetime, savename, device='cuda'):
        self.i = i
        self.max_distance = max_distance
        self.initial_location_matrix = initial_location_matrix
        self.location_matrix = score_matrix[:, :4]
        self.P_weight = P_weight
        self.S_weight = S_weight
        self.number_weight = number_weight
        self.time_weight = time_weight
        self.magnitude_weight = magnitude_weight
        self.time_type = time_type
        self.number_type = number_type
        self.dis0 = dis0
        self.dis1 = dis1
        self.write_dict = write_dict
        self.sum_eve_num = sum_eve_num
        self.sum_p_num = sum_p_num
        self.sum_s_num = sum_s_num
        self.sum_both_num = sum_both_num
        self.station_matrix = station_matrix
        self.station_dic = station_dic
        self.p_phase_matrix = p_phase_matrix
        self.s_phase_matrix = s_phase_matrix
        self.p_tt_matrix = p_tt_matrix
        self.s_tt_matrix = s_tt_matrix
        self.p_number = p_number
        self.s_number = s_number
        self.sum_number = sum_number
        self.both_number = both_number
        self.p_tol_min = p_tol_min
        self.p_tol_max = p_tol_max
        self.s_tol_min = s_tol_min
        self.s_tol_max = s_tol_max
        self.only_double = only_double
        self.magnitude_type = magnitude_type
        self.datetime = datetime
        self.savename = savename
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

        mask_above_threshold = self.distances_matrix[:, :, 0] > self.max_distance
        valid_mask = ~mask_above_threshold

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

    def delete_raw_phase(self, pick_time, idx, type):
        t = int(pick_time // 5)
        if type == 'P':
            mask = self.initial_location_matrix[:, 3] != pick_time
            self.initial_location_matrix = self.initial_location_matrix[mask]
            mask = self.p_phase_matrix[:, idx, t, 0] == pick_time
            self.p_phase_matrix[:, idx, t, :][mask] = float('nan')
        else:
            mask = self.s_phase_matrix[:, idx, t, 0] == pick_time
            self.s_phase_matrix[:, idx, t, :][mask] = float('nan')

    def delete_phase(self, input_p, input_s):
        p_mask = self.p_pick == input_p.unsqueeze(0)
        s_mask = self.s_pick == input_s.unsqueeze(0)

        self.p_pick = torch.where(p_mask, torch.nan, self.p_pick)
        self.s_pick = torch.where(s_mask, torch.nan, self.s_pick)

        self.p_err = torch.where(p_mask, torch.nan, self.p_err)
        self.s_err = torch.where(s_mask, torch.nan, self.s_err)

        self.p_prob = torch.where(p_mask, torch.nan, self.p_prob)
        self.s_prob = torch.where(s_mask, torch.nan, self.s_prob)

        self.p_amp = torch.where(p_mask, torch.nan, self.p_amp)
        self.s_amp = torch.where(s_mask, torch.nan, self.s_amp)

    def write_results(self):
        self.calcualte_distances()
        self.get_theoretical_time()
        self.cal_score_P()
        self.cal_score_S()
        self.cal_weight_score()

        self.score_index = self.score_index.squeeze(-1)
        sorted_score_indices = torch.argsort(self.score_index, descending=True)
        sorted_location_matrix = self.location_matrix[sorted_score_indices]
        sorted_distance_matrix = self.distances_matrix[sorted_score_indices]
        self.p_amp = self.p_amp[sorted_score_indices]
        self.s_amp = self.s_amp[sorted_score_indices]
        self.p_err = self.p_err[sorted_score_indices]
        self.s_err = self.s_err[sorted_score_indices]
        self.p_prob = self.p_prob[sorted_score_indices]
        self.s_prob = self.s_prob[sorted_score_indices]
        self.p_pick = self.p_pick[sorted_score_indices]
        self.s_pick = self.s_pick[sorted_score_indices]

        N = sorted_location_matrix.size(0)
        used_rows = torch.zeros(N, dtype=torch.bool, device=self.device)

        self.event_number = 0
        self.p_sum_number = 0
        self.s_sum_number = 0
        self.both_sum_number = 0


        buffer = io.StringIO()

        for i in range(N):
            if used_rows[i]:
                continue

            p_valid = ~torch.isnan(self.p_pick[i])
            s_valid = ~torch.isnan(self.s_pick[i])
            both_valid = p_valid & s_valid

            count_p = p_valid.sum()
            count_s = s_valid.sum()
            count_both = both_valid.sum()
            count_sum = count_p + count_s

            if self.only_double:
                count_p = count_both
                count_s = count_both
                count_both = count_both
                count_sum = count_p + count_s

            if count_p < self.p_number or count_s < self.s_number or count_both < self.both_number or count_sum < self.sum_number:
                continue

            self.p_sum_number += count_p
            self.s_sum_number += count_s
            self.both_sum_number += count_both
            self.event_number += 1

            lat, lon, dep, time = sorted_location_matrix[i, :4]
            lat_d, lon_d = torch.deg2rad(lat), torch.deg2rad(lon)
            dist_mask = (
                    (torch.abs(sorted_location_matrix[:, 0] - lat) <= 0.1) &
                    (torch.abs(sorted_location_matrix[:, 1] - lon) <= 0.1) &
                    (torch.abs(sorted_location_matrix[:, 3] - time) <= 1)
            )

            used_rows |= dist_mask
            matching_matrix = sorted_location_matrix[dist_mask]
            err_distance = haversine_distance(lat_d, lon_d, torch.deg2rad(matching_matrix[:, 0]),
                                              torch.deg2rad(matching_matrix[:, 1]))
            err_depth = dep - matching_matrix[:, 2]
            err_time = time - matching_matrix[:, 3]

            err_x = torch.std(err_distance)
            err_y = torch.std(torch.abs(err_depth))
            err_t = torch.std(torch.abs(err_time))
            rms = torch.sqrt(torch.nanmean(torch.cat([self.p_err[i], self.s_err[i]]) ** 2))

            event_time = self.datetime + time.item()
            event_key = time.item()
            self.write_dict[event_key] = []
            y, m, d, h, minute = event_time.year, event_time.month, event_time.day, event_time.hour, event_time.minute
            sec = event_time - UTCDateTime(y, m, d, h, minute)

            ms = MagnitudeScore(self.p_amp[i], self.s_amp[i], 0.5, 0.5, sorted_distance_matrix[i, :, 0],
                                self.magnitude_type)
            p_mag, s_mag, mag = ms.cal_median_mag()

            header = f"# {y:04d} {m:02d} {d:02d} {h:02d} {minute:02d} {sec:06.3f} {lat.item():.6f} {lon.item():.6f} {dep.item():.3f} {mag:.2f} {err_x:.3f} {err_y:.3f} {err_t:.2f} {rms:.2f} {count_p} {count_s} {count_both} {count_sum}"
            self.write_dict[event_key].append(header)

            valid_idx = ~torch.isnan(sorted_distance_matrix[i, :, 0])
            if valid_idx.sum() == 0:
                continue

            idx_all = torch.arange(sorted_distance_matrix.shape[1], device=self.device)[valid_idx]
            sorted_dis = sorted_distance_matrix[i, valid_idx, 0]
            sort_order = torch.argsort(sorted_dis)
            idx_all = idx_all[sort_order]

            p_pick_all = self.p_pick[i, idx_all]
            s_pick_all = self.s_pick[i, idx_all]
            p_prob_all = self.p_prob[i, idx_all]
            s_prob_all = self.s_prob[i, idx_all]
            p_err_all = self.p_err[i, idx_all]
            s_err_all = self.s_err[i, idx_all]
            p_amp_all = self.p_amp[i, idx_all]
            s_amp_all = self.s_amp[i, idx_all]
            p_mag_all = p_mag[idx_all]
            s_mag_all = s_mag[idx_all]
            dis_all = sorted_distance_matrix[i, idx_all, 0]
            time_shift = time

            if self.only_double:
                valid_mask = (~torch.isnan(p_pick_all)) & (~torch.isnan(s_pick_all))
            else:
                valid_mask = (~torch.isnan(p_pick_all)) | (~torch.isnan(s_pick_all))

            idx_all = idx_all[valid_mask]
            dis_all = dis_all[valid_mask]
            p_pick_all = p_pick_all[valid_mask]
            s_pick_all = s_pick_all[valid_mask]
            p_prob_all = p_prob_all[valid_mask]
            s_prob_all = s_prob_all[valid_mask]
            p_err_all = p_err_all[valid_mask]
            s_err_all = s_err_all[valid_mask]
            p_amp_all = p_amp_all[valid_mask]
            s_amp_all = s_amp_all[valid_mask]
            p_mag_all = p_mag_all[valid_mask]
            s_mag_all = s_mag_all[valid_mask]

            for j, idx in enumerate(idx_all.tolist()):
                net, station = self.station_dic[idx]

                if not torch.isnan(p_pick_all[j]):
                    self.delete_raw_phase(p_pick_all[j], idx, 'P')
                    p_line = f"{net:<3} {station:<5} {dis_all[j]:>7.3f} {(p_pick_all[j] - time_shift):>6.3f} {float(p_prob_all[j]):.2f} Pg {float(p_err_all[j]):>5.2f} ML {float(p_mag_all[j]):>5.2f} {float(p_amp_all[j]):.3f}"
                    self.write_dict[event_key].append(p_line)

                if not torch.isnan(s_pick_all[j]):
                    self.delete_raw_phase(s_pick_all[j], idx, 'S')
                    s_line = f"{net:<3} {station:<5} {dis_all[j]:>7.3f} {(s_pick_all[j] - time_shift):>6.3f} {float(s_prob_all[j]):.2f} Sg {float(s_err_all[j]):>5.2f} ML {float(s_mag_all[j]):>5.2f} {float(s_amp_all[j]):.3f}"
                    self.write_dict[event_key].append(s_line)

            p_event = self.p_pick[i]
            s_event = self.s_pick[i]
            if self.only_double:
                mask = torch.isnan(p_event) | torch.isnan(s_event)
                p_event[mask] = float('nan')
                s_event[mask] = float('nan')
            self.delete_phase(p_event, s_event)

        if self.event_number != 0 and self.i < 1:
            return self.initial_location_matrix, self.p_phase_matrix, self.s_phase_matrix, self.write_dict, self.event_number + self.sum_eve_num, self.p_sum_number + self.sum_p_num, self.s_sum_number + self.sum_s_num, self.both_sum_number + self.sum_both_num
        else:
            for num, (k, v) in enumerate(sorted(self.write_dict.items()), 1):
                buffer.write(f"{v[0]} {num}\n")
                buffer.write("\n".join(v[1:]) + "\n")
            with open(self.savename, 'w') as f:
                f.write(buffer.getvalue())
            return self.event_number + self.sum_eve_num, self.p_sum_number + self.sum_p_num, self.s_sum_number + self.sum_s_num, self.both_sum_number + self.sum_both_num
