import torch
import numpy as np

from time import perf_counter

from .perf import add_time, timed
from .weight import MagnitudeScore


def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371
    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1

    a = torch.sin(delta_lat / 2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(delta_lon / 2) ** 2
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))
    return R * c


class GetResult:
    def __init__(self, i, max_distance, initial_location_matrix, score_matrix, station_matrix, final_lower_bound, final_upper_bound, initial_seed_pick_uids, station_dic,
                 p_tol_min, p_tol_max, s_tol_min, s_tol_max,
                 phase_index, p_tt_matrix, s_tt_matrix, tt_distance_step_km, tt_depth_step_km,
                 P_weight, S_weight, number_weight, time_weight, magnitude_weight,
                 time_type, number_type, magnitude_type, dis0, dis1,
                 write_dict, sum_eve_num, sum_p_num, sum_s_num, sum_both_num,
                 p_number, s_number, sum_number, both_number, only_double, datetime, savename,
                 result_batch_size=256, device='cuda', initial_batch_ids=None, batch_ref_msec=None, batch_window_msec=None,
                 reference_global_msec=0, initial_used_pick_mask=None, enable_repeat=True, pick_global_msec=None):
        self.i = i
        self.max_distance = max_distance
        self.initial_location_matrix = initial_location_matrix
        self.initial_seed_pick_uids = initial_seed_pick_uids
        self.initial_batch_ids = initial_batch_ids
        self.batch_ref_msec = batch_ref_msec or {}
        self.batch_window_msec = batch_window_msec or {}
        self.reference_global_msec = int(reference_global_msec)
        self.initial_used_pick_mask = initial_used_pick_mask
        self.enable_repeat = bool(enable_repeat)
        self.pick_global_msec = pick_global_msec
        self.location_matrix = score_matrix[:, :4]
        self.input_score = score_matrix[:, 4]
        self.final_lower_bound = final_lower_bound
        self.final_upper_bound = final_upper_bound
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
        self.phase_index = phase_index
        self.p_tt_matrix = p_tt_matrix
        self.s_tt_matrix = s_tt_matrix
        self.tt_distance_step_km = tt_distance_step_km
        self.tt_depth_step_km = tt_depth_step_km
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
        self.result_batch_size = result_batch_size
        self.device = torch.device(device)
        self.max_p_tt = float(torch.nan_to_num(self.p_tt_matrix, nan=float("-inf")).max().item()) if self.p_tt_matrix.numel() > 0 else 0.0
        self.max_s_tt = float(torch.nan_to_num(self.s_tt_matrix, nan=float("-inf")).max().item()) if self.s_tt_matrix.numel() > 0 else 0.0

    def _candidate_chunk_size(self, total_candidates):
        if self.result_batch_size and self.result_batch_size < 10 ** 8:
            return max(1, min(int(self.result_batch_size), total_candidates))
        return max(1, min(2048, total_candidates))

    def _calculate_distances_for(self, location_matrix):
        lat1 = torch.deg2rad(location_matrix[:, 0])
        lon1 = torch.deg2rad(location_matrix[:, 1])
        lat2 = torch.deg2rad(self.station_matrix[:, 1])
        lon2 = torch.deg2rad(self.station_matrix[:, 2])
        depth = location_matrix[:, 2]

        lat1_expanded = lat1.unsqueeze(1)
        lon1_expanded = lon1.unsqueeze(1)
        lat2_expanded = lat2.unsqueeze(0)
        lon2_expanded = lon2.unsqueeze(0)

        distances = haversine_distance(lat1_expanded, lon1_expanded, lat2_expanded, lon2_expanded)
        return distances, depth, location_matrix[:, 3]

    def _get_theoretical_time_for(self, distances_raw, depths_raw, times_raw):
        N, S = distances_raw.shape
        depths_raw = depths_raw.unsqueeze(1).expand(-1, S)
        times_raw = times_raw.unsqueeze(1)
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
            station_idx = torch.arange(S, device=self.device).unsqueeze(0).expand(N, -1)
            p_time_values = self.p_tt_matrix[station_idx, distance_idx, depth_idx]
            s_time_values = self.s_tt_matrix[station_idx, distance_idx, depth_idx]

        p_tt_distance = times_raw + p_time_values
        s_tt_distance = times_raw + s_time_values
        p_tt_distance = torch.where(valid_mask, p_tt_distance, torch.full_like(p_tt_distance, float("nan")))
        s_tt_distance = torch.where(valid_mask, s_tt_distance, torch.full_like(s_tt_distance, float("nan")))
        return p_tt_distance, s_tt_distance

    def _lookup_for(self, lookup_index, phase, predicted_times, tolerance):
        return lookup_index.lookup(phase, predicted_times, tolerance)

    def _passes_thresholds(self, count_p, count_s, count_both, count_sum):
        return (
            count_p >= self.p_number
            and count_s >= self.s_number
            and count_both >= self.both_number
            and count_sum >= self.sum_number
        )

    def _refresh_event_counts(self, event_record):
        station_phases = {}
        for pick in event_record["picks"]:
            key = (pick["net"], pick["station"])
            station_phases.setdefault(key, set()).add(pick["phase"])

        if self.only_double:
            keep_stations = {key for key, phases in station_phases.items() if {"P", "S"}.issubset(phases)}
            event_record["picks"] = [
                pick for pick in event_record["picks"] if (pick["net"], pick["station"]) in keep_stations
            ]
            station_phases = {key: phases for key, phases in station_phases.items() if key in keep_stations}

        count_p = sum(1 for pick in event_record["picks"] if pick["phase"] == "P")
        count_s = sum(1 for pick in event_record["picks"] if pick["phase"] == "S")
        count_both = sum(1 for phases in station_phases.values() if {"P", "S"}.issubset(phases))
        count_sum = count_p + count_s
        err_values = [float(pick["err"]) for pick in event_record["picks"]]
        if err_values:
            event_record["rms"] = float((sum(err * err for err in err_values) / len(err_values)) ** 0.5)
        else:
            event_record["rms"] = float("nan")
        event_record["count_p"] = int(count_p)
        event_record["count_s"] = int(count_s)
        event_record["count_both"] = int(count_both)
        event_record["count_sum"] = int(count_sum)
        return event_record

    def write_results(self):
        logger = self.write_dict.get("logger")
        with timed(logger, "result.sort_candidates"):
            score_values = self.input_score.reshape(-1)
            sorted_score_indices = torch.argsort(score_values, descending=True)
            sorted_location_matrix = self.location_matrix[sorted_score_indices]
            sorted_lower_bound = self.final_lower_bound[sorted_score_indices]
            sorted_upper_bound = self.final_upper_bound[sorted_score_indices]
            sorted_scores = score_values[sorted_score_indices]
            sorted_seed_pick_uids = self.initial_seed_pick_uids[sorted_score_indices]
            sorted_batch_ids = (
                self.initial_batch_ids[sorted_score_indices]
                if self.initial_batch_ids is not None else None
            )
        sorted_seed_pick_uids_cpu = sorted_seed_pick_uids.cpu()

        self.event_number = 0
        self.p_sum_number = 0
        self.s_sum_number = 0
        self.both_sum_number = 0
        self.events = []
        used_pick_ids = set()
        successful_batch_ids = set()
        max_pick_uid = int(self.phase_index.df["pick_uid"].max()) if not self.phase_index.df.empty else -1
        if self.initial_used_pick_mask is not None:
            used_pick_mask = self.initial_used_pick_mask.clone()
            if max_pick_uid >= int(used_pick_mask.shape[0]):
                expanded = torch.zeros(max_pick_uid + 1, dtype=torch.bool)
                expanded[: used_pick_mask.shape[0]] = used_pick_mask
                used_pick_mask = expanded
        else:
            used_pick_mask = torch.zeros(max_pick_uid + 1, dtype=torch.bool) if max_pick_uid >= 0 else None
        if self.pick_global_msec is not None:
            pick_global_msec = self.pick_global_msec
        else:
            pick_global_msec = np.full(max_pick_uid + 1, -1, dtype=np.int64) if max_pick_uid >= 0 else np.empty(0, dtype=np.int64)
            if max_pick_uid >= 0:
                pick_uid_values = self.phase_index.df["pick_uid"].to_numpy(dtype=np.int32, copy=False)
                pick_global_values = self.phase_index.df["GlobalMsec"].to_numpy(dtype=np.int64, copy=False)
                pick_global_msec[pick_uid_values] = pick_global_values
        magnitude_time = 0.0
        pick_loop_time = 0.0
        sort_filter_time = 0.0
        total_candidates = int(sorted_location_matrix.shape[0])
        candidate_chunk_size = self._candidate_chunk_size(total_candidates)

        with timed(logger, "result.extract_events"):
            for start in range(0, total_candidates, candidate_chunk_size):
                end = min(start + candidate_chunk_size, total_candidates)

                with timed(logger, "result.extract_events.seed_skip"):
                    active_mask_cpu = torch.ones(end - start, dtype=torch.bool)
                    if used_pick_mask is not None:
                        active_mask_cpu &= ~used_pick_mask[sorted_seed_pick_uids_cpu[start:end].long()]
                    active_indices = torch.nonzero(active_mask_cpu, as_tuple=False).squeeze(1)
                if active_indices.numel() == 0:
                    continue
                chunk_location_cpu = sorted_location_matrix[start:end].index_select(0, active_indices)
                chunk_lower_cpu = sorted_lower_bound[start:end].index_select(0, active_indices)
                chunk_upper_cpu = sorted_upper_bound[start:end].index_select(0, active_indices)
                chunk_scores_cpu = sorted_scores[start:end].index_select(0, active_indices)
                chunk_seed_uids_cpu = sorted_seed_pick_uids_cpu[start:end].index_select(0, active_indices)
                chunk_batch_ids_cpu = (
                    sorted_batch_ids[start:end].index_select(0, active_indices).cpu()
                    if sorted_batch_ids is not None else None
                )

                if chunk_batch_ids_cpu is None:
                    raise ValueError("Local reference times require batch ids in write_results")
                chunk_batch_ids_cpu = chunk_batch_ids_cpu.to(torch.long)
                n_candidates = int(chunk_location_cpu.shape[0])
                n_stations = int(self.station_matrix.shape[0])
                chunk_distances = torch.full((n_candidates, n_stations), float("nan"), dtype=torch.float32)
                p_err = torch.full((n_candidates, n_stations), float("nan"), dtype=torch.float32)
                s_err = torch.full((n_candidates, n_stations), float("nan"), dtype=torch.float32)
                p_prob = torch.full((n_candidates, n_stations), float("nan"), dtype=torch.float32)
                s_prob = torch.full((n_candidates, n_stations), float("nan"), dtype=torch.float32)
                p_amp = torch.full((n_candidates, n_stations), float("nan"), dtype=torch.float32)
                s_amp = torch.full((n_candidates, n_stations), float("nan"), dtype=torch.float32)
                p_pick = torch.full((n_candidates, n_stations), float("nan"), dtype=torch.float32)
                s_pick = torch.full((n_candidates, n_stations), float("nan"), dtype=torch.float32)
                p_pick_uid = torch.full((n_candidates, n_stations), -1, dtype=torch.int32)
                s_pick_uid = torch.full((n_candidates, n_stations), -1, dtype=torch.int32)
                with timed(logger, "result.relookup_candidates"):
                    for event_batch_id in torch.unique(chunk_batch_ids_cpu).tolist():
                        local_ref_msec = self.batch_ref_msec.get(int(event_batch_id))
                        if local_ref_msec is None:
                            raise ValueError(f"Missing local reference time for batch id {event_batch_id}")
                        group_indices = torch.nonzero(chunk_batch_ids_cpu == int(event_batch_id), as_tuple=False).squeeze(1)
                        group_location_cpu = chunk_location_cpu.index_select(0, group_indices)
                        window_msec = self.batch_window_msec.get(int(event_batch_id))
                        if window_msec is None:
                            group_lower_cpu = chunk_lower_cpu.index_select(0, group_indices)
                            group_upper_cpu = chunk_upper_cpu.index_select(0, group_indices)
                            origin_min_local = float(group_lower_cpu[:, 3].min().item())
                            origin_max_local = float(group_upper_cpu[:, 3].max().item())
                            origin_min_abs = (local_ref_msec / 1000.0) + origin_min_local
                            origin_max_abs = (local_ref_msec / 1000.0) + origin_max_local
                            group_phase_index = self.phase_index.window(
                                origin_min_abs - self.p_tol_max,
                                origin_max_abs + self.p_tol_max + self.max_p_tt,
                                s_start_time=origin_min_abs - self.s_tol_max,
                                s_end_time=origin_max_abs + self.s_tol_max + self.max_s_tt,
                                local_ref_msec=local_ref_msec,
                            )
                        else:
                            p_start_msec, p_end_msec, s_start_msec, s_end_msec = window_msec
                            group_phase_index = self.phase_index.window(
                                p_start_msec / 1000.0,
                                p_end_msec / 1000.0,
                                s_start_time=s_start_msec / 1000.0,
                                s_end_time=s_end_msec / 1000.0,
                                local_ref_msec=local_ref_msec,
                            )
                        group_location = group_location_cpu.to(self.device)
                        distances_raw, depths_raw, times_raw = self._calculate_distances_for(group_location)
                        p_tt_distance, s_tt_distance = self._get_theoretical_time_for(distances_raw, depths_raw, times_raw)
                        p_time_offset = (distances_raw / self.max_distance) * (self.p_tol_max - self.p_tol_min) + self.p_tol_min
                        s_time_offset = (distances_raw / self.max_distance) * (self.s_tol_max - self.s_tol_min) + self.s_tol_min
                        p_err_i, p_prob_i, p_amp_i, p_pick_i, p_pick_uid_i = self._lookup_for(group_phase_index, "P", p_tt_distance, p_time_offset)
                        s_err_i, s_prob_i, s_amp_i, s_pick_i, s_pick_uid_i = self._lookup_for(group_phase_index, "S", s_tt_distance, s_time_offset)
                        chunk_distances[group_indices] = distances_raw.cpu()
                        p_err[group_indices] = p_err_i.cpu()
                        s_err[group_indices] = s_err_i.cpu()
                        p_prob[group_indices] = p_prob_i.cpu()
                        s_prob[group_indices] = s_prob_i.cpu()
                        p_amp[group_indices] = p_amp_i.cpu()
                        s_amp[group_indices] = s_amp_i.cpu()
                        p_pick[group_indices] = p_pick_i.cpu()
                        s_pick[group_indices] = s_pick_i.cpu()
                        p_pick_uid[group_indices] = p_pick_uid_i.cpu()
                        s_pick_uid[group_indices] = s_pick_uid_i.cpu()

                with timed(logger, "result.extract_events.prefilter"):
                    p_valid_all = (p_pick_uid >= 0) & torch.isfinite(p_pick)
                    s_valid_all = (s_pick_uid >= 0) & torch.isfinite(s_pick)
                    both_valid_all = p_valid_all & s_valid_all
                    count_p_all = p_valid_all.sum(dim=1)
                    count_s_all = s_valid_all.sum(dim=1)
                    count_both_all = both_valid_all.sum(dim=1)
                    if self.only_double:
                        count_p_eval = count_both_all
                        count_s_eval = count_both_all
                        count_sum_eval = count_both_all * 2
                    else:
                        count_p_eval = count_p_all
                        count_s_eval = count_s_all
                        count_sum_eval = count_p_all + count_s_all
                    accept_mask = (
                        (count_p_eval >= self.p_number)
                        & (count_s_eval >= self.s_number)
                        & (count_both_all >= self.both_number)
                        & (count_sum_eval >= self.sum_number)
                    )
                if not bool(accept_mask.any().item()):
                    continue

                with timed(logger, "result.extract_events.to_cpu"):
                    accept_mask_cpu = accept_mask.cpu()
                    chunk_location = chunk_location_cpu[accept_mask_cpu]
                    chunk_lower = chunk_lower_cpu[accept_mask_cpu]
                    chunk_upper = chunk_upper_cpu[accept_mask_cpu]
                    chunk_scores = chunk_scores_cpu[accept_mask_cpu]
                    chunk_seed_uids = chunk_seed_uids_cpu[accept_mask_cpu]
                    if chunk_batch_ids_cpu is not None:
                        chunk_batch_ids = chunk_batch_ids_cpu[accept_mask_cpu]
                    else:
                        chunk_batch_ids = None
                    chunk_distances = chunk_distances[accept_mask_cpu]
                    p_err = p_err[accept_mask_cpu]
                    s_err = s_err[accept_mask_cpu]
                    p_prob = p_prob[accept_mask_cpu]
                    s_prob = s_prob[accept_mask_cpu]
                    p_amp = p_amp[accept_mask_cpu]
                    s_amp = s_amp[accept_mask_cpu]
                    p_pick = p_pick[accept_mask_cpu]
                    s_pick = s_pick[accept_mask_cpu]
                    p_pick_uid = p_pick_uid[accept_mask_cpu]
                    s_pick_uid = s_pick_uid[accept_mask_cpu]

                station_indices = torch.arange(chunk_distances.shape[1], device="cpu")
                for i in range(chunk_location.size(0)):
                    seed_pick_uid = int(chunk_seed_uids[i].item())
                    if used_pick_mask is not None and used_pick_mask[seed_pick_uid]:
                        continue
                    event_batch_id = int(chunk_batch_ids[i].item()) if chunk_batch_ids is not None else -1

                    lat, lon, dep, time = chunk_location[i, :4]
                    p_sq = torch.where(torch.isfinite(p_err[i]), p_err[i] * p_err[i], torch.zeros_like(p_err[i]))
                    s_sq = torch.where(torch.isfinite(s_err[i]), s_err[i] * s_err[i], torch.zeros_like(s_err[i]))
                    valid_err_count = torch.isfinite(p_err[i]).sum() + torch.isfinite(s_err[i]).sum()
                    if valid_err_count.item() > 0:
                        rms = torch.sqrt((p_sq.sum() + s_sq.sum()) / valid_err_count)
                    else:
                        rms = torch.tensor(float("nan"), dtype=torch.float32)

                    local_ref_msec = self.batch_ref_msec.get(event_batch_id)
                    if local_ref_msec is None:
                        raise ValueError(f"Missing local reference time for batch id {event_batch_id}")
                    origin_global_msec = int(local_ref_msec + round(float(time.item()) * 1000.0))
                    event_time = self.datetime + ((origin_global_msec - self.reference_global_msec) / 1000.0)
                    t0 = perf_counter()
                    ms = MagnitudeScore(
                        p_amp[i],
                        s_amp[i],
                        0.5,
                        0.5,
                        chunk_distances[i],
                        self.magnitude_type,
                        device="cpu",
                    )
                    p_mag, s_mag, mag = ms.cal_median_mag()
                    magnitude_time += perf_counter() - t0

                    event_record = {
                        "origin_time": origin_global_msec / 1000.0,
                        "origin_global_msec": origin_global_msec,
                        "origin_datetime": event_time,
                        "location": {
                            "lat": float(lat.item()),
                            "lon": float(lon.item()),
                            "dep": float(dep.item()),
                        },
                        "score": float(chunk_scores[i].item()),
                        "magnitude": float(mag),
                        "rms": float(rms),
                        "count_p": 0,
                        "count_s": 0,
                        "count_both": 0,
                        "count_sum": 0,
                        "err_lat": float((chunk_upper[i, 0] - chunk_lower[i, 0]).item() / 2.0),
                        "err_lon": float((chunk_upper[i, 1] - chunk_lower[i, 1]).item() / 2.0),
                        "err_dep": float((chunk_upper[i, 2] - chunk_lower[i, 2]).item() / 2.0),
                        "err_time": float((chunk_upper[i, 3] - chunk_lower[i, 3]).item() / 2.0),
                        "picks": [],
                    }

                    t0 = perf_counter()
                    valid_idx = ~torch.isnan(chunk_distances[i])
                    if valid_idx.sum() == 0:
                        sort_filter_time += perf_counter() - t0
                        continue

                    idx_all = station_indices[valid_idx]
                    sorted_dis = chunk_distances[i, valid_idx]
                    sort_order = torch.argsort(sorted_dis)
                    idx_all = idx_all[sort_order]

                    p_pick_all = p_pick[i, idx_all]
                    s_pick_all = s_pick[i, idx_all]
                    p_pick_uid_all = p_pick_uid[i, idx_all]
                    s_pick_uid_all = s_pick_uid[i, idx_all]
                    p_prob_all = p_prob[i, idx_all]
                    s_prob_all = s_prob[i, idx_all]
                    p_err_all = p_err[i, idx_all]
                    s_err_all = s_err[i, idx_all]
                    p_amp_all = p_amp[i, idx_all]
                    s_amp_all = s_amp[i, idx_all]
                    p_mag_all = p_mag[idx_all]
                    s_mag_all = s_mag[idx_all]
                    dis_all = chunk_distances[i, idx_all]

                    if self.only_double:
                        valid_mask = (p_pick_uid_all >= 0) & (s_pick_uid_all >= 0)
                    else:
                        valid_mask = (p_pick_uid_all >= 0) | (s_pick_uid_all >= 0)

                    idx_all = idx_all[valid_mask]
                    dis_all = dis_all[valid_mask]
                    p_pick_all = p_pick_all[valid_mask]
                    s_pick_all = s_pick_all[valid_mask]
                    p_pick_uid_all = p_pick_uid_all[valid_mask]
                    s_pick_uid_all = s_pick_uid_all[valid_mask]
                    p_prob_all = p_prob_all[valid_mask]
                    s_prob_all = s_prob_all[valid_mask]
                    p_err_all = p_err_all[valid_mask]
                    s_err_all = s_err_all[valid_mask]
                    p_amp_all = p_amp_all[valid_mask]
                    s_amp_all = s_amp_all[valid_mask]
                    p_mag_all = p_mag_all[valid_mask]
                    s_mag_all = s_mag_all[valid_mask]
                    sort_filter_time += perf_counter() - t0

                    t0 = perf_counter()
                    for j, idx in enumerate(idx_all.tolist()):
                        net, station = self.station_dic[idx]

                        if p_pick_uid_all[j] >= 0 and (used_pick_mask is None or not used_pick_mask[int(p_pick_uid_all[j].item())]):
                            p_uid = int(p_pick_uid_all[j].item())
                            relative_pick = (int(pick_global_msec[p_uid]) - origin_global_msec) / 1000.0
                            event_record["picks"].append({
                                "phase": "P",
                                "net": net,
                                "station": station,
                                "pick_uid": p_uid,
                                "distance_km": float(dis_all[j]),
                                "relative_pick": float(relative_pick),
                                "pick_time": float(p_pick_all[j]),
                                "prob": float(p_prob_all[j]),
                                "err": float(p_err_all[j]),
                                "mag": float(p_mag_all[j]),
                                "amp": float(p_amp_all[j]),
                            })

                        if s_pick_uid_all[j] >= 0 and (used_pick_mask is None or not used_pick_mask[int(s_pick_uid_all[j].item())]):
                            s_uid = int(s_pick_uid_all[j].item())
                            relative_pick = (int(pick_global_msec[s_uid]) - origin_global_msec) / 1000.0
                            event_record["picks"].append({
                                "phase": "S",
                                "net": net,
                                "station": station,
                                "pick_uid": s_uid,
                                "distance_km": float(dis_all[j]),
                                "relative_pick": float(relative_pick),
                                "pick_time": float(s_pick_all[j]),
                                "prob": float(s_prob_all[j]),
                                "err": float(s_err_all[j]),
                                "mag": float(s_mag_all[j]),
                                "amp": float(s_amp_all[j]),
                            })
                    pick_loop_time += perf_counter() - t0

                    event_record = self._refresh_event_counts(event_record)
                    if not self._passes_thresholds(
                        event_record["count_p"],
                        event_record["count_s"],
                        event_record["count_both"],
                        event_record["count_sum"],
                    ):
                        continue

                    for pick in event_record["picks"]:
                        pick_uid = int(pick["pick_uid"])
                        used_pick_ids.add(pick_uid)
                        if used_pick_mask is not None:
                            used_pick_mask[pick_uid] = True

                    self.events.append(event_record)
                    successful_batch_ids.add(event_batch_id)
                    self.event_number += 1
                    self.p_sum_number += event_record["count_p"]
                    self.s_sum_number += event_record["count_s"]
                    self.both_sum_number += event_record["count_both"]

        add_time(logger, "result.extract_events.magnitude", magnitude_time, magnitude_time)
        add_time(logger, "result.extract_events.sort_filter", sort_filter_time, sort_filter_time)
        add_time(logger, "result.extract_events.pick_loop", pick_loop_time, pick_loop_time)

        counts = (
            self.event_number + self.sum_eve_num,
            self.p_sum_number + self.sum_p_num,
            self.s_sum_number + self.sum_s_num,
            self.both_sum_number + self.sum_both_num,
        )

        if self.enable_repeat and self.event_number != 0 and self.i < 1:
            remaining_phase_index = self.phase_index.remove_pick_ids(used_pick_ids)
            if used_pick_mask is not None:
                keep_mask = ~used_pick_mask[self.initial_seed_pick_uids.cpu().long()]
            else:
                keep_mask = torch.ones_like(self.initial_seed_pick_uids, dtype=torch.bool, device="cpu")
            if self.initial_batch_ids is not None and successful_batch_ids:
                successful_batch_list = torch.tensor(
                    sorted(successful_batch_ids),
                    dtype=self.initial_batch_ids.dtype,
                    device="cpu",
                )
                # Batch IDs are assigned densely per outer batch, so a lookup table
                # avoids the O(num_candidates * num_successful_batches) broadcast.
                max_batch_id = int(self.initial_batch_ids.max().item())
                successful_batch_lookup = torch.zeros(
                    max_batch_id + 1,
                    dtype=torch.bool,
                    device="cpu",
                )
                successful_batch_lookup[successful_batch_list] = True
                successful_batch_mask = successful_batch_lookup[self.initial_batch_ids.cpu()]
                keep_mask &= successful_batch_mask
            elif self.initial_batch_ids is not None:
                keep_mask &= torch.zeros_like(keep_mask, dtype=torch.bool, device="cpu")
            next_location_matrix = self.initial_location_matrix[keep_mask]
            next_seed_pick_uids = self.initial_seed_pick_uids[keep_mask]
            next_seed_times = torch.from_numpy(pick_global_msec[next_seed_pick_uids.cpu().numpy()]).to(dtype=torch.long)
            if next_location_matrix.shape[0] > 0:
                return {
                    "continue": True,
                    "events": self.events,
                    "counts": counts,
                    "location_matrix": next_location_matrix[:, :2],
                    "initial_seed_pick_uids": next_seed_pick_uids,
                    "seed_times": next_seed_times,
                    "phase_index": remaining_phase_index,
                    "used_pick_ids": used_pick_ids,
                    "used_pick_mask": used_pick_mask,
                    "successful_batch_ids": successful_batch_ids,
                }

        return {
            "continue": False,
            "events": self.events,
            "counts": counts,
            "used_pick_ids": used_pick_ids,
            "used_pick_mask": used_pick_mask,
            "successful_batch_ids": successful_batch_ids,
        }
