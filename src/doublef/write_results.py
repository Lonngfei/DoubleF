import torch

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
                 result_batch_size=256, device='cuda', initial_batch_ids=None):
        self.i = i
        self.max_distance = max_distance
        self.initial_location_matrix = initial_location_matrix
        self.initial_seed_pick_uids = initial_seed_pick_uids
        self.initial_batch_ids = initial_batch_ids
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
        self.device = device

    def _candidate_chunk_size(self, total_candidates):
        if self.result_batch_size and self.result_batch_size < 10 ** 8:
            return max(1, min(int(self.result_batch_size), total_candidates))
        return max(1, min(2048, total_candidates))

    def _calculate_distances_for(self, location_matrix):
        lat1 = torch.deg2rad(location_matrix[:, 0])
        lon1 = torch.deg2rad(location_matrix[:, 1])
        lat2 = torch.deg2rad(self.station_matrix[:, 1])
        lon2 = torch.deg2rad(self.station_matrix[:, 2])
        depth_time = location_matrix[:, 2:4]

        lat1_expanded = lat1.unsqueeze(1)
        lon1_expanded = lon1.unsqueeze(1)
        lat2_expanded = lat2.unsqueeze(0)
        lon2_expanded = lon2.unsqueeze(0)

        distances = haversine_distance(lat1_expanded, lon1_expanded, lat2_expanded, lon2_expanded)
        depth_time_expanded = depth_time.unsqueeze(1).expand(-1, distances.shape[1], -1)
        return torch.cat([distances.unsqueeze(-1), depth_time_expanded], dim=-1)

    def _get_theoretical_time_for(self, distances_matrix):
        N, S, _ = distances_matrix.shape
        distances_raw = distances_matrix[:, :, 0]
        depths_raw = distances_matrix[:, :, 1]
        times_raw = distances_matrix[:, :, 2]
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

    def _lookup_for(self, phase, predicted_times, tolerance):
        return self.phase_index.lookup(phase, predicted_times, tolerance)

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
            sorted_score_indices = torch.argsort(self.input_score.squeeze(-1), descending=True)
            sorted_location_matrix = self.location_matrix[sorted_score_indices]
            sorted_lower_bound = self.final_lower_bound[sorted_score_indices]
            sorted_upper_bound = self.final_upper_bound[sorted_score_indices]
            sorted_scores = self.input_score.squeeze(-1)[sorted_score_indices]
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
        used_pick_mask = torch.zeros(max_pick_uid + 1, dtype=torch.bool) if max_pick_uid >= 0 else None
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
                        active_mask_cpu &= ~used_pick_mask[sorted_seed_pick_uids_cpu[start:end]]
                    active_indices = torch.nonzero(active_mask_cpu, as_tuple=False).squeeze(1)
                if active_indices.numel() == 0:
                    continue
                active_indices_gpu = active_indices.to(self.device)

                chunk_location = sorted_location_matrix[start:end].index_select(0, active_indices_gpu)
                chunk_lower = sorted_lower_bound[start:end].index_select(0, active_indices_gpu)
                chunk_upper = sorted_upper_bound[start:end].index_select(0, active_indices_gpu)
                chunk_scores = sorted_scores[start:end].index_select(0, active_indices_gpu)
                chunk_seed_uids = sorted_seed_pick_uids[start:end].index_select(0, active_indices_gpu)
                chunk_batch_ids = (
                    sorted_batch_ids[start:end].index_select(0, active_indices_gpu)
                    if sorted_batch_ids is not None else None
                )

                with timed(logger, "result.calculate_distances"):
                    chunk_distances = self._calculate_distances_for(chunk_location)
                with timed(logger, "result.get_theoretical_time"):
                    p_tt_distance, s_tt_distance = self._get_theoretical_time_for(chunk_distances)
                with timed(logger, "result.lookup_p"):
                    p_time_offset = (chunk_distances[:, :, 0] / self.max_distance) * (self.p_tol_max - self.p_tol_min) + self.p_tol_min
                    p_err, p_prob, p_amp, p_pick, p_pick_uid = self._lookup_for("P", p_tt_distance, p_time_offset)
                with timed(logger, "result.lookup_s"):
                    s_time_offset = (chunk_distances[:, :, 0] / self.max_distance) * (self.s_tol_max - self.s_tol_min) + self.s_tol_min
                    s_err, s_prob, s_amp, s_pick, s_pick_uid = self._lookup_for("S", s_tt_distance, s_time_offset)

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
                    chunk_location = chunk_location[accept_mask].cpu()
                    chunk_lower = chunk_lower[accept_mask].cpu()
                    chunk_upper = chunk_upper[accept_mask].cpu()
                    chunk_scores = chunk_scores[accept_mask].cpu()
                    chunk_seed_uids = chunk_seed_uids[accept_mask].cpu()
                    if chunk_batch_ids is not None:
                        chunk_batch_ids = chunk_batch_ids[accept_mask].cpu()
                    chunk_distances = chunk_distances[accept_mask].cpu()
                    p_err = p_err[accept_mask].cpu()
                    s_err = s_err[accept_mask].cpu()
                    p_prob = p_prob[accept_mask].cpu()
                    s_prob = s_prob[accept_mask].cpu()
                    p_amp = p_amp[accept_mask].cpu()
                    s_amp = s_amp[accept_mask].cpu()
                    p_pick = p_pick[accept_mask].cpu()
                    s_pick = s_pick[accept_mask].cpu()
                    p_pick_uid = p_pick_uid[accept_mask].cpu()
                    s_pick_uid = s_pick_uid[accept_mask].cpu()

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

                    event_time = self.datetime + time.item()
                    t0 = perf_counter()
                    ms = MagnitudeScore(
                        p_amp[i],
                        s_amp[i],
                        0.5,
                        0.5,
                        chunk_distances[i, :, 0],
                        self.magnitude_type,
                        device="cpu",
                    )
                    p_mag, s_mag, mag = ms.cal_median_mag()
                    magnitude_time += perf_counter() - t0

                    event_record = {
                        "origin_time": float(time.item()),
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
                    valid_idx = ~torch.isnan(chunk_distances[i, :, 0])
                    if valid_idx.sum() == 0:
                        sort_filter_time += perf_counter() - t0
                        continue

                    idx_all = station_indices[valid_idx]
                    sorted_dis = chunk_distances[i, valid_idx, 0]
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
                    dis_all = chunk_distances[i, idx_all, 0]
                    time_shift = time

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
                            event_record["picks"].append({
                                "phase": "P",
                                "net": net,
                                "station": station,
                                "pick_uid": int(p_pick_uid_all[j].item()),
                                "distance_km": float(dis_all[j]),
                                "relative_pick": float(p_pick_all[j] - time_shift),
                                "pick_time": float(p_pick_all[j]),
                                "prob": float(p_prob_all[j]),
                                "err": float(p_err_all[j]),
                                "mag": float(p_mag_all[j]),
                                "amp": float(p_amp_all[j]),
                            })

                        if s_pick_uid_all[j] >= 0 and (used_pick_mask is None or not used_pick_mask[int(s_pick_uid_all[j].item())]):
                            event_record["picks"].append({
                                "phase": "S",
                                "net": net,
                                "station": station,
                                "pick_uid": int(s_pick_uid_all[j].item()),
                                "distance_km": float(dis_all[j]),
                                "relative_pick": float(s_pick_all[j] - time_shift),
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

        if self.event_number != 0 and self.i < 1:
            remaining_phase_index = self.phase_index.remove_pick_ids(used_pick_ids)
            if used_pick_mask is not None:
                keep_mask = (~used_pick_mask[self.initial_seed_pick_uids.cpu()]).to(self.device)
            else:
                keep_mask = torch.ones_like(self.initial_seed_pick_uids, dtype=torch.bool, device=self.device)
            if self.initial_batch_ids is not None and successful_batch_ids:
                successful_batch_mask = torch.zeros_like(keep_mask, dtype=torch.bool, device=self.device)
                successful_batch_list = torch.tensor(
                    sorted(successful_batch_ids),
                    dtype=self.initial_batch_ids.dtype,
                    device=self.device,
                )
                successful_batch_mask = (self.initial_batch_ids.unsqueeze(1) == successful_batch_list.unsqueeze(0)).any(dim=1)
                keep_mask &= successful_batch_mask
            elif self.initial_batch_ids is not None:
                keep_mask &= torch.zeros_like(keep_mask, dtype=torch.bool, device=self.device)
            next_location_matrix = self.initial_location_matrix[keep_mask]
            next_seed_pick_uids = self.initial_seed_pick_uids[keep_mask]
            next_batch_ids = self.initial_batch_ids[keep_mask] if self.initial_batch_ids is not None else None
            if next_location_matrix.shape[0] > 0:
                next_location_matrix = torch.cat(
                    [
                        next_location_matrix[:, :2],
                        torch.zeros((next_location_matrix.shape[0], 1), dtype=torch.float32, device=self.device),
                        next_location_matrix[:, 3:4],
                    ],
                    dim=1,
                )
                return {
                    "continue": True,
                    "events": self.events,
                    "counts": counts,
                    "location_matrix": next_location_matrix,
                    "initial_seed_pick_uids": next_seed_pick_uids,
                    "initial_batch_ids": next_batch_ids,
                    "phase_index": remaining_phase_index,
                }

        return {
            "continue": False,
            "events": self.events,
            "counts": counts,
        }
