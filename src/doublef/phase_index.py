from collections import OrderedDict

import numpy as np
import pandas as pd
import torch

from .perf import timed


class PhasePickIndex:
    _WINDOW_CACHE_SIZE = 2

    def __init__(
        self,
        phase_df,
        max_station_id,
        device="cuda",
        phase_arrays=None,
        counts=None,
        time_step=0.01,
        window_lookup=None,
        logger=None,
        lookup_mode="searchsorted",
        lookup_query_chunk_size=4096,
        local_ref_msec=None,
        reference_global_msec=None,
    ):
        self.max_station_id = int(max_station_id)
        self.compute_device = torch.device(device)
        self.device = self.compute_device
        self.time_step = float(time_step)
        self.window_lookup = window_lookup if window_lookup is not None else OrderedDict()
        self.logger = logger
        self.lookup_mode = str(lookup_mode)
        self.lookup_query_chunk_size = max(1, int(lookup_query_chunk_size))
        self.local_ref_msec = None if local_ref_msec is None else int(local_ref_msec)
        self.reference_global_msec = None if reference_global_msec is None else int(reference_global_msec)

        if phase_arrays is not None:
            self.df = self._prepare_df(phase_df)
            self.counts = counts or {"P": 0, "S": 0}
            self.padded_phase = phase_arrays
            self.storage_device = self._phase_arrays_device(phase_arrays)
        else:
            if "pick_uid" not in phase_df.columns:
                phase_df = phase_df.copy()
                if len(phase_df) > np.iinfo(np.int32).max:
                    raise ValueError("Too many picks for int32 pick_uid")
                phase_df["pick_uid"] = np.arange(len(phase_df), dtype=np.int32)
            self.df = self._prepare_df(phase_df)
            self.counts = {
                "P": int((self.df["phasetype"] == "P").sum()),
                "S": int((self.df["phasetype"] == "S").sum()),
            }
            self.storage_device = torch.device("cpu")
            self.padded_phase = {}
            self._build_padded_arrays()

        time_sort_col = "GlobalMsec" if "GlobalMsec" in self.df.columns else "RelativeTime"
        time_dtype = np.int64 if time_sort_col == "GlobalMsec" else np.float64
        self._time_sort_col = time_sort_col
        self._sort_times = (
            self.df[time_sort_col].to_numpy(dtype=time_dtype, copy=False)
            if not self.df.empty else np.empty(0, dtype=time_dtype)
        )

    @staticmethod
    def _prepare_df(phase_df):
        if phase_df is None:
            return pd.DataFrame()
        if phase_df.empty:
            return phase_df.copy()
        sort_col = "GlobalMsec" if "GlobalMsec" in phase_df.columns else "RelativeTime"
        if phase_df[sort_col].is_monotonic_increasing:
            return phase_df
        return phase_df.sort_values(sort_col).reset_index(drop=True)

    @staticmethod
    def _phase_arrays_device(phase_arrays):
        for phase_data in phase_arrays.values():
            if isinstance(phase_data, dict) and "times" in phase_data:
                return phase_data["times"].device
        return torch.device("cpu")

    def _empty_phase_arrays(self, device):
        return {
            "times": torch.empty((self.max_station_id, 0), dtype=torch.float32, device=device),
            "prob": torch.empty((self.max_station_id, 0), dtype=torch.float32, device=device),
            "amp": torch.empty((self.max_station_id, 0), dtype=torch.float32, device=device),
            "pick_uid": torch.empty((self.max_station_id, 0), dtype=torch.int32, device=device),
            "lengths": torch.zeros(self.max_station_id, dtype=torch.long, device=device),
        }

    def _phase_data_to_device(self, phase_data, device):
        if phase_data["times"].device == device:
            return phase_data
        return {
            key: value.to(device=device)
            for key, value in phase_data.items()
        }

    def _cache_window_result(self, key, result):
        self.window_lookup[key] = result
        self.window_lookup.move_to_end(key)
        while len(self.window_lookup) > self._WINDOW_CACHE_SIZE:
            self.window_lookup.popitem(last=False)

    def _build_padded_arrays(self):
        for phase in ("P", "S"):
            phase_df = self.df[self.df["phasetype"] == phase]
            if phase_df.empty:
                self.padded_phase[phase] = self._empty_phase_arrays(self.storage_device)
                continue

            counts = phase_df.groupby("id").size().sort_index()
            lengths = torch.zeros(self.max_station_id, dtype=torch.long, device=self.storage_device)
            station_ids_np = counts.index.to_numpy(dtype=np.int64, copy=False)
            station_lengths_np = counts.to_numpy(dtype=np.int64, copy=False)
            station_ids = torch.as_tensor(station_ids_np, dtype=torch.long, device=self.storage_device)
            station_lengths = torch.as_tensor(station_lengths_np, dtype=torch.long, device=self.storage_device)
            lengths[station_ids] = station_lengths
            max_len = int(station_lengths.max().item())

            time_dtype = torch.float32 if self.local_ref_msec is not None else torch.float64
            times = torch.full((self.max_station_id, max_len), float("inf"), dtype=time_dtype, device=self.storage_device)
            prob = torch.full((self.max_station_id, max_len), float("nan"), dtype=torch.float32, device=self.storage_device)
            amp = torch.full((self.max_station_id, max_len), float("nan"), dtype=torch.float32, device=self.storage_device)
            pick_uid = torch.full((self.max_station_id, max_len), -1, dtype=torch.int32, device=self.storage_device)

            for station_id, group in phase_df.groupby("id", sort=False):
                station_id = int(station_id)
                sort_col = "GlobalMsec" if "GlobalMsec" in group.columns else "RelativeTime"
                group = group.sort_values(sort_col)
                length = len(group)
                if self.local_ref_msec is not None and "GlobalMsec" in group.columns:
                    time_values = (group["GlobalMsec"].to_numpy(dtype=np.int64, copy=False) - self.local_ref_msec) / 1000.0
                elif self.reference_global_msec is not None and "GlobalMsec" in group.columns:
                    time_values = (group["GlobalMsec"].to_numpy(dtype=np.int64, copy=False) - self.reference_global_msec) / 1000.0
                else:
                    time_values = group["RelativeTime"].to_numpy(copy=False)
                times[station_id, :length] = torch.as_tensor(
                    time_values,
                    dtype=time_dtype,
                    device=self.storage_device,
                )
                prob[station_id, :length] = torch.as_tensor(
                    group["Probability"].to_numpy(copy=False),
                    dtype=torch.float32,
                    device=self.storage_device,
                )
                amp[station_id, :length] = torch.as_tensor(
                    group["Amplitude"].to_numpy(copy=False),
                    dtype=torch.float32,
                    device=self.storage_device,
                )
                pick_uid[station_id, :length] = torch.as_tensor(
                    group["pick_uid"].to_numpy(copy=False),
                    dtype=torch.int32,
                    device=self.storage_device,
                )

            self.padded_phase[phase] = {
                "times": times,
                "prob": prob,
                "amp": amp,
                "pick_uid": pick_uid,
                "lengths": lengths,
            }

    def lookup(self, phase, predicted_times, tolerance):
        predicted_times = torch.as_tensor(predicted_times, dtype=torch.float32, device=self.compute_device)
        tolerance = torch.as_tensor(tolerance, dtype=torch.float32, device=self.compute_device)
        result_shape = predicted_times.shape
        phase_data = self._phase_data_to_device(self.padded_phase[phase], self.compute_device)
        max_len = phase_data["times"].shape[1]

        if max_len == 0:
            return (
                torch.full(result_shape, float("nan"), dtype=torch.float32, device=self.compute_device),
                torch.full(result_shape, float("nan"), dtype=torch.float32, device=self.compute_device),
                torch.full(result_shape, float("nan"), dtype=torch.float32, device=self.compute_device),
                torch.full(result_shape, float("nan"), dtype=torch.float32, device=self.compute_device),
                torch.full(result_shape, -1, dtype=torch.int32, device=self.compute_device),
            )

        flat_query = predicted_times.reshape(-1, self.max_station_id).transpose(0, 1).contiguous()
        flat_tol = tolerance.reshape(-1, self.max_station_id).transpose(0, 1).contiguous()
        with timed(self.logger, "phase_index.lookup"):
            if self.lookup_mode == "searchsorted":
                err, prob, amp, pick, pick_uid = self._lookup_searchsorted(phase_data, flat_query, flat_tol)
            elif self.lookup_mode == "direct-station":
                err, prob, amp, pick, pick_uid = self._lookup_direct_station(phase_data, flat_query, flat_tol)
            elif self.lookup_mode == "direct-chunked":
                err, prob, amp, pick, pick_uid = self._lookup_direct_chunked(phase_data, flat_query, flat_tol)
            else:
                raise ValueError(f"Unsupported lookup mode: {self.lookup_mode}")

        return (
            err.transpose(0, 1).reshape(result_shape),
            prob.transpose(0, 1).reshape(result_shape),
            amp.transpose(0, 1).reshape(result_shape),
            pick.transpose(0, 1).reshape(result_shape),
            pick_uid.transpose(0, 1).reshape(result_shape),
        )

    def _lookup_searchsorted(self, phase_data, flat_query, flat_tol):
        with timed(self.logger, "phase_index.lookup.searchsorted"):
            valid_query = torch.isfinite(flat_query) & torch.isfinite(flat_tol)

            lengths = phase_data["lengths"].unsqueeze(1)
            has_pick = lengths > 0
            safe_query = torch.where(valid_query, flat_query, torch.zeros_like(flat_query))
            insert_idx = torch.searchsorted(phase_data["times"], safe_query, right=False)
            last_valid = torch.clamp(lengths - 1, min=0)
            right_idx = torch.minimum(insert_idx, last_valid)
            left_idx = torch.minimum(torch.clamp(insert_idx - 1, min=0), last_valid)

            left_times = torch.gather(phase_data["times"], 1, left_idx)
            right_times = torch.gather(phase_data["times"], 1, right_idx)
            left_diff = torch.abs(flat_query - left_times)
            right_diff = torch.abs(flat_query - right_times)
            use_left = left_diff <= right_diff
            best_idx = torch.where(use_left, left_idx, right_idx)
            best_pick = torch.where(use_left, left_times, right_times)
            best_diff = torch.where(use_left, left_diff, right_diff)

            best_prob = torch.gather(phase_data["prob"], 1, best_idx)
            best_amp = torch.gather(phase_data["amp"], 1, best_idx)
            best_uid = torch.gather(phase_data["pick_uid"], 1, best_idx)

            matched = valid_query & has_pick & torch.isfinite(best_pick) & (best_diff <= flat_tol)

            err = torch.full_like(flat_query, float("nan"))
            prob = torch.full_like(flat_query, float("nan"))
            amp = torch.full_like(flat_query, float("nan"))
            pick = torch.full_like(flat_query, float("nan"))
            pick_uid = torch.full(flat_query.shape, -1, dtype=torch.int32, device=self.compute_device)

            err = torch.where(matched, flat_query - best_pick, err)
            prob = torch.where(matched, best_prob, prob)
            amp = torch.where(matched, best_amp, amp)
            pick = torch.where(matched, best_pick, pick)
            pick_uid = torch.where(matched, best_uid, pick_uid)
            return err, prob, amp, pick, pick_uid

    def _lookup_direct_station(self, phase_data, flat_query, flat_tol):
        with timed(self.logger, "phase_index.lookup.direct_station"):
            valid_query = torch.isfinite(flat_query) & torch.isfinite(flat_tol)
            err = torch.full_like(flat_query, float("nan"))
            prob = torch.full_like(flat_query, float("nan"))
            amp = torch.full_like(flat_query, float("nan"))
            pick = torch.full_like(flat_query, float("nan"))
            pick_uid = torch.full(flat_query.shape, -1, dtype=torch.int32, device=self.compute_device)

            for station_idx in range(self.max_station_id):
                length = int(phase_data["lengths"][station_idx].item())
                if length == 0:
                    continue
                station_valid_idx = torch.nonzero(valid_query[station_idx], as_tuple=False).squeeze(1)
                if station_valid_idx.numel() == 0:
                    continue

                station_query = flat_query[station_idx, station_valid_idx]
                station_tol = flat_tol[station_idx, station_valid_idx]
                station_times = phase_data["times"][station_idx, :length]
                station_prob = phase_data["prob"][station_idx, :length]
                station_amp = phase_data["amp"][station_idx, :length]
                station_uid = phase_data["pick_uid"][station_idx, :length]

                station_diff = station_query.unsqueeze(1) - station_times.unsqueeze(0)
                station_abs_diff = torch.abs(station_diff)
                best_abs_diff, best_idx = torch.min(station_abs_diff, dim=1)
                matched = best_abs_diff <= station_tol
                if not bool(matched.any().item()):
                    continue

                matched_query_idx = station_valid_idx[matched]
                matched_pick_idx = best_idx[matched]
                matched_pick = station_times[matched_pick_idx]

                err[station_idx, matched_query_idx] = station_query[matched] - matched_pick
                prob[station_idx, matched_query_idx] = station_prob[matched_pick_idx]
                amp[station_idx, matched_query_idx] = station_amp[matched_pick_idx]
                pick[station_idx, matched_query_idx] = matched_pick
                pick_uid[station_idx, matched_query_idx] = station_uid[matched_pick_idx]

            return err, prob, amp, pick, pick_uid

    def _lookup_direct_chunked(self, phase_data, flat_query, flat_tol):
        with timed(self.logger, "phase_index.lookup.direct_chunked"):
            valid_query = torch.isfinite(flat_query) & torch.isfinite(flat_tol)
            err = torch.full_like(flat_query, float("nan"))
            prob = torch.full_like(flat_query, float("nan"))
            amp = torch.full_like(flat_query, float("nan"))
            pick = torch.full_like(flat_query, float("nan"))
            pick_uid = torch.full(flat_query.shape, -1, dtype=torch.int32, device=self.compute_device)

            phase_times = phase_data["times"]
            phase_prob = phase_data["prob"]
            phase_amp = phase_data["amp"]
            phase_uid = phase_data["pick_uid"]
            query_count = flat_query.shape[1]

            for start in range(0, query_count, self.lookup_query_chunk_size):
                end = min(start + self.lookup_query_chunk_size, query_count)
                query_chunk = flat_query[:, start:end]
                tol_chunk = flat_tol[:, start:end]
                valid_chunk = valid_query[:, start:end]
                safe_query_chunk = torch.where(valid_chunk, query_chunk, torch.zeros_like(query_chunk))

                diff_chunk = safe_query_chunk.unsqueeze(-1) - phase_times.unsqueeze(1)
                abs_diff_chunk = torch.abs(diff_chunk)
                best_abs_diff, best_idx = torch.min(abs_diff_chunk, dim=2)

                best_pick = torch.gather(phase_times, 1, best_idx)
                best_prob = torch.gather(phase_prob, 1, best_idx)
                best_amp = torch.gather(phase_amp, 1, best_idx)
                best_uid = torch.gather(phase_uid, 1, best_idx)

                matched = valid_chunk & torch.isfinite(best_pick) & (best_abs_diff <= tol_chunk)
                err[:, start:end] = torch.where(matched, query_chunk - best_pick, err[:, start:end])
                prob[:, start:end] = torch.where(matched, best_prob, prob[:, start:end])
                amp[:, start:end] = torch.where(matched, best_amp, amp[:, start:end])
                pick[:, start:end] = torch.where(matched, best_pick, pick[:, start:end])
                pick_uid[:, start:end] = torch.where(matched, best_uid, pick_uid[:, start:end])

            return err, prob, amp, pick, pick_uid

    def remove_pick_ids(self, pick_ids):
        if not pick_ids:
            return self
        pick_ids = set(int(value) for value in pick_ids)
        with timed(self.logger, "phase_index.remove_pick_ids.filter"):
            filtered = self.df[~self.df["pick_uid"].isin(pick_ids)].copy()
        with timed(self.logger, "phase_index.remove_pick_ids.rebuild"):
            return PhasePickIndex(
                filtered,
                self.max_station_id,
                device=self.compute_device,
                time_step=self.time_step,
                logger=self.logger,
                lookup_mode=self.lookup_mode,
                lookup_query_chunk_size=self.lookup_query_chunk_size,
                local_ref_msec=self.local_ref_msec,
                reference_global_msec=self.reference_global_msec,
            )

    def remaining_p_df(self, excluded_pick_ids=None):
        df = self.df
        if excluded_pick_ids:
            excluded_pick_ids = set(int(value) for value in excluded_pick_ids)
            df = df[~df["pick_uid"].isin(excluded_pick_ids)]
        return df[df["phasetype"] == "P"].copy()

    def _slice_phase_window(self, phase_data, start_time, end_time):
        phase_device = phase_data["times"].device
        max_len = phase_data["times"].shape[1]
        if max_len == 0:
            empty_phase = self._empty_phase_arrays(phase_device)
            return empty_phase, 0

        time_dtype = phase_data["times"].dtype
        start_values = torch.full((self.max_station_id, 1), float(start_time), dtype=time_dtype, device=phase_device)
        end_values = torch.full((self.max_station_id, 1), float(end_time), dtype=time_dtype, device=phase_device)

        start_idx = torch.searchsorted(phase_data["times"], start_values, right=False).squeeze(1)
        end_idx = torch.searchsorted(phase_data["times"], end_values, right=True).squeeze(1)
        lengths = torch.clamp(end_idx - start_idx, min=0)
        max_window_len = int(lengths.max().item())

        if max_window_len == 0:
            empty_phase = self._empty_phase_arrays(phase_device)
            empty_phase["lengths"] = lengths
            return empty_phase, 0

        offsets = torch.arange(max_window_len, device=phase_device).unsqueeze(0)
        gather_idx = start_idx.unsqueeze(1) + offsets
        valid = offsets < lengths.unsqueeze(1)
        safe_idx = torch.where(valid, gather_idx, torch.zeros_like(gather_idx))

        times = torch.gather(phase_data["times"], 1, safe_idx)
        prob = torch.gather(phase_data["prob"], 1, safe_idx)
        amp = torch.gather(phase_data["amp"], 1, safe_idx)
        pick_uid = torch.gather(phase_data["pick_uid"], 1, safe_idx)

        times = torch.where(valid, times, torch.full_like(times, float("inf")))
        prob = torch.where(valid, prob, torch.full_like(prob, float("nan")))
        amp = torch.where(valid, amp, torch.full_like(amp, float("nan")))
        pick_uid = torch.where(valid, pick_uid, torch.full_like(pick_uid, -1))

        return {
            "times": times,
            "prob": prob,
            "amp": amp,
            "pick_uid": pick_uid,
            "lengths": lengths,
        }, int(lengths.sum().item())

    def window(self, start_time, end_time, s_start_time=None, s_end_time=None, local_ref_msec=None, reference_global_msec=None):
        child_reference_global_msec = self.reference_global_msec if reference_global_msec is None else int(reference_global_msec)
        if self.df.empty:
            return self

        start_abs_msec = int(round(float(start_time) * 1000.0))
        end_abs_msec = int(round(float(end_time) * 1000.0))
        if s_start_time is None:
            s_start_time = start_time
        if s_end_time is None:
            s_end_time = end_time
        s_start_abs_msec = int(round(float(s_start_time) * 1000.0))
        s_end_abs_msec = int(round(float(s_end_time) * 1000.0))

        local_window = local_ref_msec is not None and self._time_sort_col == "GlobalMsec"
        if local_window:
            local_ref_msec = int(local_ref_msec)
            start_time = (start_abs_msec - local_ref_msec) / 1000.0
            end_time = (end_abs_msec - local_ref_msec) / 1000.0
            s_start_time = (s_start_abs_msec - local_ref_msec) / 1000.0
            s_end_time = (s_end_abs_msec - local_ref_msec) / 1000.0
        else:
            start_time = float(start_time)
            end_time = float(end_time)
            s_start_time = float(s_start_time)
            s_end_time = float(s_end_time)
        if start_time > end_time:
            start_time, end_time = end_time, start_time
            start_abs_msec, end_abs_msec = end_abs_msec, start_abs_msec
        if s_start_time > s_end_time:
            s_start_time, s_end_time = s_end_time, s_start_time
            s_start_abs_msec, s_end_abs_msec = s_end_abs_msec, s_start_abs_msec

        if local_window:
            overall_start = min(start_abs_msec, s_start_abs_msec)
            overall_end = max(end_abs_msec, s_end_abs_msec)
        else:
            overall_start = min(start_time, s_start_time)
            overall_end = max(end_time, s_end_time)

        key = (
            start_abs_msec if local_window else int(np.floor(start_time / self.time_step)),
            end_abs_msec if local_window else int(np.ceil(end_time / self.time_step)),
            s_start_abs_msec if local_window else int(np.floor(s_start_time / self.time_step)),
            s_end_abs_msec if local_window else int(np.ceil(s_end_time / self.time_step)),
            int(local_ref_msec) if local_ref_msec is not None else -1,
        )
        cached = self.window_lookup.get(key)
        if cached is not None:
            self.window_lookup.move_to_end(key)
            return cached

        with timed(self.logger, "phase_index.window"):
            with timed(self.logger, "phase_index.window.df_slice"):
                start_idx = int(np.searchsorted(self._sort_times, overall_start, side="left"))
                end_idx = int(np.searchsorted(self._sort_times, overall_end, side="right"))
                window_df = self.df.iloc[start_idx:end_idx].copy()
                if not window_df.empty:
                    if local_window:
                        p_times = window_df["GlobalMsec"]
                        s_times = p_times
                        mask = (
                            ((window_df["phasetype"] == "P") & (p_times >= start_abs_msec) & (p_times <= end_abs_msec))
                            | ((window_df["phasetype"] == "S") & (s_times >= s_start_abs_msec) & (s_times <= s_end_abs_msec))
                        )
                    else:
                        mask = (
                            ((window_df["phasetype"] == "P") & (window_df["RelativeTime"] >= start_time) & (window_df["RelativeTime"] <= end_time))
                            | ((window_df["phasetype"] == "S") & (window_df["RelativeTime"] >= s_start_time) & (window_df["RelativeTime"] <= s_end_time))
                        )
                    window_df = window_df[mask].copy()
                if local_window and not window_df.empty:
                    window_df["RelativeTime"] = (window_df["GlobalMsec"].to_numpy(dtype=np.int64, copy=False) - local_ref_msec) / 1000.0

            if local_window:
                ref_msec = child_reference_global_msec
                ref_shift = (ref_msec - local_ref_msec) / 1000.0
                p_start_ref = (start_abs_msec - ref_msec) / 1000.0
                p_end_ref = (end_abs_msec - ref_msec) / 1000.0
                s_start_ref = (s_start_abs_msec - ref_msec) / 1000.0
                s_end_ref = (s_end_abs_msec - ref_msec) / 1000.0
                with timed(self.logger, "phase_index.window.slice_p"):
                    p_phase, p_count = self._slice_phase_window(self.padded_phase["P"], p_start_ref, p_end_ref)
                with timed(self.logger, "phase_index.window.slice_s"):
                    s_phase, s_count = self._slice_phase_window(self.padded_phase["S"], s_start_ref, s_end_ref)
                with timed(self.logger, "phase_index.window.materialize_p"):
                    p_phase = self._phase_data_to_device(p_phase, self.compute_device)
                    p_phase["times"] = (p_phase["times"] + ref_shift).to(dtype=torch.float32)
                with timed(self.logger, "phase_index.window.materialize_s"):
                    s_phase = self._phase_data_to_device(s_phase, self.compute_device)
                    s_phase["times"] = (s_phase["times"] + ref_shift).to(dtype=torch.float32)
                phase_arrays = {"P": p_phase, "S": s_phase}
                counts = {"P": p_count, "S": s_count}
            else:
                with timed(self.logger, "phase_index.window.slice_p"):
                    p_phase, p_count = self._slice_phase_window(self.padded_phase["P"], start_time, end_time)
                with timed(self.logger, "phase_index.window.slice_s"):
                    s_phase, s_count = self._slice_phase_window(self.padded_phase["S"], s_start_time, s_end_time)
                with timed(self.logger, "phase_index.window.materialize_p"):
                    p_phase = self._phase_data_to_device(p_phase, self.compute_device)
                with timed(self.logger, "phase_index.window.materialize_s"):
                    s_phase = self._phase_data_to_device(s_phase, self.compute_device)
                phase_arrays = {"P": p_phase, "S": s_phase}
                counts = {"P": p_count, "S": s_count}

            with timed(self.logger, "phase_index.window.build_index"):
                result = PhasePickIndex(
                    window_df,
                    self.max_station_id,
                    device=self.compute_device,
                    phase_arrays=phase_arrays,
                    counts=counts,
                    time_step=self.time_step,
                    window_lookup=self.window_lookup,
                    logger=self.logger,
                    lookup_mode=self.lookup_mode,
                    lookup_query_chunk_size=self.lookup_query_chunk_size,
                    local_ref_msec=local_ref_msec,
                    reference_global_msec=child_reference_global_msec,
                )
        self._cache_window_result(key, result)
        return result
