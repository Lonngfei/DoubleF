import os

import numpy as np
import pandas as pd
import torch

from .get_tt import station_key_to_filename, step_to_decimals

PICK_TIME_BIN_SECONDS = 1
PICK_CANONICAL_MSEC_THRESHOLD = 10


def _days_before_year(year):
    year = np.asarray(year, dtype=np.int64)
    y = year - 1
    return 365 * y + y // 4 - y // 100 + y // 400


def _day_number(year, julday):
    return _days_before_year(year) + np.asarray(julday, dtype=np.int64) - 1


def datetime_to_time_code(datetimes):
    dt = pd.to_datetime(datetimes, utc=True)
    code = np.empty((len(dt), 6), dtype=np.float32)
    code[:, 0] = dt.dt.year.to_numpy(dtype=np.int32, copy=False)
    code[:, 1] = dt.dt.dayofyear.to_numpy(dtype=np.int32, copy=False)
    code[:, 2] = dt.dt.hour.to_numpy(dtype=np.int32, copy=False)
    code[:, 3] = dt.dt.minute.to_numpy(dtype=np.int32, copy=False)
    code[:, 4] = dt.dt.second.to_numpy(dtype=np.int32, copy=False)
    code[:, 5] = (dt.dt.microsecond.to_numpy(dtype=np.int64, copy=False) // 1000).astype(np.int32, copy=False)
    return code


def time_code_diff_seconds(time_code, reference_code):
    code = np.asarray(time_code, dtype=np.float32)
    ref = np.asarray(reference_code, dtype=np.float32)
    code_i = code.astype(np.int64, copy=False)
    ref_i = ref.astype(np.int64, copy=False)
    day_diff = _day_number(code_i[:, 0], code_i[:, 1]) - _day_number(ref_i[0], ref_i[1])
    msec = (((code_i[:, 2] * 60 + code_i[:, 3]) * 60 + code_i[:, 4]) * 1000 + code_i[:, 5])
    ref_msec = (((ref_i[2] * 60 + ref_i[3]) * 60 + ref_i[4]) * 1000 + ref_i[5])
    diff_msec = day_diff * 86_400_000 + msec - ref_msec
    return diff_msec.astype(np.float64) / 1000.0


def time_code_to_global_msec(time_code):
    code = np.asarray(time_code, dtype=np.float32).astype(np.int64, copy=False)
    day_number = _day_number(code[:, 0], code[:, 1])
    msec_of_day = (((code[:, 2] * 60 + code[:, 3]) * 60 + code[:, 4]) * 1000 + code[:, 5])
    return day_number * 86_400_000 + msec_of_day


def assign_canonical_pick_uid(df, threshold_msec=PICK_CANONICAL_MSEC_THRESHOLD):
    if df.empty:
        df["pick_uid"] = np.empty(0, dtype=np.int32)
        return df

    order = np.lexsort((
        df["GlobalMsec"].to_numpy(dtype=np.int64, copy=False),
        df["phasetype"].astype(str).to_numpy(),
        df["id"].to_numpy(dtype=np.int64, copy=False),
    ))
    station_values = df["id"].to_numpy(dtype=np.int64, copy=False)[order]
    phase_values = df["phasetype"].astype(str).to_numpy()[order]
    time_values = df["GlobalMsec"].to_numpy(dtype=np.int64, copy=False)[order]

    uid_sorted = np.empty(len(df), dtype=np.int32)
    uid = np.int64(0)
    group_start_time = time_values[0]
    uid_sorted[0] = np.int32(uid)
    for idx in range(1, len(df)):
        same_pick = (
            station_values[idx] == station_values[idx - 1]
            and phase_values[idx] == phase_values[idx - 1]
            and time_values[idx] - group_start_time < threshold_msec
        )
        if not same_pick:
            uid += 1
            if uid > np.iinfo(np.int32).max:
                raise ValueError("Too many canonical picks for int32 pick_uid")
            group_start_time = time_values[idx]
        uid_sorted[idx] = np.int32(uid)

    pick_uid = np.empty(len(df), dtype=np.int32)
    pick_uid[order] = uid_sorted
    df = df.copy()
    df["pick_uid"] = pick_uid
    return df


class CsvTorch(object):
    def __init__(
        self,
        cache_dir,
        project_name,
        phase_csv,
        device,
        year=0,
        month=0,
        day=0,
        station_file=None,
        consider_station_elevation=False,
        lookup_grid_step_km=0.01,
        tt_depth_step_km=None,
        phase_df=None,
        reference_time=None,
    ):
        self.cache_dir = cache_dir
        self.project_name = project_name
        self.phase_csv = phase_csv
        self.device = device
        self.year = year
        self.month = month
        self.day = day
        self.station_file = station_file
        self.consider_station_elevation = consider_station_elevation
        self.lookup_grid_step_km = lookup_grid_step_km
        self.tt_depth_step_km = tt_depth_step_km if tt_depth_step_km is not None else lookup_grid_step_km
        self.pick_time_round_decimals = max(step_to_decimals(lookup_grid_step_km), 2)
        self.phase_df = phase_df
        self.reference_time = reference_time

    def _table_path(self, phase, station_key=None):
        base = station_key_to_filename(station_key) if station_key is not None else self.project_name
        return os.path.join(self.cache_dir, f"{base}.{phase}.npy")

    def _load_station_catalog(self):
        if not self.station_file:
            return None
        station_df = pd.read_csv(self.station_file)
        station_df["net_sta"] = station_df["network"].astype(str) + "_" + station_df["station"].astype(str)
        station_df = station_df[["network", "station", "latitude", "longitude", "elevation", "net_sta"]]
        station_df = station_df.drop_duplicates("net_sta").reset_index(drop=True)
        return station_df

    def _load_source_df(self):
        if self.phase_df is not None:
            return self.phase_df
        return pd.read_csv(self.phase_csv)

    def _get_reference_time(self, df):
        if self.reference_time is not None:
            return self.reference_time
        if self.year and self.month and self.day:
            return pd.Timestamp(year=self.year, month=self.month, day=self.day, tz="UTC")
        min_time = pd.to_datetime(df["Time"], utc=True).min()
        return min_time.normalize()

    def _read_phase_csv(self):
        self.df = self._load_source_df()
        self.df["Time"] = pd.to_datetime(self.df["Time"], utc=True)
        self.reference_time = self._get_reference_time(self.df)
        time_code = datetime_to_time_code(self.df["Time"])
        reference_code = datetime_to_time_code(pd.Series([self.reference_time]))[0]
        self.df["TimeYear"] = time_code[:, 0]
        self.df["TimeJulday"] = time_code[:, 1]
        self.df["TimeHour"] = time_code[:, 2]
        self.df["TimeMinute"] = time_code[:, 3]
        self.df["TimeSecond"] = time_code[:, 4]
        self.df["TimeMillisecond"] = time_code[:, 5]
        self.df["GlobalMsec"] = time_code_to_global_msec(time_code)
        self.df["RelativeTime"] = time_code_diff_seconds(time_code, reference_code)
        self.df["RelativeTime"] = self.df["RelativeTime"].round(self.pick_time_round_decimals)
        self.df["net_sta"] = self.df["network"].astype(str) + "_" + self.df["station"].astype(str)

        station_df = self._load_station_catalog()
        if station_df is None:
            required = {"latitude", "longitude", "elevation"}
            missing = required - set(self.df.columns)
            if missing:
                raise ValueError(
                    f"Missing station columns in phase CSV: {sorted(missing)}. "
                    "Provide input.station-file or include station coordinates in picks."
                )
            return

        drop_cols = [col for col in ("latitude", "longitude", "elevation") if col in self.df.columns]
        if drop_cols:
            self.df = self.df.drop(columns=drop_cols)

        self.df = self.df.merge(
            station_df[["net_sta", "latitude", "longitude", "elevation"]],
            on="net_sta",
            how="left",
        )

        missing_station_rows = self.df["latitude"].isna() | self.df["longitude"].isna() | self.df["elevation"].isna()
        if missing_station_rows.any():
            missing_keys = self.df.loc[missing_station_rows, "net_sta"].drop_duplicates().tolist()
            example = ", ".join(missing_keys[:5])
            raise ValueError(f"Stations missing from station catalog ({len(missing_keys)}): {example}")

    def load_tt(self, selected_station_keys=None):
        depth_stride = max(1, int(round(self.tt_depth_step_km / self.lookup_grid_step_km)))

        def _load_table(path):
            arr = np.load(path, mmap_mode="r")
            if depth_stride > 1:
                arr = arr[:, ::depth_stride]
            return np.asarray(arr, dtype=np.float32)

        if self.consider_station_elevation:
            if not selected_station_keys:
                raise ValueError("selected_station_keys is required when station elevation is enabled")
            p_npy = np.stack([_load_table(self._table_path("P", key)) for key in selected_station_keys], axis=0)
            s_npy = np.stack([_load_table(self._table_path("S", key)) for key in selected_station_keys], axis=0)
        else:
            p_npy = _load_table(self._table_path("P"))
            s_npy = _load_table(self._table_path("S"))

        self.p_tt = torch.tensor(p_npy, dtype=torch.float32, device=self.device)
        self.s_tt = torch.tensor(s_npy, dtype=torch.float32, device=self.device)
        return self.p_tt, self.s_tt

    def max_tt_by_phase_from_cache(
        self,
        max_distance_km,
        max_depth_km,
        distance_step_km,
        depth_step_km,
        selected_station_keys=None,
    ):
        distance_idx = max(0, int(round(max_distance_km / distance_step_km)))
        depth_idx = max(0, int(round(max_depth_km / depth_step_km)))
        depth_stride = max(1, int(round(self.tt_depth_step_km / self.lookup_grid_step_km)))

        def _slice_max(path):
            arr = np.load(path, mmap_mode="r")
            sliced = arr[: distance_idx + 1, : depth_idx * depth_stride + 1 : depth_stride]
            value = np.nanmax(sliced)
            return float(value) if np.isfinite(value) else float("-inf")

        if self.consider_station_elevation:
            if not selected_station_keys:
                raise ValueError("selected_station_keys is required when station elevation is enabled")
            max_p_value = float("-inf")
            max_s_value = float("-inf")
            for key in selected_station_keys:
                max_p_value = max(max_p_value, _slice_max(self._table_path("P", key)))
                max_s_value = max(max_s_value, _slice_max(self._table_path("S", key)))
            return max_p_value, max_s_value

        return (
            _slice_max(self._table_path("P")),
            _slice_max(self._table_path("S")),
        )

    def max_tt_from_cache(
        self,
        max_distance_km,
        max_depth_km,
        distance_step_km,
        depth_step_km,
        selected_station_keys=None,
    ):
        max_p_tt, max_s_tt = self.max_tt_by_phase_from_cache(
            max_distance_km,
            max_depth_km,
            distance_step_km,
            depth_step_km,
            selected_station_keys=selected_station_keys,
        )
        return max(max_p_tt, max_s_tt)

    def generate_station_data(self):
        self._read_phase_csv()
        if len(self.df) == 0:
            return False

        duplicate_cols = [col for col in ("id", "RelativeTime_Bin", "group_key") if col in self.df.columns]
        if duplicate_cols:
            self.df = self.df.drop(columns=duplicate_cols)

        self.unique_stations = self.df[["net_sta", "latitude", "longitude", "elevation"]].drop_duplicates()
        self.unique_stations.reset_index(drop=True, inplace=True)
        self.unique_stations["id"] = self.unique_stations.index
        self.station_keys = self.unique_stations["net_sta"].tolist()
        self.station_tensor = torch.tensor(
            self.unique_stations[["id", "latitude", "longitude", "elevation"]].to_numpy(),
            dtype=torch.float32,
            device=self.device,
        )
        self.sta_dict = {row["id"]: row["net_sta"].split("_") for _, row in self.unique_stations.iterrows()}
        self.df = pd.merge(self.df, self.unique_stations[["net_sta", "id"]], on="net_sta", how="left")
        self.df = assign_canonical_pick_uid(self.df)
        self.num_stations = self.unique_stations["id"].nunique()
        self.max_id = self.station_tensor.shape[0]
        return self.station_tensor, self.num_stations, self.sta_dict

    def set_initial_points_from_df(self, p_df):
        p_df = p_df.copy()
        p_df = p_df.sort_values("RelativeTime").reset_index(drop=True)
        self.initial_point_df = p_df
        initial_point = torch.tensor(
            p_df[["latitude", "longitude", "RelativeTime"]].values,
            dtype=torch.float64,
            device=self.device,
        )
        number = initial_point.shape[0]
        constant_depth = torch.full((number, 1), 0, device=self.device, dtype=torch.float64)
        initial_point = torch.hstack((initial_point[:, :2], constant_depth, initial_point[:, 2:]))
        zero_matrix = torch.zeros((initial_point.shape[0], 1), dtype=torch.float64, device=self.device)
        score_matrix = torch.hstack((initial_point, zero_matrix))
        return initial_point, score_matrix

    def generate_initial_point(self):
        df_P = self.df[self.df["phasetype"] == "P"].copy()
        df_P = df_P[df_P["RelativeTime"] < 86400]
        return self.set_initial_points_from_df(df_P)

    def generate_phase_tensor(self, df_subset):
        df_subset = df_subset.copy()
        df_subset["RelativeTime_Bin"] = (df_subset["RelativeTime"] // PICK_TIME_BIN_SECONDS).astype(int)

        ids = torch.tensor(df_subset["id"].to_numpy(), dtype=torch.long, device=self.device)
        bins = torch.tensor(df_subset["RelativeTime_Bin"].to_numpy(), dtype=torch.long, device=self.device)
        rel_time = torch.tensor(df_subset["RelativeTime"].to_numpy(), dtype=torch.float64, device=self.device)
        prob = torch.tensor(df_subset["Probability"].to_numpy(), dtype=torch.float64, device=self.device)
        amp = torch.tensor(df_subset["Amplitude"].to_numpy(), dtype=torch.float64, device=self.device)

        df_subset["group_key"] = list(zip(ids.cpu().tolist(), bins.cpu().tolist()))
        group_counts = df_subset["group_key"].value_counts()
        max_rows = group_counts.max()

        time_bins = int(df_subset["RelativeTime_Bin"].max()) + 1

        result = torch.full(
            (max_rows, self.max_id, time_bins, 3),
            float("nan"),
            dtype=torch.float64,
            device=self.device,
        )

        row_indices = torch.empty(len(df_subset), dtype=torch.long, device=self.device)
        current_positions = {}

        for idx, (i, h) in enumerate(zip(ids.tolist(), bins.tolist())):
            key = (i, h)
            pos = current_positions.get(key, 0)
            row_indices[idx] = pos
            current_positions[key] = pos + 1

        indices = (
            row_indices,
            ids,
            bins,
        )

        result.index_put_(indices + (torch.tensor([0], device=self.device),), rel_time, accumulate=False)
        result.index_put_(indices + (torch.tensor([1], device=self.device),), prob, accumulate=False)
        result.index_put_(indices + (torch.tensor([2], device=self.device),), amp, accumulate=False)
        return result

    def generate_all_data(self):
        df_P = self.df[self.df["phasetype"] == "P"].copy()
        df_S = self.df[self.df["phasetype"] == "S"].copy()
        if len(df_P) == 0:
            raise ValueError("No P picks remain in current batch")
        P_tensor = self.generate_phase_tensor(df_P)
        S_tensor = self.generate_phase_tensor(df_S) if len(df_S) > 0 else torch.full(
            (1, self.max_id, P_tensor.shape[2], 3),
            float("nan"),
            dtype=torch.float64,
            device=self.device,
        )
        return P_tensor, S_tensor, len(df_P), len(df_S)
