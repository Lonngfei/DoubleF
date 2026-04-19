import math
import os

import numpy as np
import pandas as pd
import yaml
from pyrocko import cake


def station_key_to_filename(station_key: str) -> str:
    return station_key.replace("_", ".")


def step_to_decimals(step: float) -> int:
    text = f"{step:.10f}".rstrip("0").rstrip(".")
    if "." not in text:
        return 0
    return len(text.split(".")[1])


class TravelTime(object):
    def __init__(
        self,
        filename,
        logger,
        cache_dir,
        project_name,
        sdepth_max=31,
        depth_step=1,
        distance_max=2.0,
        distance_step=0.01,
        lookup_grid_step_km=0.01,
        station_df=None,
        consider_station_elevation=False,
    ):
        self.model = cake.load_model(filename)
        self.filename = filename
        self.logger = logger
        self.cache_dir = cache_dir
        self.project_name = project_name
        self.sdepth_max = sdepth_max
        self.depth_step = depth_step
        self.distance_max = distance_max
        self.distance_step = distance_step
        self.lookup_grid_step_km = lookup_grid_step_km
        self.station_df = station_df
        self.consider_station_elevation = consider_station_elevation

        self.coarse_depths_km = np.arange(0, sdepth_max + 1e-6, depth_step, dtype=np.float32)
        self.coarse_distances_deg = np.arange(0, distance_max + 1e-6, distance_step, dtype=np.float32)
        self.coarse_distances_km = (self.coarse_distances_deg * cake.d2m / 1000.0).astype(np.float32)

        max_distance_km = float(self.coarse_distances_km[-1])
        self.lookup_distances_km = np.arange(0, max_distance_km + 1e-6, lookup_grid_step_km, dtype=np.float32)
        self.lookup_depths_km = np.arange(0, sdepth_max + 1e-6, lookup_grid_step_km, dtype=np.float32)
        self.lookup_decimals = step_to_decimals(lookup_grid_step_km)

    @staticmethod
    def _ensure_dir(path):
        os.makedirs(path, exist_ok=True)

    @staticmethod
    def _station_key_frame(station_df):
        station_df = station_df.copy()
        station_df["net_sta"] = station_df["network"].astype(str) + "_" + station_df["station"].astype(str)
        station_df = station_df[["network", "station", "elevation", "net_sta"]].drop_duplicates("net_sta")
        station_df = station_df.sort_values("net_sta").reset_index(drop=True)
        return station_df

    def _table_path(self, phase, station_key=None):
        base = station_key_to_filename(station_key) if station_key else self.project_name
        return os.path.join(self.cache_dir, f"{base}.{phase}.npy")

    def _phase_grid(self, model):
        p = cake.PhaseDef("p")
        s = cake.PhaseDef("s")
        P = cake.PhaseDef("P<(moho)")
        S = cake.PhaseDef("S<(moho)")
        Pn = cake.PhaseDef("Pv_(moho)p")
        Sn = cake.PhaseDef("Sv_(moho)s")

        p_grid = np.full((len(self.coarse_distances_km), len(self.coarse_depths_km)), np.nan, dtype=np.float32)
        s_grid = np.full((len(self.coarse_distances_km), len(self.coarse_depths_km)), np.nan, dtype=np.float32)

        for depth_idx, source_depth in enumerate(self.coarse_depths_km):
            p_hits = {}
            s_hits = {}

            for arrival in model.arrivals(self.coarse_distances_deg, phases=p, zstart=float(source_depth) * 1000.0, zstop=0.0):
                p_hits.setdefault(round(arrival.x * cake.d2m / 1000.0, 3), []).append(arrival.t)
            for arrival in model.arrivals(self.coarse_distances_deg, phases=P, zstart=float(source_depth) * 1000.0, zstop=0.0):
                p_hits.setdefault(round(arrival.x * cake.d2m / 1000.0, 3), []).append(arrival.t)
            for arrival in model.arrivals(self.coarse_distances_deg, phases=Pn, zstart=float(source_depth) * 1000.0, zstop=0.0):
                p_hits.setdefault(round(arrival.x * cake.d2m / 1000.0, 3), []).append(arrival.t)

            for arrival in model.arrivals(self.coarse_distances_deg, phases=s, zstart=float(source_depth) * 1000.0, zstop=0.0):
                s_hits.setdefault(round(arrival.x * cake.d2m / 1000.0, 3), []).append(arrival.t)
            for arrival in model.arrivals(self.coarse_distances_deg, phases=S, zstart=float(source_depth) * 1000.0, zstop=0.0):
                s_hits.setdefault(round(arrival.x * cake.d2m / 1000.0, 3), []).append(arrival.t)
            for arrival in model.arrivals(self.coarse_distances_deg, phases=Sn, zstart=float(source_depth) * 1000.0, zstop=0.0):
                s_hits.setdefault(round(arrival.x * cake.d2m / 1000.0, 3), []).append(arrival.t)

            for distance_idx, distance_km in enumerate(self.coarse_distances_km):
                key = round(float(distance_km), 3)
                if key in p_hits:
                    p_grid[distance_idx, depth_idx] = min(p_hits[key])
                if key in s_hits:
                    s_grid[distance_idx, depth_idx] = min(s_hits[key])

        return p_grid, s_grid

    @staticmethod
    def _fill_missing_along_axis(values, axis):
        arr = np.array(values, copy=True)
        if axis == 1:
            arr = arr.T

        x = np.arange(arr.shape[1], dtype=np.float32)
        for i in range(arr.shape[0]):
            row = arr[i]
            valid = np.isfinite(row)
            if valid.all():
                continue
            if valid.any():
                row[~valid] = np.interp(x[~valid], x[valid], row[valid])
            arr[i] = row

        if axis == 1:
            arr = arr.T
        return arr

    def _fill_missing(self, values):
        filled = self._fill_missing_along_axis(values, axis=0)
        filled = self._fill_missing_along_axis(filled, axis=1)
        return filled.astype(np.float32)

    def _resample_grid(self, values):
        coarse = self._fill_missing(values)
        depth_resampled = np.empty((coarse.shape[0], len(self.lookup_depths_km)), dtype=np.float32)
        for i in range(coarse.shape[0]):
            depth_resampled[i] = np.interp(self.lookup_depths_km, self.coarse_depths_km, coarse[i]).astype(np.float32)

        dense = np.empty((len(self.lookup_distances_km), len(self.lookup_depths_km)), dtype=np.float32)
        for j in range(depth_resampled.shape[1]):
            dense[:, j] = np.interp(self.lookup_distances_km, self.coarse_distances_km, depth_resampled[:, j]).astype(np.float32)

        return np.round(dense, decimals=max(self.lookup_decimals + 1, 3)).astype(np.float32)

    def _save_pair(self, p_grid, s_grid, station_key=None):
        np.save(self._table_path("P", station_key), self._resample_grid(p_grid))
        np.save(self._table_path("S", station_key), self._resample_grid(s_grid))

    def _write_metadata(self, metadata):
        metadata_path = os.path.join(self.cache_dir, f"{self.project_name}.travel_time.yaml")
        with open(metadata_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(metadata, f, sort_keys=False, allow_unicode=True)

    def _run_global(self):
        p_grid, s_grid = self._phase_grid(self.model)
        self._save_pair(p_grid, s_grid)

        self._write_metadata({
            "mode": "project-wide",
            "project_name": self.project_name,
            "velocity_model": self.filename,
            "cache_directory": self.cache_dir,
            "lookup_grid_step_km": float(self.lookup_grid_step_km),
            "lookup_decimals": int(self.lookup_decimals),
            "distance_grid_km": self.lookup_distances_km.tolist(),
            "depth_grid_km": self.lookup_depths_km.tolist(),
            "p_file": self._table_path("P"),
            "s_file": self._table_path("S"),
        })

    def _run_station_elevation(self):
        if self.station_df is None:
            raise ValueError("station_df is required when consider_station_elevation=True")

        station_df = self._station_key_frame(self.station_df)
        if station_df.empty:
            raise ValueError("station_df is empty, cannot build station-elevation travel-time tables")

        station_df["elevation_key"] = station_df["elevation"].astype(float).round(6)
        grouped = station_df.groupby("elevation_key")

        metadata = {
            "mode": "station-elevation",
            "project_name": self.project_name,
            "velocity_model": self.filename,
            "cache_directory": self.cache_dir,
            "lookup_grid_step_km": float(self.lookup_grid_step_km),
            "lookup_decimals": int(self.lookup_decimals),
            "distance_grid_km": self.lookup_distances_km.tolist(),
            "depth_grid_km": self.lookup_depths_km.tolist(),
            "stations": [],
        }

        for elevation, group in grouped:
            model = self.model.copy_with_elevation(float(elevation) * 1000.0)
            p_grid, s_grid = self._phase_grid(model)

            for _, row in group.iterrows():
                station_key = row["net_sta"]
                self._save_pair(p_grid, s_grid, station_key=station_key)
                metadata["stations"].append({
                    "station_key": station_key,
                    "elevation_km": float(row["elevation"]),
                    "p_file": self._table_path("P", station_key),
                    "s_file": self._table_path("S", station_key),
                })

        self._write_metadata(metadata)

    def run(self):
        self._ensure_dir(self.cache_dir)
        if self.consider_station_elevation:
            self._run_station_elevation()
        else:
            self._run_global()
