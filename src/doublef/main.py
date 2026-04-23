#!/usr/bin/env python3
from __future__ import annotations
import gc
import logging
import os
import re
from datetime import date, datetime
from glob import glob

import pandas as pd
import torch
import yaml
from obspy import UTCDateTime
from tqdm import tqdm

from .csv_tensor import CsvTorch
from .get_tt import TravelTime
from .perf import get_time, timed
from .phase_index import PhasePickIndex
from .sampling import Sampler
from .write_results import GetResult


_MISSING = object()
_PICK_READ_CHUNK_ROWS = 200_000
_PICK_DATE_PATTERN = re.compile(r"(?<!\d)(\d{4})[-_]?(\d{2})[-_]?(\d{2})(?!\d)")
_BATCH_CLEANUP_INTERVAL = 32

_CONFIG_ALIASES = {
    "pick_dir": [("input", "pick-directory"), ("input", "pick-dir")],
    "station_file": [("input", "station-file")],
    "vel_model": [("travel-time", "velocity-model")],
    "cal_tt": [("travel-time", "recompute-travel-time")],
    "consider_station_elevation": [
        ("travel-time", "consider-station-elevation"),
        ("travel-time", "use-station-specific-travel-time"),
    ],
    "sdepth_max": [("travel-time", "maximum-source-depth-km"), ("travel-time", "sdepth-max-km")],
    "depth_step": [("travel-time", "source-depth-step-km"), ("travel-time", "depth-step-km")],
    "distance_max": [("travel-time", "maximum-distance-deg"), ("travel-time", "distance-max-deg")],
    "distance_step": [("travel-time", "distance-step-deg")],
    "lookup_grid_step_km": [("travel-time", "lookup-grid-step-km")],
    "iteration": [("sampling", "number-of-iterations"), ("sampling", "iteration")],
    "samples": [("sampling", "number-of-samples"), ("sampling", "samples")],
    "multiple": [("sampling", "sample-reduction-factor"), ("sampling", "multiple")],
    "top": [("sampling", "number-of-tops"), ("sampling", "top")],
    "quantile": [("sampling", "confidence-quantile"), ("sampling", "quantile")],
    "search_lat": [("sampling", "search-range-latitude-deg"), ("sampling", "search-lat")],
    "search_lon": [("sampling", "search-range-longitude-deg"), ("sampling", "search-lon")],
    "search_depth": [("sampling", "search-range-depth-km"), ("sampling", "search-depth-km")],
    "search_time": [("sampling", "search-range-time-s"), ("sampling", "search-time-s")],
    "repeat_rounds": [
        ("sampling", "repeat-rounds"),
        ("sampling", "association-rounds"),
        ("sampling", "allow-repeat-association"),
        ("sampling", "is-repeat"),
    ],
    "max_distance": [("scoring", "maximum-association-distance-km"), ("scoring", "max-distance-km")],
    "number_weight": [("scoring", "phase-count-weight"), ("scoring", "number-weight")],
    "mag_weight": [("scoring", "magnitude-consistency-weight"), ("scoring", "mag-weight")],
    "P_weight": [("scoring", "p-phase-weight"), ("scoring", "p-weight")],
    "number_type": [("scoring", "phase-count-method"), ("scoring", "number-type")],
    "time_type": [("scoring", "residual-method"), ("scoring", "time-type")],
    "dis0": [("scoring", "distance-weight-lower-bound-km"), ("scoring", "distance-weight-dis0-km")],
    "dis1": [("scoring", "distance-weight-upper-bound-km"), ("scoring", "distance-weight-dis1-km")],
    "min_P_tolerance": [("tolerance", "minimum-p-residual-s"), ("tolerance", "min-p-s")],
    "max_P_tolerance": [("tolerance", "maximum-p-residual-s"), ("tolerance", "max-p-s")],
    "min_S_tolerance": [("tolerance", "minimum-s-residual-s"), ("tolerance", "min-s-s")],
    "max_S_tolerance": [("tolerance", "maximum-s-residual-s"), ("tolerance", "max-s-s")],
    "code": [("output", "experiment-code"), ("output", "code")],
    "logger_dir": [("output", "output-directory"), ("output", "logger-dir")],
    "min_p": [("output", "minimum-p-phases"), ("output", "min-p")],
    "min_s": [("output", "minimum-s-phases"), ("output", "min-s")],
    "min_both": [("output", "minimum-p-and-s-stations"), ("output", "min-both")],
    "min_sum": [("output", "minimum-total-phases"), ("output", "min-sum")],
    "only_double": [("output", "require-p-and-s-at-same-station"), ("output", "only-double")],
    "device": [("runtime", "device")],
    "glob_batch_size": [("runtime", "glob-batch-size")],
    "lookup_mode": [("runtime", "lookup-mode")],
    "lookup_query_chunk_size": [("runtime", "lookup-query-chunk-size")],
    "profile_cuda_sync": [("runtime", "profile-cuda-sync")],
    "silent_screen": [("runtime", "silent-screen")],
    "year0": [("input", "year0")],
    "month0": [("input", "month0")],
    "day0": [("input", "day0")],
    "nday": [("input", "number-of-days"), ("input", "nday")],
}


def _get_nested(raw_cfg: dict, section: str, key: str, default: object = _MISSING):
    section_data = raw_cfg.get(section, {})
    if not isinstance(section_data, dict):
        if default is _MISSING:
            raise TypeError(f"Config section '{section}' must be a mapping")
        return default
    if key in section_data:
        return section_data[key]
    if default is _MISSING:
        raise KeyError(f"Missing required config key: {section}.{key}")
    return default



def _get_with_aliases(raw_cfg: dict, internal_name: str, default: object = _MISSING):
    for section, key in _CONFIG_ALIASES[internal_name]:
        try:
            value = _get_nested(raw_cfg, section, key)
        except KeyError:
            continue
        if value is not _MISSING:
            return value
    if default is _MISSING:
        aliases = ", ".join(f"{section}.{key}" for section, key in _CONFIG_ALIASES[internal_name])
        raise KeyError(f"Missing required config key for '{internal_name}'. Expected one of: {aliases}")
    return default



def _parse_start_date(raw_cfg: dict) -> tuple[int, int, int]:
    start_date = _get_nested(raw_cfg, "input", "start-date", None)
    if start_date is not None:
        if isinstance(start_date, datetime):
            start_date = start_date.date()
        if isinstance(start_date, date):
            return start_date.year, start_date.month, start_date.day
        if isinstance(start_date, str):
            parsed = datetime.strptime(start_date, "%Y-%m-%d")
            return parsed.year, parsed.month, parsed.day
        raise TypeError("Config key 'input.start-date' must be YYYY-MM-DD, date, or datetime")

    try:
        year0 = _get_with_aliases(raw_cfg, "year0")
        month0 = _get_with_aliases(raw_cfg, "month0")
        day0 = _get_with_aliases(raw_cfg, "day0")
    except KeyError:
        year0 = month0 = day0 = _MISSING
    if year0 is not _MISSING and month0 is not _MISSING and day0 is not _MISSING:
        return int(year0), int(month0), int(day0)

    pick_dir = _get_with_aliases(raw_cfg, "pick_dir", "")
    if isinstance(pick_dir, str):
        match = _PICK_DATE_PATTERN.search(os.path.basename(pick_dir))
        if match:
            year, month, day = (int(value) for value in match.groups())
            try:
                inferred_date = date(year, month, day)
            except ValueError:
                pass
            else:
                return inferred_date.year, inferred_date.month, inferred_date.day

    return 1970, 1, 1



def _normalize_raw_config(raw_cfg: dict) -> dict:
    year0, month0, day0 = _parse_start_date(raw_cfg)
    repeat_value = _get_with_aliases(raw_cfg, "repeat_rounds", 0)
    if isinstance(repeat_value, bool):
        repeat_rounds = 1 if repeat_value else 0
    else:
        repeat_rounds = int(repeat_value)

    cfg = {
        "pick_dir": _get_with_aliases(raw_cfg, "pick_dir"),
        "station_file": _get_with_aliases(raw_cfg, "station_file"),
        "vel_model": _get_with_aliases(raw_cfg, "vel_model"),
        "cal_tt": bool(_get_with_aliases(raw_cfg, "cal_tt")),
        "consider_station_elevation": bool(_get_with_aliases(raw_cfg, "consider_station_elevation", False)),
        "sdepth_max": float(_get_with_aliases(raw_cfg, "sdepth_max")),
        "depth_step": float(_get_with_aliases(raw_cfg, "depth_step")),
        "distance_max": float(_get_with_aliases(raw_cfg, "distance_max")),
        "distance_step": float(_get_with_aliases(raw_cfg, "distance_step")),
        "lookup_grid_step_km": float(_get_with_aliases(raw_cfg, "lookup_grid_step_km", 0.01)),
        "iteration": int(_get_with_aliases(raw_cfg, "iteration")),
        "samples": int(_get_with_aliases(raw_cfg, "samples")),
        "multiple": float(_get_with_aliases(raw_cfg, "multiple")),
        "top": int(_get_with_aliases(raw_cfg, "top")),
        "quantile": float(_get_with_aliases(raw_cfg, "quantile")),
        "search_lat": float(_get_with_aliases(raw_cfg, "search_lat")),
        "search_lon": float(_get_with_aliases(raw_cfg, "search_lon")),
        "search_depth": float(_get_with_aliases(raw_cfg, "search_depth")),
        "search_time": float(_get_with_aliases(raw_cfg, "search_time")),
        "repeat_rounds": repeat_rounds,
        "max_distance": float(_get_with_aliases(raw_cfg, "max_distance")),
        "number_weight": float(_get_with_aliases(raw_cfg, "number_weight")),
        "mag_weight": float(_get_with_aliases(raw_cfg, "mag_weight")),
        "P_weight": float(_get_with_aliases(raw_cfg, "P_weight")),
        "number_type": str(_get_with_aliases(raw_cfg, "number_type")),
        "time_type": str(_get_with_aliases(raw_cfg, "time_type")),
        "dis0": float(_get_with_aliases(raw_cfg, "dis0")),
        "dis1": float(_get_with_aliases(raw_cfg, "dis1")),
        "min_P_tolerance": float(_get_with_aliases(raw_cfg, "min_P_tolerance")),
        "max_P_tolerance": float(_get_with_aliases(raw_cfg, "max_P_tolerance")),
        "min_S_tolerance": float(_get_with_aliases(raw_cfg, "min_S_tolerance")),
        "max_S_tolerance": float(_get_with_aliases(raw_cfg, "max_S_tolerance")),
        "code": str(_get_with_aliases(raw_cfg, "code")),
        "logger_dir": _get_with_aliases(raw_cfg, "logger_dir"),
        "min_p": int(_get_with_aliases(raw_cfg, "min_p")),
        "min_s": int(_get_with_aliases(raw_cfg, "min_s")),
        "min_both": int(_get_with_aliases(raw_cfg, "min_both")),
        "min_sum": int(_get_with_aliases(raw_cfg, "min_sum")),
        "only_double": bool(_get_with_aliases(raw_cfg, "only_double")),
        "device": str(_get_with_aliases(raw_cfg, "device")),
        "glob_batch_size": int(_get_with_aliases(raw_cfg, "glob_batch_size", 0)),
        "lookup_mode": str(_get_with_aliases(raw_cfg, "lookup_mode", "searchsorted")),
        "lookup_query_chunk_size": int(_get_with_aliases(raw_cfg, "lookup_query_chunk_size", 4096)),
        "profile_cuda_sync": bool(_get_with_aliases(raw_cfg, "profile_cuda_sync", False)),
        "silent_screen": bool(_get_with_aliases(raw_cfg, "silent_screen", False)),
        "year0": year0,
        "month0": month0,
        "day0": day0,
        "nday": int(_get_with_aliases(raw_cfg, "nday", 1)),
    }

    if not 0.0 <= cfg["number_weight"] <= 1.0:
        raise ValueError("'scoring.phase-count-weight' must be between 0 and 1")
    if not 0.0 <= cfg["P_weight"] <= 1.0:
        raise ValueError("'scoring.p-phase-weight' must be between 0 and 1")
    if not 0.0 < cfg["quantile"] <= 1.0:
        raise ValueError("'sampling.quantile' must be in (0, 1]")
    if cfg["iteration"] < 1:
        raise ValueError("'sampling.number-of-iterations' must be at least 1")
    if cfg["repeat_rounds"] != -1 and cfg["repeat_rounds"] < 0:
        raise ValueError("'sampling.repeat-rounds' must be -1 or at least 0")
    if cfg["samples"] < 1:
        raise ValueError("'sampling.number-of-samples' must be at least 1")
    if cfg["top"] < 1:
        raise ValueError("'sampling.number-of-tops' must be at least 1")
    if cfg["glob_batch_size"] < 0:
        raise ValueError("'runtime.glob-batch-size' must be >= 0")
    if cfg["lookup_mode"] not in {"searchsorted", "direct-station", "direct-chunked"}:
        raise ValueError("'runtime.lookup-mode' must be one of: searchsorted, direct-station, direct-chunked")
    if cfg["lookup_query_chunk_size"] < 1:
        raise ValueError("'runtime.lookup-query-chunk-size' must be >= 1")
    if not cfg["station_file"]:
        raise ValueError("'input.station-file' is required")

    return cfg



def read_config(filepath: str) -> dict:
    with open(filepath, "r", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f)

    if not isinstance(raw_cfg, dict):
        raise TypeError("YAML config root must be a mapping")

    return _normalize_raw_config(raw_cfg)



def resolve_path(path_value: str, base_dir: str) -> str:
    if not isinstance(path_value, str):
        return path_value
    if os.path.isabs(path_value):
        return path_value
    return os.path.normpath(os.path.join(base_dir, path_value))



def normalize_config_paths(cfg: dict, config_dir: str) -> dict:
    path_like_keys = [
        "logger_dir",
        "vel_model",
        "pick_dir",
        "station_file",
    ]

    for key in path_like_keys:
        if cfg.get(key):
            cfg[key] = resolve_path(cfg[key], config_dir)

    return cfg



def to_effective_yaml(cfg: dict) -> dict:
    return {
        "input": {
            "pick-directory": cfg["pick_dir"],
            "start-date": f"{cfg['year0']:04d}-{cfg['month0']:02d}-{cfg['day0']:02d}",
            "station-file": cfg["station_file"],
            "processing-mode": "global-picks",
        },
        "travel-time": {
            "recompute-travel-time": cfg["cal_tt"],
            "velocity-model": cfg["vel_model"],
            "consider-station-elevation": cfg["consider_station_elevation"],
            "maximum-source-depth-km": cfg["sdepth_max"],
            "source-depth-step-km": cfg["depth_step"],
            "maximum-distance-deg": cfg["distance_max"],
            "distance-step-deg": cfg["distance_step"],
            "lookup-grid-step-km": cfg["lookup_grid_step_km"],
        },
        "sampling": {
            "number-of-iterations": cfg["iteration"],
            "number-of-samples": cfg["samples"],
            "multiple": cfg["multiple"],
            "number-of-tops": cfg["top"],
            "quantile": cfg["quantile"],
            "search-range-latitude-deg": cfg["search_lat"],
            "search-range-longitude-deg": cfg["search_lon"],
            "search-range-depth-km": cfg["search_depth"],
            "search-range-time-s": cfg["search_time"],
            "repeat-rounds": cfg["repeat_rounds"],
        },
        "scoring": {
            "maximum-association-distance-km": cfg["max_distance"],
            "phase-count-weight": cfg["number_weight"],
            "magnitude-consistency-weight": cfg["mag_weight"],
            "p-phase-weight": cfg["P_weight"],
            "phase-count-method": cfg["number_type"],
            "residual-method": cfg["time_type"],
            "distance-weight-lower-bound-km": cfg["dis0"],
            "distance-weight-upper-bound-km": cfg["dis1"],
        },
        "tolerance": {
            "minimum-p-residual-s": cfg["min_P_tolerance"],
            "maximum-p-residual-s": cfg["max_P_tolerance"],
            "minimum-s-residual-s": cfg["min_S_tolerance"],
            "maximum-s-residual-s": cfg["max_S_tolerance"],
        },
        "output": {
            "experiment-code": cfg["code"],
            "output-directory": cfg["logger_dir"],
            "minimum-p-phases": cfg["min_p"],
            "minimum-s-phases": cfg["min_s"],
            "minimum-p-and-s-stations": cfg["min_both"],
            "minimum-total-phases": cfg["min_sum"],
            "require-p-and-s-at-same-station": cfg["only_double"],
        },
        "runtime": {
            "device": cfg["device"],
            "glob-batch-size": cfg["glob_batch_size"],
            "lookup-mode": cfg["lookup_mode"],
            "lookup-query-chunk-size": cfg["lookup_query_chunk_size"],
            "silent-screen": cfg["silent_screen"],
        },
    }


def write_effective_config(path: str, cfg: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(to_effective_yaml(cfg), f, sort_keys=False, allow_unicode=True)


def create_logger(log_dir: str, code: str, sync_cuda_timing: bool = False, silent_screen: bool = False) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{code}.log")

    logger = logging.getLogger("doublef")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger._doublef_sync_cuda = bool(sync_cuda_timing)

    if logger.handlers:
        logger.handlers.clear()

    class ColorFormatter(logging.Formatter):
        BLUE = "\033[34m"
        RESET = "\033[0m"

        def format(self, record):
            original = self._style._fmt
            self._style._fmt = f"[{self.BLUE}%(asctime)s{self.RESET}] %(message)s"
            try:
                return super().format(record)
            finally:
                self._style._fmt = original

    class ConsoleFilter(logging.Filter):
        SUPPRESSED_PREFIXES = (
            "Timing self summary:",
            "Timing total summary:",
            "Run completed in ",
        )

        def filter(self, record):
            message = record.getMessage()
            return not any(message.startswith(prefix) for prefix in self.SUPPRESSED_PREFIXES)

    formatter = logging.Formatter(
        fmt="[%(asctime)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    color_formatter = ColorFormatter(
        fmt="[%(asctime)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)

    if not silent_screen:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(color_formatter)
        stream_handler.addFilter(ConsoleFilter())
        logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger


def get_travel_time_cache_dir(cfg: dict) -> str:
    return os.path.dirname(cfg["vel_model"])


def get_distance_step_km(cfg: dict) -> float:
    return float(cfg["lookup_grid_step_km"])


def _list_pick_files(pick_path: str, station_file: str | None) -> list[str]:
    pick_path = os.path.abspath(pick_path)
    if os.path.isfile(pick_path):
        if not pick_path.lower().endswith(".csv"):
            raise ValueError(f"Pick input must be a CSV file: {pick_path}")
        return [pick_path]

    if os.path.isdir(pick_path):
        pick_files = sorted(
            os.path.join(pick_path, filename)
            for filename in os.listdir(pick_path)
            if filename.lower().endswith(".csv")
        )
        if not pick_files:
            raise ValueError(f"No pick CSV files were found in directory: {pick_path}")
        return pick_files

    raise FileNotFoundError(f"Pick path does not exist: {pick_path}")


def _pick_csv_columns(path: str, require_station_columns: bool) -> list[str]:
    header = pd.read_csv(path, nrows=0)
    available = set(header.columns)
    required = {"network", "station", "phasetype", "Time", "Probability", "Amplitude"}
    if require_station_columns:
        required |= {"latitude", "longitude", "elevation"}
    missing = required - available
    if missing:
        raise ValueError(f"Pick CSV is missing required columns: {sorted(missing)} in {path}")

    ordered_columns = [
        "network",
        "station",
        "phasetype",
        "Time",
        "Probability",
        "Amplitude",
        "latitude",
        "longitude",
        "elevation",
        "pick_uid",
    ]
    return [column for column in ordered_columns if column in available]


def _read_pick_csv(path: str, require_station_columns: bool) -> pd.DataFrame:
    usecols = _pick_csv_columns(path, require_station_columns)
    dtype_map = {
        "network": "string",
        "station": "string",
        "phasetype": "string",
        "Time": "string",
        "Probability": "float32",
        "Amplitude": "float32",
        "latitude": "float32",
        "longitude": "float32",
        "elevation": "float32",
        "pick_uid": "int64",
    }
    read_kwargs = {
        "usecols": usecols,
        "dtype": {column: dtype_map[column] for column in usecols if column in dtype_map},
    }

    chunks = pd.read_csv(path, chunksize=_PICK_READ_CHUNK_ROWS, **read_kwargs)
    frames = [chunk for chunk in chunks]
    if not frames:
        return pd.DataFrame(columns=usecols)
    return pd.concat(frames, ignore_index=True, copy=False)


def load_pick_dataframe(pick_path: str, station_file: str | None) -> pd.DataFrame:
    pick_files = _list_pick_files(pick_path, station_file)
    df = _read_pick_csv(pick_files[0], require_station_columns=False)
    required = {"network", "station", "phasetype", "Time", "Probability", "Amplitude"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Pick CSV is missing required columns: {sorted(missing)}")
    return df


def _format_number(value: float, decimals: int = 3) -> str:
    if pd.isna(value):
        return "nan"
    return f"{float(value):.{decimals}f}"

def write_phase_report(output_file: str, events: list[dict]) -> None:
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    event_rows: list[list[str]] = []
    pick_rows_per_event: list[list[list[str]]] = []
    for event in events:
        event_time = event["origin_datetime"]
        y = event_time.year
        m = event_time.month
        d = event_time.day
        hour = event_time.hour
        minute = event_time.minute
        sec = event_time - UTCDateTime(y, m, d, hour, minute)
        event_rows.append([
            "#",
            f"{y:04d}",
            f"{m:02d}",
            f"{d:02d}",
            f"{hour:02d}",
            f"{minute:02d}",
            _format_number(sec, 3),
            _format_number(event["location"]["lat"], 6),
            _format_number(event["location"]["lon"], 6),
            _format_number(event["location"]["dep"], 3),
            _format_number(event["magnitude"], 1),
            _format_number(event["err_lat"], 6),
            _format_number(event["err_lon"], 6),
            _format_number(event["err_dep"], 3),
            _format_number(event["err_time"], 3),
            _format_number(event["rms"], 3),
            str(int(event["count_p"])),
            str(int(event["count_s"])),
            str(int(event["count_both"])),
            str(int(event["count_sum"])),
            str(int(event["final_event_id"])),
        ])

        event["picks"].sort(key=lambda item: (item["distance_km"], item["phase"], item["pick_time"]))
        pick_rows = []
        for pick in event["picks"]:
            pick_rows.append([
                str(pick["net"]),
                str(pick["station"]),
                _format_number(pick["distance_km"], 3),
                _format_number(pick["relative_pick"], 3),
                _format_number(pick["prob"], 2),
                str(pick["phase"]),
                _format_number(pick["err"], 3),
                _format_number(event["magnitude"], 1),
                _format_number(pick["amp"], 3),
            ])
        pick_rows_per_event.append(pick_rows)

    event_widths = [1] * 21
    for row in event_rows:
        for idx, value in enumerate(row):
            event_widths[idx] = max(event_widths[idx], len(value))

    pick_widths = [1] * 9
    for pick_rows in pick_rows_per_event:
        for row in pick_rows:
            for idx, value in enumerate(row):
                pick_widths[idx] = max(pick_widths[idx], len(value))

    with open(output_file, "w", encoding="utf-8") as f:
        for event_row, pick_rows in zip(event_rows, pick_rows_per_event):
            event_fields = [
                f"{value:<{event_widths[idx]}}" if idx == 0 else f"{value:>{event_widths[idx]}}"
                for idx, value in enumerate(event_row)
            ]
            f.write(" ".join(event_fields) + "\n")
            for row in pick_rows:
                pick_fields = []
                for idx, value in enumerate(row):
                    if idx in (0, 1, 5):
                        pick_fields.append(f"{value:<{pick_widths[idx]}}")
                    else:
                        pick_fields.append(f"{value:>{pick_widths[idx]}}")
                f.write(" ".join(pick_fields) + "\n")


def get_output_file_path(exp_dir: str, pick_file: str) -> str:
    prefix = os.path.splitext(os.path.basename(pick_file))[0]
    filename = f"{prefix}.phase"
    return os.path.join(exp_dir, filename)


def _reset_perf_stats(logger: logging.Logger) -> None:
    logger._doublef_perf_total = {}
    logger._doublef_perf_self = {}
    logger._doublef_perf_stack = []



def assign_final_event_ids(events: list[dict]) -> list[dict]:
    events.sort(key=lambda item: item["origin_time"])
    for index, event in enumerate(events, start=1):
        event["final_event_id"] = index
    return events


def _mean_event_metric(events: list[dict], key: str) -> float:
    values = [
        float(event[key])
        for event in events
        if key in event and pd.notna(event[key])
    ]
    if not values:
        return float("nan")
    return float(sum(values) / len(values))



def _associate_pick_dataframe(
    raw_pick_df: pd.DataFrame,
    cfg: dict,
    logger: logging.Logger,
    cache_dir: str,
    distance_step_km: float,
    depth_step_km: float,
    p_tt_matrix: torch.Tensor | None,
    s_tt_matrix: torch.Tensor | None,
    static_max_tt: float | None,
    sample_index: list[int],
    top_number_index: list[int],
    number_weight_index: list[float],
    time_weight_index: list[float],
    confidence_level_index: list[float],
    sampling_batch_size: int,
    score_event_batch_size: int,
    result_batch_size: int,
    file_tag: str = "",
) -> tuple[list[dict], dict]:
    global_csv_tensor = CsvTorch(
        cache_dir,
        cfg["code"],
        "",
        cfg["device"],
        station_file=cfg["station_file"],
        consider_station_elevation=cfg["consider_station_elevation"],
        lookup_grid_step_km=cfg["lookup_grid_step_km"],
        tt_depth_step_km=depth_step_km,
        phase_df=raw_pick_df,
    )
    with timed(logger, "prep.generate_station_data"):
        return_value = global_csv_tensor.generate_station_data()
    if not return_value:
        return [], {"total": 0, "p_total": 0, "s_total": 0, "reference_time": None}

    full_phase_df = global_csv_tensor.df.sort_values(["RelativeTime", "phasetype", "net_sta"]).reset_index(drop=True)
    reference_time = global_csv_tensor.reference_time
    reference_utc = UTCDateTime(reference_time.to_pydatetime())
    p_pick_df = full_phase_df[full_phase_df["phasetype"] == "P"].sort_values("RelativeTime").reset_index(drop=True)
    s_pick_count = int((full_phase_df["phasetype"] == "S").sum())
    station_matrix, _station_number, station_dic = return_value
    stats = {
        "total": len(full_phase_df),
        "p_total": len(p_pick_df),
        "s_total": s_pick_count,
        "reference_time": reference_time,
    }

    if len(p_pick_df) == 0:
        global_csv_tensor.phase_df = None
        global_csv_tensor.df = None
        return [], stats

    with timed(logger, "prep.build_phase_index"):
        phase_index = PhasePickIndex(
            full_phase_df,
            global_csv_tensor.max_id,
            device=cfg["device"],
            time_step=10 ** (-global_csv_tensor.pick_time_round_decimals),
            logger=logger,
            lookup_mode=cfg["lookup_mode"],
            lookup_query_chunk_size=cfg["lookup_query_chunk_size"],
        )
    global_csv_tensor.phase_df = None
    global_csv_tensor.df = None

    p_seed_location_values = torch.tensor(
        p_pick_df[["latitude", "longitude", "RelativeTime"]].to_numpy(),
        dtype=torch.float32,
        device="cpu",
    )
    p_seed_pick_uids = torch.tensor(
        p_pick_df["pick_uid"].to_numpy(),
        dtype=torch.long,
        device="cpu",
    )
    p_seed_times = p_seed_location_values[:, 2].clone()

    local_p_tt_matrix = p_tt_matrix
    local_s_tt_matrix = s_tt_matrix
    local_max_p_tt, local_max_s_tt = static_max_tt
    if cfg["consider_station_elevation"]:
        with timed(logger, "prep.load_tt_all_stations"):
            local_p_tt_matrix, local_s_tt_matrix = global_csv_tensor.load_tt(selected_station_keys=global_csv_tensor.station_keys)
        with timed(logger, "prep.max_tt_from_cache"):
            local_max_p_tt, local_max_s_tt = global_csv_tensor.max_tt_by_phase_from_cache(
                cfg["max_distance"],
                cfg["sdepth_max"],
                distance_step_km,
                depth_step_km,
                selected_station_keys=global_csv_tensor.station_keys,
            )

    outer_batch_size = cfg["glob_batch_size"] if cfg["glob_batch_size"] > 0 else 256
    global_event_counter = 0

    def run_global_round(
        seed_df: pd.DataFrame | None,
        current_phase_index: PhasePickIndex,
        round_index: int,
        seed_location_values: torch.Tensor | None = None,
        seed_pick_uids: torch.Tensor | None = None,
        seed_times: torch.Tensor | None = None,
        allow_continue: bool = False,
    ) -> dict:
        nonlocal global_event_counter
        if seed_location_values is None:
            if seed_df is None or len(seed_df) == 0:
                return {"continue": False, "events": [], "counts": (0, 0, 0, 0)}
        else:
            if seed_location_values.shape[0] == 0:
                return {"continue": False, "events": [], "counts": (0, 0, 0, 0)}

        if seed_location_values is None or seed_pick_uids is None or seed_times is None:
            seed_location_values = torch.tensor(
                seed_df[["latitude", "longitude", "RelativeTime"]].to_numpy(),
                dtype=torch.float32,
                device="cpu",
            )
            seed_pick_uids = torch.tensor(
                seed_df["pick_uid"].to_numpy(),
                dtype=torch.long,
                device="cpu",
            )
            seed_times = seed_location_values[:, 2].clone()

        seed_count = int(seed_location_values.shape[0])
        round_initial_location_matrix = torch.empty(
            (seed_count, 4),
            dtype=seed_location_values.dtype,
            device="cpu",
        )
        round_lower_bound = None
        round_upper_bound = None
        round_score_matrix = None
        round_seed_pick_uids = torch.empty(
            (seed_count,),
            dtype=seed_pick_uids.dtype,
            device="cpu",
        )
        round_batch_ids = torch.empty(
            (seed_count,),
            dtype=torch.long,
            device="cpu",
        )

        batch_starts = range(0, seed_count, outer_batch_size)
        with timed(logger, f"round{round_index}.sample_batches"):
            for batch_start in tqdm(batch_starts, desc=f"Round {round_index}", unit="batch", leave=False):
                batch_end = min(batch_start + outer_batch_size, seed_count)
                batch_location_values = seed_location_values[batch_start:batch_end]
                batch_pick_uids = seed_pick_uids[batch_start:batch_end]
                batch_times = seed_times[batch_start:batch_end]
                batch_p_min = float(batch_times[0].item())
                batch_p_max = float(batch_times[-1].item())
                origin_min = batch_p_min - cfg["search_time"]
                origin_max = batch_p_max + 1.0
                p_pick_min = origin_min - cfg["max_P_tolerance"]
                p_pick_max = origin_max + cfg["max_P_tolerance"] + local_max_p_tt
                s_pick_min = origin_min - cfg["max_S_tolerance"]
                s_pick_max = origin_max + cfg["max_S_tolerance"] + local_max_s_tt
                with timed(logger, "batch.window_phase_index"):
                    batch_phase_index = current_phase_index.window(
                        p_pick_min,
                        p_pick_max,
                        s_start_time=s_pick_min,
                        s_end_time=s_pick_max,
                    )

                batch_location_values_gpu = batch_location_values.to(cfg["device"])
                location_matrix = torch.cat(
                    [
                        batch_location_values_gpu[:, :2],
                        torch.zeros((batch_location_values.shape[0], 1), dtype=torch.float32, device=cfg["device"]),
                        batch_location_values_gpu[:, 2:],
                    ],
                    dim=1,
                )

                sampler = Sampler(
                    logger,
                    cfg["max_distance"],
                    location_matrix,
                    station_matrix,
                    cfg["min_P_tolerance"],
                    cfg["max_P_tolerance"],
                    cfg["min_S_tolerance"],
                    cfg["max_S_tolerance"],
                    batch_phase_index,
                    local_p_tt_matrix,
                    local_s_tt_matrix,
                    distance_step_km,
                    depth_step_km,
                    cfg["P_weight"],
                    1 - cfg["P_weight"],
                    cfg["mag_weight"],
                    cfg["time_type"],
                    cfg["number_type"],
                    "Continuous",
                    cfg["dis0"],
                    cfg["dis1"],
                    cfg["search_lat"],
                    cfg["search_lon"],
                    cfg["search_depth"],
                    cfg["search_time"],
                    sample_index,
                    top_number_index,
                    number_weight_index,
                    time_weight_index,
                    confidence_level_index,
                    sampling_batch_size,
                    score_event_batch_size,
                    cfg["device"],
                )

                with timed(logger, "batch.sampler_only"):
                    batch_top_samples = sampler.run().squeeze(-1)

                if round_score_matrix is None:
                    round_score_matrix = torch.empty(
                        (seed_count, batch_top_samples.shape[1]),
                        dtype=batch_top_samples.dtype,
                        device="cpu",
                    )
                    round_lower_bound = torch.empty(
                        (seed_count, sampler.final_lower_bound.shape[1]),
                        dtype=sampler.final_lower_bound.dtype,
                        device="cpu",
                    )
                    round_upper_bound = torch.empty(
                        (seed_count, sampler.final_upper_bound.shape[1]),
                        dtype=sampler.final_upper_bound.dtype,
                        device="cpu",
                    )

                round_initial_location_matrix[batch_start:batch_end] = location_matrix.cpu()
                round_score_matrix[batch_start:batch_end] = batch_top_samples.cpu()
                round_lower_bound[batch_start:batch_end] = sampler.final_lower_bound.cpu()
                round_upper_bound[batch_start:batch_end] = sampler.final_upper_bound.cpu()
                round_seed_pick_uids[batch_start:batch_end] = batch_pick_uids
                round_batch_ids[batch_start:batch_end] = batch_start // outer_batch_size

                del batch_phase_index, sampler, batch_top_samples, batch_location_values_gpu
                if (batch_end // outer_batch_size) % _BATCH_CLEANUP_INTERVAL == 0:
                    with timed(logger, "batch.cleanup"):
                        if torch.cuda.is_available():
                            with timed(logger, "batch.cleanup.cuda_empty_cache"):
                                torch.cuda.empty_cache()
                        with timed(logger, "batch.cleanup.gc_collect"):
                            gc.collect()

        round_write_dict = {"logger": logger}
        get_result_i = 0 if allow_continue else 1
        logger.info(f"Round {round_index}: writing results...")
        with timed(logger, f"round{round_index}.write_results"):
            gr = GetResult(
                get_result_i,
                cfg["max_distance"],
                round_initial_location_matrix,
                round_score_matrix,
                station_matrix,
                round_lower_bound,
                round_upper_bound,
                round_seed_pick_uids,
                station_dic,
                cfg["min_P_tolerance"],
                cfg["max_P_tolerance"],
                cfg["min_S_tolerance"],
                cfg["max_S_tolerance"],
                current_phase_index,
                local_p_tt_matrix,
                local_s_tt_matrix,
                distance_step_km,
                depth_step_km,
                cfg["P_weight"],
                1 - cfg["P_weight"],
                0.95,
                0.05,
                0,
                cfg["time_type"],
                cfg["number_type"],
                "Continuous",
                cfg["dis0"],
                cfg["dis1"],
                round_write_dict,
                0,
                0,
                0,
                0,
                cfg["min_p"],
                cfg["min_s"],
                cfg["min_sum"],
                cfg["min_both"],
                cfg["only_double"],
                reference_utc,
                "",
                result_batch_size,
                device=cfg["device"],
                initial_batch_ids=round_batch_ids,
            )
            round_result = gr.write_results()

        for local_index, event in enumerate(round_result["events"], start=1):
            global_event_counter += 1
            event["round_id"] = round_index
            event["event_rank_in_round"] = local_index
            event["global_event_id"] = (
                f"{file_tag}R{round_index:02d}-"
                f"E{local_index:04d}-{global_event_counter:08d}"
            )

        return round_result

    final_events = []
    current_phase_index = phase_index
    current_seed_df = p_pick_df
    current_seed_location_values = p_seed_location_values
    current_seed_pick_uids = p_seed_pick_uids
    current_seed_times = p_seed_times
    round_index = 1
    while True:
        allow_continue = cfg["repeat_rounds"] == -1 or round_index <= cfg["repeat_rounds"]
        with timed(logger, f"round{round_index}.total"):
            round_result = run_global_round(
                current_seed_df,
                current_phase_index,
                round_index,
                seed_location_values=current_seed_location_values,
                seed_pick_uids=current_seed_pick_uids,
                seed_times=current_seed_times,
                allow_continue=allow_continue,
            )
        final_events.extend(round_result["events"])

        if not allow_continue or not round_result.get("continue") or len(round_result["events"]) == 0:
            break

        with timed(logger, f"post.prepare_round{round_index + 1}"):
            current_phase_index = round_result["phase_index"]
            current_seed_df = None
            remaining_location_matrix = round_result["location_matrix"]
            current_seed_pick_uids = round_result["initial_seed_pick_uids"]
            current_seed_location_values = torch.cat(
                [remaining_location_matrix[:, :2], remaining_location_matrix[:, 3:4]],
                dim=1,
            )
            current_seed_times = remaining_location_matrix[:, 3].clone()
        round_index += 1

    with timed(logger, "post.cleanup"):
        if torch.cuda.is_available():
            with timed(logger, "post.cleanup.cuda_empty_cache"):
                torch.cuda.empty_cache()
        with timed(logger, "post.cleanup.gc_collect"):
            gc.collect()
    return final_events, stats



def run_from_config(config_path: str) -> None:
    if not isinstance(config_path, str) or not config_path.strip():
        raise ValueError("Config path must be a non-empty string")
    config_ext = os.path.splitext(config_path)[1].lower()
    if config_ext not in {".yaml", ".yml"}:
        raise ValueError(f"Config file must be a YAML file: {config_path}")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config_path = os.path.abspath(config_path)
    config_dir = os.path.dirname(config_path)

    cfg = read_config(config_path)
    cfg = normalize_config_paths(cfg, config_dir)

    os.makedirs(cfg["logger_dir"], exist_ok=True)
    experiment_index = len(glob(os.path.join(cfg["logger_dir"], "*")))
    exp_dir = os.path.join(cfg["logger_dir"], f"{experiment_index:03d}-{cfg['code']}")

    logger = create_logger(
        exp_dir,
        cfg["code"],
        sync_cuda_timing=cfg["profile_cuda_sync"],
        silent_screen=cfg["silent_screen"],
    )

    logger.info("Association run initialized.")
    logger.info(f"Configuration file: {config_path}")
    logger.info(f"Compute device: {cfg['device']}")
    if cfg["profile_cuda_sync"]:
        logger.info("CUDA-synchronized profiling enabled.")

    os.makedirs(exp_dir, exist_ok=True)
    pick_files = _list_pick_files(cfg["pick_dir"], cfg["station_file"])
    parsed_cfg_path = os.path.join(exp_dir, "Config.yaml")
    write_effective_config(parsed_cfg_path, cfg)
    logger.info(f"Normalized configuration: {parsed_cfg_path}")
    start_time = UTCDateTime()
    cache_dir = get_travel_time_cache_dir(cfg)
    distance_step_km = get_distance_step_km(cfg)
    depth_step_km = float(cfg["depth_step"])

    if cfg.get("cal_tt", False):
        logger.info("Computing travel-time tables...")
        station_df = pd.read_csv(cfg["station_file"]) if cfg["consider_station_elevation"] else None
        TravelTime(
            filename=cfg["vel_model"],
            cache_dir=cache_dir,
            project_name=cfg["code"],
            sdepth_max=cfg["sdepth_max"],
            depth_step=cfg["depth_step"],
            distance_max=cfg["distance_max"],
            distance_step=cfg["distance_step"],
            lookup_grid_step_km=cfg["lookup_grid_step_km"],
            logger=logger,
            station_df=station_df,
            consider_station_elevation=cfg["consider_station_elevation"],
        ).run()
        logger.info("Travel-time tables ready.")
    else:
        logger.info("Loading travel-time tables...")

    p_tt_matrix = None
    s_tt_matrix = None
    static_max_tt = (0.0, 0.0)
    if not cfg["consider_station_elevation"]:
        csv_tensor = CsvTorch(
            cache_dir,
            cfg["code"],
            "",
            cfg["device"],
            lookup_grid_step_km=cfg["lookup_grid_step_km"],
            tt_depth_step_km=depth_step_km,
        )
        static_max_tt = csv_tensor.max_tt_by_phase_from_cache(
            cfg["max_distance"],
            cfg["sdepth_max"],
            distance_step_km,
            depth_step_km,
        )
        p_tt_matrix, s_tt_matrix = csv_tensor.load_tt()
    if not cfg.get("cal_tt", False):
        logger.info("Travel-time tables ready.")

    sample_index = []
    top_number_index = []
    number_weight_index = []
    time_weight_index = []
    confidence_level_index = []
    for i in range(cfg["iteration"]):
        sample_index.append(int(cfg["samples"] * (cfg["multiple"] ** i)))
        top_number_index.append(cfg["top"])
        number_weight_index.append(cfg["number_weight"])
        time_weight_index.append(1 - cfg["number_weight"])
        confidence_level_index.append(cfg["quantile"])

    sampling_batch_size = max(sample_index) if sample_index else cfg["samples"]
    score_event_batch_size = 10 ** 9
    result_batch_size = 10 ** 9

    for pick_file in pick_files:
        _reset_perf_stats(logger)
        logger.info(f"Processing pick file: {pick_file}")
        with timed(logger, "prep.load_pick_dataframe"):
            raw_pick_df = load_pick_dataframe(pick_file, cfg["station_file"])
        final_events, file_stats = _associate_pick_dataframe(
            raw_pick_df,
            cfg,
            logger,
            cache_dir,
            distance_step_km,
            depth_step_km,
            p_tt_matrix,
            s_tt_matrix,
            static_max_tt,
            sample_index,
            top_number_index,
            number_weight_index,
            time_weight_index,
            confidence_level_index,
            sampling_batch_size,
            score_event_batch_size,
            result_batch_size,
            file_tag="",
        )
        del raw_pick_df
        with timed(logger, "prep.file_gc_collect"):
            gc.collect()

        total_input_count = int(file_stats["total"])
        total_p_count = int(file_stats["p_total"])
        total_s_count = int(file_stats["s_total"])
        earliest_reference_time = file_stats["reference_time"]

        if total_p_count == 0:
            logger.warning("No P-phase records were found in the input.")
            continue

        if len(pick_files) == 1 and earliest_reference_time is not None:
            cfg["year0"] = int(earliest_reference_time.year)
            cfg["month0"] = int(earliest_reference_time.month)
            cfg["day0"] = int(earliest_reference_time.day)
            write_effective_config(parsed_cfg_path, cfg)

        logger.info(f"Input summary: total={total_input_count}, P={total_p_count}, S={total_s_count}")

        final_events = assign_final_event_ids(final_events)
        output_file = get_output_file_path(exp_dir, pick_file)
        with timed(logger, "post.write_report"):
            write_phase_report(output_file, final_events)

        total_p_used = sum(event["count_p"] for event in final_events)
        total_s_used = sum(event["count_s"] for event in final_events)
        total_both_used = sum(event["count_both"] for event in final_events)
        total_paired_phases = total_both_used * 2
        total_associated_phases = total_p_used + total_s_used
        paired_phase_ratio = (total_paired_phases * 100 / total_associated_phases) if total_associated_phases > 0 else 0.0
        p_ratio = (total_p_used * 100 / total_p_count) if total_p_count > 0 else 0.0
        s_ratio = (total_s_used * 100 / total_s_count) if total_s_count > 0 else 0.0
        logger.info(f"Accepted events: {len(final_events)}")
        logger.info(f"P used: {total_p_used} / {total_p_count} ({p_ratio:.2f}%)")
        if total_s_count > 0:
            logger.info(f"S used: {total_s_used} / {total_s_count} ({s_ratio:.2f}%)")
        logger.info(f"P-S paired: {total_paired_phases} / {total_associated_phases} ({paired_phase_ratio:.2f}%)")
        if final_events:
            avg_p_per_event = total_p_used / len(final_events)
            avg_s_per_event = total_s_used / len(final_events)
            avg_both_per_event = total_both_used / len(final_events)
            avg_rms = _mean_event_metric(final_events, "rms")
            avg_err_lat = _mean_event_metric(final_events, "err_lat")
            avg_err_lon = _mean_event_metric(final_events, "err_lon")
            avg_err_dep = _mean_event_metric(final_events, "err_dep")
            avg_err_time = _mean_event_metric(final_events, "err_time")
            logger.info(
                f"Catalog averages: P/event={avg_p_per_event:.2f}, "
                f"S/event={avg_s_per_event:.2f}, PS-both/event={avg_both_per_event:.2f}, "
                f"RMS={avg_rms:.3f}s"
            )
            logger.info(
                f"Catalog mean uncertainty: lat={avg_err_lat:.4f} deg, "
                f"lon={avg_err_lon:.4f} deg, dep={avg_err_dep:.3f} km, "
                f"time={avg_err_time:.3f} s"
            )
        logger.info(f"Output file: {output_file}")
        timing_parts = []
        perf_total_stats = getattr(logger, "_doublef_perf_total", {})
        completed_rounds = max(
            (
                int(key.split(".")[0][5:])
                for key in perf_total_stats
                if key.startswith("round") and key.endswith(".total")
            ),
            default=0,
        )
        for idx in range(1, completed_rounds + 1):
            round_total = get_time(logger, f"round{idx}.total", mode="total")
            round_sample = get_time(logger, f"round{idx}.sample_batches", mode="total")
            round_write = get_time(logger, f"round{idx}.write_results", mode="total")
            if round_total > 0:
                timing_parts.append(
                    f"round{idx}={round_total:.2f}s"
                    f" (sample={round_sample:.2f}s, write-results={round_write:.2f}s)"
                )
        report_time = get_time(logger, "post.write_report", mode="total")
        if report_time > 0:
            timing_parts.append(f"write-report={report_time:.2f}s")
        if timing_parts:
            logger.info("Timing summary: " + ", ".join(timing_parts))

    total_time = UTCDateTime() - start_time
    logger.info(f"Run completed in {total_time:.2f} seconds.")
    logger.info("Processing completed.")



def main(config_path: str) -> None:
    run_from_config(config_path)
