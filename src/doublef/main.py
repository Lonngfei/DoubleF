#!/usr/bin/env python3
from __future__ import annotations
import ast
import gc
import logging
import os
import sys
from glob import glob
import torch
from obspy import UTCDateTime
import shutil
from .csv_tensor import CsvTorch
from .get_tt import TravelTime
from .multiple_iteration import Mutiple_Iteration


# ==============================================================
# Read simple text config (key = value, support # comments)
# ==============================================================

def read_config(filepath: str) -> tuple[dict, dict]:
    config = {}
    comments = {}

    with open(filepath, "r", encoding="utf-8") as f:
        last_comment = None
        for line in f:
            raw_line = line.strip()

            if not raw_line:
                continue

            if raw_line.startswith("#"):
                last_comment = raw_line
                continue

            if "=" not in raw_line:
                continue

            key, value = raw_line.split("=", 1)
            key = key.strip()

            # Separate inline comment
            if "#" in value:
                value, comment = value.split("#", 1)
                comments[key] = "#" + comment.strip()

            value = value.strip()

            # Type inference
            if value.lower() in ("true", "false"):
                config[key] = value.lower() == "true"
            else:
                try:
                    config[key] = ast.literal_eval(value)
                except Exception:
                    config[key] = value

            if last_comment is not None:
                comments[key] = last_comment
                last_comment = None

    return config, comments


# ==============================================================
# Path normalization
# ==============================================================

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
        "tt_csv",
        "ptt_npy",
        "stt_npy",
        "pick_dir",
    ]

    for key in path_like_keys:
        if key in cfg and isinstance(cfg[key], str):
            cfg[key] = resolve_path(cfg[key], config_dir)

    return cfg


# ==============================================================
# Logger setup
# ==============================================================

def create_logger(log_dir: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "DoubleF.log")

    logger = logging.getLogger("doublef")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Avoid duplicate handlers if called multiple times
    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="[%(asctime)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger


# ==============================================================
# Main process
# ==============================================================

def run_from_config(config_path: str) -> None:
    config_path = os.path.abspath(config_path)
    config_dir = os.path.dirname(config_path)

    cfg, comments = read_config(config_path)
    cfg = normalize_config_paths(cfg, config_dir)

    if "logger_dir" not in cfg:
        raise KeyError("Missing required config key: logger_dir")
    if "code" not in cfg:
        raise KeyError("Missing required config key: code")

    os.makedirs(cfg["logger_dir"], exist_ok=True)
    if cfg["is_plot"]:
        if os.path.exists(cfg["plot_dir"]):
            shutil.rmtree(cfg["plot_dir"])
        os.makedirs(cfg["plot_dir"], exist_ok=True)
    experiment_index = len(glob(os.path.join(cfg["logger_dir"], "*")))
    exp_dir = os.path.join(cfg["logger_dir"], f"{experiment_index:03d}-{cfg['code']}")

    logger = create_logger(exp_dir)

    logger.info("=== Starting Event Association ===")
    logger.info(f"Loaded configuration from: {config_path}")
    logger.info(f"Device: {cfg['device']}")

    # Save parsed configuration
    os.makedirs(exp_dir, exist_ok=True)
    parsed_cfg_path = os.path.join(exp_dir, "Config")
    with open(parsed_cfg_path, "w", encoding="utf-8") as f:
        for key, value in cfg.items():
            _ = comments.get(key, "")
            f.write(f"{key} = {value}\n")
    logger.info(f"Saved parsed configuration to {parsed_cfg_path}")

    start_time = UTCDateTime()

    # ==============================================================
    # Step 1: Travel-time calculation
    # ==============================================================

    if cfg.get("cal_tt", False):
        logger.info("Calculating theoretical travel-time table...")
        base = os.path.splitext(cfg["vel_model"])[0]
        TravelTime(
            filename=cfg["vel_model"],
            tt_csv=base + "_tt.csv",
            ptt_npy=cfg["ptt_npy"],
            stt_npy=cfg["stt_npy"],
            sdepth_max=cfg["sdepth_max"],
            depth_step=cfg["depth_step"],
            distance_max=cfg["distance_max"],
            distance_step=cfg["distance_step"],
            logger=logger,
        ).run()
    else:
        logger.info("Using existing travel-time tables.")

    # ==============================================================
    # Step 2: Initialize CsvTorch and load travel-time tables
    # ==============================================================

    csv_tensor = CsvTorch(cfg["ptt_npy"], cfg["stt_npy"], "", cfg["device"])
    p_tt_matrix, s_tt_matrix = csv_tensor.load_tt()
    logger.info("Travel-time tables loaded successfully.")

    # ==============================================================
    # Step 3: Iterative sampling setup
    # ==============================================================

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

    # ==============================================================
    # Step 4: Process each day
    # ==============================================================

    for i in range(cfg["nday"]):
        ite0_time = UTCDateTime()
        initial_day = UTCDateTime(cfg["year0"], cfg["month0"], cfg["day0"]) + (i * 86400)
        year, month, day = initial_day.year, initial_day.month, initial_day.day
        logger.info(f"Processing {year:04d}-{month:02d}-{day:02d}")

        phase_csv = os.path.join(cfg["pick_dir"], f"{year:04d}{month:02d}{day:02d}.csv")
        results_dir = os.path.join(exp_dir, "phase_report")
        os.makedirs(results_dir, exist_ok=True)
        output_file = os.path.join(results_dir, f"{year:04d}{month:02d}{day:02d}.dat")

        if not os.path.exists(phase_csv):
            logger.warning(f"No phase CSV found for {year}-{month}-{day}: {phase_csv}")
            continue

        csv_tensor = CsvTorch(cfg["ptt_npy"], cfg["stt_npy"], phase_csv, cfg["device"], year, month, day)
        return_value = csv_tensor.generate_station_data()
        if not return_value:
            logger.warning(f"No valid phase data in {phase_csv}")
            continue

        station_matrix, station_number, station_dic = return_value
        location_matrix, score_matrix = csv_tensor.generate_initial_point()
        p_phase, s_phase, p_num, s_num = csv_tensor.generate_all_data()

        logger.info(f"Collected {station_number} stations, {p_num} P-phases, {s_num} S-phases")

        mi = Mutiple_Iteration(
            logger,
            cfg["is_repeat"],
            cfg["max_distance"],
            location_matrix,
            station_matrix,
            station_dic,
            p_phase,
            s_phase,
            p_tt_matrix,
            s_tt_matrix,
            cfg["P_weight"],
            1 - cfg["P_weight"],
            cfg["mag_weight"],
            cfg["time_type"],
            cfg["number_type"],
            'Continuous',
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
            cfg["min_p"],
            cfg["min_s"],
            cfg["min_sum"],
            cfg["min_both"],
            cfg["min_P_tolerance"],
            cfg["max_P_tolerance"],
            cfg["min_S_tolerance"],
            cfg["max_S_tolerance"],
            cfg["only_double"],
            initial_day,
            output_file,
            cfg["max_batch_size"],
            cfg["is_plot"],
            device=cfg["device"],
            plot_dir=cfg["plot_dir"]
        )

        event_number, P_number, S_number, PS_both = mi.get_results()

        if event_number != 0:
            logger.info(f"Associated {event_number} events!!!")
            logger.info(
                f"Associated {P_number} P-phases, the utilization rate is "
                f"{P_number * 100 / p_num:.2f}%, with an average of "
                f"{P_number / event_number:.3f} P-phases per event"
            )
            logger.info(
                f"Associated {S_number} S-phases, the utilization rate is "
                f"{S_number * 100 / s_num:.2f}%, with an average of "
                f"{S_number / event_number:.3f} S-phases per event"
            )
            logger.info(
                f"Associated {PS_both} PS-both stations, with an average of "
                f"{PS_both / event_number:.3f} ps_both stations per event"
            )
            logger.info(f"Results written to {output_file}")
        else:
            logger.info("Associated 0 events!!!")

        logger.info(f"Day processing time: {UTCDateTime() - ite0_time:.2f} s")

        del location_matrix, score_matrix, p_phase, s_phase, station_matrix, mi

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    total_time = UTCDateTime() - start_time
    logger.info(f"=== All tasks finished in {total_time:.2f} seconds ===")


# ==============================================================
# Backward-compatible entry point
# ==============================================================

def main(config_path: str) -> None:
    run_from_config(config_path)
