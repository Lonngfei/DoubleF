import math

import torch
from .perf import timed

SCORE_CHUNK_TARGET_ELEMENTS = 8_000_000


def nan_std(x, dim, keepdim=False):
    mask = ~torch.isnan(x)
    count = mask.sum(dim=dim, keepdim=keepdim)
    safe_count = count.clamp_min(1)

    mean = torch.nan_to_num(x, nan=0.0).sum(dim=dim, keepdim=keepdim) / safe_count
    var = torch.nan_to_num((x - mean) ** 2, nan=0.0).sum(dim=dim, keepdim=keepdim) / safe_count
    std = var.sqrt()

    std = torch.where(count > 0, std, torch.full_like(std, float("nan")))
    return std


class NumberScore(object):
    """
    Number-score implementation that works for both:
    - event x station
    - batch x event x station
    """

    def __init__(self, p_prob, s_prob, pweight, sweight, number_station, type, device="cuda"):
        self.p_prob = torch.as_tensor(p_prob, dtype=torch.float32, device=device)
        self.s_prob = torch.as_tensor(s_prob, dtype=torch.float32, device=device)
        self.pweight = float(pweight)
        self.sweight = float(sweight)
        self.number_station = int(number_station)
        self.type = type
        self.device = device

        self.p_nan_mask = ~torch.isnan(self.p_prob)
        self.s_nan_mask = ~torch.isnan(self.s_prob)
        self.station_dim = self.p_prob.ndim - 1

    def number(self):
        p_matrix = self.p_nan_mask.sum(dim=self.station_dim)
        s_matrix = self.s_nan_mask.sum(dim=self.station_dim)
        number_score_matrix = self.pweight * p_matrix + self.sweight * s_matrix
        return number_score_matrix / self.number_station

    def prob_number(self):
        p_matrix = torch.where(self.p_nan_mask, self.p_prob, torch.zeros_like(self.p_prob)).sum(dim=self.station_dim)
        s_matrix = torch.where(self.s_nan_mask, self.s_prob, torch.zeros_like(self.s_prob)).sum(dim=self.station_dim)
        number_score_matrix = self.pweight * p_matrix + self.sweight * s_matrix
        return number_score_matrix / self.number_station

    def number_both(self):
        p_matrix = self.p_nan_mask.sum(dim=self.station_dim)
        s_matrix = self.s_nan_mask.sum(dim=self.station_dim)
        both_matrix = (self.p_nan_mask & self.s_nan_mask).sum(dim=self.station_dim)
        number_score_matrix = self.pweight * p_matrix + self.sweight * s_matrix + both_matrix
        return number_score_matrix / (2 * self.number_station)

    def prob_number_both(self):
        matrixp_pweight = self.pweight * self.p_prob
        matrixs_sweight = self.sweight * self.s_prob
        sum_matrixp = torch.where(self.p_nan_mask, matrixp_pweight, torch.zeros_like(matrixp_pweight)).sum(dim=self.station_dim)
        sum_matrixs = torch.where(self.s_nan_mask, matrixs_sweight, torch.zeros_like(matrixs_sweight)).sum(dim=self.station_dim)
        sum_both = torch.where(
            self.p_nan_mask & self.s_nan_mask,
            matrixp_pweight + matrixs_sweight,
            torch.zeros_like(matrixp_pweight),
        ).sum(dim=self.station_dim)
        number_score_matrix = sum_matrixp + sum_matrixs + sum_both
        return number_score_matrix / (2 * self.number_station)

    def cal(self):
        if self.type == "number":
            return self.number()
        if self.type == "prob_number":
            return self.prob_number()
        if self.type == "number_both":
            return self.number_both()
        if self.type == "prob_number_both":
            return self.prob_number_both()
        raise ValueError(f"Calculating Number Score: Function {self.type} not supported")


class TimeScore(object):
    """
    Time-score implementation that works for both:
    - event x station
    - batch x event x station
    """

    def __init__(
        self,
        p_max_offset,
        s_max_offset,
        p_err,
        s_err,
        p_prob,
        s_prob,
        pweight,
        sweight,
        distance_matrix,
        dis0,
        dis1,
        type,
        device="cuda",
        logger=None,
    ):
        self.p_max_offset = float(p_max_offset)
        self.s_max_offset = float(s_max_offset)

        self.p_sigma = max(self.p_max_offset / 2.0, 1e-6)
        self.s_sigma = max(self.s_max_offset / 2.0, 1e-6)
        self.p_scale = -1.0 / (2.0 * self.p_sigma * self.p_sigma)
        self.s_scale = -1.0 / (2.0 * self.s_sigma * self.s_sigma)

        self.p_err = torch.as_tensor(p_err, dtype=torch.float32, device=device)
        self.s_err = torch.as_tensor(s_err, dtype=torch.float32, device=device)
        self.p_prob = torch.as_tensor(p_prob, dtype=torch.float32, device=device)
        self.s_prob = torch.as_tensor(s_prob, dtype=torch.float32, device=device)

        self.pweight = float(pweight)
        self.sweight = float(sweight)

        self.distance_matrix = torch.as_tensor(distance_matrix, dtype=torch.float32, device=device)
        self.dis0 = float(dis0)
        self.dis1 = float(dis1)
        self.type = type
        self.device = device
        self.station_dim = self.distance_matrix.ndim - 1
        self.logger = logger

        with timed(self.logger, "score.time_score.distance_weight"):
            self.weight_matrix = self._compute_distance_weights()

    def _compute_distance_weights(self):
        if self.dis1 <= self.dis0:
            raise ValueError(f"dis1 must be larger than dis0, but got dis0={self.dis0}, dis1={self.dis1}")

        d = self.distance_matrix
        scale = math.pi / (2.0 * (self.dis1 - self.dis0))
        weight_matrix = d.sub(self.dis0).mul(scale)
        weight_matrix.clamp_(0.0, math.pi / 2.0).cos_()
        return weight_matrix

    def _gaussian_score(self, err, sigma, weight=None):
        valid = torch.isfinite(err)
        safe_err = torch.where(valid, err, torch.zeros_like(err))
        rho = 1.0 - torch.exp(-(safe_err * safe_err) / (2.0 * sigma * sigma))

        if weight is None:
            weighted = torch.where(valid, rho, torch.zeros_like(rho))
        else:
            safe_weight = torch.where(valid, torch.nan_to_num(weight, nan=0.0, posinf=0.0, neginf=0.0), torch.zeros_like(weight))
            safe_weight = torch.clamp_min(safe_weight, 0.0)
            weighted = safe_weight * rho

        num = torch.sum(weighted, dim=self.station_dim)
        count = torch.sum(valid, dim=self.station_dim)
        score = num / count.clamp_min(1)
        return torch.where(count > 0, score, torch.full_like(score, float("nan"))).to(torch.float32)

    def _prob_distance_score(self, err, prob, scale):
        valid = torch.isfinite(err)
        safe_err = torch.where(valid, err, torch.zeros_like(err))
        rho = safe_err.mul(safe_err).mul_(scale).exp_().neg_().add_(1.0)

        safe_prob = torch.where(valid, prob, torch.zeros_like(prob))
        safe_prob = torch.nan_to_num(safe_prob, nan=0.0, posinf=0.0, neginf=0.0)
        safe_prob.clamp_min_(0.0)
        safe_prob.mul_(self.weight_matrix)

        num = torch.sum(safe_prob * rho, dim=self.station_dim)
        count = torch.sum(valid, dim=self.station_dim)
        score = num / count.clamp_min(1)
        return torch.where(count > 0, score, torch.full_like(score, float("nan"))).to(torch.float32)

    def rms(self):
        p_score_matrix = self._gaussian_score(self.p_err, self.p_sigma)
        s_score_matrix = self._gaussian_score(self.s_err, self.s_sigma)
        return self.pweight * p_score_matrix + self.sweight * s_score_matrix

    def distance_rms(self):
        p_score_matrix = self._gaussian_score(self.p_err, self.p_sigma, self.weight_matrix)
        s_score_matrix = self._gaussian_score(self.s_err, self.s_sigma, self.weight_matrix)
        return self.pweight * p_score_matrix + self.sweight * s_score_matrix

    def prob_rms(self):
        p_score_matrix = self._gaussian_score(self.p_err, self.p_sigma, self.p_prob)
        s_score_matrix = self._gaussian_score(self.s_err, self.s_sigma, self.s_prob)
        return self.pweight * p_score_matrix + self.sweight * s_score_matrix

    def prob_distance_rms(self):
        with timed(self.logger, "score.time_score.prob_distance.p"):
            p_score_matrix = self._prob_distance_score(self.p_err, self.p_prob, self.p_scale)
        with timed(self.logger, "score.time_score.prob_distance.s"):
            s_score_matrix = self._prob_distance_score(self.s_err, self.s_prob, self.s_scale)
        return self.pweight * p_score_matrix + self.sweight * s_score_matrix

    def cal(self):
        if self.type == "rms":
            return self.rms()
        if self.type == "distance_rms":
            return self.distance_rms()
        if self.type == "prob_rms":
            return self.prob_rms()
        if self.type == "prob_distance_rms":
            return self.prob_distance_rms()
        raise ValueError(f"Calculating Time Score: Function {self.type} not supported")


class TimeScoreDepth(object):
    def __init__(self, max_offset, ps_err, distance_matrix, ddis0, ddis1, type, device="cuda"):
        self.max_offset = max_offset
        self.ps_err = torch.as_tensor(ps_err, dtype=torch.float32, device=device)
        self.distance_matrix = torch.as_tensor(distance_matrix, dtype=torch.float32, device=device)
        self.ddis0 = ddis0
        self.ddis1 = ddis1
        self.type = type
        self.device = device
        self.weight_matrix = self._compute_distance_weights()

    def _compute_distance_weights(self):
        weight_matrix = self.distance_matrix.detach().clone()
        weight_matrix[self.distance_matrix < self.ddis0] = 1
        mask = (self.distance_matrix >= self.ddis0) & (self.distance_matrix <= self.ddis1)
        weight_matrix[mask] = torch.cos((math.pi / 2) * (self.distance_matrix[mask] - self.ddis0) / (self.ddis1 - self.ddis0))
        weight_matrix[self.distance_matrix > self.ddis1] = float("nan")
        return weight_matrix

    def nearest_station(self):
        n_events, _ = self.distance_matrix.shape
        distance_matrix = self.distance_matrix.clone()
        distance_matrix[torch.isnan(self.ps_err)] = float("nan")

        distance_matrix_cleaned = torch.where(torch.isnan(distance_matrix), torch.tensor(float("inf"), device=self.device), distance_matrix)

        indices = torch.argmin(distance_matrix_cleaned, dim=1)
        time_score_matrix = torch.full((n_events,), float("nan"), device=self.device)

        valid_mask = ~torch.isnan(distance_matrix).all(dim=1)
        time_score_matrix[valid_mask] = torch.abs(self.ps_err[valid_mask, indices[valid_mask]])

        return time_score_matrix.view(-1, 1) / (2 * self.max_offset)

    def distance_weight(self):
        weight_matrix = self.weight_matrix.clone()
        squared_sum = torch.nansum(torch.square(self.ps_err) * weight_matrix, dim=1, keepdim=True)
        count_non_nan = torch.sum(~torch.isnan(self.ps_err * weight_matrix), dim=1, keepdim=True)
        time_score_matrix = torch.sqrt(squared_sum / count_non_nan)
        time_score_matrix[torch.isnan(time_score_matrix)] = float("nan")
        return time_score_matrix / (2 * self.max_offset)

    def cal(self):
        if self.type == "nearest_station":
            return self.nearest_station()
        if self.type == "distance_weight":
            return self.distance_weight()
        raise ValueError(f"Calculating Time Score Under Only Update Depth Mode: Function {self.type} not supported")


class MagnitudeScore(object):
    def __init__(self, p_amp, s_amp, p_weight, s_weight, distance_matrix, type, device="cuda"):
        self.p_amp = torch.as_tensor(p_amp, dtype=torch.float32, device=device)
        self.s_amp = torch.as_tensor(s_amp, dtype=torch.float32, device=device)
        self.p_weight = float(p_weight)
        self.s_weight = float(s_weight)
        self.distance_matrix = torch.as_tensor(distance_matrix, dtype=torch.float32, device=device)
        self.type = type
        self.device = device
        self.station_dim = self.distance_matrix.ndim - 1

    def cal_mag(self, amp):
        if self.type == "Continuous":
            return torch.log10(amp) + 1.110 * torch.log10(self.distance_matrix / 100) + 0.00189 * (self.distance_matrix - 100) + 3.0
        if self.type == "Discrete":
            valid_distance_matrix = self.distance_matrix
            empirical_matrix = torch.where(valid_distance_matrix <= 10, 2.0,
                                   torch.where(valid_distance_matrix <= 15, 2.1,
                                   torch.where(valid_distance_matrix <= 20, 2.2,
                                   torch.where(valid_distance_matrix <= 25, 2.4,
                                   torch.where(valid_distance_matrix <= 30, 2.6,
                                   torch.where(valid_distance_matrix <= 35, 2.7,
                                   torch.where(valid_distance_matrix <= 40, 2.8,
                                   torch.where(valid_distance_matrix <= 45, 2.9,
                                   torch.where(valid_distance_matrix <= 50, 3.0,
                                   torch.where(valid_distance_matrix <= 55, 3.1,
                                   torch.where(valid_distance_matrix <= 70, 3.2,
                                   torch.where(valid_distance_matrix <= 85, 3.3,
                                   torch.where(valid_distance_matrix <= 100, 3.4,
                                   torch.where(valid_distance_matrix <= 120, 3.5,
                                   torch.where(valid_distance_matrix <= 140, 3.6,
                                   torch.where(valid_distance_matrix <= 160, 3.7,
                                   torch.where(valid_distance_matrix <= 180, 3.8,
                                   torch.where(valid_distance_matrix <= 220, 3.9,
                                   torch.where(valid_distance_matrix <= 250, 4.0, 4.1)))))))))))))))))))
            return torch.log10(amp * 2080 * 20) + empirical_matrix
        raise ValueError(f"Calculating Magnitude Score: Function {self.type} not supported")

    def nan_std(self):
        p_magnitude_matrix = self.cal_mag(self.p_amp)
        s_magnitude_matrix = self.cal_mag(self.s_amp)

        p_magnitude_matrix = torch.where(torch.isinf(p_magnitude_matrix), torch.full_like(p_magnitude_matrix, float("nan")), p_magnitude_matrix)
        s_magnitude_matrix = torch.where(torch.isinf(s_magnitude_matrix), torch.full_like(s_magnitude_matrix, float("nan")), s_magnitude_matrix)

        p_magnitude_score_matrix = nan_std(p_magnitude_matrix, dim=self.station_dim)
        s_magnitude_score_matrix = nan_std(s_magnitude_matrix, dim=self.station_dim)

        magnitude_score_matrix = self.p_weight * p_magnitude_score_matrix + self.s_weight * s_magnitude_score_matrix
        return magnitude_score_matrix / 10

    def cal_median_mag(self):
        p_magnitude_matrix = self.cal_mag(self.p_amp)
        s_magnitude_matrix = self.cal_mag(self.s_amp)

        magnitude_matrix = torch.cat((p_magnitude_matrix, s_magnitude_matrix), dim=0)
        data_no_nan = magnitude_matrix[~torch.isnan(magnitude_matrix)]
        median_value = torch.median(data_no_nan)
        return p_magnitude_matrix, s_magnitude_matrix, median_value
