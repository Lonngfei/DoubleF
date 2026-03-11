import torch
import math


class NumberScore(object):
    """
    Multiple Types for Number Score
    p_number: Number of P-phases
    s_number: Number of S-phases
    both_ps_number: Number of stations with both p and s

    Args:
        pweight: User-defined weight for P-phases
        sweight: User-defined weight for S-phases (=1 - pweight)
        p_prob: Input probability of picked P-phases
        s_prob: Input probability of picked S-phases
        number_station: Number of stations

    Returns:
        number_score_matrix
    """
    def __init__(self, p_prob, s_prob, pweight, sweight, number_station, type, device='cuda'):
        self.p_prob = torch.as_tensor(p_prob, dtype=torch.float32, device=device)
        self.s_prob = torch.as_tensor(s_prob, dtype=torch.float32, device=device)
        self.pweight = pweight
        self.sweight = sweight
        self.number_station = number_station
        self.type = type
        self.device = device

        self.p_nan_mask = ~torch.isnan(self.p_prob)
        self.s_nan_mask = ~torch.isnan(self.s_prob)

    # number_score = (p_number * pweight + s_number * sweight) / (number_station)
    def number(self):
        p_matrix = self.p_nan_mask.sum(dim=1, keepdim=True)
        s_matrix = self.s_nan_mask.sum(dim=1, keepdim=True)

        number_score_matrix = self.pweight * p_matrix
        number_score_matrix.add_(self.sweight * s_matrix)

        return number_score_matrix / self.number_station

    # number_score = (p_prob * p_number * pweight + s_prob * s_number * sweight) / (number_station)
    def prob_number(self):
        p_matrix = torch.where(self.p_nan_mask, self.p_prob, torch.zeros_like(self.p_prob)).sum(dim=1, keepdim=True)
        s_matrix = torch.where(self.s_nan_mask, self.s_prob, torch.zeros_like(self.s_prob)).sum(dim=1, keepdim=True)

        number_score_matrix = self.pweight * p_matrix
        number_score_matrix.add_(self.sweight * s_matrix)

        return number_score_matrix / self.number_station

    # number_score = (p_number * pweight + s_number * sweight + both_ps_number) / (2 * number_station)
    def number_both(self):
        p_matrix = self.p_nan_mask.sum(dim=1, keepdim=True)
        s_matrix = self.s_nan_mask.sum(dim=1, keepdim=True)
        both_matrix = (self.p_nan_mask & self.s_nan_mask).sum(dim=1, keepdim=True)

        number_score_matrix = self.pweight * p_matrix
        number_score_matrix.add_(self.sweight * s_matrix)
        number_score_matrix.add_(both_matrix)

        return number_score_matrix / (2 * self.number_station)

    # number_score = (p_prob * p_number * pweight + s_prob * s_number * sweight + both_ps_number * (pweight * p_prob + sweight * s_prob)) / (2 * number_station)
    def prob_number_both(self):
        matrixp_pweight = self.pweight * self.p_prob
        matrixs_sweight = self.sweight * self.s_prob

        sum_matrixp = torch.where(self.p_nan_mask, matrixp_pweight, torch.zeros_like(matrixp_pweight)).sum(dim=1, keepdim=True)
        sum_matrixs = torch.where(self.s_nan_mask, matrixs_sweight, torch.zeros_like(matrixs_sweight)).sum(dim=1, keepdim=True)
        sum_both = torch.where(self.p_nan_mask & self.s_nan_mask, matrixp_pweight + matrixs_sweight, torch.zeros_like(matrixp_pweight)).sum(dim=1, keepdim=True)

        number_score_matrix = sum_matrixp
        number_score_matrix.add_(sum_matrixs)
        number_score_matrix.add_(sum_both)

        return number_score_matrix / (2 * self.number_station)

    def cal(self):
        if self.type == 'number':
            return self.number()
        elif self.type == 'prob_number':
            return self.prob_number()
        elif self.type == 'number_both':
            return self.number_both()
        elif self.type == 'prob_number_both':
            return self.prob_number_both()
        else:
            raise ValueError(f'Calculating Number Score: Function {self.type} not supported')


class TimeScore(object):
    """
    Multiple Types for Time Score

    Robust Gaussian residual:
        rho(e; sigma) = 1 - exp(-e^2 / (2*sigma^2))

    Final score:
        sum(w * rho) / N_valid

    where:
        sigma_P = p_max_offset / 2
        sigma_S = s_max_offset / 2

    Lower score is better.
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
        device='cuda'
    ):
        self.p_max_offset = p_max_offset
        self.s_max_offset = s_max_offset

        self.p_sigma = p_max_offset / 2.0
        self.s_sigma = s_max_offset / 2.0

        self.p_err = torch.as_tensor(p_err, dtype=torch.float32, device=device)
        self.s_err = torch.as_tensor(s_err, dtype=torch.float32, device=device)
        self.p_prob = torch.as_tensor(p_prob, dtype=torch.float32, device=device)
        self.s_prob = torch.as_tensor(s_prob, dtype=torch.float32, device=device)

        self.pweight = pweight
        self.sweight = sweight

        self.distance_matrix = torch.as_tensor(distance_matrix, dtype=torch.float32, device=device)
        self.dis0 = dis0
        self.dis1 = dis1
        self.type = type
        self.device = device

        self.weight_matrix = self._compute_distance_weights()

    def _compute_distance_weights(self):
        weight_matrix = self.distance_matrix.detach().clone()
        weight_matrix[self.distance_matrix < self.dis0] = 1.0

        mask = (self.distance_matrix >= self.dis0) & (self.distance_matrix <= self.dis1)
        weight_matrix[mask] = torch.cos(
            (math.pi / 2) * (self.distance_matrix[mask] - self.dis0) / (self.dis1 - self.dis0)
        )

        weight_matrix[self.distance_matrix > self.dis1] = float('nan')
        return weight_matrix

    def _weighted_gaussian_score(self, err, sigma, weight=None):
        """
        score = sum(w * rho(err; sigma)) / N_valid
        rho(err; sigma) = 1 - exp(-err^2 / (2*sigma^2))

        Notes:
        1. NaN residuals are ignored
        2. NaN / negative / zero weights are treated as 0 contribution
        3. denominator is the number of valid phases, not sum(weight)
        """
        valid = ~torch.isnan(err)

        if weight is None:
            weight = torch.ones_like(err)
        else:
            weight = torch.as_tensor(weight, dtype=torch.float32, device=self.device)

        safe_weight = torch.where(valid, weight, torch.zeros_like(weight))
        safe_err = torch.where(valid, err, torch.zeros_like(err))

        safe_weight = torch.nan_to_num(safe_weight, nan=0.0, posinf=0.0, neginf=0.0)
        safe_weight = torch.clamp(safe_weight, min=0.0)

        sigma = max(float(sigma), 1e-6)

        rho = 1.0 - torch.exp(-torch.square(safe_err) / (2.0 * sigma * sigma))

        num = torch.sum(safe_weight * rho, dim=1, keepdim=True)
        count = torch.sum(valid, dim=1, keepdim=True)

        score = num / count.clamp_min(1)
        score[count <= 0] = float('nan')
        return score

    def rms(self):
        p_score_matrix = self._weighted_gaussian_score(self.p_err, self.p_sigma, None)
        s_score_matrix = self._weighted_gaussian_score(self.s_err, self.s_sigma, None)

        time_score_matrix = (
            self.pweight * p_score_matrix
            + self.sweight * s_score_matrix
        )
        return time_score_matrix

    def distance_rms(self):
        p_score_matrix = self._weighted_gaussian_score(self.p_err, self.p_sigma, self.weight_matrix)
        s_score_matrix = self._weighted_gaussian_score(self.s_err, self.s_sigma, self.weight_matrix)

        time_score_matrix = (
            self.pweight * p_score_matrix
            + self.sweight * s_score_matrix
        )
        return time_score_matrix

    def prob_rms(self):
        p_score_matrix = self._weighted_gaussian_score(self.p_err, self.p_sigma, self.p_prob)
        s_score_matrix = self._weighted_gaussian_score(self.s_err, self.s_sigma, self.s_prob)

        time_score_matrix = (
            self.pweight * p_score_matrix
            + self.sweight * s_score_matrix
        )
        return time_score_matrix

    def prob_distance_rms(self):
        p_weight = self.weight_matrix * self.p_prob
        s_weight = self.weight_matrix * self.s_prob

        p_score_matrix = self._weighted_gaussian_score(self.p_err, self.p_sigma, p_weight)
        s_score_matrix = self._weighted_gaussian_score(self.s_err, self.s_sigma, s_weight)

        time_score_matrix = (
            self.pweight * p_score_matrix
            + self.sweight * s_score_matrix
        )
        return time_score_matrix

    def cal(self):
        if self.type == 'rms':
            return self.rms()
        elif self.type == 'distance_rms':
            return self.distance_rms()
        elif self.type == 'prob_rms':
            return self.prob_rms()
        elif self.type == 'prob_distance_rms':
            return self.prob_distance_rms()
        else:
            raise ValueError(f'Calculating Time Score: Function {self.type} not supported')


class TimeScoreDepth(object):
    """
    Multiple Types for Time Score under only update depth mode
    s-p: s travel time - p travel time
    RMS: Root Mean Square of deviation for every s-p phase

    Args:
        max_offset: maximum offset for the deviation between the theoretical travel time and the observed travel time
        ps_err: (theoretical travel time - observed travel time) of s-p phases
        distance_matrix: distance matrix for every station
        ddis0: distance < ddis0, weight = 1
        ddis1: distance > ddis1, weight = 0
        if ddis0 <= distance <= ddis1: weight = cos(pi * (distance - ddis0) / (ddis1 - ddis0) / 2)

    Returns:
        time_score_matrix
    """

    def __init__(self, max_offset, ps_err, distance_matrix, ddis0, ddis1, type, device='cuda'):
        self.max_offset = max_offset
        self.ps_err = torch.as_tensor(ps_err, dtype=torch.float32, device=device)
        self.distance_matrix = torch.as_tensor(distance_matrix, dtype=torch.float32, device=device)
        self.ddis0 = ddis0
        self.ddis1 = ddis1
        self.type = type
        self.device = device

        # Precompute distance weight matrix
        self.weight_matrix = self._compute_distance_weights()

    def _compute_distance_weights(self):
        weight_matrix = self.distance_matrix.detach().clone()
        weight_matrix[self.distance_matrix < self.ddis0] = 1
        mask = (self.distance_matrix >= self.ddis0) & (self.distance_matrix <= self.ddis1)
        weight_matrix[mask] = torch.cos((math.pi / 2) * (self.distance_matrix[mask] - self.ddis0) / (self.ddis1 - self.ddis0))
        weight_matrix[self.distance_matrix > self.ddis1] = float('nan')
        return weight_matrix

    # time_score = abs(s-p) / (2 * max_offset)
    def nearest_station(self):
        N, _ = self.distance_matrix.shape
        distance_matrix = self.distance_matrix.clone()
        distance_matrix[torch.isnan(self.ps_err)] = float('nan')

        distance_matrix_cleaned = torch.where(torch.isnan(distance_matrix), torch.tensor(float('inf')), distance_matrix)

        indices = torch.argmin(distance_matrix_cleaned, dim=1)
        time_score_matrix = torch.full((N,), float('nan'), device=self.device)

        valid_mask = ~torch.isnan(distance_matrix).all(dim=1)
        time_score_matrix[valid_mask] = torch.abs(self.ps_err[valid_mask, indices[valid_mask]])

        return time_score_matrix.view(-1, 1) / (2 * self.max_offset)

    # time_score = distance_weight * RMS(s-p) / (2 * max_offset)
    def distance_weight(self):
        weight_matrix = self.weight_matrix.clone()  # Use the precomputed weight matrix

        squared_sum = torch.nansum(torch.square(self.ps_err) * weight_matrix, dim=1, keepdim=True)
        count_non_nan = torch.sum(~torch.isnan(self.ps_err * weight_matrix), dim=1, keepdim=True)

        time_score_matrix = torch.sqrt(squared_sum / count_non_nan)
        time_score_matrix[torch.isnan(time_score_matrix)] = float('nan')

        return time_score_matrix / (2 * self.max_offset)

    def cal(self):
        if self.type == 'nearest_station':
            return self.nearest_station()
        elif self.type == 'distance_weight':
            return self.distance_weight()
        else:
            raise ValueError(f'Calculating Time Score Under Only Update Depth Mode: Function {self.type} not supported')


class MagnitudeScore(object):
    """
    Multiple Types for Magnitude Score

    Args:
        distance_matrix: distance matrix for every station
        p_amp: Amplitude for P-phases
        s_amp: Amplitude for S-phases
        p_weight: User-defined weight for P-phases
        s_weight: User-defined weight for S-phases (=1 - p_weight)

    Returns:
        magnitude_score_matrix
    """

    def __init__(self, p_amp, s_amp, p_weight, s_weight, distance_matrix, type, device='cuda'):
        self.p_amp = torch.as_tensor(p_amp, dtype=torch.float32, device=device)
        self.s_amp = torch.as_tensor(s_amp, dtype=torch.float32, device=device)
        self.p_weight = p_weight
        self.s_weight = s_weight
        self.distance_matrix = torch.as_tensor(distance_matrix, dtype=torch.float32, device=device)
        self.type = type
        self.device = device

    def cal_mag(self, amp):
        if self.type == 'Continuous':
            magnitude_matrix = torch.log10(amp) + 1.110 * torch.log10(self.distance_matrix / 100) + 0.00189 * (self.distance_matrix - 100) + 3.0  # Hutton and Boore (1987)
        elif self.type == 'Discrete':
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
                                   torch.where(valid_distance_matrix <=100, 3.4,
                                   torch.where(valid_distance_matrix <=120, 3.5,
                                   torch.where(valid_distance_matrix <=140, 3.6,
                                   torch.where(valid_distance_matrix <=160, 3.7,
                                   torch.where(valid_distance_matrix <=180, 3.8,
                                   torch.where(valid_distance_matrix <=220, 3.9,
                                   torch.where(valid_distance_matrix <=250, 4.0, 4.1)))))))))))))))))))
            magnitude_matrix = torch.log10(amp * 2080 * 20) + empirical_matrix
        else:
            raise ValueError(f'Calculating Magnitude Score: Function {self.type} not supported')
        return magnitude_matrix

    def nan_std(self):
        p_magnitude_matrix = self.cal_mag(self.p_amp)
        s_magnitude_matrix = self.cal_mag(self.s_amp)

        p_magnitude_matrix[torch.isinf(p_magnitude_matrix)] = float('nan')
        p_magnitude_score_matrix = torch.nanstd(p_magnitude_matrix, dim=1, keepdim=True)

        s_magnitude_matrix[torch.isinf(s_magnitude_matrix)] = float('nan')
        s_magnitude_score_matrix = torch.nanstd(s_magnitude_matrix, dim=1, keepdim=True)

        all_nan_rows_p = torch.isnan(p_magnitude_matrix).all(dim=1)
        all_nan_rows_s = torch.isnan(s_magnitude_matrix).all(dim=1)

        p_magnitude_score_matrix[all_nan_rows_p] = float('nan')
        s_magnitude_score_matrix[all_nan_rows_s] = float('nan')

        magnitude_score_matrix = self.p_weight * p_magnitude_score_matrix + self.s_weight * s_magnitude_score_matrix
        return magnitude_score_matrix / 10

    def cal_median_mag(self):
        p_magnitude_matrix = self.cal_mag(self.p_amp)
        s_magnitude_matrix = self.cal_mag(self.s_amp)

        magnitude_matrix = torch.cat((p_magnitude_matrix, s_magnitude_matrix), dim=0)
        data_no_nan = magnitude_matrix[~torch.isnan(magnitude_matrix)]

        median_value = torch.median(data_no_nan)
        return p_magnitude_matrix, s_magnitude_matrix, median_value

