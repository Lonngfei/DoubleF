import torch
import math

def nan_std(x, dim, keepdim=False):
    mask = ~torch.isnan(x)
    count = mask.sum(dim=dim, keepdim=keepdim)

    mean = torch.nan_to_num(x, nan=0.0).sum(dim=dim, keepdim=keepdim) / count
    var = torch.nan_to_num((x - mean) ** 2, nan=0.0).sum(dim=dim, keepdim=keepdim) / count
    std = var.sqrt()

    std[count == 0] = float('nan')
    return std


class BatchNumberScore(object):
    """
    Multiple Types for Number Score
    p_number: Number of P-phases
    s_number: Number of S-phases
    both_ps_number: Number of stations with both p and s

    Args:
        pweight: User-defined weight for P-phases
        sweight: User-defined weight for S-phases (=1 - pweight)
        p_prob: Input probability of picked P-phases (batch_size x N x S)
        s_prob: Input probability of picked S-phases (batch_size x N x S)
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

    def number(self):
        p_matrix = self.p_nan_mask.sum(dim=2)
        s_matrix = self.s_nan_mask.sum(dim=2)

        number_score_matrix = self.pweight * p_matrix + self.sweight * s_matrix

        return number_score_matrix / self.number_station

    def prob_number(self):
        p_matrix = torch.where(self.p_nan_mask, self.p_prob, torch.zeros_like(self.p_prob)).sum(dim=2)
        s_matrix = torch.where(self.s_nan_mask, self.s_prob, torch.zeros_like(self.s_prob)).sum(dim=2)

        number_score_matrix = self.pweight * p_matrix + self.sweight * s_matrix

        return number_score_matrix / self.number_station

    def number_both(self):
        p_matrix = self.p_nan_mask.sum(dim=2)
        s_matrix = self.s_nan_mask.sum(dim=2)
        both_matrix = (self.p_nan_mask & self.s_nan_mask).sum(dim=2)

        number_score_matrix = self.pweight * p_matrix + self.sweight * s_matrix + both_matrix

        return number_score_matrix / (2 * self.number_station)

    def prob_number_both(self):
        matrixp_pweight = self.pweight * self.p_prob
        matrixs_sweight = self.sweight * self.s_prob

        sum_matrixp = torch.where(self.p_nan_mask, matrixp_pweight, torch.zeros_like(matrixp_pweight)).sum(dim=2)
        sum_matrixs = torch.where(self.s_nan_mask, matrixs_sweight, torch.zeros_like(matrixs_sweight)).sum(dim=2)
        sum_both = torch.where(self.p_nan_mask & self.s_nan_mask, matrixp_pweight + matrixs_sweight, torch.zeros_like(matrixp_pweight)).sum(dim=2)

        number_score_matrix = sum_matrixp + sum_matrixs + sum_both

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

class BatchTimeScore(object):
    """
    Multiple types for time score.

    Supported types:
        rms
        distance_rms
        prob_rms
        prob_distance_rms

    Robust Gaussian residual:
        rho(e; sigma) = 1 - exp(-e^2 / (2*sigma^2))

    Final score:
        score = sum(w * rho) / N_valid

    where:
        sigma_P = p_max_offset / 2
        sigma_S = s_max_offset / 2

    Notes:
        1. Lower score is better.
        2. NaN residuals are ignored.
        3. NaN / negative weights are treated as 0.
        4. Denominator is the number of valid phases, not sum(weight).

    Args:
        p_max_offset: maximum allowed P residual
        s_max_offset: maximum allowed S residual
        p_err: theoretical travel time - observed travel time of P phases
               shape = (batch_size, N, S)
        s_err: theoretical travel time - observed travel time of S phases
               shape = (batch_size, N, S)
        p_prob: input probability of picked P phases
                shape = (batch_size, N, S)
        s_prob: input probability of picked S phases
                shape = (batch_size, N, S)
        distance_matrix: distance matrix for every station
                         shape = (batch_size, N, S)
        dis0: distance < dis0, weight = 1
        dis1: distance > dis1, weight = 0
              if dis0 <= distance <= dis1:
              weight = cos(pi/2 * (distance - dis0) / (dis1 - dis0))

    Returns:
        time_score_matrix, shape = (batch_size, N, 1)
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
        self.p_max_offset = float(p_max_offset)
        self.s_max_offset = float(s_max_offset)

        # sigma = tolerance / 2
        self.p_sigma = max(self.p_max_offset / 2.0, 1e-6)
        self.s_sigma = max(self.s_max_offset / 2.0, 1e-6)

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

        self.weight_matrix = self._compute_distance_weights()

    def _compute_distance_weights(self):
        """
        Distance weighting:
            d < dis0:     weight = 1
            d > dis1:     weight = 0
            dis0 <= d <= dis1:
                         weight = cos(pi/2 * (d - dis0)/(dis1 - dis0))
        """
        d = self.distance_matrix.detach().clone()

        if self.dis1 <= self.dis0:
            raise ValueError(
                f'dis1 must be larger than dis0, but got dis0={self.dis0}, dis1={self.dis1}'
            )

        weight_matrix = torch.zeros_like(d)

        weight_matrix[d < self.dis0] = 1.0

        mask = (d >= self.dis0) & (d <= self.dis1)
        weight_matrix[mask] = torch.cos(
            (math.pi / 2.0) * (d[mask] - self.dis0) / (self.dis1 - self.dis0)
        )

        weight_matrix[d > self.dis1] = 0.0
        return weight_matrix

    def _prepare_weight(self, err, weight=None):
        """
        Make weight broadcastable to err and clean invalid values.
        """
        if weight is None:
            weight = torch.ones_like(err)
        else:
            weight = torch.as_tensor(weight, dtype=torch.float32, device=self.device)
            weight = torch.broadcast_to(weight, err.shape)

        valid = ~torch.isnan(err)

        safe_err = torch.where(valid, err, torch.zeros_like(err))
        safe_weight = torch.where(valid, weight, torch.zeros_like(err))

        safe_weight = torch.nan_to_num(safe_weight, nan=0.0, posinf=0.0, neginf=0.0)
        safe_weight = torch.clamp(safe_weight, min=0.0)

        return valid, safe_err, safe_weight

    def _weighted_gaussian_score(self, err, sigma, weight=None):
        """
        score = sum(w * rho(err; sigma)) / N_valid
        rho(err; sigma) = 1 - exp(-err^2 / (2*sigma^2))

        Computed along dim=2.
        """
        valid, safe_err, safe_weight = self._prepare_weight(err, weight)

        rho = 1.0 - torch.exp(-torch.square(safe_err) / (2.0 * sigma * sigma))

        num = torch.sum(safe_weight * rho, dim=2, keepdim=True)
        count = torch.sum(valid, dim=2, keepdim=True)

        score = num / count.clamp_min(1)
        score = score.to(torch.float32)
        score[count <= 0] = float('nan')
        return score

    def rms(self):
        """
        Unweighted Gaussian robust score.
        """
        p_score_matrix = self._weighted_gaussian_score(self.p_err, self.p_sigma)
        s_score_matrix = self._weighted_gaussian_score(self.s_err, self.s_sigma)

        time_score_matrix = (
            self.pweight * p_score_matrix
            + self.sweight * s_score_matrix
        )
        return time_score_matrix

    def distance_rms(self):
        """
        Distance-weighted Gaussian robust score.
        """
        p_score_matrix = self._weighted_gaussian_score(self.p_err, self.p_sigma, self.weight_matrix)
        s_score_matrix = self._weighted_gaussian_score(self.s_err, self.s_sigma, self.weight_matrix)

        time_score_matrix = (
            self.pweight * p_score_matrix
            + self.sweight * s_score_matrix
        )
        return time_score_matrix

    def prob_rms(self):
        """
        Probability-weighted Gaussian robust score.
        """
        p_score_matrix = self._weighted_gaussian_score(self.p_err, self.p_sigma, self.p_prob)
        s_score_matrix = self._weighted_gaussian_score(self.s_err, self.s_sigma, self.s_prob)

        time_score_matrix = (
            self.pweight * p_score_matrix
            + self.sweight * s_score_matrix
        )
        return time_score_matrix

    def prob_distance_rms(self):
        """
        Probability * distance weighted Gaussian robust score.
        """
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

class BatchMagnitudeScore(object):
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
            magnitude_matrix = torch.log10(amp) + 1.110 * torch.log10(self.distance_matrix / 100) + 0.00189 * (
                        self.distance_matrix - 100) + 3.0  # Hutton and Boore (1987)
        elif self.type == 'Discrete':
            valid_distance_matrix = amp * 4000 / (2080 * 20)
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
                                           torch.where(valid_distance_matrix <= 100,3.4,
                                           torch.where(valid_distance_matrix <= 120,3.5,
                                           torch.where(valid_distance_matrix <= 140,3.6,
                                           torch.where(valid_distance_matrix <= 160,3.7,
                                           torch.where(valid_distance_matrix <= 180,3.8,
                                           torch.where(valid_distance_matrix <= 220,3.9,
                                           torch.where(valid_distance_matrix <= 250,4.0,
                                           4.1)))))))))))))))))))
            magnitude_matrix = torch.log10(amp) + empirical_matrix
        else:
            raise ValueError(f'Calculating Magnitude Score: Function {self.type} not supported')
        return magnitude_matrix

    def nan_std(self):
        p_magnitude_matrix = self.cal_mag(self.p_amp)
        s_magnitude_matrix = self.cal_mag(self.s_amp)

        p_magnitude_score_matrix = nan_std(p_magnitude_matrix, 2, True)
        s_magnitude_score_matrix = nan_std(s_magnitude_matrix, 2, True)

        magnitude_score_matrix = self.p_weight * p_magnitude_score_matrix + self.s_weight * s_magnitude_score_matrix
        return (magnitude_score_matrix / 10).squeeze(-1)