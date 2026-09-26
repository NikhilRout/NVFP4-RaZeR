import torch


def quant_scale(scale_fp, exp_bits, man_bits, exp_min=None):
    if exp_min is None:
        exp_min = -2**(exp_bits-1) + 2
    scale_sign = scale_fp.sign()
    assert (scale_sign == -1).any().logical_not(), "The scaling factor CANNOT be negative. Something is WRONG..."
    scale_exp  = (
        scale_fp + (scale_fp == 0).type(scale_fp.dtype)
    ).log2().floor().clamp_(min=exp_min)
    scale_man  = torch.round(
        scale_fp / 2**scale_exp * 2**man_bits
    ) / (2**man_bits)
    scale_dq   = scale_sign * 2**scale_exp * scale_man 

    return scale_dq


def quant_scale_lns(scale_fp, exp_int_bits, exp_frac_bits):
    """
        Quantize a (positive) scaling factor to a logarithmic number system,
        whose base-2 logarithm is a signed two's-complement fixed-point number
        with `exp_int_bits` integer bits and `exp_frac_bits` fractional bits.
    """
    exp_step  = 2**(-exp_frac_bits)
    scale_exp = torch.log2(
        scale_fp + (scale_fp == 0).type(scale_fp.dtype)
    )
    scale_exp = torch.round(scale_exp / exp_step) * exp_step
    scale_exp = scale_exp.clamp(
        min=-2**(exp_int_bits - 1),
        max=lns_exp_max(exp_int_bits, exp_frac_bits)
    )

    return torch.exp2(scale_exp)


def lns_exp_max(exp_int_bits, exp_frac_bits):
    """
        Largest exponent of an LNS format with the given fixed-point split.
    """
    return 2**(exp_int_bits - 1) - 2**(-exp_frac_bits)


def fp_scale_max(exp_bits, man_bits):
    """
        Largest value of a floating-point scale format with `exp_bits` exponent
        bits and `man_bits` mantissa bits, following the E4M3 convention of
        reserving the all-ones mantissa at the top exponent (so 4, 3 -> 448).
    """
    return (2 - 2**(1 - man_bits)) * 2**(2**(exp_bits - 1))


def fp_scale_min(exp_bits, man_bits):
    """
        Smallest non-zero (subnormal) value of a floating-point scale format.
    """
    return 2**(2 - 2**(exp_bits - 1) - man_bits)
