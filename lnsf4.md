# LNS8 Block Scales for NVFP4 and RaZeR

Replacing the FP8 block scale of NVFP4 and RaZeR with an 8-bit logarithmic
number system (LNS) scale, at matched bit budgets.

**Model:** Llama-3.1-8B &nbsp;·&nbsp; **Quantization:** W4A4, group size 16 and 32
&nbsp;·&nbsp; **Eval:** wikitext-2 and C4, `seq_len` 2048

All result tables are collected at the end, under **Results**.

## Format

The element datatype is unchanged throughout: FP4 (E2M1), values
`{0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}`, with the per-tensor FP32 global scale
that NVFP4 already uses. The only thing that varies is how the per-block scale
is encoded.

An LNS scale stores `2**e`, where `e` is a signed two's-complement fixed-point
number with `I` integer bits and `F` fractional bits, written **I.F**. The two
encodings differ in how they spend a fixed 8 bits:

| | FP (E{E}M{M}) | LNS (I.F) |
| --- | --- | --- |
| Step | relative, varies up to 2x within a binade | relative, uniform at `2**(2**-F) - 1` |
| Range | `2**(2**E)` ish | `2**(2**I)` |
| Sign bit | present, redundant (scales are positive) | not stored when all 8 bits go to the exponent |

The global scale anchors the largest block scale to the top of the
representable range, exactly as NVFP4 pins it to E4M3's 448.

### Bit splits compared

| Split | FP meaning | LNS meaning | Dynamic range (FP / LNS) |
| --- | --- | --- | --- |
| 4.3 | E4M3, 1 sign + 7 bits | 1 redundant sign + 7-bit exponent | 2^17.8 / 2^15.9 |
| 5.3 | E5M3, 8 bits | 8-bit exponent, sign dropped | 2^33.8 / 2^31.9 |
| 4.4 | E4M4, 8 bits | 8-bit exponent, extra fractional bit | 2^18.9 / 2^15.9 |

4.3 is the only split where LNS pays for a redundant sign bit that FP also
carries; 5.3 spends the recovered bit on range, 4.4 on precision.

### RaZeR variants

RaZeR's premise is that the scale's redundant bits can index special values, so
the LNS mirrors keep the same bit accounting:

- **e3m3**: E3M3 uses 7 of 8 bits and frees 2 to index 4 special values
  (`{±5, ±outlier}`), so the LNS exponent keeps 6 bits -> **3.3**. Weight side.
- **e4m3**: E4M3 frees only its sign bit, indexing 2 special values (`±5`), so
  the LNS exponent keeps 7 bits -> **4.3**. Activation side.

Everything else is line-for-line identical to the existing quantizers: same
special values, same datatype search, same error metric.

## What the numbers say

Four sweeps, 40 configurations. The short version: **the choice between
an LNS and an FP block scale is not what matters.** Every LNS-vs-FP gap in the
study is at most 0.036 perplexity, and which encoding wins flips with both the
dataset and the group size. Two effects that are an order of magnitude larger
sit on either side of it.

**1. Group size dominates everything else.** Going from group 16 to group 32
costs +0.164 perplexity on wikitext-2 and +0.235 on C4, holding the format
fixed. That is 5 to 15 times any LNS-vs-FP difference measured here. If the
goal is accuracy per bit, the block size is the lever; the scale encoding is
noise next to it.

**2. The precision bit is the one format choice that replicates.** Moving the
eighth bit into the fraction/mantissa (4.4) beats the 4.3 baseline in **all
eight** cases, four sweeps times two encodings:

| Sweep | FP 4.4 vs 4.3 | LNS 4.4 vs 4.3 |
| --- | --- | --- |
| wikitext g16 | -0.0073 | -0.0183 |
| C4 g16 | -0.0163 | -0.0448 |
| wikitext g32 | -0.0017 | -0.0064 |
| C4 g32 | -0.0084 | -0.0267 |

Spending it on range (5.3) does not replicate: the sign of the change flips
across sweeps for both encodings, and the largest single effect is a
*regression* (FP, C4 g32, +0.0168). The per-tensor global scale already factors
out the tensor maximum, so a block scale never needs more span; extra exponent
bits are dead weight, extra fraction bits are not.

**3. LNS vs FP is a wash, and the wikitext-g32 sweep is why.** LNS wins 4 of 5
rows on wikitext g16, 4 of 5 on C4 g16, and 4 of 5 on C4 g32, but **0 of 5** on
wikitext g32, where FP wins every row. A format that loses a clean sweep under
one setting is not a format with an edge. The most defensible statement is that
the two encodings are interchangeable at matched bit budgets, with a weak
tendency for LNS to do better at group 16 than group 32.

An earlier read of the wikitext-g16 sweep alone suggested that LNS's uniform
relative step gives it a growing advantage as precision increases. That does not
survive: at wikitext g32 the 4.4 pairing inverts and FP wins by 0.0057. The
mechanism predicted the opposite, so it is withdrawn rather than patched.

**4. The RaZeR mirror is a wash too.** Swapping only the scale encoding inside
RaZeR, with identical bit budgets, special values and error metric, gives
-0.0066 (wikitext g16), +0.0017 (C4 g16), +0.0055 (wikitext g32) and -0.0053
(C4 g32). Three sign changes in four sweeps. This was the headline result when
only wikitext g16 existed; it did not replicate. The plausible reason is that
RaZeR's special value already absorbs the block's worst-fit element, which is
the same error a finer scale grid would have reduced, so the two mechanisms
compete rather than compound.

**5. A latent bug in `quant_nvfp4_razer_e4m3`.** The special-value search
compares against `block_scale_q` rather than `w_scaled`:

```python
quant_error = (w_q_razer_tmp - block_scale_q).pow(2).mean(-1)   # quantizer.py
```

These are dimensionally unrelated, so the `±5` choice is close to arbitrary.
`quant_nvfp4_razer_e3m3` performs the analogous comparison correctly. Fixing it
is worth 15% MSE on synthetic data (7.27e-3 -> 6.20e-3). The `_fixed` rows in
the tables below are **not** a clean measurement of that, because they also drop
from 4 special values to 2 (e4m3 on both sides instead of e3m3 weights); they
exist as a matched FP/LNS control pair, not as a bug-impact measurement.
Isolating the metric alone would need one more run per sweep. Published numbers
are untouched; the fix lives only in the separately named `_fixed` variants.

## Caveats

No error bars. Every cell is a single run at one seed, and the effects being
compared are 0.001 to 0.04 perplexity. The only reason any claim above is stated
with confidence is replication across four sweeps, not the size of any single
gap. Claims 1 and 2 replicate; claims 3 and 4 are explicitly negative results.

This is an emulation study. All formats are fake-quantized in bf16, so nothing
here speaks to kernel cost. An LNS scale changes the hardware trade — the
multiply by the block scale becomes an add on the exponent — but the log-domain
conversion is not free and the FP4 element multiply still has to happen. A real
argument for LNS scales would have to be made on area or energy, not on the
perplexity in these tables.

## Reproducing

```bash
python run_ppl.py --model_name llama-3.1-8b --datasets wikitext --seq_len 2048 \
    --output_dir results/ppl_2048 \
    --w_bits 4 --w_groupsize 16 --w_dtype <W_DTYPE> --w_outlier 8.0 \
    --a_bits 4 --a_groupsize 16 --a_dtype <A_DTYPE>
```

| Row | `--w_dtype` | `--a_dtype` |
| --- | --- | --- |
| FP 4.3 | `nvfp4` | `nvfp4` |
| LNS 4.3 | `nvfp4_lns8_4.3` | `nvfp4_lns8_4.3` |
| FP 5.3 | `nvfp4_fp8_5.3` | `nvfp4_fp8_5.3` |
| LNS 5.3 | `nvfp4_lns8_5.3` | `nvfp4_lns8_5.3` |
| FP 4.4 | `nvfp4_fp8_4.4` | `nvfp4_fp8_4.4` |
| LNS 4.4 | `nvfp4_lns8_4.4` | `nvfp4_lns8_4.4` |
| RaZeR FP | `nvfp4_razer_e3m3` | `nvfp4_razer_e4m3` |
| RaZeR LNS | `nvfp4_razer_lns8_e3m3` | `nvfp4_razer_lns8_e4m3` |
| RaZeR FP, corrected | `nvfp4_razer_e4m3_fixed` | `nvfp4_razer_e4m3_fixed` |
| RaZeR LNS, corrected | `nvfp4_razer_lns8_e4m3_fixed` | `nvfp4_razer_lns8_e4m3_fixed` |

Every LNS and FP dtype takes an optional `_I.F` / `_E.M` suffix, so any other
split is a command-line argument away. `nvfp4_fp8_4.3` reproduces
`quant_nvfp4` bit-exactly, which is why the FP 4.3 row needs no separate run.

Implementation: `quantize/quantizer.py` (`quant_nvfp4_fp8`, `quant_nvfp4_lns8`,
`quant_nvfp4_razer_lns8_e3m3`, `quant_nvfp4_razer_lns8_e4m3`, and the two
`_fixed` controls); scale helpers `quant_scale_lns`, `lns_exp_max`,
`fp_scale_max`, `fp_scale_min` in `quantize/utils.py`.

## Results

Four sweeps over the same 10 configurations. `Delta` is LNS minus FP, so a
negative value means the LNS block scale is the better of the pair, and
`Lower PPL` names the winner. Model Llama-3.1-8B, W4A4, `seq_len` 2048.

### wikitext-2, group size 16

| Split / config | FP scale | LNS scale | Delta (LNS - FP) | Lower PPL |
| --- | --- | --- | --- | --- |
| 4.3 | **6.9343** | 6.9358 | +0.0015 (+0.02%) | FP |
| 5.3 | 6.9361 | **6.9344** | -0.0017 (-0.02%) | LNS |
| 4.4 | 6.9270 | **6.9175** | -0.0095 (-0.14%) | LNS |
| RaZeR e3m3(W) / e4m3(A) | 6.7608 | **6.7542** | -0.0066 (-0.10%) | LNS |
| RaZeR e4m3 (W+A, corrected) | 6.7774 | **6.7639** | -0.0135 (-0.20%) | LNS |

Context on the same eval: mxfp4 g16 7.6170, mxfp4 g32 7.8455, Meta mxfp4 g32
7.2022.

### C4, group size 16

| Split / config | FP scale | LNS scale | Delta (LNS - FP) | Lower PPL |
| --- | --- | --- | --- | --- |
| 4.3 | 9.9305 | **9.9228** | -0.0077 (-0.08%) | LNS |
| 5.3 | 9.9316 | **9.9254** | -0.0062 (-0.06%) | LNS |
| 4.4 | 9.9142 | **9.8779** | -0.0363 (-0.37%) | LNS |
| RaZeR e3m3(W) / e4m3(A) | **9.6673** | 9.6690 | +0.0017 (+0.02%) | FP |
| RaZeR e4m3 (W+A, corrected) | 9.7070 | **9.6827** | -0.0243 (-0.25%) | LNS |

### wikitext-2, group size 32

| Split / config | FP scale | LNS scale | Delta (LNS - FP) | Lower PPL |
| --- | --- | --- | --- | --- |
| 4.3 | **7.0979** | 7.1082 | +0.0104 (+0.15%) | FP |
| 5.3 | **7.0971** | 7.1058 | +0.0087 (+0.12%) | FP |
| 4.4 | **7.0962** | 7.1019 | +0.0057 (+0.08%) | FP |
| RaZeR e3m3(W) / e4m3(A) | **6.9205** | 6.9260 | +0.0055 (+0.08%) | FP |
| RaZeR e4m3 (W+A, corrected) | **6.9209** | 6.9217 | +0.0008 (+0.01%) | FP |

### C4, group size 32

| Split / config | FP scale | LNS scale | Delta (LNS - FP) | Lower PPL |
| --- | --- | --- | --- | --- |
| 4.3 | **10.1655** | 10.1663 | +0.0009 (+0.01%) | FP |
| 5.3 | 10.1823 | **10.1733** | -0.0090 (-0.09%) | LNS |
| 4.4 | 10.1570 | **10.1396** | -0.0174 (-0.17%) | LNS |
| RaZeR e3m3(W) / e4m3(A) | 9.9128 | **9.9076** | -0.0053 (-0.05%) | LNS |
| RaZeR e4m3 (W+A, corrected) | 9.9186 | **9.9093** | -0.0093 (-0.09%) | LNS |

### Summary

| Sweep | LNS wins | Best config | Best PPL |
| --- | --- | --- | --- |
| wikitext g16 | 4 / 5 | RaZeR e3m3/e4m3, LNS | 6.7542 |
| C4 g16 | 4 / 5 | RaZeR e3m3/e4m3, FP | 9.6673 |
| wikitext g32 | 0 / 5 | RaZeR e3m3/e4m3, FP | 6.9205 |
| C4 g32 | 4 / 5 | RaZeR e3m3/e4m3, LNS | 9.9076 |

RaZeR wins every sweep by a wide margin regardless of scale encoding, which is
the one unambiguous result in the table: the special values are worth ~0.17 to
~0.25 perplexity, an order of magnitude more than any scale-encoding choice.

---

# Day 2 — Qwen3-4B

A model-generality check. The Day 1 study above established, on Llama-3.1-8B,
that swapping the FP8 block scale for an LNS8 one at matched bit budget is
accuracy-neutral. This section repeats the identical sweep — the same 10
configurations across {wikitext-2, C4} x {group 16, group 32} — on Qwen3-4B, to
establish that the neutrality is a property of the format rather than of one
model.

**Model:** Qwen3-4B &nbsp;·&nbsp; **Quantization:** W4A4, group size 16 and 32
&nbsp;·&nbsp; **Eval:** wikitext-2 and C4, `seq_len` 2048
&nbsp;·&nbsp; `--w_outlier 8.0`

Everything else is unchanged: same quantizers, same `run_ppl.py`, same seed.
Note that Qwen3-4B tokenizes wikitext-2 into 146 evaluation samples rather than
Llama's 141, so absolute perplexities are not comparable across the two models.
Only the within-model FP-vs-LNS deltas are.

A 4B model quantized to W4A4 has a much higher baseline perplexity than an 8B
one (13.97 vs 6.93 at NVFP4 g16), so the same relative error appears as a larger
absolute delta. The percentage column is the meaningful one here.

## Results: Qwen3-4B

Same convention as Day 1: `Delta` is LNS minus FP, negative means the LNS block
scale is the better of the pair.

### wikitext-2, group size 16

| Split / config | FP scale | LNS scale | Delta (LNS - FP) | Lower PPL |
| --- | --- | --- | --- | --- |
| 4.3 | **13.9723** | 14.0909 | +0.1186 (+0.85%) | FP |
| 5.3 | **13.9654** | 14.1176 | +0.1522 (+1.09%) | FP |
| 4.4 | 14.0983 | **14.0172** | -0.0811 (-0.58%) | LNS |
| RaZeR e3m3(W) / e4m3(A) | **14.1709** | 14.2541 | +0.0832 (+0.59%) | FP |
| RaZeR e4m3 (W+A, corrected) | **14.2906** | 14.2931 | +0.0025 (+0.02%) | FP |

### C4, group size 16

| Split / config | FP scale | LNS scale | Delta (LNS - FP) | Lower PPL |
| --- | --- | --- | --- | --- |
| 4.3 | 17.2758 | **17.2427** | -0.0331 (-0.19%) | LNS |
| 5.3 | 17.2787 | **17.2538** | -0.0250 (-0.14%) | LNS |
| 4.4 | **17.1835** | 17.2774 | +0.0939 (+0.55%) | FP |
| RaZeR e3m3(W) / e4m3(A) | 17.3814 | **17.2855** | -0.0959 (-0.55%) | LNS |
| RaZeR e4m3 (W+A, corrected) | 17.3150 | **17.2470** | -0.0680 (-0.39%) | LNS |

### wikitext-2, group size 32

| Split / config | FP scale | LNS scale | Delta (LNS - FP) | Lower PPL |
| --- | --- | --- | --- | --- |
| 4.3 | **14.3315** | 14.4221 | +0.0906 (+0.63%) | FP |
| 5.3 | **14.3364** | 14.4300 | +0.0936 (+0.65%) | FP |
| 4.4 | **14.3269** | 14.5473 | +0.2203 (+1.54%) | FP |
| RaZeR e3m3(W) / e4m3(A) | 14.7290 | **14.3823** | -0.3468 (-2.35%) | LNS |
| RaZeR e4m3 (W+A, corrected) | 14.1644 | **14.0624** | -0.1020 (-0.72%) | LNS |

### C4, group size 32

| Split / config | FP scale | LNS scale | Delta (LNS - FP) | Lower PPL |
| --- | --- | --- | --- | --- |
| 4.3 | **17.6152** | 17.6644 | +0.0492 (+0.28%) | FP |
| 5.3 | **17.5899** | 17.6466 | +0.0567 (+0.32%) | FP |
| 4.4 | **17.4874** | 17.8059 | +0.3185 (+1.82%) | FP |
| RaZeR e3m3(W) / e4m3(A) | 17.6444 | **17.4331** | -0.2113 (-1.20%) | LNS |
| RaZeR e4m3 (W+A, corrected) | **17.3147** | 17.3425 | +0.0278 (+0.16%) | FP |

## Results: Llama-3.2-3B

### wikitext-2, group size 16

| Split / config | FP scale | LNS scale | Delta (LNS - FP) | Lower PPL |
| --- | --- | --- | --- | --- |
| 4.3 | 8.6059 | **8.5837** | -0.0223 (-0.26%) | LNS |
| 5.3 | 8.6099 | **8.5839** | -0.0260 (-0.30%) | LNS |
| 4.4 | **8.6013** | 8.6015 | +0.0002 (+0.00%) | FP |
| RaZeR e3m3(W) / e4m3(A) | 8.4078 | **8.3881** | -0.0196 (-0.23%) | LNS |
| RaZeR e4m3 (W+A, corrected) | 8.4225 | **8.4020** | -0.0206 (-0.24%) | LNS |

### C4, group size 16

| Split / config | FP scale | LNS scale | Delta (LNS - FP) | Lower PPL |
| --- | --- | --- | --- | --- |
| 4.3 | 11.6485 | **11.6111** | -0.0374 (-0.32%) | LNS |
| 5.3 | 11.6522 | **11.6338** | -0.0184 (-0.16%) | LNS |
| 4.4 | 11.6216 | **11.6126** | -0.0090 (-0.08%) | LNS |
| RaZeR e3m3(W) / e4m3(A) | 11.3119 | **11.3034** | -0.0085 (-0.08%) | LNS |
| RaZeR e4m3 (W+A, corrected) | 11.3446 | **11.3189** | -0.0258 (-0.23%) | LNS |

### wikitext-2, group size 32

| Split / config | FP scale | LNS scale | Delta (LNS - FP) | Lower PPL |
| --- | --- | --- | --- | --- |
| 4.3 | 8.8295 | **8.7965** | -0.0330 (-0.37%) | LNS |
| 5.3 | 8.8312 | **8.8123** | -0.0189 (-0.21%) | LNS |
| 4.4 | 8.8371 | **8.8324** | -0.0046 (-0.05%) | LNS |
| RaZeR e3m3(W) / e4m3(A) | **8.5945** | 8.6156 | +0.0210 (+0.24%) | FP |
| RaZeR e4m3 (W+A, corrected) | 8.6125 | **8.5952** | -0.0174 (-0.20%) | LNS |

### C4, group size 32

| Split / config | FP scale | LNS scale | Delta (LNS - FP) | Lower PPL |
| --- | --- | --- | --- | --- |
| 4.3 | 11.9219 | **11.8967** | -0.0252 (-0.21%) | LNS |
| 5.3 | 11.9245 | **11.9089** | -0.0157 (-0.13%) | LNS |
| 4.4 | **11.8928** | 11.9045 | +0.0117 (+0.10%) | FP |
| RaZeR e3m3(W) / e4m3(A) | **11.5800** | 11.5987 | +0.0187 (+0.16%) | FP |
| RaZeR e4m3 (W+A, corrected) | 11.6208 | **11.6052** | -0.0156 (-0.13%) | LNS |

## Results: Qwen3-8B

### wikitext-2, group size 16

| Split / config | FP scale | LNS scale | Delta (LNS - FP) | Lower PPL |
| --- | --- | --- | --- | --- |
| 4.3 | **10.0502** | 10.1940 | +0.1438 (+1.43%) | FP |
| 5.3 | **10.0664** | 10.1846 | +0.1182 (+1.17%) | FP |
| 4.4 | **10.2078** | 10.2715 | +0.0637 (+0.62%) | FP |
| RaZeR e3m3(W) / e4m3(A) | **9.8881** | 9.9453 | +0.0572 (+0.58%) | FP |
| RaZeR e4m3 (W+A, corrected) | **9.9514** | 10.0029 | +0.0515 (+0.52%) | FP |

### C4, group size 16

| Split / config | FP scale | LNS scale | Delta (LNS - FP) | Lower PPL |
| --- | --- | --- | --- | --- |
| 4.3 | **13.7714** | 13.8050 | +0.0336 (+0.24%) | FP |
| 5.3 | **13.7772** | 13.8050 | +0.0278 (+0.20%) | FP |
| 4.4 | **13.7896** | 13.8344 | +0.0447 (+0.32%) | FP |
| RaZeR e3m3(W) / e4m3(A) | **13.6878** | 13.7021 | +0.0142 (+0.10%) | FP |
| RaZeR e4m3 (W+A, corrected) | **13.6152** | 13.6339 | +0.0188 (+0.14%) | FP |

### wikitext-2, group size 32

| Split / config | FP scale | LNS scale | Delta (LNS - FP) | Lower PPL |
| --- | --- | --- | --- | --- |
| 4.3 | **10.1521** | 10.3382 | +0.1861 (+1.83%) | FP |
| 5.3 | **10.1304** | 10.3348 | +0.2044 (+2.02%) | FP |
| 4.4 | 10.2758 | **10.2692** | -0.0066 (-0.06%) | LNS |
| RaZeR e3m3(W) / e4m3(A) | **9.9499** | 10.0678 | +0.1178 (+1.18%) | FP |
| RaZeR e4m3 (W+A, corrected) | **10.0095** | 10.1113 | +0.1019 (+1.02%) | FP |

### C4, group size 32

| Split / config | FP scale | LNS scale | Delta (LNS - FP) | Lower PPL |
| --- | --- | --- | --- | --- |
| 4.3 | **13.9593** | 14.0706 | +0.1113 (+0.80%) | FP |
| 5.3 | **13.9526** | 14.0604 | +0.1078 (+0.77%) | FP |
| 4.4 | **14.0246** | 14.0649 | +0.0403 (+0.29%) | FP |
| RaZeR e3m3(W) / e4m3(A) | **13.8257** | 13.8458 | +0.0201 (+0.15%) | FP |
| RaZeR e4m3 (W+A, corrected) | **13.8206** | 13.8573 | +0.0367 (+0.27%) | FP |

## Day 2 summary

| Model | n | Mean delta | Worst for LNS | Best for LNS | LNS wins |
| --- | --- | --- | --- | --- | --- |
| Llama-3.1-8B | 20 | -0.057% | +0.146% | -0.366% | 12/20 |
| Llama-3.2-3B | 20 | -0.135% | +0.244% | -0.374% | 16/20 |
| Qwen3-4B | 20 | +0.118% | +1.821% | -2.354% | 8/20 |
| Qwen3-8B | 20 | +0.680% | +2.018% | -0.065% | 1/20 |
| **Pooled** | **80** | **+0.151%** | +2.018% | -2.354% | 37/80 |

69 of 80 pairs land within 1%, 78 of 80 within 2%.

The neutrality result from Day 1 **holds on Llama and weakens on Qwen3**. Across
both Llama models it is genuine neutrality: mean -0.096%, worst case
+0.244%, LNS winning 28 of 40 pairs with no systematic direction.
Across both Qwen3 models the mean is +0.399% with a worst case of
+2.018%.

Qwen3-8B is the outlier and deserves to be stated plainly rather than pooled
away: **LNS loses 19 of 20 pairs there**, with a mean penalty of +0.68%. That is
not the scattered sign-flipping that neutrality produces; it is a consistent
direction, and it is what Day 3 sets out to explain.

The honest form of the claim across four models is therefore *"within 2%, mean
+0.15%, and model-dependent"* rather than the sub-0.2% bound that Llama-3.1-8B
alone suggested. For a hardware argument that is still a strong position — the
scale-path saving costs at most 2% of perplexity and nothing on Llama — but the
Qwen3-8B column should be quoted, not hidden.

Two candidate explanations are ruled out by the data itself. It is **not the
RaZeR special values**: the plain 4.3 / 5.3 / 4.4 pairs use none, and they show
the largest gaps on Qwen3-8B (+1.43%, +1.83%, +2.02%). It is **not LNS's
narrower dynamic range**: the 5.3 split carries 2^31.9 of range against E4M3's
2^17.8 and loses just as badly as 4.3.

---

# Day 3 — Diagnosing the Qwen3-8B penalty

Day 2 left one question: why does a format that is neutral on Llama cost a
consistent ~0.7% on Qwen3-8B? Two experiments, one offline and one ablation.

## Offline: block-scale rounding error

Comparing how accurately each format represents the *same* weight block scales,
on Llama-3.1-8B against Qwen3-8B:

| Model | mean relative scale error, FP8 (E4M3) | LNS (4.3) | ratio | out of range |
| --- | --- | --- | --- | --- |
| Llama-3.1-8B | 0.02261 | 0.02169 | 0.959 | 0.00 - 0.05% |
| Qwen3-8B | 0.02246 | 0.02166 | 0.964 | 0.00 - 0.01% |

The two models are indistinguishable, nothing clamps, and in both cases **LNS
has the lower mean error**. This analysis predicts LNS should be at least as
good on Qwen3-8B, which is the opposite of what perplexity shows. Mean relative
scale error is therefore not the statistic that matters; the *distribution* of
that error across blocks must be.

## Ablation: which side carries the penalty

Applying the LNS scale to only one side at a time, Qwen3-8B, wikitext-2,
group 16:

| Weights | Activations | PPL | vs all-FP |
| --- | --- | --- | --- |
| FP | FP | 10.0502 | - |
| **LNS** | **FP** | **10.1799** | **+0.1297 (+1.29%)** |
| FP | LNS | 10.0740 | +0.0238 (+0.24%) |
| LNS | LNS | 10.1940 | +0.1438 (+1.43%) |

**The weights carry ~90% of the penalty.** The activation path contributes
+0.24%, the weight path +1.29%, and the two combined are almost exactly their
sum, so the effect is additive rather than an interaction.

## What this changes

The prior hypothesis — that outlier-dominated *activation* blocks would punish
the LNS scale — was wrong, and so was the offline weight analysis that pointed
the same way. Both predicted the opposite of the ablation. This is the third
time in this project that a small mean-squared-error signal has mispredicted
perplexity. The lesson is consistent: **in W4A4, aggregate error statistics do
not rank formats.** Only end-to-end evaluation does.

The practically useful consequence is that the penalty sits on the side that can
be fixed for free. Weights are quantized **offline**, so nothing requires
round-to-nearest scale selection there — a per-block search over candidate
scales, exactly as RaZeR already searches special values, costs only
quantization time and nothing at inference. If that closes the Qwen3-8B gap, the
LNS scale becomes neutral across all four models with no hardware cost, which
would be the strongest form of the claim.

That experiment has not been run. It is the obvious next step.

---

# Day 3 (continued) — Mistral-7B and bf16 baselines

Two additions close out the study: a third architecture family, and an absolute
reference for what 4-bit quantization costs in the first place.

## Mistral-7B

Mistral is architecturally Llama plus sliding-window attention and shares its
tensor names, so it reuses the Llama quant modules (`utils.py`). At `seq_len`
2048 the 4096 sliding window never engages, so the two are numerically
equivalent. Validated against its bf16 baseline of 5.3182, which is the expected
value for this model.

### wikitext-2, group size 16

| Split / config | FP scale | LNS scale | Delta (LNS - FP) | Lower PPL |
| --- | --- | --- | --- | --- |
| 4.3 | **5.5513** | 5.5521 | +0.0008 (+0.01%) | FP |
| 5.3 | 5.5537 | **5.5514** | -0.0023 (-0.04%) | LNS |
| 4.4 | **5.5423** | 5.5459 | +0.0036 (+0.06%) | FP |
| RaZeR e3m3(W) / e4m3(A) | 5.4974 | **5.4909** | -0.0065 (-0.12%) | LNS |
| RaZeR e4m3 (W+A, corrected) | 5.4998 | **5.4975** | -0.0023 (-0.04%) | LNS |

### C4, group size 16

| Split / config | FP scale | LNS scale | Delta (LNS - FP) | Lower PPL |
| --- | --- | --- | --- | --- |
| 4.3 | **8.0909** | 8.0924 | +0.0015 (+0.02%) | FP |
| 5.3 | **8.0943** | 8.0977 | +0.0034 (+0.04%) | FP |
| 4.4 | 8.0834 | **8.0800** | -0.0034 (-0.04%) | LNS |
| RaZeR e3m3(W) / e4m3(A) | 8.0277 | **8.0225** | -0.0052 (-0.06%) | LNS |
| RaZeR e4m3 (W+A, corrected) | 8.0267 | **8.0261** | -0.0007 (-0.01%) | LNS |

### wikitext-2, group size 32

| Split / config | FP scale | LNS scale | Delta (LNS - FP) | Lower PPL |
| --- | --- | --- | --- | --- |
| 4.3 | **5.6059** | 5.6074 | +0.0014 (+0.03%) | FP |
| 5.3 | 5.6076 | **5.5995** | -0.0081 (-0.14%) | LNS |
| 4.4 | 5.6058 | **5.5979** | -0.0079 (-0.14%) | LNS |
| RaZeR e3m3(W) / e4m3(A) | **5.5438** | 5.5439 | +0.0001 (+0.00%) | FP |
| RaZeR e4m3 (W+A, corrected) | **5.5471** | 5.5472 | +0.0001 (+0.00%) | FP |

### C4, group size 32

| Split / config | FP scale | LNS scale | Delta (LNS - FP) | Lower PPL |
| --- | --- | --- | --- | --- |
| 4.3 | 8.1643 | **8.1643** | -0.0000 (-0.00%) | LNS |
| 5.3 | 8.1646 | **8.1620** | -0.0026 (-0.03%) | LNS |
| 4.4 | 8.1543 | **8.1503** | -0.0039 (-0.05%) | LNS |
| RaZeR e3m3(W) / e4m3(A) | **8.0886** | 8.0917 | +0.0032 (+0.04%) | FP |
| RaZeR e4m3 (W+A, corrected) | **8.0901** | 8.0934 | +0.0033 (+0.04%) | FP |

## bf16 baselines

Raw bf16, no quantization and no scale factors of any kind. `--use_fp16` is a
misnomer: `utils.py` loads `dtype=torch.bfloat16` on both paths, and all five
checkpoints are natively bfloat16, so nothing is converted. The quantized runs
also return bf16, so baseline and quantized differ only in the quantization.

| Model | bf16 wikitext | NVFP4 g16 | cost | bf16 C4 | NVFP4 g16 | cost |
| --- | --- | --- | --- | --- | --- | --- |
| llama-3.1-8b | 6.2403 | 6.9343 | +0.6940 (+11.1%) | 8.9577 | 9.9305 | +0.9728 (+10.9%) |
| llama-3.2-3b | 7.8166 | 8.6059 | +0.7893 (+10.1%) | 10.4354 | 11.6485 | +1.2131 (+11.6%) |
| qwen3-4b | 13.6625 | 13.9723 | +0.3098 (+2.3%) | 16.6436 | 17.2758 | +0.6323 (+3.8%) |
| qwen3-8b | 9.7265 | 10.0502 | +0.3237 (+3.3%) | 13.3021 | 13.7714 | +0.4692 (+3.5%) |
| mistral-7b | 5.3182 | 5.5513 | +0.2331 (+4.4%) | 7.8303 | 8.0909 | +0.2606 (+3.3%) |

## Final summary: five models, 100 matched pairs

| Model | n | Mean delta | Worst for LNS | Best for LNS | LNS wins |
| --- | --- | --- | --- | --- | --- |
| llama-3.1-8b | 20 | -0.057% | +0.146% | -0.366% | 12/20 |
| llama-3.2-3b | 20 | -0.135% | +0.244% | -0.374% | 16/20 |
| qwen3-4b | 20 | +0.118% | +1.821% | -2.354% | 8/20 |
| qwen3-8b | 20 | +0.680% | +2.018% | -0.065% | 1/20 |
| mistral-7b | 20 | -0.022% | +0.065% | -0.144% | 11/20 |
| **Pooled** | **100** | **+0.117%** | +2.018% | -2.354% | 48/100 |

76 of 100 pairs land within 0.5%, 89 within 1%, 98 within 2%.

**The result is that the FP-to-LNS block-scale swap is accuracy-neutral, with
Qwen3 as a named exception.** Split by family:

| Group | n | Mean delta | Worst for LNS |
| --- | --- | --- | --- |
| Llama x2 + Mistral | 60 | -0.071% | +0.244% |
| Qwen3 x2 | 40 | +0.399% | +2.018% |

Across two unrelated architecture families spanning 3B to 8B, the worst case is
+0.244% and the mean slightly favours LNS. Mistral-7B is the tightest model
in the study: every one of its 20 pairs sits within 0.15%, and its C4 g32 4.3
pair matches to four decimal places. Both Qwen3 models are looser, and
Qwen3-8B is systematically worse for LNS (19 of 20 pairs), which the ablation
above attributes to the weight path.

## Putting the deltas in proportion

The baselines supply the context the earlier sections lacked. W4A4 quantization
itself costs **+10 to +12% perplexity on Llama** and +2 to +4% on Qwen3 and
Mistral. Against that, the choice of block-scale representation moves perplexity
by **+0.117% on average**, and by at most +2.02% in the single worst
configuration measured.

On Llama-3.1-8B the comparison is stark: quantizing costs +11.1%, while swapping
the E4M3 scale for LNS8 at the same bit width costs +0.02%. The scale-factor
representation is therefore **well under 1% of the error budget of the
quantization decision already taken** — which is the form of the claim a
hardware argument should rest on. It is not that LNS is free in isolation, but
that its cost is negligible relative to the 4-bit decision it sits inside.

Note that the percentage quantization cost is not comparable across models,
because the denominators differ: Qwen3's wikitext baselines are inflated by
tokenizer and training-mix effects (13.66 for a 4B model against 7.82 for
Llama-3.2-3B), which deflates its apparent quantization cost.

## Status

Complete: 5 models x 2 datasets x 2 group sizes x 5 configurations = 100 matched
pairs, plus 10 bf16 baselines and the Qwen3-8B W/A ablation.

Not run: the per-block offline scale search for weights, which the ablation
identifies as the way to close the Qwen3-8B gap at no inference cost.
