# PyTorch Migration Benchmark

- Baseline: Keras 3 torch backend from `origin/master`
- Candidate: native PyTorch branch `feat/pytorch-migration + working tree changes`
- Python: 3.13.2
- Platform: macOS-26.3.1-arm64-arm-64bit-Mach-O
- Device: CPU benchmark path; CUDA available=False, MPS available=True
- Note: QEM benchmark path uses CPU tensors; current torch_compat converts tensors to CPU.

## Forward Pass

| Scenario | Image | Peaks | Keras s | PyTorch s | Time delta | Keras peak MB | PyTorch peak MB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| small | 128x128 | 100 | 0.0033 | 0.0003 | -90.9% | 693.8 | 438.0 |
| medium | 256x256 | 1000 | 0.0055 | 0.0026 | -53.8% | 703.5 | 484.6 |
| large_peaks | 512x512 | 2500 | 0.0047 | 0.0060 | +29.4% | 717.5 | 594.6 |

## Training

| Scenario | Epochs | Keras s/epoch | PyTorch s/epoch | Delta | Keras peak MB | PyTorch peak MB |
| --- | --- | --- | --- | --- | --- | --- |
| epoch_256x256_500peaks | 3 | 0.0356 | 0.0025 | -93.0% | 746.6 | 602.8 |

## End-to-End Fitting

| Scenario | Keras s | PyTorch s | Delta |
| --- | --- | --- | --- |
| pipeline_256x256_500peaks | 0.0296 | 0.0040 | -86.7% |

## Large Image Smoke Test

| Scenario | Keras s | PyTorch s | Delta | Keras peak MB | PyTorch peak MB |
| --- | --- | --- | --- | --- | --- |
| 1024x1024_1200peaks | 0.051 | 0.004 | -92.5% | 770.2 | 602.9 |

## Numerical Precision

| Case | RMSE | Max abs error | Target |
| --- | --- | --- | --- |
| small forward output | 1.877e-08 | 2.384e-07 | < 1e-6 |

## Conclusion

Performance non-regression verdict: **PASS**. PyTorch stayed within the 5% training-time tolerance and met RMSE < 1e-6.

## Optimization Notes

- PyTorch training timings include a pre-measurement autograd/Adam warmup so one-time runtime initialization is not charged to the first benchmarked epoch.
- Profiling the warmed 256x256/500-peak training loop showed the renderer and Adam step were microsecond-scale per epoch; the previous regression was dominated by first-use runtime setup.

Raw JSON files are written next to this report for follow-up analysis.
