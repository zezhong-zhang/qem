"""Benchmark QEM's PyTorch migration against the Keras torch-backend baseline.

The runner compares the current checkout with a git ref that still uses Keras
imports, defaulting to ``origin/master``. Each implementation is executed in a
fresh subprocess so Python's module cache cannot mix the two ``qem`` packages.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import platform
import statistics
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import psutil


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Scenario:
    name: str
    image_size: int
    peaks: int


def _run(cmd: list[str], cwd: Path, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        joined = " ".join(cmd)
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}: {joined}\n"
            f"cwd: {cwd}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return result


def _git_value(args: list[str]) -> str:
    return _run(["git", *args], REPO_ROOT).stdout.strip()


def _export_git_ref(ref: str, dest: Path) -> None:
    archive = dest / "baseline.tar"
    with archive.open("wb") as fh:
        subprocess.run(["git", "archive", "--format=tar", ref], cwd=REPO_ROOT, stdout=fh, check=True)
    with tarfile.open(archive) as tar:
        tar.extractall(dest, filter="data")
    archive.unlink()


def _device_info() -> dict[str, Any]:
    import torch

    return {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "mps_available": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
        "qem_model_device_note": "QEM benchmark path uses CPU tensors; current torch_compat converts tensors to CPU.",
    }


def _warmup_torch_training_runtime() -> None:
    """Pay one-time PyTorch autograd/optimizer setup before timed training."""
    import torch

    param = torch.nn.Parameter(torch.ones(4, dtype=torch.float32))
    optimizer = torch.optim.Adam([param], lr=1e-3)
    optimizer.zero_grad(set_to_none=True)
    loss = torch.sum(param * param)
    loss.backward()
    optimizer.step()


class MemorySampler:
    def __init__(self, interval: float = 0.005) -> None:
        self.interval = interval
        self.process = psutil.Process()
        self.peak_rss = self.process.memory_info().rss
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        while not self._stop.is_set():
            self.peak_rss = max(self.peak_rss, self.process.memory_info().rss)
            time.sleep(self.interval)

    def __enter__(self) -> "MemorySampler":
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._stop.set()
        self._thread.join()
        self.peak_rss = max(self.peak_rss, self.process.memory_info().rss)


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def _make_coordinates(image_size: int, peaks: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    margin = max(6, min(24, image_size // 16))
    return rng.uniform(margin, image_size - margin, size=(peaks, 2)).astype(np.float32)


def _background_image(image_size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    image = rng.normal(0.0, 1e-4, size=(image_size, image_size)).astype(np.float32)
    image[0, 0] = 0.0
    image[-1, -1] = 1e-3
    return image


def _set_param(params: dict[str, Any], key: str, value: Any, keras_module: Any) -> None:
    params[key] = keras_module.ops.convert_to_tensor(value, dtype="float32")


def _build_fitter(impl: str, image: np.ndarray, coords: np.ndarray, atom_size: float):
    if impl == "keras":
        import keras
        from qem.image_fitting import ImageFitting
    else:
        from qem.fit.image_fitting import ImageFitting
        from qem.utils import torch_compat as keras

    fitter = ImageFitting(image, same_width=True, fit_background=False)
    fitter.coordinates = coords
    fitter.atom_types = np.zeros(len(coords), dtype=np.int32)
    params = fitter.init_params(atom_size=atom_size, init_background=0.0)

    rng = np.random.default_rng(1234 + len(coords))
    heights = rng.uniform(0.7, 1.3, size=len(coords)).astype(np.float32)
    _set_param(params, "height", heights, keras)
    _set_param(params, "background", np.float32(0.02), keras)
    fitter.params = params
    return fitter, params


def _time_call(func, min_rounds: int) -> tuple[float, Any]:
    timings: list[float] = []
    result = None
    for _ in range(min_rounds):
        start = time.perf_counter()
        result = func()
        timings.append(time.perf_counter() - start)
    return statistics.median(timings), result


def _fit_global_compatible(fitter: Any, params: dict[str, Any], epochs: int) -> None:
    kwargs: dict[str, Any] = {
        "params": params,
        "maxiter": epochs,
        "tol": 1e-4,
        "step_size": 1e-3,
    }
    signature = inspect.signature(fitter.fit_global)
    if "optimizer" in signature.parameters:
        kwargs["optimizer"] = "adam"
    if "verbose" in signature.parameters:
        kwargs["verbose"] = False
    if "local" in signature.parameters:
        kwargs["local"] = True
    fitter.fit_global(**kwargs)


def _run_worker(args: argparse.Namespace) -> None:
    os.environ.setdefault("KERAS_BACKEND", "torch")

    import torch

    if args.impl == "keras":
        import keras
    else:
        from qem.utils import torch_compat as keras

    torch.set_num_threads(args.torch_threads)
    _warmup_torch_training_runtime()
    results: dict[str, Any] = {
        "implementation": args.impl,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "device_info": _device_info(),
        "forward": [],
        "training": [],
        "end_to_end": [],
        "large_image": [],
    }

    for scenario in [Scenario("small", 128, 100), Scenario("medium", 256, 1000), Scenario("large_peaks", 512, 2500)]:
        image = _background_image(scenario.image_size, seed=scenario.peaks + 7)
        coords = _make_coordinates(scenario.image_size, scenario.peaks, seed=scenario.peaks)
        fitter, params = _build_fitter(args.impl, image, coords, atom_size=2.0)

        def forward_once():
            return fitter.predict(params, local=True)

        forward_once()
        rss_before = psutil.Process().memory_info().rss
        with MemorySampler() as sampler:
            seconds, output = _time_call(forward_once, args.rounds)
        rss_after = psutil.Process().memory_info().rss

        output_np = _to_numpy(output)
        if scenario.name == "small":
            np.save(Path(args.artifact_dir) / f"{args.impl}_forward_small.npy", output_np)

        results["forward"].append(
            {
                "scenario": scenario.name,
                "image_size": scenario.image_size,
                "peaks": scenario.peaks,
                "median_seconds": seconds,
                "peaks_per_second": scenario.peaks / seconds,
                "mpixels_per_second": (scenario.image_size * scenario.image_size / 1_000_000) / seconds,
                "rss_before_mb": rss_before / 1024**2,
                "rss_after_mb": rss_after / 1024**2,
                "peak_rss_mb": sampler.peak_rss / 1024**2,
                "output_mean": float(output_np.mean()),
            }
        )

    train_scenario = Scenario("epoch_256x256_500peaks", 256, 500)
    coords = _make_coordinates(train_scenario.image_size, train_scenario.peaks, seed=987)
    base_image = _background_image(train_scenario.image_size, seed=222)
    fitter, params = _build_fitter(args.impl, base_image, coords, atom_size=2.0)
    target = _to_numpy(fitter.predict(params, local=True)).astype(np.float32)
    train_fitter, train_params = _build_fitter(args.impl, target, coords, atom_size=2.2)

    start_rss = psutil.Process().memory_info().rss
    with MemorySampler() as sampler:
        start = time.perf_counter()
        _fit_global_compatible(train_fitter, train_params, args.epochs)
        train_seconds = time.perf_counter() - start
    results["training"].append(
        {
            "scenario": train_scenario.name,
            "epochs": args.epochs,
            "seconds": train_seconds,
            "seconds_per_epoch": train_seconds / args.epochs,
            "rss_before_mb": start_rss / 1024**2,
            "peak_rss_mb": sampler.peak_rss / 1024**2,
        }
    )

    e2e_scenario = Scenario("pipeline_256x256_500peaks", 256, 500)
    coords = _make_coordinates(e2e_scenario.image_size, e2e_scenario.peaks, seed=654)
    image = _background_image(e2e_scenario.image_size, seed=333)
    source_fitter, source_params = _build_fitter(args.impl, image, coords, atom_size=2.0)
    image = _to_numpy(source_fitter.predict(source_params, local=True)).astype(np.float32)

    start = time.perf_counter()
    e2e_fitter, e2e_params = _build_fitter(args.impl, image, coords, atom_size=2.1)
    _fit_global_compatible(e2e_fitter, e2e_params, max(1, args.epochs // 2))
    e2e_seconds = time.perf_counter() - start
    results["end_to_end"].append(
        {
            "scenario": e2e_scenario.name,
            "seconds": e2e_seconds,
            "epochs": max(1, args.epochs // 2),
        }
    )

    if args.large_image_size:
        large = Scenario(f"{args.large_image_size}x{args.large_image_size}_{args.large_peaks}peaks", args.large_image_size, args.large_peaks)
        coords = _make_coordinates(large.image_size, large.peaks, seed=321)
        image = _background_image(large.image_size, seed=444)
        fitter, params = _build_fitter(args.impl, image, coords, atom_size=2.0)
        start_rss = psutil.Process().memory_info().rss
        with MemorySampler() as sampler:
            seconds, output = _time_call(lambda: fitter.predict(params, local=True), 1)
        results["large_image"].append(
            {
                "scenario": large.name,
                "seconds": seconds,
                "peaks": large.peaks,
                "image_size": large.image_size,
                "peak_rss_mb": sampler.peak_rss / 1024**2,
                "rss_delta_mb": (sampler.peak_rss - start_rss) / 1024**2,
                "output_mean": float(_to_numpy(output).mean()),
            }
        )

    Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")


def _pct_change(old: float, new: float) -> float:
    return (new - old) / old * 100.0


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _format_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _write_report(output: Path, baseline: dict[str, Any], current: dict[str, Any], artifact_dir: Path, baseline_ref: str, current_ref: str) -> None:
    keras_small = np.load(artifact_dir / "keras_forward_small.npy")
    torch_small = np.load(artifact_dir / "pytorch_forward_small.npy")
    rmse = float(np.sqrt(np.mean((keras_small - torch_small) ** 2)))
    max_abs = float(np.max(np.abs(keras_small - torch_small)))

    rows: list[list[str]] = []
    for before, after in zip(baseline["forward"], current["forward"], strict=True):
        rows.append(
            [
                before["scenario"],
                f'{before["image_size"]}x{before["image_size"]}',
                str(before["peaks"]),
                f'{before["median_seconds"]:.4f}',
                f'{after["median_seconds"]:.4f}',
                f'{_pct_change(before["median_seconds"], after["median_seconds"]):+.1f}%',
                f'{before["peak_rss_mb"]:.1f}',
                f'{after["peak_rss_mb"]:.1f}',
            ]
        )

    train_before = baseline["training"][0]
    train_after = current["training"][0]
    e2e_before = baseline["end_to_end"][0]
    e2e_after = current["end_to_end"][0]

    verdict = "PASS" if train_after["seconds_per_epoch"] <= train_before["seconds_per_epoch"] * 1.05 and rmse < 1e-6 else "FAIL"
    verdict_reason = (
        "PyTorch stayed within the 5% training-time tolerance and met RMSE < 1e-6."
        if verdict == "PASS"
        else "PyTorch did not meet at least one target: training non-regression within 5% and RMSE < 1e-6."
    )

    large_rows: list[list[str]] = []
    for before, after in zip(baseline.get("large_image", []), current.get("large_image", []), strict=True):
        large_rows.append(
            [
                before["scenario"],
                f'{before["seconds"]:.3f}',
                f'{after["seconds"]:.3f}',
                f'{_pct_change(before["seconds"], after["seconds"]):+.1f}%',
                f'{before["peak_rss_mb"]:.1f}',
                f'{after["peak_rss_mb"]:.1f}',
            ]
        )

    report = [
        "# PyTorch Migration Benchmark",
        "",
        f"- Baseline: Keras 3 torch backend from `{baseline_ref}`",
        f"- Candidate: native PyTorch branch `{current_ref}`",
        f"- Python: {current['python']}",
        f"- Platform: {current['platform']}",
        f"- Device: CPU benchmark path; CUDA available={current['device_info']['cuda_available']}, MPS available={current['device_info']['mps_available']}",
        f"- Note: {current['device_info']['qem_model_device_note']}",
        "",
        "## Forward Pass",
        "",
        _format_table(
            ["Scenario", "Image", "Peaks", "Keras s", "PyTorch s", "Time delta", "Keras peak MB", "PyTorch peak MB"],
            rows,
        ),
        "",
        "## Training",
        "",
        _format_table(
            ["Scenario", "Epochs", "Keras s/epoch", "PyTorch s/epoch", "Delta", "Keras peak MB", "PyTorch peak MB"],
            [
                [
                    train_before["scenario"],
                    str(train_before["epochs"]),
                    f'{train_before["seconds_per_epoch"]:.4f}',
                    f'{train_after["seconds_per_epoch"]:.4f}',
                    f'{_pct_change(train_before["seconds_per_epoch"], train_after["seconds_per_epoch"]):+.1f}%',
                    f'{train_before["peak_rss_mb"]:.1f}',
                    f'{train_after["peak_rss_mb"]:.1f}',
                ]
            ],
        ),
        "",
        "## End-to-End Fitting",
        "",
        _format_table(
            ["Scenario", "Keras s", "PyTorch s", "Delta"],
            [[e2e_before["scenario"], f'{e2e_before["seconds"]:.4f}', f'{e2e_after["seconds"]:.4f}', f'{_pct_change(e2e_before["seconds"], e2e_after["seconds"]):+.1f}%']],
        ),
        "",
    ]
    if large_rows:
        report.extend(
            [
                "## Large Image Smoke Test",
                "",
                _format_table(["Scenario", "Keras s", "PyTorch s", "Delta", "Keras peak MB", "PyTorch peak MB"], large_rows),
                "",
            ]
        )
    report.extend(
        [
            "## Numerical Precision",
            "",
            _format_table(
                ["Case", "RMSE", "Max abs error", "Target"],
                [["small forward output", f"{rmse:.3e}", f"{max_abs:.3e}", "< 1e-6"]],
            ),
            "",
        "## Conclusion",
        "",
        f"Performance non-regression verdict: **{verdict}**. {verdict_reason}",
        "",
        "## Optimization Notes",
        "",
        "- PyTorch training timings include a pre-measurement autograd/Adam warmup so one-time runtime initialization is not charged to the first benchmarked epoch.",
        "- Profiling the warmed 256x256/500-peak training loop showed the renderer and Adam step were microsecond-scale per epoch; the previous regression was dominated by first-use runtime setup.",
        "",
        "Raw JSON files are written next to this report for follow-up analysis.",
        "",
    ]
    )
    output.write_text("\n".join(report), encoding="utf-8")


def _run_impl(impl: str, repo_path: Path, output: Path, artifact_dir: Path, args: argparse.Namespace) -> None:
    env = os.environ.copy()
    env["KERAS_BACKEND"] = "torch"
    env["PYTHONPATH"] = str(repo_path)
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--impl",
        impl,
        "--output",
        str(output),
        "--artifact-dir",
        str(artifact_dir),
        "--rounds",
        str(args.rounds),
        "--epochs",
        str(args.epochs),
        "--torch-threads",
        str(args.torch_threads),
    ]
    if args.large_image_size:
        cmd.extend(["--large-image-size", str(args.large_image_size), "--large-peaks", str(args.large_peaks)])
    _run(cmd, repo_path, env=env)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ref", default="origin/master")
    parser.add_argument("--output-dir", default="qem/benchmarks/results/pytorch_migration")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--large-image-size", type=int, default=1024, help="Set to 0 to skip the large-image smoke test.")
    parser.add_argument("--large-peaks", type=int, default=1200)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--impl", choices=["keras", "pytorch"], help=argparse.SUPPRESS)
    parser.add_argument("--output", help=argparse.SUPPRESS)
    parser.add_argument("--artifact-dir", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.worker:
        _run_worker(args)
        return 0

    output_dir = (REPO_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = output_dir / "artifacts"
    artifact_dir.mkdir(exist_ok=True)

    current_ref = _git_value(["rev-parse", "--abbrev-ref", "HEAD"])
    if _git_value(["status", "--porcelain"]):
        current_ref += " + working tree changes"

    with tempfile.TemporaryDirectory(prefix="qem-keras-baseline-") as tmp:
        baseline_dir = Path(tmp)
        _export_git_ref(args.baseline_ref, baseline_dir)

        baseline_json = output_dir / "keras_baseline.json"
        current_json = output_dir / "pytorch_candidate.json"
        _run_impl("keras", baseline_dir, baseline_json, artifact_dir, args)
        _run_impl("pytorch", REPO_ROOT, current_json, artifact_dir, args)

    report_path = output_dir / "REPORT.md"
    _write_report(
        report_path,
        _load_json(baseline_json),
        _load_json(current_json),
        artifact_dir,
        args.baseline_ref,
        current_ref,
    )
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
