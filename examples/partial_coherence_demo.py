"""
Demonstration of partial coherence envelope effects on CTF and PSF.

This script shows how temporal and spatial coherence affect the
contrast transfer function and point spread function.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from qem.instruments.ctf import SSB_CTF, ADF_CTF, create_aberration_list
from qem.instruments.probe import Probe


def plot_envelope_comparison():
    """Plot different coherence envelopes for comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    q = np.linspace(0, 0.5, 200)  # Spatial frequency in 1/A

    # Temporal coherence envelope
    ax = axes[0]
    for Cc, deltaE in [(None, None), (1e7, 0.3), (2e7, 0.5), (3e7, 1.0)]:
        if Cc is None:
            label = "No temporal damping"
        else:
            label = f"Cc={Cc/1e7:.1f}mm, ΔE={deltaE}eV"
        probe = Probe(eV=60e3, Cc=Cc, deltaE=deltaE)
        E_temp = probe.temporal_coherence_envelope(q)
        ax.plot(q, E_temp, label=label, linewidth=2)

    ax.set_xlabel("Spatial frequency q (1/Å)")
    ax.set_ylabel("Envelope E(q)")
    ax.set_title("Temporal Coherence Envelope\n(Energy spread + Chromatic aberration)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])

    # Spatial coherence envelope.  The quasi-coherent formula
    # exp(-(σ_α/2)² · |∇χ|²) is identically 1 for an unaberrated probe
    # because ∇χ = 0; aberrations are required to see source-size damping.
    # Use a representative uncorrected STEM: defocus = 50 Å, Cs = 1 mm.
    ax = axes[1]
    df_demo = 50.0  # Å
    cs_demo = 1e7   # Å (= 1 mm, typical uncorrected STEM)
    aberrations_demo = create_aberration_list(
        defocus=df_demo,
        spherical_aberration=cs_demo,
    )
    for source_size in [None, 0.05, 0.1, 0.2]:
        if source_size is None:
            label = "No spatial damping"
        else:
            label = f"Source size = {source_size} mrad"
        probe = Probe(eV=60e3, aberrations=aberrations_demo, source_size=source_size)
        E_spatial = probe.spatial_coherence_envelope(q)
        ax.plot(q, E_spatial, label=label, linewidth=2)

    ax.set_xlabel("Spatial frequency q (1/Å)")
    ax.set_ylabel("Envelope E(q)")
    ax.set_title(
        "Spatial Coherence Envelope\n"
        f"(Source size; defocus = {df_demo:g} Å, Cs = {cs_demo / 1e7:g} mm)"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])

    plt.tight_layout()
    return fig


def plot_combined_envelope():
    """Plot combined temporal and spatial coherence envelope."""
    fig, ax = plt.subplots(figsize=(10, 6))

    q = np.linspace(0, 0.5, 200)

    # All probes share aberrations so the (aberration-dependent) spatial
    # envelope is non-trivial.  ∇χ = 0 ⇒ spatial envelope ≡ 1.
    df_demo = 50.0
    cs_demo = 1e7
    src = 0.1  # mrad
    aberrations_demo = create_aberration_list(
        defocus=df_demo, spherical_aberration=cs_demo
    )

    # No coherence
    probe_no = Probe(eV=60e3, aberrations=aberrations_demo)
    E_no = probe_no.partial_coherence_envelope(q)
    ax.plot(q, E_no, 'k--', label="No damping", linewidth=2)

    # Temporal only
    probe_temp = Probe(
        eV=60e3, aberrations=aberrations_demo, Cc=2e7, deltaE=0.5
    )
    E_temp = probe_temp.partial_coherence_envelope(q)
    ax.plot(q, E_temp, 'b-', label="Temporal only (Cc=2mm, ΔE=0.5eV)", linewidth=2)

    # Spatial only
    probe_spat = Probe(
        eV=60e3, aberrations=aberrations_demo, source_size=src
    )
    E_spatial = probe_spat.partial_coherence_envelope(q)
    ax.plot(q, E_spatial, 'r-', label=f"Spatial only (src={src} mrad)", linewidth=2)

    # Combined
    probe_both = Probe(
        eV=60e3, aberrations=aberrations_demo, Cc=2e7, deltaE=0.5, source_size=src
    )
    E_both = probe_both.partial_coherence_envelope(q)
    ax.plot(q, E_both, 'g-', label="Combined", linewidth=2.5)

    ax.set_xlabel("Spatial frequency q (1/Å)", fontsize=12)
    ax.set_ylabel("Envelope E(q)", fontsize=12)
    ax.set_title(
        f"Combined Partial Coherence Envelope\n"
        f"(defocus = {df_demo:g} Å, Cs = {cs_demo / 1e7:g} mm)",
        fontsize=14,
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])

    plt.tight_layout()
    return fig


def plot_psf_comparison():
    """Compare PSF with and without partial coherence."""
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

    # Parameters
    pix_dim = (128, 128)
    real_dim = (64.0, 64.0)

    # SSB PSF comparison
    for i, (with_coherence, title) in enumerate([
        (False, "No Coherence"),
        (True, "With Coherence\n(Cc=2mm, ΔE=0.5eV, src=0.1mrad)")
    ]):
        if with_coherence:
            ctf = SSB_CTF(alpha=20.0, eV=60e3, Cc=2e7, deltaE=0.5, source_size=0.1)
        else:
            ctf = SSB_CTF(alpha=20.0, eV=60e3)

        psf = ctf.get_psf(pix_dim, real_dim)

        # Image
        ax = fig.add_subplot(gs[0, i])
        im = ax.imshow(psf, cmap='RdBu', vmin=-np.max(np.abs(psf)), vmax=np.max(np.abs(psf)))
        ax.set_title(f"SSB PSF: {title}")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.axis('off')

    # ADF PSF comparison
    for i, (with_coherence, title) in enumerate([
        (False, "No Coherence"),
        (True, "With Coherence\n(Cc=2mm, ΔE=0.5eV)")
    ]):
        if with_coherence:
            ctf = ADF_CTF(alpha=20.0, eV=60e3, detector_inner=50, detector_outer=200,
                         Cc=2e7, deltaE=0.5)
        else:
            ctf = ADF_CTF(alpha=20.0, eV=60e3, detector_inner=50, detector_outer=200)

        psf = ctf.get_psf(pix_dim, real_dim)

        # Image
        ax = fig.add_subplot(gs[1, i])
        im = ax.imshow(psf, cmap='viridis')
        ax.set_title(f"ADF PSF: {title}")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.axis('off')

    # Cross-section comparison
    ax = fig.add_subplot(gs[2, :])

    center = pix_dim[0] // 2
    x = np.arange(pix_dim[1]) - center

    # SSB cross-sections
    ctf_ssb_no = SSB_CTF(alpha=20.0, eV=60e3)
    psf_ssb_no = ctf_ssb_no.get_psf(pix_dim, real_dim)

    ctf_ssb_yes = SSB_CTF(alpha=20.0, eV=60e3, Cc=2e7, deltaE=0.5, source_size=0.1)
    psf_ssb_yes = ctf_ssb_yes.get_psf(pix_dim, real_dim)

    ax.plot(x, psf_ssb_no[center, :], 'b-', label="SSB, no coherence", linewidth=2)
    ax.plot(x, psf_ssb_yes[center, :], 'b--', label="SSB, with coherence", linewidth=2)

    # ADF cross-sections (normalized)
    ctf_adf_no = ADF_CTF(alpha=20.0, eV=60e3, detector_inner=50, detector_outer=200)
    psf_adf_no = ctf_adf_no.get_psf(pix_dim, real_dim)

    ctf_adf_yes = ADF_CTF(alpha=20.0, eV=60e3, detector_inner=50, detector_outer=200,
                          Cc=2e7, deltaE=0.5)
    psf_adf_yes = ctf_adf_yes.get_psf(pix_dim, real_dim)

    # Normalize ADF for comparison (guard against all-zero PSFs)
    def _safe_normalize(psf, reference):
        peak = np.max(psf)
        if not np.isfinite(peak) or peak == 0:
            return psf
        return psf / peak * np.max(np.abs(reference))

    psf_adf_no = _safe_normalize(psf_adf_no, psf_ssb_no)
    psf_adf_yes = _safe_normalize(psf_adf_yes, psf_ssb_yes)

    ax.plot(x, psf_adf_no[center, :], 'r-', label="ADF, no coherence", linewidth=2)
    ax.plot(x, psf_adf_yes[center, :], 'r--', label="ADF, with coherence", linewidth=2)

    ax.set_xlabel("Position (pixels)")
    ax.set_ylabel("PSF intensity")
    ax.set_title("PSF Cross-section Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-60, 60])

    return fig


def plot_ctf_with_envelope():
    """Show how envelope modifies the CTF."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    pix_dim = (256, 256)
    real_dim = (20.0, 20.0)

    # Get q-space (with azimuth so the spatial-coherence envelope can use ∇χ).
    from qem.processing import q_space_array
    q = q_space_array(pix_dim, real_dim)
    q_mag = np.sqrt(q[0]**2 + q[1]**2)
    qphi = np.arctan2(q[0], q[1])

    df_demo = 50.0
    cs_demo = 1e7
    src = 0.1
    aberrations_demo = create_aberration_list(
        defocus=df_demo, spherical_aberration=cs_demo
    )

    # Get radial profile of CTF
    ctf_no = SSB_CTF(alpha=20.0, eV=60e3, aberrations=aberrations_demo)
    ctf_array_no = ctf_no.calculate_ctf(pix_dim, real_dim)

    ctf_yes = SSB_CTF(
        alpha=20.0, eV=60e3, aberrations=aberrations_demo,
        Cc=2e7, deltaE=0.5, source_size=src,
    )
    ctf_array_yes = ctf_yes.calculate_ctf(pix_dim, real_dim)

    # Apply envelope for comparison
    probe_env = Probe(
        eV=60e3, aberrations=aberrations_demo,
        Cc=2e7, deltaE=0.5, source_size=src,
    )
    envelope = probe_env.partial_coherence_envelope(q_mag, qphi=qphi)
    ctf_with_env = ctf_array_yes * envelope

    # Get radial profiles (average over angles)
    center = (pix_dim[0] // 2, pix_dim[1] // 2)
    y, x = np.indices(pix_dim)
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    r_flat = r.ravel()
    ctf_flat = np.abs(ctf_array_no).ravel()
    ctf_env_flat = np.abs(ctf_with_env).ravel()

    # Bin by radius
    max_r = int(np.max(r))
    r_bins = np.arange(0, max_r, 2)
    ctf_profile = []
    ctf_env_profile = []
    r_values = []

    for i in range(len(r_bins) - 1):
        mask = (r_flat >= r_bins[i]) & (r_flat < r_bins[i+1])
        if np.any(mask):
            r_values.append((r_bins[i] + r_bins[i+1]) / 2)
            ctf_profile.append(np.mean(ctf_flat[mask]))
            ctf_env_profile.append(np.mean(ctf_env_flat[mask]))

    # Plot
    ax = axes[0]
    ax.plot(r_values, ctf_profile, 'b-', label="CTF without envelope", linewidth=2)
    ax.plot(r_values, ctf_env_profile, 'r-', label="CTF with envelope", linewidth=2)
    ax.set_xlabel("Radius (pixels)")
    ax.set_ylabel("CTF magnitude")
    ax.set_title("Radial CTF Profile")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot envelope
    ax = axes[1]
    q_1d = np.linspace(0, 0.5, 100)
    probe_all = Probe(
        eV=60e3, aberrations=aberrations_demo,
        Cc=2e7, deltaE=0.5, source_size=src,
    )
    env_1d = probe_all.partial_coherence_envelope(q_1d)

    # Temporal component
    probe_t = Probe(eV=60e3, aberrations=aberrations_demo, Cc=2e7, deltaE=0.5)
    env_temp = probe_t.temporal_coherence_envelope(q_1d)

    # Spatial component (aberrations required, otherwise ≡ 1)
    probe_s = Probe(eV=60e3, aberrations=aberrations_demo, source_size=src)
    env_spatial = probe_s.spatial_coherence_envelope(q_1d)

    ax.plot(q_1d, env_temp, 'b--', label="Temporal envelope", linewidth=2)
    ax.plot(q_1d, env_spatial, 'r--', label="Spatial envelope", linewidth=2)
    ax.plot(q_1d, env_1d, 'k-', label="Combined envelope", linewidth=2.5)
    ax.set_xlabel("Spatial frequency q (1/Å)")
    ax.set_ylabel("Envelope E(q)")
    ax.set_title("Coherence Envelope Components")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    print("Generating partial coherence envelope demonstrations...")

    print("1. Envelope comparison...")
    fig1 = plot_envelope_comparison()
    fig1.savefig("/Users/zhangzz/code/qem/examples/envelope_comparison.png", dpi=150)
    print("   Saved to envelope_comparison.png")

    print("2. Combined envelope...")
    fig2 = plot_combined_envelope()
    fig2.savefig("/Users/zhangzz/code/qem/examples/combined_envelope.png", dpi=150)
    print("   Saved to combined_envelope.png")

    print("3. PSF comparison...")
    fig3 = plot_psf_comparison()
    fig3.savefig("/Users/zhangzz/code/qem/examples/psf_coherence_comparison.png", dpi=150)
    print("   Saved to psf_coherence_comparison.png")

    print("4. CTF with envelope...")
    fig4 = plot_ctf_with_envelope()
    fig4.savefig("/Users/zhangzz/code/qem/examples/ctf_with_envelope.png", dpi=150)
    print("   Saved to ctf_with_envelope.png")

    print("\nAll figures generated successfully!")
    print("\nKey observations:")
    print("- Temporal envelope (from Cc and ΔE) dampens high spatial frequencies")
    print("- Spatial envelope (from source size) also dampens high frequencies")
    print("- The combined envelope is the product of temporal and spatial envelopes")
    print("- With damping, the PSF becomes broader in real space (less sharp)")
    print("- The negative halo in SSB is also affected by coherence")
