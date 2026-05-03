"""
Visualization of how aberrations affect the CTF and PSF.

This script demonstrates the impact of various aberrations on:
- The Contrast Transfer Function (CTF) in reciprocal space
- The Point Spread Function (PSF) in real space
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from qem.instruments.ctf import (
    SSB_CTF,
    create_aberration_list,
)
from qem.instruments.probe import Aberration


def plot_aberration_effects_on_ctf():
    """Plot how different aberrations affect the SSB CTF."""
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(4, 4, figure=fig, hspace=0.3, wspace=0.3)

    # Base parameters
    alpha = 20.0  # mrad
    eV = 60e3    # 60 kV
    pix_dim = (128, 128)
    real_dim = (32.0, 32.0)

    # Define aberration configurations
    aberration_configs = [
        {
            "name": "No Aberrations",
            "aberrations": [],
            "df": 0,
        },
        {
            "name": "Defocus: 50 Å",
            "aberrations": create_aberration_list(defocus=50),
            "df": 0,  # Using explicit C10 instead
        },
        {
            "name": "2-fold Astigmatism\n10 Å at 45°",
            "aberrations": create_aberration_list(
                two_fold_astigmatism=10,
                two_fold_angle=np.pi/4,
            ),
            "df": 0,
        },
        {
            "name": "3-fold Astigmatism\n15 Å at 30°",
            "aberrations": create_aberration_list(
                three_fold_astigmatism=15,
                three_fold_angle=np.pi/6,
            ),
            "df": 0,
        },
        {
            "name": "Coma\n500 Å at 60°",
            "aberrations": create_aberration_list(
                coma=500,
                coma_angle=np.pi/3,
            ),
            "df": 0,
        },
        {
            "name": "Cs: 0.5 mm\n(3rd order spherical)",
            "aberrations": create_aberration_list(
                spherical_aberration=0.5e7,  # 0.5 mm
            ),
            "df": 0,
        },
        {
            "name": "Defocus + Cs",
            "aberrations": create_aberration_list(
                defocus=50,
                spherical_aberration=0.5e7,
            ),
            "df": 0,
        },
        {
            "name": "Defocus + 2-fold Astig.",
            "aberrations": create_aberration_list(
                defocus=50,
                two_fold_astigmatism=15,
                two_fold_angle=np.pi/4,
            ),
            "df": 0,
        },
        {
            "name": "Defocus + Coma",
            "aberrations": create_aberration_list(
                defocus=50,
                coma=800,
                coma_angle=np.pi/3,
            ),
            "df": 0,
        },
        {
            "name": "2-fold + 3-fold\nAstigmatism",
            "aberrations": create_aberration_list(
                two_fold_astigmatism=10,
                two_fold_angle=np.pi/4,
                three_fold_astigmatism=15,
                three_fold_angle=np.pi/6,
            ),
            "df": 0,
        },
        {
            "name": "Cs + Coma",
            "aberrations": create_aberration_list(
                spherical_aberration=0.5e7,
                coma=500,
                coma_angle=np.pi/3,
            ),
            "df": 0,
        },
        {
            "name": "Severe: Cs=1mm + Defocus",
            "aberrations": create_aberration_list(
                defocus=80,
                spherical_aberration=1.0e7,  # 1 mm
            ),
            "df": 0,
        },
    ]

    # Plot CTF magnitude for each configuration
    for i, config in enumerate(aberration_configs[:12]):
        ax = fig.add_subplot(gs[i // 4, i % 4])

        # Create CTF with aberrations
        ctf = SSB_CTF(alpha=alpha, eV=eV, aberrations=config["aberrations"])
        ctf_array = ctf.calculate_ctf(pix_dim, real_dim)

        # Plot magnitude (for visualization, we can also look at phase)
        magnitude = np.abs(ctf_array)

        # Display
        im = ax.imshow(magnitude, cmap='viridis', origin='lower')
        ax.set_title(config["name"], fontsize=9)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f'Effect of Aberrations on SSB CTF Magnitude\n(60 kV, {alpha} mrad)',
                 fontsize=14, fontweight='bold')

    return fig


def plot_aberration_effects_on_psf():
    """Plot how different aberrations affect the SSB PSF."""
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)

    # Base parameters
    alpha = 20.0
    eV = 60e3
    pix_dim = (128, 128)
    real_dim = (64.0, 64.0)

    aberration_configs = [
        {
            "name": "No Aberrations",
            "aberrations": [],
        },
        {
            "name": "Defocus: 50 Å",
            "aberrations": create_aberration_list(defocus=50),
        },
        {
            "name": "2-fold Astig.\n10 Å at 45°",
            "aberrations": create_aberration_list(
                two_fold_astigmatism=10,
                two_fold_angle=np.pi/4,
            ),
        },
        {
            "name": "3-fold Astig.\n15 Å at 30°",
            "aberrations": create_aberration_list(
                three_fold_astigmatism=15,
                three_fold_angle=np.pi/6,
            ),
        },
        {
            "name": "Coma\n500 Å at 60°",
            "aberrations": create_aberration_list(
                coma=500,
                coma_angle=np.pi/3,
            ),
        },
        {
            "name": "Cs: 0.5 mm",
            "aberrations": create_aberration_list(
                spherical_aberration=0.5e7,
            ),
        },
        {
            "name": "Defocus + Cs",
            "aberrations": create_aberration_list(
                defocus=50,
                spherical_aberration=0.5e7,
            ),
        },
        {
            "name": "Defocus + 2-fold",
            "aberrations": create_aberration_list(
                defocus=50,
                two_fold_astigmatism=15,
                two_fold_angle=np.pi/4,
            ),
        },
        {
            "name": "Defocus + Coma",
            "aberrations": create_aberration_list(
                defocus=50,
                coma=800,
                coma_angle=np.pi/3,
            ),
        },
        {
            "name": "Cs + 2-fold Astig.",
            "aberrations": create_aberration_list(
                spherical_aberration=0.5e7,
                two_fold_astigmatism=15,
                two_fold_angle=np.pi/4,
            ),
        },
        {
            "name": "Severe: Cs=1mm\nDefocus=80Å",
            "aberrations": create_aberration_list(
                defocus=80,
                spherical_aberration=1.0e7,
            ),
        },
        {
            "name": "Full Set\n(Starter Pack)",
            "aberrations": None,  # Use aberration_starter_pack
        },
    ]

    # Plot PSF for each configuration
    for i, config in enumerate(aberration_configs):
        ax = fig.add_subplot(gs[i // 4, i % 4])

        # Handle full aberration set
        if config["aberrations"] is None:
            from qem.instruments.probe import aberration_starter_pack
            ab_list = aberration_starter_pack()
            # Set some non-zero values for demonstration
            for ab in ab_list:
                if ab.Krivanek == "C10":
                    ab.amplitude = 50  # Defocus
                elif ab.Krivanek == "C30":
                    ab.amplitude = 0.5e7  # 0.5 mm Cs
                elif ab.Krivanek == "C12":
                    ab.amplitude = 15
                    ab.angle = np.pi/4
            aberrations = ab_list
        else:
            aberrations = config["aberrations"]

        # Create CTF with aberrations
        ctf = SSB_CTF(alpha=alpha, eV=eV, aberrations=aberrations)
        psf = ctf.get_psf(pix_dim, real_dim)

        # For PSF, use RdBu colormap to show positive/negative values
        vmax = np.max(np.abs(psf))
        im = ax.imshow(psf, cmap='RdBu', vmin=-vmax, vmax=vmax, origin='lower')
        ax.set_title(config["name"], fontsize=9)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f'Effect of Aberrations on SSB PSF\n(60 kV, {alpha} mrad)',
                 fontsize=14, fontweight='bold')

    return fig


def plot_ctf_radial_profiles():
    """Plot radial CTF profiles comparing different aberrations."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    alpha = 20.0
    eV = 60e3
    pix_dim = (256, 256)
    real_dim = (32.0, 32.0)

    # Get q-space for radial plotting
    from qem.processing import q_space_array
    q = q_space_array(pix_dim, real_dim)
    q_mag = np.sqrt(q[0]**2 + q[1]**2)

    # Define aberration configurations to compare
    configs = [
        ("No aberrations", []),
        ("Defocus: 50 Å", create_aberration_list(defocus=50)),
        ("2-fold astig: 10 Å", create_aberration_list(two_fold_astigmatism=10, two_fold_angle=np.pi/4)),
        ("3-fold astig: 15 Å", create_aberration_list(three_fold_astigmatism=15, three_fold_angle=np.pi/6)),
        ("Cs: 0.5 mm", create_aberration_list(spherical_aberration=0.5e7)),
        ("Coma: 500 Å", create_aberration_list(coma=500, coma_angle=np.pi/3)),
        ("Defocus + Cs", create_aberration_list(defocus=50, spherical_aberration=0.5e7)),
    ]

    # Get center and radius
    center = (pix_dim[0] // 2, pix_dim[1] // 2)
    y, x = np.indices(pix_dim)
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2)

    # Plot 1: CTF magnitude vs radius
    ax = axes[0, 0]
    for name, aberrations in configs:
        ctf = SSB_CTF(alpha=alpha, eV=eV, aberrations=aberrations)
        ctf_array = ctf.calculate_ctf(pix_dim, real_dim)
        magnitude = np.abs(ctf_array)

        # Radial average
        r_flat = r.ravel()
        mag_flat = magnitude.ravel()
        max_r = int(np.max(r))
        r_bins = np.arange(0, max_r, 2)
        mag_profile = []
        r_values = []

        for j in range(len(r_bins) - 1):
            mask = (r_flat >= r_bins[j]) & (r_flat < r_bins[j+1])
            if np.any(mask):
                r_values.append((r_bins[j] + r_bins[j+1]) / 2)
                mag_profile.append(np.mean(mag_flat[mask]))

        ax.plot(r_values, mag_profile, label=name, linewidth=2)

    ax.set_xlabel("Radius (pixels)")
    ax.set_ylabel("CTF Magnitude")
    ax.set_title("CTF Magnitude vs Spatial Frequency")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 60])

    # Plot 2: CTF phase vs radius
    ax = axes[0, 1]
    for name, aberrations in configs[:5]:  # Fewer for clarity
        ctf = SSB_CTF(alpha=alpha, eV=eV, aberrations=aberrations)
        ctf_array = ctf.calculate_ctf(pix_dim, real_dim)
        phase = np.angle(ctf_array)

        # Radial average
        r_flat = r.ravel()
        phase_flat = phase.ravel()
        max_r = int(np.max(r))
        r_bins = np.arange(0, max_r, 2)
        phase_profile = []
        r_values = []

        for j in range(len(r_bins) - 1):
            mask = (r_flat >= r_bins[j]) & (r_flat < r_bins[j+1])
            if np.any(mask):
                r_values.append((r_bins[j] + r_bins[j+1]) / 2)
                # Use circular mean for phase
                complex_vals = np.exp(1j * phase_flat[mask])
                phase_profile.append(np.angle(np.mean(complex_vals)))

        ax.plot(r_values, phase_profile, label=name, linewidth=2)

    ax.set_xlabel("Radius (pixels)")
    ax.set_ylabel("CTF Phase (radians)")
    ax.set_title("CTF Phase vs Spatial Frequency")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 60])

    # Plot 3: PSF cross-sections
    ax = axes[1, 0]
    center_x = pix_dim[1] // 2
    x = np.arange(pix_dim[1]) - center_x

    for name, aberrations in configs[:6]:
        ctf = SSB_CTF(alpha=alpha, eV=eV, aberrations=aberrations)
        psf = ctf.get_psf(pix_dim, real_dim)

        # Cross-section through center
        cross_section = psf[pix_dim[0] // 2, :]
        ax.plot(x, cross_section, label=name, linewidth=2)

    ax.set_xlabel("Position (pixels)")
    ax.set_ylabel("PSF Intensity")
    ax.set_title("PSF Cross-section Comparison")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-60, 60])

    # Plot 4: Negative halo magnitude comparison
    ax = axes[1, 1]
    names = []
    neg_halo = []

    for name, aberrations in configs:
        ctf = SSB_CTF(alpha=alpha, eV=eV, aberrations=aberrations)
        psf = ctf.get_psf(pix_dim, real_dim)

        # Measure negative halo (most negative value / peak positive value)
        peak_pos = np.max(psf)
        peak_neg = np.min(psf)
        halo_ratio = abs(peak_neg) / peak_pos if peak_pos > 0 else 0

        names.append(name.replace("\n", " "))
        neg_halo.append(halo_ratio)

    bars = ax.barh(names, neg_halo, color='steelblue')
    ax.set_xlabel("Negative/Positive Ratio")
    ax.set_title("Negative Halo Strength Comparison")
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    return fig


def plot_aberration_phase_maps():
    """Plot the phase of the aberration function chi(q, phi)."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    alpha = 20.0
    eV = 60e3
    pix_dim = (128, 128)
    real_dim = (32.0, 32.0)

    from qem.processing import q_space_array
    from qem.instruments.probe import Probe
    q = q_space_array(pix_dim, real_dim)
    q_mag = np.sqrt(q[0]**2 + q[1]**2)
    qphi = np.arctan2(q[0], q[1])

    aberration_configs = [
        ("No aberrations", []),
        ("Defocus: 50 Å", [Aberration("C10", "C1", "Defocus", 50, 0.0, 1, 0)]),
        ("2-fold astig.", [Aberration("C12", "A1", "2-Fold", 10, np.pi/4, 1, 2)]),
        ("3-fold astig.", [Aberration("C23", "A2", "3-Fold", 15, np.pi/6, 2, 3)]),
        ("Cs: 0.5mm", [Aberration("C30", "C3", "Spherical", 0.5e7, 0.0, 3, 0)]),
        ("Coma", [Aberration("C21", "B2", "Coma", 500, np.pi/3, 2, 1)]),
    ]

    for i, (name, aberrations) in enumerate(aberration_configs):
        ax = axes[i // 3, i % 3]

        # Calculate chi (aberration phase)
        probe = Probe(eV=eV, aberrations=aberrations)
        chi_phase = probe.chi(q_mag, qphi)

        # Plot wrapped phase (-pi to pi)
        im = ax.imshow(chi_phase, cmap='twilight', origin='lower')
        ax.set_title(name)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='χ (rad)')

    fig.suptitle('Aberration Phase Function χ(q, φ)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    print("Generating aberration effect visualizations...")

    print("1. CTF magnitude with different aberrations...")
    fig1 = plot_aberration_effects_on_ctf()
    fig1.savefig("/Users/zhangzz/code/qem/examples/ctf_aberration_effects.png", dpi=150)
    print("   Saved to ctf_aberration_effects.png")

    print("2. PSF with different aberrations...")
    fig2 = plot_aberration_effects_on_psf()
    fig2.savefig("/Users/zhangzz/code/qem/examples/psf_aberration_effects.png", dpi=150)
    print("   Saved to psf_aberration_effects.png")

    print("3. Radial profiles comparison...")
    fig3 = plot_ctf_radial_profiles()
    fig3.savefig("/Users/zhangzz/code/qem/examples/ctf_aberration_profiles.png", dpi=150)
    print("   Saved to ctf_aberration_profiles.png")

    print("4. Aberration phase maps...")
    fig4 = plot_aberration_phase_maps()
    fig4.savefig("/Users/zhangzz/code/qem/examples/aberration_phase_maps.png", dpi=150)
    print("   Saved to aberration_phase_maps.png")

    print("\nAll figures generated successfully!")
    print("\nKey observations:")
    print("- Defocus adds a quadratic phase shift (radial symmetry)")
    print("- 2-fold astigmatism breaks circular symmetry (2-fold pattern)")
    print("- 3-fold astigmatism creates 3-fold symmetric pattern")
    print("- Coma introduces asymmetric phase shift")
    print("- Spherical aberration (Cs) creates higher-order phase distortion")
    print("- The PSF shows how aberrations affect the real-space probe shape")
    print("- Negative halo in SSB is modified by aberrations")
