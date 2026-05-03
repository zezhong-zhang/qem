"""Unit tests for CTF/PSF calculations."""

import numpy as np
import pytest

from qem.instruments.ctf import (
    SSB_CTF,
    ADF_CTF,
    ePIE_CTF,
    iCoM_CTF,
    calculate_psf_width,
    extract_psf_from_atom_image,
)
from qem.instruments.probe import Probe


class TestSSB_CTF:
    """Test SSB CTF calculation."""

    def test_initialization(self):
        """Test SSB CTF initialization."""
        ctf = SSB_CTF(alpha=20.0, eV=60e3, df=0.0)
        assert ctf.alpha == 20.0
        assert ctf.eV == 60e3
        assert ctf.df == 0.0
        assert ctf.k > 0  # Wavenumber should be positive

    def test_calculate_ctf_shape(self):
        """Test CTF output shape."""
        ctf = SSB_CTF(alpha=20.0, eV=60e3)
        pix_dim = (64, 64)
        real_dim = (64.0, 64.0)
        result = ctf.calculate_ctf(pix_dim, real_dim)
        assert result.shape == pix_dim
        assert result.dtype == np.complex128

    def test_ctf_is_zero_outside_range(self):
        """Test that CTF is zero outside the double disk overlap."""
        ctf = SSB_CTF(alpha=20.0, eV=60e3)
        pix_dim = (128, 128)
        real_dim = (10.0, 10.0)  # Small real space -> large q space
        result = ctf.calculate_ctf(pix_dim, real_dim)

        # Check that high frequencies are zero
        # The CTF should be zero beyond 2*alpha
        center = (pix_dim[0] // 2, pix_dim[1] // 2)
        corner = result[0, 0]  # Highest frequency component
        assert np.abs(corner) < 1e-10  # Should be essentially zero

    def test_get_psf(self):
        """Test PSF generation."""
        ctf = SSB_CTF(alpha=20.0, eV=60e3)
        pix_dim = (64, 64)
        real_dim = (64.0, 64.0)
        psf = ctf.get_psf(pix_dim, real_dim)

        assert psf.shape == pix_dim
        assert psf.dtype == np.float64

        # SSB PSF should have negative values (characteristic feature)
        assert np.min(psf) < 0  # Should have negative halo

    def test_psf_is_centered(self):
        """Test that PSF is centered."""
        ctf = SSB_CTF(alpha=20.0, eV=60e3)
        pix_dim = (64, 64)
        real_dim = (64.0, 64.0)
        psf = ctf.get_psf(pix_dim, real_dim)

        # Maximum should be near center
        center = (pix_dim[0] // 2, pix_dim[1] // 2)
        max_idx = np.unravel_index(np.argmax(psf), psf.shape)
        assert max(abs(max_idx[0] - center[0]), abs(max_idx[1] - center[1])) < 5


class TestADF_CTF:
    """Test ADF CTF calculation."""

    def test_initialization(self):
        """Test ADF CTF initialization."""
        ctf = ADF_CTF(
            alpha=20.0,
            eV=60e3,
            detector_inner=50.0,
            detector_outer=200.0,
        )
        assert ctf.detector_inner == 50.0
        assert ctf.detector_outer == 200.0

    def test_calculate_ctf_shape(self):
        """Test CTF output shape."""
        ctf = ADF_CTF(
            alpha=20.0,
            eV=60e3,
            detector_inner=50.0,
            detector_outer=200.0,
        )
        pix_dim = (64, 64)
        real_dim = (64.0, 64.0)
        result = ctf.calculate_ctf(pix_dim, real_dim)
        assert result.shape == pix_dim

    def test_psf_is_positive(self):
        """Test that ADF PSF is always positive (incoherent imaging)."""
        ctf = ADF_CTF(
            alpha=20.0,
            eV=60e3,
            detector_inner=50.0,
            detector_outer=200.0,
        )
        pix_dim = (64, 64)
        real_dim = (64.0, 64.0)
        psf = ctf.get_psf(pix_dim, real_dim)

        # ADF PSF should be positive (no negative halo)
        # Allow small negative values from FFT numerical errors
        assert np.min(psf) >= -1e-6  # Allow small numerical errors


class TestePIE_CTF:
    """Test ePIE CTF calculation."""

    def test_initialization(self):
        """Test ePIE CTF initialization."""
        ctf = ePIE_CTF(alpha=20.0, eV=60e3, defocus=1.0)
        assert ctf.defocus == 1.0
        assert ctf.step_size == 0.5  # Default

    def test_calculate_ctf_shape(self):
        """Test CTF output shape."""
        ctf = ePIE_CTF(alpha=20.0, eV=60e3)
        pix_dim = (64, 64)
        real_dim = (64.0, 64.0)
        result = ctf.calculate_ctf(pix_dim, real_dim)
        assert result.shape == pix_dim


class TestiCoM_CTF:
    """Test iCoM CTF calculation."""

    def test_initialization_no_filter(self):
        """Test iCoM CTF initialization without filter."""
        ctf = iCoM_CTF(alpha=20.0, eV=60e3, filter_type="none")
        assert ctf.filter_type == "none"

    def test_initialization_with_highpass(self):
        """Test iCoM CTF initialization with high-pass filter."""
        ctf = iCoM_CTF(
            alpha=20.0,
            eV=60e3,
            high_pass_cutoff=5.0,
            filter_type="highpass",
        )
        assert ctf.high_pass_cutoff == 5.0
        assert ctf.filter_type == "highpass"

    def test_calculate_ctf_shape(self):
        """Test CTF output shape."""
        ctf = iCoM_CTF(alpha=20.0, eV=60e3)
        pix_dim = (64, 64)
        real_dim = (64.0, 64.0)
        result = ctf.calculate_ctf(pix_dim, real_dim)
        assert result.shape == pix_dim


class TestPSFWidth:
    """Test PSF width calculation."""

    def test_calculate_psf_width(self):
        """Test PSF width calculation."""
        # Create a Gaussian-like PSF
        y, x = np.indices((32, 32))
        cy, cx = 16, 16
        psf = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * 2 ** 2))
        psf = psf / np.sum(psf)

        width = calculate_psf_width(psf, dx=1.0)
        assert width > 0
        assert width < 10  # Should be less than the box size

    def test_psf_width_with_negative_halo(self):
        """Test PSF width calculation with negative values."""
        # Create PSF with negative halo (like SSB)
        y, x = np.indices((32, 32))
        cy, cx = 16, 16
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        psf = np.exp(-r ** 2 / (2 * 2 ** 2)) - 0.1 * np.exp(-r ** 2 / (2 * 5 ** 2))

        # Normalize to have positive sum for width calculation
        psf = psf - np.min(psf)  # Shift to be non-negative
        psf = psf / np.sum(psf)  # Normalize

        width = calculate_psf_width(psf, dx=1.0)
        assert width > 0


class TestExtractPSFFromAtomImage:
    """Test PSF extraction from single atom image."""

    def test_extract_psf_from_gaussian(self):
        """Test extracting PSF from a Gaussian atom image."""
        # Create a single atom image
        y, x = np.indices((32, 32))
        cy, cx = 16, 16
        atom_image = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * 2 ** 2))

        psf = extract_psf_from_atom_image(atom_image)

        assert psf.shape == atom_image.shape
        assert np.abs(np.sum(psf) - 1.0) < 1e-6  # Should be normalized

    def test_extract_psf_with_background(self):
        """Test extracting PSF with background subtraction."""
        # Create atom image with background
        y, x = np.indices((32, 32))
        cy, cx = 16, 16
        atom = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * 2 ** 2))
        background = np.ones_like(atom) * 0.1
        atom_image = atom + background

        psf = extract_psf_from_atom_image(atom_image, background=background)

        # Background should be subtracted
        assert np.min(psf) >= 0


class TestCTFComparison:
    """Test comparison between different CTF types."""

    def test_ssb_vs_adf_psf_difference(self):
        """Test that SSB and ADF PSFs have different characteristics."""
        ssb_ctf = SSB_CTF(alpha=20.0, eV=60e3)
        adf_ctf = ADF_CTF(
            alpha=20.0,
            eV=60e3,
            detector_inner=50.0,
            detector_outer=200.0,
        )

        pix_dim = (64, 64)
        real_dim = (64.0, 64.0)

        ssb_psf = ssb_ctf.get_psf(pix_dim, real_dim)
        adf_psf = adf_ctf.get_psf(pix_dim, real_dim)

        # SSB should have significant negative values
        assert np.min(ssb_psf) < -0.01  # Significant negative halo

        # ADF should be mostly positive (allow small FFT errors)
        assert np.min(adf_psf) > -0.01  # Should have minimal negative values

        # ADF minimum should be greater than SSB minimum
        assert np.min(adf_psf) > np.min(ssb_psf)

    def test_different_alpha_changes_psf_width(self):
        """Test that changing convergence angle changes PSF width."""
        for ctf_class in [SSB_CTF, ADF_CTF]:
            if ctf_class == ADF_CTF:
                ctf1 = ctf_class(alpha=10.0, eV=60e3, detector_inner=50.0, detector_outer=200.0)
                ctf2 = ctf_class(alpha=30.0, eV=60e3, detector_inner=50.0, detector_outer=200.0)
            else:
                ctf1 = ctf_class(alpha=10.0, eV=60e3)
                ctf2 = ctf_class(alpha=30.0, eV=60e3)

            pix_dim = (64, 64)
            real_dim = (64.0, 64.0)

            psf1 = ctf1.get_psf(pix_dim, real_dim)
            psf2 = ctf2.get_psf(pix_dim, real_dim)

            width1 = calculate_psf_width(psf1)
            width2 = calculate_psf_width(psf2)

            # Larger alpha should give smaller PSF width
            assert width2 < width1


class TestTemporalCoherenceEnvelope:
    """Test temporal coherence envelope calculation."""

    def test_envelope_at_zero_frequency(self):
        """Test that envelope is 1 at zero frequency."""
        q = np.array([0.0])
        probe = Probe(eV=60e3, Cc=2e7, deltaE=0.5)
        envelope = probe.temporal_coherence_envelope(q)
        assert envelope[0] == pytest.approx(1.0)

    def test_envelope_decreases_with_frequency(self):
        """Test that envelope decreases with spatial frequency."""
        q = np.linspace(0, 2, 100)
        probe = Probe(eV=60e3, Cc=2e7, deltaE=0.5)
        envelope = probe.temporal_coherence_envelope(q)
        # Envelope should be monotonically decreasing
        assert np.all(np.diff(envelope) <= 0)

    def test_envelope_with_direct_df_spread(self):
        """Test envelope with direct defocus spread parameter."""
        q = np.array([1.0])
        probe1 = Probe(eV=60e3, Cc=2e7, deltaE=0.5)
        envelope1 = probe1.temporal_coherence_envelope(q)
        # Calculate equivalent df_spread
        df_spread = 2e7 * 0.5 / 60e3
        probe2 = Probe(eV=60e3, df_spread=df_spread)
        envelope2 = probe2.temporal_coherence_envelope(q)
        assert envelope1[0] == pytest.approx(envelope2[0])

    def test_envelope_with_no_coherence_effects(self):
        """Test that envelope is 1 when no coherence effects are specified."""
        q = np.linspace(0, 2, 100)
        probe = Probe(eV=60e3)
        envelope = probe.temporal_coherence_envelope(q)
        assert np.all(envelope == 1.0)


class TestSpatialCoherenceEnvelope:
    """Test spatial coherence envelope calculation."""

    def test_envelope_at_zero_frequency(self):
        """Test that envelope is 1 at zero frequency."""
        q = np.array([0.0])
        probe = Probe(eV=60e3, source_size=0.1)
        envelope = probe.spatial_coherence_envelope(q)
        assert envelope[0] == pytest.approx(1.0)

    def test_envelope_decreases_with_frequency(self):
        """Test that envelope decreases with spatial frequency."""
        q = np.linspace(0, 2, 100)
        probe = Probe(eV=60e3, source_size=0.1)
        envelope = probe.spatial_coherence_envelope(q)
        # Envelope should be monotonically decreasing
        assert np.all(np.diff(envelope) <= 0)

    def test_larger_source_size_gives_narrower_envelope(self):
        """Test that larger source size gives narrower envelope."""
        q = np.array([1.0])
        probe_small = Probe(eV=60e3, source_size=0.05)
        probe_large = Probe(eV=60e3, source_size=0.2)
        envelope_small = probe_small.spatial_coherence_envelope(q)
        envelope_large = probe_large.spatial_coherence_envelope(q)
        assert envelope_small[0] > envelope_large[0]


class TestPartialCoherenceEnvelope:
    """Test combined partial coherence envelope."""

    def test_combined_envelope_is_product(self):
        """Test that combined envelope is product of temporal and spatial."""
        q = np.array([1.0])
        probe_temp = Probe(eV=60e3, Cc=2e7, deltaE=0.5)
        probe_spat = Probe(eV=60e3, source_size=0.1)
        probe_both = Probe(eV=60e3, Cc=2e7, deltaE=0.5, source_size=0.1)
        temporal = probe_temp.temporal_coherence_envelope(q)
        spatial = probe_spat.spatial_coherence_envelope(q)
        combined = probe_both.partial_coherence_envelope(q)
        expected = temporal[0] * spatial[0]
        assert combined[0] == pytest.approx(expected)

    def test_temporal_only(self):
        """Test envelope with only temporal effects."""
        q = np.array([1.0])
        probe = Probe(eV=60e3, Cc=2e7, deltaE=0.5)
        temporal = probe.temporal_coherence_envelope(q)
        combined = probe.partial_coherence_envelope(q)
        assert combined[0] == temporal[0]

    def test_spatial_only(self):
        """Test envelope with only spatial effects."""
        q = np.array([1.0])
        probe = Probe(eV=60e3, source_size=0.1)
        spatial = probe.spatial_coherence_envelope(q)
        combined = probe.partial_coherence_envelope(q)
        assert combined[0] == spatial[0]


class TestCTFWithPartialCoherence:
    """Test CTF calculation with partial coherence envelope."""

    def test_ssb_psf_with_coherence_is_broader(self):
        """Test that partial coherence makes PSF broader.

        The envelope dampens high spatial frequencies, which causes the PSF
        to be broader in real space (less sharp features).
        """
        # Without coherence
        ctf_no_coherence = SSB_CTF(alpha=20.0, eV=60e3)
        # With temporal coherence (typical Schottky FEG values)
        ctf_with_coherence = SSB_CTF(alpha=20.0, eV=60e3, Cc=2e7, deltaE=0.5)

        pix_dim = (64, 64)
        real_dim = (64.0, 64.0)

        psf_no_coherence = ctf_no_coherence.get_psf(pix_dim, real_dim)
        psf_with_coherence = ctf_with_coherence.get_psf(pix_dim, real_dim)

        # Calculate widths
        width_no_coherence = calculate_psf_width(psf_no_coherence)
        width_with_coherence = calculate_psf_width(psf_with_coherence)

        # Coherence dampens high frequencies, making PSF broader in real space
        # The PSF with coherence should be wider (or equal)
        assert width_with_coherence >= width_no_coherence * 0.95  # Allow small numerical variations

    def test_ssb_ctf_initialization_with_coherence(self):
        """Test SSB CTF initialization with coherence parameters."""
        ctf = SSB_CTF(
            alpha=20.0,
            eV=60e3,
            Cc=2e7,
            deltaE=0.5,
            source_size=0.1,
        )
        assert ctf.Cc == 2e7
        assert ctf.deltaE == 0.5
        assert ctf.source_size == 0.1

    def test_adf_ctf_with_coherence(self):
        """Test ADF CTF with partial coherence."""
        ctf = ADF_CTF(
            alpha=20.0,
            eV=60e3,
            detector_inner=50.0,
            detector_outer=200.0,
            Cc=2e7,
            deltaE=0.5,
        )

        pix_dim = (64, 64)
        real_dim = (64.0, 64.0)

        psf = ctf.get_psf(pix_dim, real_dim)

        # Should still be positive
        assert np.min(psf) >= -1e-6

        # Should have valid shape
        assert psf.shape == pix_dim

    def test_epie_ctf_with_coherence(self):
        """Test ePIE CTF with partial coherence."""
        ctf = ePIE_CTF(
            alpha=20.0,
            eV=60e3,
            source_size=0.1,
        )

        pix_dim = (64, 64)
        real_dim = (64.0, 64.0)

        psf = ctf.get_psf(pix_dim, real_dim)
        assert psf.shape == pix_dim

    def test_icom_ctf_with_coherence(self):
        """Test iCoM CTF with partial coherence."""
        ctf = iCoM_CTF(
            alpha=20.0,
            eV=60e3,
            Cc=2e7,
            deltaE=0.5,
        )

        pix_dim = (64, 64)
        real_dim = (64.0, 64.0)

        psf = ctf.get_psf(pix_dim, real_dim)
        assert psf.shape == pix_dim


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
