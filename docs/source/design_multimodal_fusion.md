# QEM Multi-Modal Fusion Design Document

## Overview

This document specifies the architecture and implementation plan for **ADF-EDX-EELS Joint Quantitative Analysis** within the QEM (Quantitative Electron Microscopy) framework. The system integrates three simultaneously-acquired signals—Annular Dark-Field (ADF), Energy-Dispersive X-ray (EDX), and Electron Energy Loss Spectroscopy (EELS)—into a unified quantitative reconstruction pipeline.

**Core Insight**: When ADF, EDX, and EELS are acquired simultaneously, cross-modal correlation itself becomes a weapon against noise and peak overlap. Noise is random and uncorrelated; signal is correlated across modalities.

**Reference Theory**: [ADF-EDX-EELS SI Joint Quantitative Analysis Framework](https://my.feishu.cn/wiki/JWCRwI3HXi4Ktzki3odcMaJKn2b)

---

## Background & Motivation

### Current State

| Component | Location | Capability | Limitation |
|-----------|----------|------------|------------|
| `qem.fit.image_fitting` | `qem/fit/image_fitting.py` | Atomic column localization, Gaussian fitting, Voronoi integration | Only ADF/HAADF; no spectroscopy |
| `vendors/multi_modal` | `vendors/multi_modal/mapfusion/` | ADF-EDX fusion via gamma-divergence + TV regularization | No EELS; no cross-modal correlation exploitation |
| Analysis scripts | `~/work/data/High_entropy/script/` | EDX.py, EELS_quantification.py, adf_quant.py, alignment.py | Jupyter-based, hardcoded paths, not modular |
| `superalign` | `~/code/superalign` | Rigid + non-rigid registration for drift correction | Not integrated into qem pipeline |
| `pyEELSMODEL` | `~/code/pyEELSMODEL` | EELS background removal, cross-section quantification | Standalone; no joint optimization with EDX/ADF |

### Pain Points

1. **Fragmented workflow**: ADF analysis (`ImageFitting`), EDX mapping (`EDXStack`), and EELS quantification (`pyEELSMODEL`) run in separate scripts with manual data handoff.
2. **No cross-modal constraints**: Each modality is analyzed independently; correlations between ADF atomic columns, EDX elemental maps, and EELS ionization edges are ignored.
3. **Scaling issues**: The `multi_modal` vendor package uses C++ extensions (`ctvlib`) that are hard to build and maintain.
4. **No unified API**: Users must orchestrate `hyperspy`, `superalign` (`~/code/superalign`), `pyEELSMODEL` (`~/code/pyEELSMODEL`), and custom scripts manually.

### Target Users

- Materials scientists analyzing complex multi-component systems (e.g., high-entropy alloys, semiconductor stacks, catalytic nanoparticles)
- Researchers who need **quantitative** (not just qualitative) elemental composition at atomic resolution
- Users who want to exploit **all three modalities** simultaneously rather than treating them as independent measurements

---

## Architecture

### Simplified Scope (Single-Frame, Pre-Aligned Data)

**Assumption**: Data has already been aligned by superalign (~/code/superalign). We work with:
- Single-frame ADF image (already aligned and averaged)
- Single-frame EDX spectrum image (already aligned)
- Single-frame EELS HL spectrum image (already aligned)
- Optional: EELS LL for thickness normalization

**Data location**: ~/work/data/High_entropy/script/ contains:
- adf_aligned.hspy (ADF after rigid + non-rigid alignment)
- edx_aligned.hspy (EDX after alignment)
- eels_hl_aligned.hspy / eels_ll_aligned.hspy (EELS after alignment)
- eels_hl_aligned_bin.hspy / eels_ll_aligned_bin.hspy (binned for faster processing)

### System Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    MultiModalDataset (Data Container)                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐ │
│  │   ADF/      │  │   EDX SI    │  │  EELS HL    │  │    EELS LL      │ │
│  │  HAADF      │  │  (elemental │  │  (core loss)│  │  (low loss/ZLP) │ │
│  │  (2D image) │  │   maps)     │  │   (3D SI)   │  │    (3D SI)      │ │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └────────┬────────┘ │
│         └────────────────┴────────────────┴──────────────────┘          │
│                                    │                                    │
│                         Preprocessing (Minimal)                         │
│                         (background removal only)                       │
│                                    │                                    │
└────────────────────────────────────┼────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│              MultiModalAnalyzer (Analysis Engine)                       │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │              Route B: Joint Least Squares (P0)                      │ │
│  │         ADF-EDX-EELS joint quantitative fusion                    │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    Results & Visualization                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐ │
│  │ Quantitative│  │  Residual   │  │  Cross-Modal│  │   Uncertainty   │ │
│  │ Composition │  │   Analysis  │  │   Consistency│  │    Maps         │ │
│  │   Maps      │  │             │  │   Check     │  │                 │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

### Module Structure

```
qem/
├── fusion/                          # NEW: Multi-modal fusion module
│   ├── __init__.py
│   ├── dataset.py                   # MultiModalDataset container
│   ├── alignment.py                 # Cross-modal registration
│   ├── preprocessing.py             # Preprocessing pipeline
│   ├── analyzer.py                  # MultiModalAnalyzer (main entry)
│   ├── routes/                      # Analysis routes
│   │   ├── __init__.py
│   │   ├── base.py                  # Abstract base for all routes
│   │   ├── route_a_mcr.py           # MCR-LLM joint extension
│   │   ├── route_b_joint_ls.py      # Joint least squares (P0)
│   │   ├── route_c_deep_learning.py # DL implicit correlation
│   │   └── route_d_bayesian.py      # Bayesian joint inversion
│   ├── constraints.py               # Cross-modal constraint definitions
│   ├── calibration.py               # Cross-section & k-factor calibration
│   └── io.py                        # Save/load fusion results
│
├── vendors/
│   └── multi_modal/                 # EXISTING (to be deprecated gradually)
│       └── mapfusion/               # Keep for backward compatibility
│
└── ... (existing modules)
```

---

## Components and Interfaces

### 1. MultiModalDataset (Data Container)

**Purpose**: Unified container for all three modalities with shared coordinate system and metadata.

```python
class MultiModalDataset:
    """
    Container for pre-aligned single-frame ADF, EDX, and EELS data.
    
    IMPORTANT: Alignment is assumed to be done by superalign (~/code/superalign)
    before creating this dataset. This class only loads already-aligned data.
    
    All modalities share a common navigation space (scanning pixels)
    but have different signal dimensions:
    - ADF: (ny, nx) — 2D image
    - EDX: (ny, nx, n_energy) — spectrum image
    - EELS HL: (ny, nx, n_energy_hl) — high-loss spectrum image  
    - EELS LL: (ny, nx, n_energy_ll) — low-loss spectrum image (optional)
    
    Parameters
    ----------
    adf : np.ndarray or hyperspy.Signal2D
        ADF/HAADF image, shape (ny, nx) — already aligned by superalign
    edx : np.ndarray or hyperspy.Signal1D
        EDX spectrum image, shape (ny, nx, n_energy) — already aligned
    eels_hl : np.ndarray or hyperspy.Signal1D
        EELS high-loss spectrum image, shape (ny, nx, n_energy_hl) — already aligned
    eels_ll : np.ndarray or hyperspy.Signal1D, optional
        EELS low-loss spectrum image for thickness normalization — already aligned
    pixel_size : float
        Spatial pixel size in nm (shared across modalities)
    elements : list of str
        Expected elements in the sample
    """
    
    def __init__(self, adf, edx, eels_hl, eels_ll=None, 
                 pixel_size=None, elements=None):
        self.adf = adf
        self.edx = edx  
        self.eels_hl = eels_hl
        self.eels_ll = eels_ll
        self.pixel_size = pixel_size
        self.elements = elements or []
        
    # --- Properties ---
    @property
    def shape_nav(self) -> tuple:
        """Navigation shape (ny, nx) shared by all modalities."""
        return self.adf.shape[:2]
    
    @property
    def n_pixels(self) -> int:
        """Total number of spatial pixels."""
        return np.prod(self.shape_nav)
    
    # --- I/O ---
    @classmethod
    def from_hspy(cls, adf_path, edx_path, eels_hl_path, 
                  eels_ll_path=None, elements=None) -> 'MultiModalDataset':
        """
        Load from hyperspy files (already aligned by superalign).
        
        Parameters
        ----------
        adf_path : str
            Path to adf_aligned.hspy
        edx_path : str
            Path to edx_aligned.hspy
        eels_hl_path : str
            Path to eels_hl_aligned.hspy
        eels_ll_path : str, optional
            Path to eels_ll_aligned.hspy
        elements : list of str
            Expected elements
            
        Returns
        -------
        dataset : MultiModalDataset
        """
        
    def save(self, path: str, format: str = 'h5'):
        """Save dataset to disk."""
        
    @classmethod
    def load(cls, path: str) -> 'MultiModalDataset':
        """Load dataset from disk."""
        
    # --- Validation ---
    def validate(self) -> dict:
        """Validate dataset consistency and return diagnostics."""
```

**Key Design Decisions**:
- Wraps `hyperspy` signals internally but exposes numpy arrays for computation
- Lazy loading support for large spectrum images (via `dask`/`hyperspy` lazy)
- Shared `axes_manager` concept from hyperspy for coordinate tracking

---

### 2. CrossModalAligner (Alignment)

**Purpose**: Rigid and non-rigid registration to correct spatial drift between modalities.

```python
class CrossModalAligner:
    """
    Applies pre-computed alignment shifts to MultiModalDataset.
    
    IMPORTANT: Alignment computation is performed by superalign (~/code/superalign),
    not by qem. This class only:
    1. Reads shift files produced by superalign (.mat format)
    2. Applies shifts to bring all modalities into common coordinate frame
    3. Validates alignment quality
    
    superalign handles:
    - Rigid registration: phase correlation on integrated signals
    - Non-rigid registration: optical flow on ADF reference
    - Shift propagation from ADF to EDX/EELS
    
    Parameters
    ----------
    shifts_source : str
        Path to superalign shift file (.mat) or 'superalign' to call directly
    """
    
    def __init__(self, shifts_source=None):
        self.shifts_source = shifts_source
        self.shifts_rigid = {}      # {modality: (dy, dx)}
        self.shifts_non_rigid = {}  # {modality: (ny, nx, 2)}
        
    def load_shifts(self, path: str) -> dict:
        """
        Load pre-computed shifts from superalign output.
        
        Parameters
        ----------
        path : str
            Path to .mat file containing shifts from superalign
            
        Returns
        -------
        shifts : dict
            {'rigid': {...}, 'non_rigid': {...}}
        """
        
    def apply(self, dataset: MultiModalDataset) -> MultiModalDataset:
        """
        Apply pre-computed shifts to dataset.
        
        Returns
        -------
        aligned_dataset : MultiModalDataset
            Dataset with all modalities in common coordinate frame
        """
        
    def validate_alignment(self, dataset: MultiModalDataset) -> dict:
        """
        Validate alignment quality by checking cross-modal correlations.
        
        Returns
        -------
        quality : dict
            {'max_displacement': float, 'mean_residual': float, 'status': str}
        """
        
    def get_displacement_field(self, modality: str) -> np.ndarray:
        """Get non-rigid displacement field for visualization."""
```
**Integration with superalign**:

- superalign (~/code/superalign) performs all alignment computation
- qem only reads shift files produced by superalign (.mat format)
- superalign handles: rigid registration, non-rigid optical flow, shift propagation
- qem applies shifts and validates alignment quality

---

### 3. PreprocessingPipeline

**Purpose**: modality-specific preprocessing with cross-modal consistency checks.

```python
class PreprocessingPipeline:
    """
    Preprocessing steps for each modality:
    
    ADF:
      - Detector gain/background calibration (via qem.detector.Calibrate_Detector)
      - PACBED thickness estimation (optional)
      
    EDX:
      - Dead time correction
      - Peak deconvolution (handle overlaps like Mn-Kβ / Fe-Kα)
      - k-factor calibration (from reference spectrum or EELS)
      
    EELS:
      - Energy alignment (ZLP centering)
      - Background removal (power-law or polynomial)
      - Thickness normalization (from LL t/λ)
      - Cross-section calibration (from tabulated GOS or reference)
    """
    
    def __init__(self, config: PreprocessingConfig = None):
        self.config = config or PreprocessingConfig()
        
    def run(self, dataset: MultiModalDataset) -> MultiModalDataset:
        """Execute full preprocessing pipeline."""
        
    # --- Individual steps ---
    def calibrate_adf_detector(self, adf, gain=None, background=None):
        """Apply detector calibration."""
        
    def deconvolve_edx_peaks(self, edx, elements, resolution=130):
        """Deconvolve overlapping EDX peaks."""
        
    def remove_eels_background(self, eels_hl, eels_ll=None, 
                                  method='power_law', fit_window=(50, 100)):
        """Remove EELS background and normalize."""
        
    def estimate_thickness(self, eels_ll) -> np.ndarray:
        """Estimate t/λ from low-loss spectrum."""
        
    def cross_modal_rescale(self, dataset) -> dict:
        """
        Rescale all modalities to common intensity units.
        Returns scaling factors for each modality.
        """
```

**Cross-Modal Consistency Checks**:
- After preprocessing, verify that EDX and EELS detect the same elements
- Flag pixels where ADF shows atomic columns but spectroscopy shows no signal (potential contamination/artefacts)
- Flag pixels where spectroscopy shows signal but ADF shows no column (potential beam damage or misalignment)

---

### 4. Analysis Routes

All routes inherit from `BaseAnalysisRoute` and implement:
- `fit(dataset)` — run analysis
- `get_results()` — return quantitative maps
- `get_uncertainty()` — return uncertainty estimates (if supported)
- `validate()` — check inputs and report diagnostics

#### 4.1 Route B: Joint Least Squares (P0 — First Implementation)

**Mathematical Framework**:

For each spatial pixel $i$, we observe:
- ADF: $b_i$ (scalar)
- EDX: $oldsymbol{x}_i^{EDX}$ (vector over energies)
- EELS: $oldsymbol{x}_i^{EELS}$ (vector over energies)

We seek elemental concentrations $oldsymbol{c}_i = [c_{i,1}, ..., c_{i,M}]$ that satisfy:

$$
\min_{\boldsymbol{c}_i \geq 0} \left\| \boldsymbol{A}_{ADF} \boldsymbol{c}_i^{\gamma} - b_i \right\|^2 + \lambda_{EDX} \left\| \boldsymbol{A}_{EDX} \boldsymbol{c}_i - \boldsymbol{x}_i^{EDX} \right\|^2 + \lambda_{EELS} \left\| \boldsymbol{A}_{EELS} \boldsymbol{c}_i - \boldsymbol{x}_i^{EELS} \right\|^2 + \lambda_{TV} \sum_{m} TV(c_{:,m})
$$

Where:
- $oldsymbol{A}_{ADF}$: Z-contrast forward model (from `multi_modal`)
- $oldsymbol{A}_{EDX}$: EDX reference spectra matrix (from `edx_calibration.mat`)
- $\boldsymbol{A}_{EELS}$: EELS cross-section matrix (from `pyEELSMODEL` at `~/code/pyEELSMODEL`)
- $oldsymbol{c}_i^{\gamma}$: element-wise power for ADF nonlinearity
- $TV$: total variation regularization across spatial dimensions

```python
class JointLeastSquaresRoute(BaseAnalysisRoute):
    """
    Route B: Known reference spectra joint least squares.
    
    Requires:
      - ADF reference: scattering cross-section library (from simulation or calibration)
      - EDX reference: fitted peak parameters (from EDX.py calibration)
      - EELS reference: ionization cross-sections (from `pyEELSMODEL` at `~/code/pyEELSMODEL` or tabulated)
    
    Parameters
    ----------
    adf_gamma : float
        Exponent for ADF Z-contrast nonlinearity (default: 1.6)
    lambda_adf : float
        Weight for ADF data fidelity term
    lambda_edx : float
        Weight for EDX data fidelity term  
    lambda_eels : float
        Weight for EELS data fidelity term
    lambda_tv : float
        Weight for spatial TV regularization
    n_iterations : int
        Number of optimization iterations
    elements : list of str
        Elements to quantify
    """
    
    def __init__(self, elements, adf_gamma=1.6, 
                 lambda_adf=None, lambda_edx=0.005, 
                 lambda_eels=0.01, lambda_tv=0.1,
                 n_iterations=50):
        self.elements = elements
        self.adf_gamma = adf_gamma
        # Default lambda_adf = 1 / n_elements
        self.lambda_adf = lambda_adf or 1.0 / len(elements)
        self.lambda_edx = lambda_edx
        self.lambda_eels = lambda_eels
        self.lambda_tv = lambda_tv
        self.n_iterations = n_iterations
        
        # Reference spectra (loaded during fit)
        self.adf_refs = None    # {element: scs_value}
        self.edx_refs = None    # {element: {edge: {'A', 'centre', 'sigma'}}}
        self.eels_refs = None   # {element: cross_section_spectrum}
        
    def load_references(self, adf_lib=None, edx_cal=None, eels_cal=None):
        """
        Load reference spectra from calibration files.
        
        Parameters
        ----------
        adf_lib : str or dict
            ADF scattering cross-section library (from simulation)
        edx_cal : str or dict  
            EDX calibration (from EDX.py output .mat)
            eels_cal : str or dict
                EELS calibration (from `pyEELSMODEL` at `~/code/pyEELSMODEL` or .mat)
        """
        
    def build_measurement_matrix(self, dataset: MultiModalDataset) -> dict:
        """
        Build forward model matrices for each modality.
        
        Returns
        -------
        matrices : dict
            {'adf': A_adf, 'edx': A_edx, 'eels': A_eels}
        """
        
    def fit(self, dataset: MultiModalDataset) -> 'JointLeastSquaresRoute':
        """
        Run joint optimization.
        
        Algorithm:
        1. Initialize concentrations from independent quantification
        2. Iteratively update concentrations via gradient descent
        3. Apply TV regularization per element map
        4. Enforce non-negativity
        """
        
    def get_results(self) -> dict:
        """
        Returns
        -------
        results : dict
            {
                'composition': {element: (ny, nx) array},
                'adf_reconstructed': (ny, nx) array,
                'edx_reconstructed': (ny, nx, n_energy) array,
                'eels_reconstructed': (ny, nx, n_energy) array,
                'residuals': {modality: residual_array},
                'convergence': cost_history
            }
        """
```

**Implementation Notes**:
- Reuse `mapfusion.DataFusion` logic for ADF-EDX fusion as baseline
- Extend with EELS term using `scipy.sparse` for memory efficiency
- TV regularization via `skimage` or `scipy.ndimage` (drop C++ dependency for portability)
- Support both pixel-wise and patch-wise optimization

---

#### 4.2 Route A: MCR with LLM Joint Extension (P1)

**Purpose**: Unknown phase composition — when you don't know what elements are present or in what chemical state.

**Mathematical Framework**:

Multivariate Curve Resolution (MCR) with non-negativity constraints:

$$
\boldsymbol{X} = \boldsymbol{C} \boldsymbol{S}^T + \boldsymbol{E}
$$

Where:
- $oldsymbol{X}$: concatenated spectrum image (EDX + EELS) flattened to $(n_{pixels}, n_{energies})$
- $oldsymbol{C}$: concentration matrix $(n_{pixels}, n_{components})$
- $oldsymbol{S}$: spectral signatures $(n_{energies}, n_{components})$
- $oldsymbol{E}$: residual

**LLM Extension**: Use language model to interpret ambiguous spectral signatures:
- Component with peaks at 6.4 keV (EDX) + 708 eV (EELS) → LLM identifies as "Fe"
- Component with no known match → LLM suggests "possible oxide contamination" or "unknown silicate"

```python
class MCRLLMRoute(BaseAnalysisRoute):
    """
    Route A: MCR-LLM joint extension for unknown phase analysis.
    
    Combines:
      - MCR-ALS (Alternating Least Squares) for spectral unmixing
      - LLM-based spectral signature interpretation
      - ADF constraint for spatial consistency
    """
    
    def __init__(self, n_components=10, max_iter=100, 
                 tol=1e-6, use_llm=True, llm_model=None):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.use_llm = use_llm
        self.llm_model = llm_model  # e.g., OpenAI API or local model
        
    def fit(self, dataset: MultiModalDataset):
        """
        1. Concatenate EDX + EELS into single spectrum matrix
        2. Run MCR-ALS with non-negativity constraints
        3. Use LLM to identify spectral components
        4. Map components back to spatial concentrations
        """
        
    def interpret_components(self, spectra: np.ndarray) -> list:
        """
        Use LLM to interpret spectral signatures.
        
        Returns list of dicts:
        [
            {
                'component_id': 0,
                'identified_elements': ['Fe', 'O'],
                'confidence': 0.92,
                'notes': 'Consistent with Fe2O3 hematite'
            },
            ...
        ]
        """
```

---

#### 4.3 Route D: Bayesian Joint Inversion (P2)

**Purpose**: High-precision quantitative analysis with uncertainty quantification, especially for low-dose data.

**Mathematical Framework**:

Probabilistic model:

$$
P(\boldsymbol{c} | \boldsymbol{b}, \boldsymbol{x}^{EDX}, \boldsymbol{x}^{EELS}) \propto P(\boldsymbol{b} | \boldsymbol{c}) \cdot P(\boldsymbol{x}^{EDX} | \boldsymbol{c}) \cdot P(\boldsymbol{x}^{EELS} | \boldsymbol{c}) \cdot P(\boldsymbol{c})
$$

Where likelihoods capture noise statistics:
- ADF: Gaussian (detector noise) or Poisson (electron counting)
- EDX: Poisson (X-ray counting statistics)
- EELS: Poisson (electron counting) + background uncertainty

Prior $P(\boldsymbol{c})$:
- Non-negativity: $c_m \geq 0$
- Sparsity: Laplace prior (most pixels contain few elements)
- Spatial smoothness: Gaussian process prior

```python
class BayesianJointInversionRoute(BaseAnalysisRoute):
    """
    Route D: Bayesian joint inversion with uncertainty quantification.
    
    Uses PyMC for probabilistic inference.
    
    Parameters
    ----------
    noise_model : dict
        {'adf': 'gaussian'|'poisson', 'edx': 'poisson', 'eels': 'poisson'}
    prior : str
        'nonnegative', 'sparse', 'smooth', or 'combined'
    n_samples : int
        Number of MCMC samples
    """
    
    def __init__(self, noise_model=None, prior='combined', 
                 n_samples=2000, n_tune=1000):
        self.noise_model = noise_model or {
            'adf': 'gaussian', 'edx': 'poisson', 'eels': 'poisson'
        }
        self.prior = prior
        self.n_samples = n_samples
        self.n_tune = n_tune
        
    def build_model(self, dataset: MultiModalDataset):
        """Build PyMC probabilistic model."""
        
    def fit(self, dataset: MultiModalDataset):
        """Run MCMC inference."""
        
    def get_uncertainty_maps(self) -> dict:
        """Return posterior standard deviation for each element map."""
```

---

#### 4.4 Route C: Deep Learning Implicit Correlation (P3)

**Purpose**: Complex backgrounds, non-linear effects, and implicit cross-modal correlations that are hard to model analytically.

**Integration with Lumen**:

Reuse the self-supervised learning framework from `~/code/lumen`:
- Train a multi-modal encoder that maps ADF + EDX + EELS to a latent representation
- Decoder reconstructs all three modalities from latent space + elemental composition
- Cross-modal consistency enforced in latent space

```python
class DeepLearningRoute(BaseAnalysisRoute):
    """
    Route C: Deep learning implicit correlation.
    
    Uses neural network to learn cross-modal correlations implicitly.
    
    Architecture:
      - Multi-modal encoder: ADF (CNN) + EDX (1D CNN) + EELS (1D CNN) → latent vector
      - Composition decoder: latent → elemental concentrations
      - Reconstruction decoder: latent + composition → reconstructed modalities
    
    Parameters
    ----------
    encoder_weights : str
        Path to pre-trained encoder (from Lumen EUPE weights)
    latent_dim : int
        Dimensionality of latent space
    """
    
    def __init__(self, encoder_weights=None, latent_dim=256,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.encoder_weights = encoder_weights
        self.latent_dim = latent_dim
        self.device = device
        self.model = None
        
    def build_model(self):
        """Build multi-modal network architecture."""
        
    def fit(self, dataset: MultiModalDataset):
        """
        Inference mode (pre-trained model).
        For training, use separate training pipeline.
        """
        
    def train(self, datasets: list, epochs=100):
        """Train on multiple datasets."""
```

---

### 5. MultiModalAnalyzer (Main Entry Point)

```python
class MultiModalAnalyzer:
    """
    Main entry point for multi-modal quantitative analysis.
    
    Orchestrates the full pipeline: alignment → preprocessing → analysis → results.
    
    Example
    -------
    >>> from qem.fusion import MultiModalAnalyzer, MultiModalDataset
    >>> 
    >>> # Load data
    >>> dataset = MultiModalDataset.load('my_experiment.h5')
    >>> 
    >>> # Create analyzer with Route B (joint least squares)
    >>> analyzer = MultiModalAnalyzer(
    ...     route='joint_ls',
    ...     elements=['Ti', 'O', 'Fe', 'Sr'],
    ...     adf_gamma=1.6,
    ...     lambda_edx=0.005,
    ...     lambda_eels=0.01
    ... )
    >>> 
    >>> # Run full pipeline
    >>> results = analyzer.run(dataset)
    >>> 
    >>> # Access results
    >>> composition = results.composition  # {element: (ny, nx) map}
    >>> ti_map = composition['Ti']
    >>> 
    >>> # Visualize
    >>> results.plot_element_maps()
    >>> results.plot_cross_modal_consistency()
    """
    
    def __init__(self, route='joint_ls', **route_kwargs):
        """
        Parameters
        ----------
        route : str
            Analysis route: 'joint_ls', 'mcr_llm', 'bayesian', 'deep_learning'
        **route_kwargs : dict
            Route-specific parameters
        """
        self.route_name = route
        self.route = self._create_route(route, **route_kwargs)
        self.aligner = CrossModalAligner()
        self.preprocessor = PreprocessingPipeline()
        self.results = None
        
    def _create_route(self, route, **kwargs) -> BaseAnalysisRoute:
        """Factory method to create analysis route."""
        routes = {
            'joint_ls': JointLeastSquaresRoute,
            'mcr_llm': MCRLLMRoute,
            'bayesian': BayesianJointInversionRoute,
            'deep_learning': DeepLearningRoute,
        }
        if route not in routes:
            raise ValueError(f"Unknown route: {route}. Choose from {list(routes.keys())}")
        return routes[route](**kwargs)
    
    def run(self, dataset: MultiModalDataset, 
            align=True, preprocess=True) -> 'AnalysisResults':
        """
        Execute full analysis pipeline.
        
        Parameters
        ----------
        dataset : MultiModalDataset
            Input multi-modal data
        align : bool
            Whether to run alignment (skip if already aligned)
        preprocess : bool
            Whether to run preprocessing (skip if already preprocessed)
            
        Returns
        -------
        results : AnalysisResults
            Container with all outputs, visualizations, and diagnostics
        """
        # Step 1: Alignment
        if align and not dataset._aligned:
            dataset = self.aligner.fit(dataset)
            
        # Step 2: Preprocessing
        if preprocess and not dataset._preprocessed:
            dataset = self.preprocessor.run(dataset)
            
        # Step 3: Analysis
        self.route.fit(dataset)
        
        # Step 4: Package results
        self.results = AnalysisResults(
            route=self.route,
            dataset=dataset,
            composition=self.route.get_results()
        )
        
        return self.results
```

---

### 6. AnalysisResults (Output Container)

```python
class AnalysisResults:
    """
    Container for analysis outputs with visualization and export.
    """
    
    def __init__(self, route, dataset, composition):
        self.route = route
        self.dataset = dataset
        self.composition = composition  # {element: (ny, nx) array}
        
    # --- Quantitative Access ---
    def get_composition_map(self, element: str) -> np.ndarray:
        """Get quantitative composition map for element."""
        
    def get_atomic_percent(self, region=None) -> dict:
        """
        Compute atomic percent composition.
        
        Parameters
        ----------
        region : np.ndarray or Region, optional
            Boolean mask or region object. If None, use full field.
            
        Returns
        -------
        composition : dict
            {element: atomic_percent}
        """
        
    def get_thickness_map(self) -> np.ndarray:
        """Get specimen thickness map from EELS low-loss."""
        
    # --- Diagnostics ---
    def get_residuals(self) -> dict:
        """Get residual maps for each modality."""
        
    def get_cross_modal_consistency(self) -> np.ndarray:
        """
        Compute cross-modal consistency score per pixel.
        High score = all three modalities agree on composition.
        Low score = inconsistency (potential artefact).
        """
        
    def get_uncertainty(self, element: str) -> np.ndarray:
        """Get uncertainty map (if route supports it)."""
        
    # --- Visualization ---
    def plot_element_maps(self, elements=None, figsize=(12, 8)):
        """Plot quantitative elemental distribution maps."""
        
    def plot_cross_sections(self, positions: list):
        """
        Plot 1D cross-sections through specified positions.
        Shows ADF + EDX + EELS profiles side by side.
        """
        
    def plot_spectrum_at_position(self, y: int, x: int):
        """
        Plot EDX and EELS spectra at specific pixel with fitted components.
        """
        
    def plot_residual_analysis(self):
        """Plot residual maps and histograms for quality assessment."""
        
    def plot_composition_scatter(self, element1: str, element2: str):
        """
        Scatter plot of element1 vs element2 composition.
        Color by ADF intensity to show structure-composition correlation.
        """
        
    # --- Export ---
    def save(self, path: str, format='h5'):
        """Save results to disk."""
        
    def to_hyperspy(self) -> dict:
        """Export as hyperspy signals for further analysis."""
        
    def to_dataframe(self, region=None) -> pd.DataFrame:
        """
        Export composition data as pandas DataFrame.
        One row per pixel, columns for each element + ADF + thickness.
        """
```

---

## Data Models

### Calibration Data Schema

```python
# ADF Calibration (from qem.detector or simulation)
ADFCalibration = {
    'element': str,           # e.g., 'Sr', 'Ti'
    'scattering_cross_section': float,  # in Å²
    'acceleration_voltage': float,      # kV
    'convergence_angle': float,         # mrad
    'collection_angle': float,          # mrad
    'source': str             # 'simulation', 'experiment', 'tabulated'
}

# EDX Calibration (from EDX.py output)
EDXCalibration = {
    'element': str,
    'edge': str,              # e.g., 'K', 'L', 'M'
    'A': float,               # Gaussian amplitude
    'centre': float,          # Peak center energy (keV)
    'sigma': float,           # Peak width
    'volume': float,          # A * sigma * sqrt(pi)
    'k_factor': float,        # Relative to reference element
    'detector_efficiency': float
}

# EELS Calibration (from pyEELSMODEL at ~/code/pyEELSMODEL)
EELSCalibration = {
    'element': str,
    'edge': str,              # e.g., 'L2,3', 'K'
    'cross_section': np.ndarray,  # Energy-dependent cross-section
    'energy_axis': np.ndarray,
    'E0': float,              # Acceleration voltage (eV)
    'alpha': float,           # Convergence angle (rad)
    'beta': float,            # Collection angle (rad)
    'source': str             # 'Zezhong', 'Kohl', 'tabulated'
}
```

### Results Schema

```python
AnalysisResultsSchema = {
    'composition': {
        element: {
            'map': np.ndarray,           # (ny, nx) quantitative map
            'units': str,                # 'atoms/nm²', 'atomic_percent', etc.
            'uncertainty': np.ndarray,   # (ny, nx) standard deviation (optional)
        }
        for element in elements
    },
    'thickness': {
        'map': np.ndarray,               # (ny, nx) t/λ or nm
        'units': str,
    },
    'reconstructed': {
        'adf': np.ndarray,               # (ny, nx) reconstructed ADF
        'edx': np.ndarray,             # (ny, nx, n_energy) reconstructed EDX
        'eels': np.ndarray,            # (ny, nx, n_energy) reconstructed EELS
    },
    'residuals': {
        'adf': np.ndarray,
        'edx': np.ndarray,
        'eels': np.ndarray,
    },
    'convergence': {
        'cost_history': np.ndarray,
        'n_iterations': int,
        'converged': bool,
    },
    'metadata': {
        'route': str,
        'parameters': dict,              # Route-specific parameters
        'processing_history': list,      # Steps applied
        'timestamp': str,
    }
}
```

---

## Integration Points

### With Existing QEM Modules

| QEM Module | Integration | Purpose |
|------------|-------------|---------|
| `qem.fit.image_fitting` | Import `ImageFitting` for ADF atomic column analysis | Atomic positions for spatial constraints |
| `qem.fit.voronoi` | Use `voronoi_integrate` for atomic column integration | Voronoi cell-based composition assignment |
| `qem.detector` | Import `Calibrate_Detector` for ADF calibration | Detector gain/background correction |
| `qem.analysis.crystal_analyzer` | Use `CrystalAnalyzer` for lattice mapping | Structural constraints on composition |
| `qem.visualization` | Extend plotting for multi-modal views | Cross-modal visualization |

### With External Packages

| Package | Usage | Integration Strategy |
|---------|-------|---------------------|
| `hyperspy` | Signal I/O, axes management, decomposition | Wrap/extend signals; use API |
| `superalign` (`~/code/superalign`) | Drift correction, registration | Call as dependency; read shift files |
| `pyEELSMODEL` (`~/code/pyEELSMODEL`) | EELS quantification, cross-sections | Call API for EELS background/model fitting |
| `scikit-image` | TV regularization, image processing | Direct import |
| `scipy.sparse` | Sparse measurement matrices | Direct import |
| `pymc` (optional) | Bayesian inference for Route D | Optional dependency |
| `torch` (optional) | Deep learning for Route C | Optional dependency; reuse Lumen weights |

---

## Implementation Roadmap

### Phase 1: Foundation (Weeks 1-2)

1. **Create `qem/fusion/` module structure**
   - `__init__.py` with exports
   - `dataset.py`: `MultiModalDataset` container
   - `alignment.py`: `CrossModalAligner` (reads superalign shift files from `~/code/superalign`, applies shifts)
   - `preprocessing.py`: `PreprocessingPipeline`
   - `io.py`: Save/load fusion datasets

2. **Refactor existing scripts**
   - Extract calibration logic from `EDX.py` into `calibration.py`
   - Extract alignment logic from `alignment.py` into `alignment.py`
   - Create unified data loading from `*.hspy` files

3. **Tests**
   - Unit tests for `MultiModalDataset` I/O
   - Unit tests for alignment with synthetic shifts
   - Integration test: load High Entropy dataset

### Phase 2: Route B — Joint Least Squares (Weeks 3-4)

1. **Implement `JointLeastSquaresRoute`**
   - Build measurement matrices from calibration data
   - Gradient descent optimization with TV regularization
   - Non-negativity constraints

2. **Integrate with existing `multi_modal`**
   - Port `DataFusion` logic into `route_b_joint_ls.py`
   - Replace C++ TV with `scikit-image`/`scipy` implementation
   - Add EELS term to objective function

3. **Calibration pipeline**
   - Load `edx_calibration.mat` and `eels_element_maps.mat`
   - Normalize cross-sections across modalities
   - Handle missing elements gracefully

4. **Tests**
   - Synthetic data test: known composition, verify recovery
   - Real data test: High Entropy alloy, compare with independent quantification

### Phase 3: Routes A, D, C (Weeks 5-8)

1. **Route A: MCR-LLM**
   - Implement MCR-ALS with `sklearn` or custom
   - LLM integration for spectral interpretation (optional, API-based)

2. **Route D: Bayesian**
   - PyMC model definition
   - MCMC sampling with convergence diagnostics
   - Uncertainty map generation

3. **Route C: Deep Learning**
   - Multi-modal encoder architecture
   - Integration with Lumen pre-trained weights
   - Training pipeline (separate from inference)

### Phase 4: Visualization & UX (Weeks 9-10)

1. **Results visualization**
   - Element maps with colorbars and scale
   - Cross-modal consistency overlay
   - Spectrum plots at selected positions

2. **Streamlit integration**
   - Add fusion tab to `qem.app`
   - Interactive parameter adjustment
   - Real-time preview during optimization

3. **Documentation**
   - API docs with examples
   - Tutorial notebooks
   - Theory background (link to Feishu doc)

---

## Risk Analysis

| Risk | Impact | Mitigation |
|------|--------|------------|
| `superalign` (`~/code/superalign`) dependency breaks | High | superalign is external tool; qem only reads its output |
| `pyEELSMODEL` (`~/code/pyEELSMODEL`) API changes | Medium | Pin version; wrap API calls |
| EELS cross-section accuracy | High | Support multiple cross-section sources (Zezhong/Kohl/tabulated) |
| Memory overflow on large SI | High | Implement chunked processing; lazy loading |
| C++ build issues for TV reg | Medium | Replace with pure Python/Numba implementation |
| LLM API availability | Low | Make LLM optional; local model fallback |

---

## Success Metrics

1. **Quantitative accuracy**: Composition maps agree with independent EDX/EELS quantification within 10% relative error
2. **Spatial resolution**: Element maps resolve atomic columns at ~0.36 nm pixel size
3. **Cross-modal consistency**: Residuals are spatially uncorrelated (white noise)
4. **Performance**: Route B completes on 100×100 pixel SI in < 5 minutes on CPU
5. **Usability**: Single Python call `analyzer.run(dataset)` replaces current 5-script workflow

---

## Appendix: Current Script Migration Guide

| Current Script | New Location | Migration Notes |
|---------------|-------------|-----------------|
| `EDX.py` | `qem/fusion/preprocessing.py` + `calibration.py` | Extract calibration logic; keep `EDXStack` usage as example |
| `EELS_quantification.py` | `qem/fusion/preprocessing.py` | Wrap `pyEELSMODEL` (`~/code/pyEELSMODEL`) calls; expose as pipeline step |
| `adf_quant.py` | `qem/fusion/calibration.py` | Reuse `ImageFitting` + Voronoi; export SCS library |
| `alignment.py` | `qem/fusion/alignment.py` | Generalize `superalign` calls; support multiple modalities |
| `EELS_alignment.py` | `qem/fusion/alignment.py` | Merge into unified aligner |

---

## References

1. **Theory Document**: [ADF-EDX-EELS SI Joint Quantitative Analysis Framework](https://my.feishu.cn/wiki/JWCRwI3HXi4Ktzki3odcMaJKn2b)
2. **Existing Multi-Modal Code**: `~/code/qem/vendors/multi_modal/mapfusion/`
3. **High Entropy Data**: `~/work/data/High_entropy/script/`
4. **QEM Image Fitting**: `~/code/qem/qem/fit/image_fitting.py`
5. **Lumen Framework**: `~/code/lumen/` (for Route C)
