# Deterministic multiple-shift lattices

MATLAB code for *A deterministic multiple-shift lattice algorithm for
function approximation in Korobov and half-period cosine spaces* by Jiarui
Du and Josef Dick. [arXiv preprint](https://arxiv.org/abs/2604.01146)

The [revision release](https://github.com/Jiarui-Du/multiple-shift/releases/tag/paper-revision-1)
contains the code, saved results, and EPS files for the revised manuscript.
The [earlier public snapshot](https://github.com/Jiarui-Du/multiple-shift/tree/public-2026-07-01)
and [changelog](CHANGELOG.md) record the changes since the first public version.

`run_korobov.m` runs parameter sets serially. It shares the multiple-shift
recovery routine with `run_cosine_pde.m` and `run_gross_comparison.m`. Setup
forms each nonempty fiber's Gram matrix and retains the conjugate transpose
of its reconstruction matrix and the lower Cholesky factor. Recovery reuses
these for every global shift. The CRT and greedy constructions count the
shared origin once and test the exponential sums of the distinct shift set.

CBC uses ceil(alpha), the corresponding rescaled weights, and arf at 113 bits
with 8 native threads. The Windows MATLAB backend is in
`functions/cbc_precision/`.

The timing fields follow the boundaries of Section 6:

- `frequency_time_matrix` measures selecting M and constructing A, the common
  input for the Gross comparison.
- `setup_time_matrix` starts with A given. It includes CBC, fiber and shift
  construction, and matrix preparation and factorization for multiple shift.
  For Gross it includes lattice construction and the FFT index maps.
- `recovery_time_matrix` includes generating sampling coordinates, evaluating
  the target there, and recovering the Fourier coefficients.
- The PDE includes frequency grouping and Laplacian eigenvalues in Setup.
  `spectral_time_matrix` records cosine coefficient conversion and the spectral
  solve for each global shift.
- `evaluation_time_matrix` includes GPU transfers and evaluation of the
  recovered approximation at the test points. Exact test values, error norms,
  and plotting are outside the stage timers.

Per-shift timings are averaged over the shifts used by the experiment. The
PDE evaluates both fields and all global shifts with shared cosine basis
blocks, and reports the total GPU evaluation time divided by the number of
shifts. The Gross comparison uses the median of the configured Setup/Recovery
repetitions.
Timing results depend on the hardware and system load.

## Running the figure scripts

On 64-bit Windows, open PowerShell in a copy of this repository and run
`.\setup_dependencies.ps1`. The script checks SHA-256 digests and obtains the
Gross et al. MATLAB routines, Dirk Nuyens's primitive-root helpers, and the
FLINT and MinGW DLLs from their upstream distributions. It extracts the DLLs
from the archives directly. MATLAB does not need a Python installation.
See [native backend setup](functions/cbc_precision/native/win64/README.md)
for the versions and runtime layout.

Set MATLAB's Current Folder to the repository root. Run the script for the
figure of interest. The archived files can be viewed or loaded directly.

| Script | Figures | Saved results |
| --- | --- | --- |
| `run_s_adaptive` | Shift strategy comparison (`s_optimal.eps`) | [S_optimal.mat](results/S_optimal.mat) |
| `run_s_prob_deter` | Shift counts and Gram-matrix condition numbers | [S_eff.mat](results/S_eff.mat) |
| `run_gross_comparison` | Gross comparison of errors, timings and oversampling, and auxiliary shifts for the multiple-shift method | [gross_comparison_Linf_eval262144.mat](results/gross_comparison_Linf_eval262144.mat) |
| `run_korobov` | Periodic approximation (`korobov_approx_alpha2_L2_N.eps` and `korobov_approx_alpha2_L2_tot.eps` in the manuscript) | [korobov_approx_alpha2_L2_eval262144.mat](results/korobov_approx_alpha2_L2_eval262144.mat) |
| `run_cosine_pde` | Neumann Poisson experiment | [cosine_pde_u_alpha1_5_L2_eval262144.mat](results/cosine_pde_u_alpha1_5_L2_eval262144.mat) |

[`figs/`](figs/) contains the EPS files used in the revised manuscript.
The periodic sampling-cost figure includes a shift-count panel and an
oversampling panel. The shift-count panel uses `s_matrix` from the Gross
comparison data, and the script exports the oversampling panel in Section 6.
To redraw the Gross comparison figures, add `functions/` to the MATLAB path,
load the saved Gross result in MATLAB, and run Sections 4–7 of
`run_gross_comparison.m`. The Korobov script adds `_eval262144` before the plot
suffix in its generated filenames.
The scripts write new results under `results/` and export figures to `figs/`.
The saved shift experiments use primes nearest to `2.^(10:20)`. The other
three saved runs use primes nearest to `2.^(8:20)` and `2^18` test points.
The Gross run uses one timing repetition. The Korobov and PDE scripts use
ten global shifts for the saved `L2` results. The two shift comparison scripts
use seed 26, and the other three scripts use seed 2026.

The supplied MEX/DLL bundle was built and tested with 64-bit Windows and
MATLAB R2024b. Other MATLAB releases require a compatibility check. It is not
a macOS/Linux binary. Full experiments use Statistics and Machine Learning
Toolbox (Sobol points) and Parallel Computing Toolbox with a compatible NVIDIA
GPU (Fourier evaluation).

Native sources, the build script and third-party licenses are included in
`functions/cbc_precision/`. Rebuilding requires a compatible compiler. The
supplied backend loads its DLLs from `native/win64/runtime/`, which is
populated by the setup script. Keep
`native/win64/build.json` with the bundle for the DLL name and build provenance.
