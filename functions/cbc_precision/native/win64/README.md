# Portable Windows 64-bit CBC runtime

Run `setup_dependencies.ps1` from the repository root to populate `runtime/`
with the verified upstream DLLs. The MEX, `build.json`, runtime DLLs and
licenses must stay together. MATLAB loads the DLLs directly.

Built with Windows 64-bit, MATLAB R2024b Update 3 and MinGW 8.1.
Other MATLAB versions require verification; macOS/Linux need a native build.
MATLAB supplies libmex.dll and libmx.dll.

Run the figure scripts from the repository root. See the root README for
the script list and the experiment toolbox and GPU requirements.

FLINT 3.3.1: https://github.com/flintlib/flint/tree/v3.3.1
The setup script extracts FLINT, GMP, MPFR and supporting DLLs from the
[python-flint 0.8.0 Windows wheel](https://pypi.org/project/python-flint/0.8.0/#files)
and the remaining runtime DLLs from MathWorks' MinGW 8.1 distribution. See
`licenses/` for the third-party license notices.
