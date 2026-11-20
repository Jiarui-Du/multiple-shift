# Changelog

## `paper-revision-1`

- Use the ceiling-CBC construction with rescaled weights and the 113-bit
  FLINT backend. Keep the original smoothness in the truncation and
  frequency set.
- Count the common origin once when forming a union of rank-1 lattices.
  Test the distinct shift set, and reuse the Gram-matrix factorization
  for coefficient recovery across global shifts.
- Add the comparison with Gross et al. on the same frequency sets and
  test points. Report pre-computation and recovery separately.
- Include five result files and the 14 EPS files used in the revised
  manuscript. Document the entry points, dependencies, and run parameters.

## `public-2026-07-01`

This tag preserves the public repository before the revision release.
[Commit 0299346](https://github.com/Jiarui-Du/multiple-shift/commit/0299346578f24e3d9b6917275b5d81627619885b)
added `k = max(k,1);` to `adaptive_construction_new.m`. In the case `V = 1`,
the fallback therefore constructs at least one prime lattice.
