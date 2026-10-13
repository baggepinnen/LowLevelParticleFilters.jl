# TODO / known issues

## UKF `cross_cov` mutates the sigma-point cache

In `src/ukf.jl` (around lines 852–859), the non-`SMatrix` `add_to_C!`
overload does `xsm .-= x` in place. `cross_cov` calls it once per
sigma point, so by the time `cross_cov` returns, every sigma point in
the cache has been replaced by its delta from the mean.

The standard `correct!` path (`src/ukf.jl` around lines 639–671) does
not re-read the cache after `cross_cov`, so this is currently safe.
Any user-supplied `cross_cov` override that re-reads the sigma points,
or any future code that adds a step after the cross-covariance
computation, will silently see corrupted data.

Fix later by computing the delta into a scratch buffer (e.g. a small
extra vector stored inside the sigma-point cache) instead of mutating
the cache in place. The `SMatrix` overload above it (lines 843–850) is
already correct because `SMatrix` is immutable.
