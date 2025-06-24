# Notes

## NCFV (node-centered finite volume)
- NCFV is using the HANIM nonlinear solver.

## SFE (stabilized finite elements)
- To help robustness during initial transients, SFE has its "ramped" smoother option active, which is a spatially uniform Laplacian smoothing with a scaling factor that ramps from 1.0 at CFL'= 50 to 0.0 at CFL' = 500, where CFL' is the highest CFL seen during the simulation, i.e., after going above CFL=500, it does not reactivate if the CFL drops below 500 again.
- SFE is using residual smoothing with an alternating coefficient that switches every 5 iterations. This creates the oscillations in the residuals with a frequency of 10 iterations seen on finer meshes. Using a constant coefficient eliminates the oscillations but makes the overall convergence slower.


# Submission Descriptions


|  Submission ID      | Description |
|--------------------:|-------------|
|                  01 | NCFV, Cadence unstructured mesh, SA-neg |
|                  02 | SFE, Cadence unstructured mesh, SA-neg |
|                  03 | NCFV, Helden mesh, SA-neg |
|                  04 | SFE, Helden mesh, SA-neg |
|                  05 | NCFV, refine/Heldenmesh adapted with initial level 6 full boundary layer, SA-neg |
|                  06 | SFE, refine/Heldenmesh adapted with initial level 6 full boundary layer, SA-neg |
