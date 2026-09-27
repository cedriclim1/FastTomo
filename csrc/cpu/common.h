#pragma once
/*
 * common.h — shared helpers included by all CPU translation units.
 *
 * Currently only houses the SAMP3 macro used by forward_project_rotmat_cpu
 * and grad_rotmat_cpu (both in projection_rotmat.c).
 */

/* Border-zero trilinear sampler for a [Ny, Nz, Nx] volume `vol`.
 * Returns 0 for any index outside its valid range; used by the 3D CPU
 * projection kernels to match CUDA cudaAddressModeBorder semantics. */
#define SAMP3(ix, iy, iz)                                                  \
    (((ix) >= 0 && (ix) < Nx && (iy) >= 0 && (iy) < Ny &&                  \
      (iz) >= 0 && (iz) < Nz)                                              \
     ? vol[((size_t)(iy) * Nz + (iz)) * Nx + (ix)] : 0.0f)
