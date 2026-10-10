"""Regenerate the golden keys changed by the loose sampling boundary and the
symmetric kink gradients (phase 1b PR G), and nothing else.

    python tests/data/regen_boundary_goldens.py

Run from the repository root with the PR G build importable, on the machine
whose arithmetic fingerprint the goldens carry (see AGENTS.md), after the
input sinograms were frozen (tests/data/freeze_golden_inputs.py).
regenerate_golden_keys.main recomputes every case of each device, stops
without writing if any key NOT listed here changes (byte-identical, or 2e-6
for the fp32-atomics keys), refuses GENFIRE / RESIRE keys and the stored
inputs, and keeps the fingerprints.

Changed, and why (the design note's section 5, with the input sinogram
frozen):
- the parallel forward (loose rule): fp, fp_ndet, fp_rect, fp_h, the
  cached proj_par_fp / proj_par_fp_h (CUDA), and bp_grad_sino,
  bp_h_grad_sino, proj_par_grad_sino (their backward is the forward);
- the CPU ray-driven transpose (edge ring): cpu bp, bp_nz, bp_wide, bp_h,
  fp_grad_vol, fp_rect_grad_vol;
- grad_angle (loose rule and kink average; the 0 deg view): fp_grad_angles;
- grad_rotmat at the identity pose of the golden's Euler table (kink
  average): fp3d_grad_euler, fp3d_shift_grad_euler, proj_rotmat_grad_mats,
  proj_rotmat_grad_mats_stale;
- every algorithm that projects with the parallel operator: sirt*, sart*,
  wsirt (not wsirt_cons: its support mask zeroes the outer ring, where the
  forward is unchanged bit for bit), fista*, gd, gd_loss, gd_sgd, gd_tv*,
  the CUDA sirt_proj*, fista_proj, and refine / refine_zyz (their volume is
  the SIRT output);
- frozen_golden_main.pt: the run_ad_recon keys ad, ad_reg, ad_staged (the
  parallel operator) and ad_pose (through grad_rotmat at its pure-tilt
  start). Not ad_support: its support mask zeroes the outer voxel ring,
  where the loose forward is bit-identical to the tight one, and the CUDA
  back projection did not change (test_frozen_identity checks it exactly).
Unchanged (checked): rotmat projection and back-projection keys, the CUDA
texture back projections, fbp*, wbp, _fft_*, mbir, *_euler,
sirt_rotmat_proj, gd_resire* (fp32-atomics noise only), GENFIRE / RESIRE.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import regenerate_golden_keys as R  # noqa: E402

ALGORITHM_KEYS = [
    "cpu/bp", "cpu/bp_grad_sino", "cpu/bp_h", "cpu/bp_h_grad_sino", "cpu/bp_nz",
    "cpu/bp_wide", "cpu/fista#0", "cpu/fista#1", "cpu/fista_L", "cpu/fista_h",
    "cpu/fista_tv#0", "cpu/fista_tv#1", "cpu/fista_tv_ch", "cpu/fp", "cpu/fp3d_grad_euler",
    "cpu/fp3d_shift_grad_euler", "cpu/fp_grad_angles", "cpu/fp_grad_vol", "cpu/fp_h",
    "cpu/fp_ndet", "cpu/fp_rect", "cpu/fp_rect_grad_vol", "cpu/gd", "cpu/gd_loss",
    "cpu/gd_sgd", "cpu/gd_tv", "cpu/gd_tv_loss", "cpu/refine", "cpu/refine_zyz",
    "cpu/sart", "cpu/sart_all", "cpu/sart_h", "cpu/sart_sub4#0", "cpu/sart_sub4#1",
    "cpu/sirt", "cpu/sirt_cons", "cpu/sirt_h", "cpu/sirt_loss#0", "cpu/sirt_loss#1",
    "cpu/sirt_support", "cpu/sirt_warm", "cpu/wsirt#0", "cpu/wsirt#1",
    "cuda/bp_grad_sino", "cuda/bp_h_grad_sino", "cuda/fista#0", "cuda/fista#1",
    "cuda/fista_L", "cuda/fista_h", "cuda/fista_proj", "cuda/fista_tv#0",
    "cuda/fista_tv#1", "cuda/fista_tv_ch", "cuda/fp", "cuda/fp3d_grad_euler",
    "cuda/fp3d_shift_grad_euler", "cuda/fp_grad_angles", "cuda/fp_h", "cuda/fp_ndet",
    "cuda/fp_rect", "cuda/gd", "cuda/gd_loss", "cuda/gd_sgd", "cuda/gd_tv",
    "cuda/gd_tv_loss", "cuda/proj_par_fp", "cuda/proj_par_fp_h", "cuda/proj_par_grad_sino",
    "cuda/proj_rotmat_grad_mats", "cuda/proj_rotmat_grad_mats_stale", "cuda/refine",
    "cuda/refine_zyz", "cuda/sart", "cuda/sart_all", "cuda/sart_h", "cuda/sart_sub4#0",
    "cuda/sart_sub4#1", "cuda/sirt", "cuda/sirt_cons", "cuda/sirt_h", "cuda/sirt_loss#0",
    "cuda/sirt_loss#1", "cuda/sirt_proj", "cuda/sirt_proj_again", "cuda/sirt_support",
    "cuda/sirt_warm", "cuda/wsirt#0", "cuda/wsirt#1",
]
AD_KEYS = ["ad", "ad_reg", "ad_pose", "ad_staged"]

if __name__ == "__main__":
    R.main(str(HERE / "algorithm_golden_main.pt"), ALGORITHM_KEYS)
    R.main(str(HERE / "frozen_golden_main.pt"), AD_KEYS)
