#!/usr/bin/env python3
"""Generate the bundled demo trajectory: a small Lennard-Jones crystal melting.

Independent, first-principles MD (velocity Verlet, truncated LJ, minimum image)
written for this project — not derived from LAMMPS source. The setup mirrors
the documented `examples/melt` input (fcc lattice at reduced density 0.8442,
initial temperature 3.0, cutoff 2.5, NVE) but at 108 atoms so the file stays
small enough to ship in the bundle.

Output is a LAMMPS-format dump with wrapped coordinates, image flags and
velocities (`id type element x y z ix iy iz vx vy vz`), so the viewer can:
  - render wrapped positions,
  - compute an EXACT mean-squared displacement from the image flags,
  - draw a speed histogram from the velocities.

Usage:  python3 scripts/generate-lj-demo.py > public/examples/lj-melt.lammpstrj
Requires numpy. Deterministic: fixed seed.
"""
import sys

import numpy as np

RHO = 0.8442          # reduced density (as in examples/melt)
CELLS = 3             # 3x3x3 fcc unit cells -> 108 atoms
T0 = 3.0              # initial temperature (as in examples/melt)
RC = 2.5              # LJ cutoff
DT = 0.005            # LAMMPS default timestep for lj units
STEPS = 2000
EVERY = 40            # dump interval -> 51 frames
SEED = 87287          # same seed value as examples/melt (our RNG differs)


def fcc_lattice(cells: int, rho: float):
    a = (4.0 / rho) ** (1.0 / 3.0)
    basis = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
    pts = [(np.array([i, j, k]) + b) * a
           for i in range(cells) for j in range(cells) for k in range(cells)
           for b in basis]
    return np.array(pts), a * cells


def forces(x: np.ndarray, L: float):
    """Truncated (unshifted) LJ forces, potential energy and virial."""
    d = x[:, None, :] - x[None, :, :]
    d -= L * np.round(d / L)                         # minimum image
    r2 = np.einsum("ijk,ijk->ij", d, d)
    np.fill_diagonal(r2, np.inf)
    mask = r2 < RC * RC
    inv2 = np.where(mask, 1.0 / r2, 0.0)
    inv6 = inv2 ** 3
    fmag = 24.0 * inv2 * inv6 * (2.0 * inv6 - 1.0)   # F·r / r^2
    f = np.einsum("ij,ijk->ik", fmag, d)
    pe = 0.5 * np.sum(np.where(mask, 4.0 * inv6 * (inv6 - 1.0), 0.0))
    # sum over pairs of r·F; np.where, not `* mask`, because the diagonal of
    # r2 is +inf and 0 * inf is NaN
    virial = 0.5 * np.sum(np.where(mask, fmag * np.where(mask, r2, 0.0), 0.0))
    return f, pe, virial


def main() -> None:
    rng = np.random.default_rng(SEED)
    x, L = fcc_lattice(CELLS, RHO)
    n = len(x)
    if L < 2 * RC:
        raise SystemExit(f"box {L:.3f} too small for cutoff {RC} (minimum image)")

    v = rng.normal(size=(n, 3))
    v -= v.mean(axis=0)                               # zero total momentum
    dof = 3 * n - 3
    v *= np.sqrt(T0 * dof / np.sum(v * v))           # exact initial temperature

    image = np.zeros((n, 3), dtype=int)
    f, pe, vir = forces(x, L)
    e0 = None
    out = sys.stdout
    stats = []

    for step in range(STEPS + 1):
        if step % EVERY == 0:
            ke = 0.5 * np.sum(v * v)
            temp = 2.0 * ke / dof
            etot = (pe + ke) / n
            e0 = etot if e0 is None else e0
            # LAMMPS convention (verified against log.8Apr21.melt): the kinetic
            # term uses the temperature's degrees of freedom, dof*T/3, not N*T.
            press = (dof * temp / 3.0 + vir / 3.0) / L ** 3
            stats.append((step, temp, pe / n, etot, press))
            out.write(f"ITEM: TIMESTEP\n{step}\nITEM: NUMBER OF ATOMS\n{n}\n")
            out.write("ITEM: BOX BOUNDS pp pp pp\n")
            for _ in range(3):
                out.write(f"0.0000000000000000e+00 {L:.16e}\n")
            out.write("ITEM: ATOMS id type element x y z ix iy iz vx vy vz\n")
            for i in range(n):
                p, im, vv = x[i], image[i], v[i]
                out.write(f"{i + 1} 1 Ar {p[0]:.5f} {p[1]:.5f} {p[2]:.5f} "
                          f"{im[0]} {im[1]} {im[2]} "
                          f"{vv[0]:.4f} {vv[1]:.4f} {vv[2]:.4f}\n")
        if step == STEPS:
            break
        # velocity Verlet (unit mass)
        v += 0.5 * DT * f
        x += DT * v
        shift = np.floor(x / L).astype(int)          # wrap + track images
        image += shift
        x -= shift * L
        f, pe, vir = forces(x, L)
        v += 0.5 * DT * f

    drift = stats[-1][3] - stats[0][3]
    print(f"# {n} atoms, L = {L:.6f}, {len(stats)} frames", file=sys.stderr)
    print("#  step    temp      pe/N     etot/N     press", file=sys.stderr)
    for s in stats[:: max(1, len(stats) // 10)]:
        print("# %5d %8.4f %9.4f %10.5f %9.4f" % s, file=sys.stderr)
    print(f"# energy drift over the run: {drift:+.2e} per atom", file=sys.stderr)


if __name__ == "__main__":
    main()
