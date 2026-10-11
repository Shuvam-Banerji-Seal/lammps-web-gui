# LAMMPS example inputs and benchmark data (third-party, GPL-2.0)

`examples/` and `bench/` are copies of the directories of the same names of the LAMMPS distribution:

- upstream: <https://github.com/lammps/lammps>
- commit: `b332f28` (the commit of the native binary used for the parity cases; `potentials/` here is the same commit)
- copied byte for byte, except:
  - native log files (`log.*`) and images and videos (`*.png`, `*.jpg`, `*.jpeg`, `*.gif`, `*.mp4`, `*.avi`, `*.ppm`,
    `*.mov`, `*.webm`) are left out;
  - symbolic links into `potentials/` are kept as links (they resolve to `third_party/lammps/potentials/`); every other
    link was replaced by a copy of the file it points to;
  - links that are already broken upstream, or that point into the upstream `cmake/` directory (plugin build scripts),
    are kept as they are and do not resolve here.

These files are part of LAMMPS and are distributed under the **GNU General Public License, version 2** (see `LICENSE` in
this directory). They are not covered by this project's licence. They are data and input scripts, read by the test
suite and by the notebook as example inputs; no LAMMPS source code is part of them.
