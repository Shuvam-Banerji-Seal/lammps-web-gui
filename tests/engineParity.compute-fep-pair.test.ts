import { paritySuite } from './parityHarness';

/* Parity with native LAMMPS (tests/parityHarness.ts): compute fep 300.0 pair <style> <name> * * v_d (d = 0.01) for the first per-type-pair name of each pair style's native extract(), pe, press and the energy difference c_[1] compared after run 0. */
paritySuite('compute-fep-pair');
