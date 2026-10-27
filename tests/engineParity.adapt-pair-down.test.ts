import { paritySuite } from './parityHarness';

/* Parity with native LAMMPS (tests/parityHarness.ts): fix adapt 0 pair <style> <name> * * v_x scale yes reset no (x = 0.9) for each cutoff and geometry name of each pair style's native extract(), pe, press and fnorm compared after run 0 and after a second run 0. */
paritySuite('adapt-pair-down');
