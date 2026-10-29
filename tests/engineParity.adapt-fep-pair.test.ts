import { paritySuite } from './parityHarness';

/* Parity with native LAMMPS (tests/parityHarness.ts): fix adapt/fep 0 pair <style> <name> * * v_x scale yes (x = 1.1) for the first per-type-pair and the first global name of each pair style's native extract(), pe, press and fnorm compared after run 0. */
paritySuite('adapt-fep-pair');
