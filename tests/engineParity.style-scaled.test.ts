import { paritySuite } from './parityHarness';

/* Parity with native LAMMPS (tests/parityHarness.ts): the first number after the style word of each pair, bond, angle, dihedral, improper and kspace style line multiplied by 1.1, pe, press and fnorm compared after run 0. */
paritySuite('style-scaled');
