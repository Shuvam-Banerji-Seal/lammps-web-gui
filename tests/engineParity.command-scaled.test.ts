import { paritySuite } from './parityHarness';

/* Parity with native LAMMPS (tests/parityHarness.ts): the first number of each other command (from word 1) multiplied by 1.1, pe, press and fnorm compared after run 0. */
paritySuite('command-scaled');
