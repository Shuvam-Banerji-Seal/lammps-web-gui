import { oracleSuite } from './oracleHarness';

/* Oracle parity with native LAMMPS, shard 6 of 8 (tests/oracleHarness.ts). One case: npx vitest run tests/engineOracle -t <case> */
oracleSuite(5);
