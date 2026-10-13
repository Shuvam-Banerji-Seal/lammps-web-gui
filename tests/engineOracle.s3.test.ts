import { oracleSuite } from './oracleHarness';

/* Oracle parity with native LAMMPS, shard 3 of 4 (tests/oracleHarness.ts). One case: npx vitest run tests/engineOracle -t <case> */
oracleSuite(2);
