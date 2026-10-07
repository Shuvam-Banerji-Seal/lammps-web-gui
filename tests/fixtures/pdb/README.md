# Real PDB entries for parser tests

Unmodified downloads from the RCSB PDB (https://files.rcsb.org/download/),
except where noted. wwPDB archive data files are available under the
CC0 1.0 Universal licence (https://www.wwpdb.org/about/usage-policies).

| File | Entry | Why it is here |
|---|---|---|
| `1L2Y-models1-3.pdb` | 1L2Y, Trp-cage NMR ensemble — trimmed to models 1-3 | MODEL/ENDMDL must become trajectory frames, not stacked atoms |
| `3NIR.pdb` | 3NIR, crambin at ultra-high resolution | alternate locations (altLoc) must not duplicate atoms: 750 atoms |
| `1MBN.pdb` | 1MBN, myoglobin | heme iron must be Fe — also tested with the element columns removed |

Used by `tests/pdbRealEntries.test.ts`.
