# AGENTS.md — map for coding agents

Browser-only LAMMPS workbench: a branching script builder, a CMake compiler
helper, a WebGL structure/trajectory viewer and an MD notebook that runs a
documented subset of LAMMPS input in the browser. There is **no backend**;
everything runs client-side.

## Stack

- Node 24, npm. React 19.3 (pinned `~19.3.0`; `@react-three/fiber@9.8.1`
  accepts React `>=19 <19.4`, so React 19.4 needs a newer fiber first), three.js 0.185, @react-three/fiber 9, @react-three/drei 10,
  Tailwind CSS 4, Vite 8, TypeScript 7, Vitest 5 (jsdom environment).
- **Do not add dependencies.** Everything needed is installed.

## Commands

```bash
npm test             # vitest run — the whole suite (~2250 tests, a few minutes; ~400 are native-LAMMPS oracle cases)
npx vitest run tests/<file>.test.ts   # one file
npm run typecheck    # tsc --noEmit — must be clean
npm run build        # tsc && vite build
npm run check:size   # gzip budgets: 420 KB initial JS, 120 KB lazy (notebook UI, small workers), 400 KB engine worker
```

There is no linter configured; `typecheck` is the static gate.

### Native parity (oracle cases)

Engine behaviour is checked against native LAMMPS (`/home/roy/.local/bin/lmp`, run as a black box only, rule 7):
a case is `tests/oracle/<case>.in` (header lines `# oracle:`, optional `# oracle-inputs:` (files next to the
case), `# oracle-potentials:` (unmodified LAMMPS potential files from `third_party/lammps/potentials`) and
`# oracle-compare:`); `LMP=/home/roy/.local/bin/lmp node scripts/oracle/run-oracle.mjs <case>` writes the native
fixture `tests/fixtures/oracle/<case>.json`, and `npx vitest run tests/engineOracle.test.ts -t <case>` compares the
engine with it. Put `thermo_modify format float %.15g` after `thermo_style`. Measured native behaviour is written in
comments as "Measured with native LAMMPS (black box): ..." without quote marks; quote marks are for doc text only.

## Layout

| Path | What lives there |
|---|---|
| `src/lammps/catalog.ts` | Every Script Builder command (`CommandDef`: params + `build()` that emits the exact LAMMPS line) |
| `src/lammps/generator.ts` | Model → script text, branch resolution (`resolvePath`), flowchart graph |
| `src/lammps/model.ts` | Lane-aware edits for branching models |
| `src/lammps/validate.ts` | Script validator — every rule cites a docs.lammps.org page |
| `src/lammps/scriptParser.ts` | `in.*` script → model (import) |
| `src/lammps/compiler.ts` | CMake package/option catalog + build-script generation |
| `src/lammps/templates.ts` | Starter pipelines (must lint clean — tested) |
| `src/services/*Parser.ts` | Structure parsers: LAMMPS data (`parser.ts`), dump, XYZ, PDB, CIF |
| `src/services/trajectoryAnalysis.ts` | RDF (cell list), MSD, density, speeds |
| `src/services/instanceMatrix.ts` | Direct instanced-matrix writes |
| `src/components/workbench/` | The four modules: `ScriptBuilder`, `CompilerHelper`, `ViewerModule`, `Notebook` (lazy-loaded) |
| `src/engine/` | In-browser MD engine for the notebook: `types.ts` (contracts), `units`, `rng`, `lattice`, `pairs`, `cpu/forces.ts` (fp64 reference), `cpu/parallel.ts` + `cpu/rangeKernel.ts` (lj/cut-only threaded forces via `src/workers/force.worker.ts`, used without cross-origin isolation), `cpu/pairThreads.ts` + `cpu/pairThreadsCore.ts` + `cpu/threadedPairs.ts` (shared-memory threads for the general engine's pair term via `src/workers/pair.worker.ts`), `force/` (pair, bond, angle, dihedral, improper, kspace styles), `fix/`, `compute/`, `commands/`, `registry/` (style factories merged in `styles.ts`), `neighbor.ts`, `gpu/webgpuForces.ts` (WGSL forces) + `gpu/resident.ts` (whole NVE steps on the GPU), `integrate`, `velocity`, `observables`, `md` (run loop), `script` + `expr` (input parsing), `interpreter.ts` (LAMMPS subset), `fixes.ts`, `host`/`client`/`protocol` (worker plumbing), `view.ts` |
| `docs/design/notebook.md` | Notebook design: scope, supported subset, acceptance test (LAMMPS `examples/melt` log) |
| `src/components/*.tsx` | three.js scene pieces (instanced meshes, camera, box, labels) |
| `src/workers/` | Parser and analysis Web Workers |
| `tests/*.test.ts` | Vitest suites, one per module |
| `third_party/lammps/` | Unmodified LAMMPS potential files (GPL-2.0, own `LICENSE` and `README.md`; not under this project's licence). Data only: rule 7 still forbids LAMMPS source code |

## Rules that are enforced

1. **LAMMPS correctness.** Anything that changes a command grammar, a
   template or a validator rule must cite the `docs.lammps.org` page it comes
   from in a code comment, quoting the sentence it relies on. Do not write a
   grammar from memory.
2. **Every template lints clean** — `tests/lammpsTemplates.test.ts` asserts zero
   validator errors and warnings.
3. **Do not touch attribution or licensing**: `LICENSE`, `NOTICE`,
   `COMMERCIAL.md`, `CITATION.cff`, the header credit in `src/App.tsx`, or
   `tests/attribution.test.ts`. The project is source-available under an
   educational / non-commercial licence, not MIT.
4. **Do not touch `.github/`.**
5. No `Math.min(...arr)` / `Math.max(...arr)` on data-sized arrays — it throws
   `RangeError` past the engine's argument limit. Loop instead.
6. Rendering keeps O(1) draw calls (instancing). Do not add per-atom meshes.
7. **The engine contains no LAMMPS source code** (LAMMPS is GPL-2.0, this
   project is not). Write engine code from textbook physics and the
   documented behaviour on docs.lammps.org, citing the page as in rule 1.
   An unsupported command must be an `EngineError` naming it — never a
   silent no-op.

## Environment notes

- **Real browser checks (WebGL2 + WebGPU):** the default headless browsers
  here have neither, but a Chromium launched with SwiftShader flags has both.
  Run a node script like this (playwright-core is NOT a repo dependency —
  borrow the installed copy; do not add it to package.json):

  ```js
  const { chromium } = require('/store/shuvam/qiskit_fallfest/node_modules/playwright-core');
  const browser = await chromium.launch({
    executablePath: '/home/roy/.cache/ms-playwright/chromium-1243/chrome-linux64/chrome',
    headless: true,
    args: ['--enable-unsafe-webgpu', '--use-webgpu-adapter=swiftshader', '--ignore-gpu-blocklist'],
  });
  ```

  The page must be a secure context (`http://127.0.0.1:…`); on `about:blank`
  `navigator.gpu` is absent. SwiftShader proves correctness, not speed.
- **Serve the app** from the production build — `vite` dev crashes here with
  `ENOSPC` (inotify limit):
  `npm run build && ln -sfn . dist/lammps-web-gui && python3 -m http.server <port> --bind 127.0.0.1 --directory dist`,
  then open `http://127.0.0.1:<port>/lammps-web-gui/`. Use your own port; do not
  kill servers you did not start.
- jsdom ignores CSS media and container queries — measure layout in a real
  browser.

## Orchestrator-only

Agents working a scoped task: ignore this section. Commits, branches and
pushes are made by the orchestrator, never by a worker.
