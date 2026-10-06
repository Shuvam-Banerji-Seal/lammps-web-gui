# AGENTS.md — map for coding agents

Browser-only LAMMPS workbench: a branching script builder, a CMake compiler
helper and a WebGL structure/trajectory viewer. There is **no backend**;
everything runs client-side.

## Stack

- Node 24, npm. React 19.2 (pinned `~19.2.8` — `@react-three/fiber@9.7.0`
  rejects 19.3), three.js 0.185, @react-three/fiber 9, @react-three/drei 10,
  Tailwind CSS 4, Vite 8, TypeScript 7, Vitest 5 (jsdom environment).
- **Do not add dependencies.** Everything needed is installed.

## Commands

```bash
npm test             # vitest run — the whole suite (~310 tests, ~10 s)
npx vitest run tests/<file>.test.ts   # one file
npm run typecheck    # tsc --noEmit — must be clean
npm run build        # tsc && vite build
npm run check:size   # gzip bundle budget (420 KB total)
```

There is no linter configured; `typecheck` is the static gate.

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
| `src/components/workbench/` | The three modules: `ScriptBuilder`, `CompilerHelper`, `ViewerModule` |
| `src/components/*.tsx` | three.js scene pieces (instanced meshes, camera, box, labels) |
| `src/workers/` | Parser and analysis Web Workers |
| `tests/*.test.ts` | Vitest suites, one per module |

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
