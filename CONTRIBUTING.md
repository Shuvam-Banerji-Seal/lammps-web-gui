# Contributing to Molecule3D

Thank you for considering a contribution! This document describes how to set up
the project locally and the rules that keep contributions easy to review.

## Code of Conduct

By participating you agree to abide by our
[Code of Conduct](CODE_OF_CONDUCT.md). Report unacceptable behavior via
[GitHub security advisories / private contact](SECURITY.md).

## Development setup

```bash
git clone https://github.com/Shuvam-Banerji-Seal/lammps-web-gui.git
cd lammps-web-gui
npm install        # Node >= 20, npm >= 10 recommended
npm run dev        # dev server at http://localhost:5173
npm test           # vitest suite (parsers, constants)
npm run typecheck  # tsc --noEmit
npm run build      # production build (tsc && vite build)
```

## How we review PRs

1. **Scope** — one logical change per PR. Refactors separate from features.
2. **Tests** — parser changes require tests. Bug fixes require a regression
   test that fails before the fix.
3. **Type safety** — `npm run typecheck` must pass with zero errors.
4. **Performance** — rendering paths must stay O(1) draw calls (instancing);
   parsing paths must not regress from O(n) to O(n²).
5. **No gradients** — UI follows a flat design system. Solid surfaces only.
6. **Accessibility** — interactive elements need titles/aria labels; keyboard
   shortcuts must keep working.
7. **Security** — no new runtime network dependencies (fonts/HDRs/CDN scripts)
   without discussion; `npm audit` must report zero high/critical issues.

## Commit style

Conventional Commits, e.g.:

```
feat(parser): parse triclinic tilt factors
fix(pdb): resolve two-letter elements from padded atom names
docs(readme): add keyboard shortcut table
perf(bonds): spatial hash grid for bond inference
```

## Branch naming

`feat/<topic>`, `fix/<topic>`, `docs/<topic>`, `perf/<topic>`.

## Filing issues

Use the issue templates. For parsing bugs attach a **minimal** snippet of the
file that fails (strip large structures down to the smallest failing case).

## Adding a file format

1. Create `services/<fmt>Parser.ts` exporting `(data: string) => MoleculeData`.
2. Resolve elements through `getAtomicNumberFromSymbol` so CPK colors work.
3. Wire extension + content sniffing into `services/fileParser.ts`.
4. Add tests under `tests/`, including a minimal valid fixture inline.
5. Update README format table + wiki.
