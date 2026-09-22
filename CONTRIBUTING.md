# Contributing to Molecule3D

Thank you for considering a contribution! This document describes how to set up
the project locally and the rules that keep contributions easy to review.

> **Licence note.** This project is source-available under an educational /
> non-commercial licence, not an open source licence. By opening a pull request
> you license your contribution to the author under [LICENSE §6](LICENSE) so it
> can ship as part of the project — including under future licence versions and
> under commercial licences granted to third parties. You keep your copyright
> and are credited in the history. If that is not acceptable to you, please open
> an issue describing the change instead of a PR.

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
npm test           # vitest suite (parsers, generator, validator, branching)
npm run typecheck  # tsc --noEmit
npm run build      # production build (tsc && vite build)
```

## Branch protection and the PR flow

`main` is protected by a repository ruleset. Direct pushes, force pushes and
branch deletion are blocked for everyone, and the history is required to be
linear. Every change therefore lands through a pull request:

```bash
git switch -c fix/<topic>
# … work, commit …
git push -u origin fix/<topic>
gh pr create --fill
```

A PR can merge when:

| Gate | What it means |
|---|---|
| `verify (node 22)` / `verify (node 24)` | typecheck, 250+ unit tests, `npm audit --audit-level=high`, production build and the bundle-size budget, on both active Node lines |
| `commit-conventions` | the **PR title** is a conventional commit — it becomes the squash commit subject |
| `dependency-review` | no new high-severity advisories, and no copyleft dependency that would conflict with this project's licence |
| `codeql (javascript-typescript)` | static analysis finds no new security or quality alerts |
| Code-owner review | [CODEOWNERS](.github/CODEOWNERS) requires the author's review; `src/lammps/`, the licence files and `.github/` are called out explicitly |
| Conversations resolved | every review thread is resolved |
| Branch up to date | required checks ran against the merge result, not a stale base |

Merges are **squash or rebase only** and the branch is deleted afterwards, so
`main` keeps one linear, conventionally-named commit per change.

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
8. **LAMMPS correctness** — anything under `src/lammps/` that touches a command
   grammar, a template or a validator rule must cite the `docs.lammps.org` page
   it came from, in a code comment, and the quoted sentence it relies on.
   Guessing a grammar from memory is the one thing that gets a PR closed
   outright: the scripts this tool emits are meant to run. Templates must lint
   clean — `tests/lammpsTemplates.test.ts` asserts zero errors and zero
   warnings from `validateScript()` for every shipped template.
9. **Attribution** — do not weaken, move or condition the author credit or the
   licence links; `tests/attribution.test.ts` guards them.

## Commit style

Conventional Commits, e.g.:

```
feat(parser): parse triclinic tilt factors
fix(pdb): resolve two-letter elements from padded atom names
docs(readme): add keyboard shortcut table
perf(bonds): spatial hash grid for bond inference
```

## Branch naming

`feat/<topic>`, `fix/<topic>`, `docs/<topic>`, `perf/<topic>`, `ci/<topic>`.

## Adding or changing a LAMMPS command

1. Open the command's page on <https://docs.lammps.org> and read its **Syntax**
   and **Restrictions** sections.
2. Add or correct the `CommandDef` in `src/lammps/catalog.ts`, with a
   `doc:` link and a `// [VERIFIED <date>] <quoted requirement>` comment.
3. If the change implies an ordering or reference rule, add it to
   `src/lammps/validate.ts` with the same citation, and add both a
   true-positive test and a check that it stays silent on the official corpus
   in `tests/lammpsValidate.test.ts`.
4. Round-trip it: `parseScript()` must recognise the line your `build()`
   emits (`tests/lammpsScriptParser.test.ts`).

## Filing issues

Use the issue templates. For parsing bugs attach a **minimal** snippet of the
file that fails (strip large structures down to the smallest failing case).

## Adding a file format

1. Create `services/<fmt>Parser.ts` exporting `(data: string) => MoleculeData`.
2. Resolve elements through `getAtomicNumberFromSymbol` so CPK colors work.
3. Wire extension + content sniffing into `services/fileParser.ts`.
4. Add tests under `tests/`, including a minimal valid fixture inline.
5. Update README format table + wiki.
