<!-- Thank you for contributing! Please read CONTRIBUTING.md first.

The PR TITLE must be a conventional commit (`type(scope): subject`) — it is
checked by CI and becomes the squash commit subject on main. -->

## What does this PR change?

<!-- One or two sentences. Reference issues with "Fixes #123". -->

## Type of change

- [ ] 🐛 Bug fix (non-breaking change that fixes an issue)
- [ ] ✨ New feature (non-breaking change that adds functionality)
- [ ] 💥 Breaking change (fix or feature that would cause existing behavior to change)
- [ ] ⚡ Performance improvement
- [ ] 📝 Documentation only
- [ ] 🔧 Build / CI / tooling
- [ ] 🧪 LAMMPS command grammar, template or validator rule

## Checklist

- [ ] `npm run typecheck` passes
- [ ] `npm test` passes (new tests added for parser changes)
- [ ] `npm run build` succeeds from a clean state
- [ ] `npm audit` reports no new high/critical vulnerabilities
- [ ] Rendering changes keep O(1) draw calls (instancing preserved)
- [ ] UI changes follow the flat design system (no gradients)
- [ ] Keyboard shortcuts still work; new ones documented in help overlay + README
- [ ] New file-format support includes tests + README/wiki table update

### If this touches `src/lammps/` (grammars, templates, validator)

- [ ] Every changed grammar cites the `docs.lammps.org` page it came from, in a
      code comment, with the sentence it relies on quoted
- [ ] `tests/lammpsTemplates.test.ts` still reports zero validator errors and
      zero warnings for every shipped template
- [ ] New validator rules have a true-positive test **and** stay silent on the
      official script corpus in `tests/lammpsValidate.test.ts`
- [ ] `parseScript()` round-trips whatever `build()` now emits

### Licence

- [ ] I have read [LICENSE §6](../LICENSE) and license this contribution to the
      author on those terms
- [ ] This PR does not weaken, move or hide the author attribution or the
      licence links (`tests/attribution.test.ts`)

## Screenshots

<!-- If the change is visual, attach before/after screenshots. -->
