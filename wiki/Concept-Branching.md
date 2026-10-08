# Concept branching

One flowchart, several competing ideas.

A molecular-dynamics study is rarely one script. You build an equilibration
sequence, then you want to know: NVT or NPT for production? `lj/cut` at 2.5 or
3.0? 50 000 steps or 200 000? Historically that meant duplicating the whole
pipeline per variant and keeping them in sync by hand.

Branching lets the shared prefix stay shared and only the divergent tail
differ.

## The model

| Term | Meaning |
|---|---|
| **Main line** | the trunk — the linear pipeline you started with |
| **Fork point** | the step a concept diverges *after* |
| **Concept** | an alternative list of steps anchored at a fork point |
| **Diverges** | the concept **replaces** everything after the fork (the default) |
| **Rejoins** | the main line **resumes** after the concept — a detour, not a replacement |

At each fork point exactly one path is taken: Main line, or one concept.
Several fork points can exist in one flowchart, and each is chosen
independently — so a fork after `velocity` that picks a thermostat composes
with a fork after `run` that picks what gets written out.

## Forking

Hover any card in the flowchart and click the branch icon, or use **Fork** in
the toolbar, or **Fork a concept here…** from any `connect` pill.

The new concept is seeded with a **copy of the rest of your pipeline**. That is
deliberate: you almost always want to *vary* what you already have, not start
from an empty tail. Edit the copy — change a fix, retune a parameter, add a
second run — and the main line is untouched.

Adding a concept with the **+** button on a fork row instead starts it empty,
for when the variant really is a different ending.

## Switching

The **Concepts** bar above the canvas and the **fork row** inside it both show
`Main line` plus a chip per concept. Click one and everything follows it at
once:

- the generated script (`Script` view, **Copy**, **Download**),
- the **Script check** diagnostics,
- the **SVG / PNG** export, which marks the fork with a decision diamond,
  rails the concept's cards in amber and names the taken concept in its
  subtitle.

Steps that a *diverging* concept cut off are not deleted — they stay visible,
ghosted, under **"not in this concept"**. Switch back to Main line and they
come straight back.

## Per-concept controls

| Control | What it does |
|---|---|
| name field | rename the concept (it appears on cards, chips and exports) |
| **diverges / rejoins** | toggle whether the main line resumes after it |
| **Tab** | flatten this concept into its own flowchart tab, for side-by-side comparison |
| **Promote** | this idea won — fold it into the main line and drop its rivals at that fork |
| 🗑 | delete the concept |

Deleting a *trunk* step that a concept forks after does not orphan the concept:
it re-anchors to the step before, or to the very start.

## Everything is undoable

Forking, switching, promoting, renaming and deleting all go through the normal
history — `Ctrl+Z` / `Ctrl+Shift+Z`. The whole multi-tab workspace, branches
included, is persisted to `localStorage`, so a reload brings your concepts back.

## Worked example

```
units lj · lattice fcc · region · create_box · create_atoms · mass
velocity · pair_style lj/cut · pair_coeff · neighbor · timestep
thermo_style · thermo · dump
                    │
              ┌─────┴─────┐  fork after dump
              │           │
         Main line    "NPT variant"
         fix nvt      fix npt
         run 50000    run 200000
                      write_data final.data
```

Switch chips to emit either script. `Promote` on **NPT variant** makes it the
main line; `Tab` gives you both as separate flowcharts to export next to each
other.

## See also

- [[Script Check]] — the validator that checks whichever concept is taken
- [[Keyboard Shortcuts]]
