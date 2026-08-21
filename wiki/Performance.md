# Performance Architecture

How Molecule3D stays smooth on large systems.

## Draw calls

| Object | Strategy | Cost |
|---|---|---|
| Atoms | single `THREE.InstancedMesh`, per-instance color+matrix | **1 draw call** |
| Bonds | single `InstancedMesh` of half-bond cylinders (2 instances/bond) | **1 draw call** |
| Labels | canvas-texture sprites, hard-capped at 400 | ≤400 |
| Box | one `LineSegments` geometry | 1 |

Legacy versions created one mesh per bond — a 10k-bond system paid 10k draw
calls. Everything is instanced now.

## Parsing

Bond inference for XYZ/CIF uses a uniform spatial hash grid: atoms are
bucketed into cubic cells sized to the maximum bond cutoff, and each atom
only tests the 27 adjacent buckets. Complexity drops from O(n²) all-pairs to
O(n·k). A 10k-atom water box infers bonds in well under 100 ms (covered by a
regression test).

## Rendering quality ladder

| System size | Sphere segments | Device pixel ratio | Shadows |
|---|---|---|---|
| ≤ 1 000 atoms | 32 | up to 2.0× | on |
| ≤ 10 000 | 20 | up to 1.5× | on |
| > 10 000 | 12 | up to 1.25× | off (>8k) |

Antialiasing switches off beyond 20k atoms; screenshots still work because the
canvas keeps `preserveDrawingBuffer`.

## No runtime network fetches

Lighting is pure three.js lights (no HDR environment files), labels use
canvas-drawn textures (no webfonts), styling ships in the bundle (no Tailwind
CDN). First paint depends only on this repo's own static assets.
