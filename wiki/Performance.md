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

## v2.2 additions

| Mechanism | Effect |
|---|---|
| Web Worker parsing | File text is parsed off the main thread — the UI stays interactive while 100k-atom files process. Falls back to synchronous parsing when Workers are unavailable. |
| Split matrix/color effects | Instance matrices rewrite only when positions or radius inputs change; toggling materials/labels/lighting costs zero per-atom work. |
| `computeBoundingSphere()` | Both instanced meshes derive true bounds from instance placements, so frustum culling stays correct when zoomed in. |
| FPS-adaptive DPR (PerformanceMonitor) | Sustained frame drops lower the device-pixel-ratio ceiling by up to 40%, recovering automatically. |
| Hover picking guard | Per-move raycasting switches off beyond 50,000 atoms; orbit/zoom remain unaffected. |
| No `preserveDrawingBuffer` | Screenshots force an explicit render before capture instead of keeping the drawing buffer alive every frame. |

## v3.5 additions

| Mechanism | Effect |
|---|---|
| Cell-list RDF | `g(r)` bins pairs through a periodic cell list instead of an all-pairs loop. Measured on a fixed-density gas: **7x faster at 10 000 atoms**, **36x at 30 000** (36.9 s → 1.0 s). The accelerated histogram is asserted **bin-for-bin identical** to a brute-force reference, including the awkward 1-cell and 2-cell wrap cases and thin-z 2D slabs. |
| Analysis Web Worker | RDF, MSD and the density profile run off the main thread. They used to be called **inline in the Analysis panel's JSX**, so every React render recomputed all of them — up to 30x/second with a trajectory playing. Now: once per loaded structure, with stale results discarded by request generation. |
| Allocation-free MSD | A stable atom ordering is resolved once instead of rebuilding an `id → atom` Map inside the (lag x origin) loop. A permutation array is allocated only for frames whose atom order actually differs — for a normal LAMMPS dump, none. |
| Direct instance-matrix writes | An atom instance is a translation plus a uniform scale, so its 4x4 is written straight into the `InstancedBufferAttribute`: no `Object3D.updateMatrix()` quaternion compose, no `setMatrixAt` copy per atom. Unit-tested against `THREE.Object3D` for random inputs so the shortcut is provably identical. |
| Typed-array bond buffers | Bonds no longer allocate 2 `Vector3` + 1 `Quaternion` per half-bond — 300k short-lived objects for a 50k-bond structure, previously rebuilt whenever `atomMap` changed identity. |
| Per-type colour cache | Both instanced meshes cached one `THREE.Color` per atom **type** instead of calling `color.set(hexString)` once per instance, which re-parsed the same handful of strings tens of thousands of times per frame. |
| Geometry disposal | The sphere and cylinder geometries are `useMemo`'d and replaced when the tessellation tier changes. R3F only disposes what it builds from `args`, so each differently sized load used to leak the previous GPU buffers. |
| Lazy `atomMap` | Only built when something needs id lookup (bonds, or a measurement selection). A LAMMPS dump has no bonds, so playback was doing 60k `Map` inserts per frame for nothing. |
| Stable camera radius | The framing radius is keyed on the simulation box, not on `data` identity. Playback previously rescanned every atom per frame and handed the camera a slightly different radius each time — the view visibly pumped. |
| No argument spreading | `computeSpeedDistribution` / `computeDensityProfile` used `Math.min(...atoms.map(...))`, which throws `RangeError` past the engine's argument limit — the bundled 60k example was already near it. Both now track their extent in one pass; regression-tested at 200 000 atoms. |

## Bundle budget

CI fails if gzipped JS exceeds **420 KB total** or any single chunk exceeds
**300 KB** (`scripts/check-bundle-size.mjs`). Current usage: ~392 KB total
across 6 chunks, including two Web Workers.
