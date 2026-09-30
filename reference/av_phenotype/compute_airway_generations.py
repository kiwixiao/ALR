# Compute anatomical generations on the AIRWAY TEASAR centerline.
#
# Mirror of compute_centerline_generations.py but adapted for airways:
#   - No CIP particles → root = highest-radius bifurcation vertex of LARGEST
#     component (effectively the trachea/main-bronchus carina)
#   - For each smaller component: root = highest-radius vertex within it (so
#     orphaned distal subtrees still get a generation, unlike the vessel
#     script which leaves them at gen=-1 awaiting particle-graph propagation
#     that the airway side does not have).
#   - BFS from each root through the tree
#   - At every bifurcation (degree ≥ 3 vertex), gen += 1
#   - Single-child segments: gen unchanged (continuation)
#
# Adds 4 fields to the airway TEASAR VTK file:
#   Generation                   int   (0 = trachea trunk; ≥0 elsewhere; never -1)
#   Generation_Type              str   ('Trachea','Main_Bronchi',...,'Bronchioles')
#   Is_Root                      int   (1 if this vertex is the chosen root for its component)
#   Distance_From_Root_mm        float (cumulative graph distance from this vertex's root;
#                                       sentinel -1.0 for any unreachable vertex)
#
# inf/nan-sanitization at write time follows the vessel-side fix to avoid
# pyvista ASCII-writer corruption (phantom 'inf' field).

import os, time
from collections import defaultdict, deque
import numpy as np
import pyvista as pv

CASE = os.environ.get('CASE_ID', 'NL001')
CASE_DIR = os.environ.get(
    'CASE_DIR',
    f'/path/to/av_phenotype/final_analysis/{CASE}/{CASE}')
AIR_VTK = f'{CASE_DIR}/airway_results/lung_airways_teasar.vtk'

# Airway-specific generation labels.
GEN_NAMES = ['Trachea',           # 0
             'Main_Bronchi',      # 1
             'Lobar_Bronchi',     # 2
             'Segmental',         # 3
             'Subsegmental',      # 4
             'Small_Bronchi_5',   # 5
             'Small_Bronchi_6',   # 6
             'Bronchioles']       # 7

t0 = time.time()
def log(msg): print(f'[{time.time()-t0:6.1f}s] {msg}', flush=True)


def gen_name(g):
    if 0 <= g < len(GEN_NAMES):
        return GEN_NAMES[g]
    if g >= len(GEN_NAMES):
        return f'Beyond_Gen{g}'
    return 'Unknown'


def process(vtk_path):
    log(f'\n══════════════ AIRWAY  ({CASE}) ══════════════')
    cl = pv.read(vtk_path)
    points = np.asarray(cl.points)
    n_v = points.shape[0]
    radii = np.asarray(cl.point_data['Radius']) if 'Radius' in cl.point_data else np.ones(n_v)
    log(f'  vertices: {n_v}')

    # Edges from VTK lines (each row of cl.lines is [2, v0, v1] for vtkLine cells)
    lines = np.asarray(cl.lines).reshape(-1, 3)
    edges = lines[:, 1:3]
    log(f'  edges: {len(edges)}')

    adj = defaultdict(list)
    for u, v in edges:
        adj[int(u)].append(int(v))
        adj[int(v)].append(int(u))

    # Connected components
    visited = np.zeros(n_v, dtype=bool)
    components = []
    for start in range(n_v):
        if visited[start]: continue
        comp = []
        q = deque([start])
        visited[start] = True
        while q:
            u = q.popleft(); comp.append(u)
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    q.append(v)
        components.append(np.array(comp))
    components.sort(key=lambda c: -len(c))
    sizes = [len(c) for c in components]
    log(f'  connected components: {len(components)} (sizes: {sizes[:5]}{"..." if len(sizes)>5 else ""})')

    gen = np.full(n_v, -1, dtype=np.int32)
    is_root = np.zeros(n_v, dtype=np.int8)
    dist_from_root = np.full(n_v, np.inf, dtype=np.float32)

    # Pre-compute edge lengths once
    edge_len = {}
    for u, v in edges:
        l = float(np.linalg.norm(points[int(u)] - points[int(v)]))
        edge_len[(int(u), int(v))] = l
        edge_len[(int(v), int(u))] = l

    # Only label the LARGEST connected component. Smaller components are
    # disconnected fragments (the airway lumen mask sometimes breaks at
    # narrow points) and their local generation chain (a tiny blob's "gen
    # 0/1/2/3") is not anatomically the trachea/main/lobar/segmental — it
    # would contaminate per-generation analyses (BV, dysanapsis, etc.).
    # Vessel-side pipeline uses the same convention (gen = -1 for small
    # components). Vertices in those components retain gen = -1.
    SKIPPED = sum(len(c) for c in components[1:])
    if SKIPPED > 0:
        log(f'  skipping {len(components)-1} small components ({SKIPPED} verts) — '
            f'they keep gen = -1 (won\'t contaminate per-gen analyses)')

    for ci, comp in enumerate(components[:1]):     # only the largest
        comp_r = radii[comp]
        comp_deg = np.array([len(adj[int(v)]) for v in comp])

        # Largest component → highest-radius BIFURCATION (trachea/carina)
        # Smaller components → highest-radius vertex (may be deg-1 endpoint)
        if ci == 0:
            bifurc = comp_deg >= 3
            if bifurc.any():
                cand_r = np.where(bifurc, comp_r, -np.inf)
                local_root = int(np.argmax(cand_r))
                log(f'  comp {ci} (largest, {len(comp)} verts): '
                    f'root = highest-radius bifurcation, r={comp_r[local_root]:.2f} mm, '
                    f'deg={comp_deg[local_root]}')
            else:
                local_root = int(np.argmax(comp_r))
                log(f'  comp {ci} (largest, {len(comp)} verts): '
                    f'no bifurcation, fallback max-radius, r={comp_r[local_root]:.2f} mm')
        else:
            local_root = int(np.argmax(comp_r))
            log(f'  comp {ci} ({len(comp)} verts): '
                f'root = max-radius, r={comp_r[local_root]:.2f} mm')

        global_root = int(comp[local_root])
        is_root[global_root] = 1
        gen[global_root] = 0
        dist_from_root[global_root] = 0.0

        # BFS from root
        seen = {global_root}
        order = [global_root]
        parent = {global_root: -1}
        q = deque([global_root])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if v in seen: continue
                seen.add(v)
                parent[v] = u
                order.append(v)
                q.append(v)

        # Assign generations: increment on bifurcation parents (deg ≥ 3)
        for v in order[1:]:
            par = parent[v]
            par_deg = len(adj[par])
            gen[v] = gen[par] + 1 if par_deg >= 3 else gen[par]
            dist_from_root[v] = dist_from_root[par] + edge_len[(par, v)]

    # ─── Collapse to a SINGLE global root ─────────────────────────────────
    # Anatomically there is exactly one root (the carina/MPA-trunk). Per-
    # component BFS gave us a "local root" in every disconnected piece, but
    # only one of them is the real anatomic root. Keep Is_Root=1 only on
    # the global highest-radius BIFURCATION across all components (fall back
    # to highest-radius any-vertex if no component has a deg≥3 vertex).
    #
    # Generation labels remain valid on every component because BFS already
    # ran from each local root — only the Is_Root flag is being filtered.
    cand_idx = np.where(is_root == 1)[0]
    if len(cand_idx) > 1:
        cand_deg = np.array([len(adj[int(v)]) for v in cand_idx])
        cand_rad = radii[cand_idx]
        bif_mask = cand_deg >= 3
        if bif_mask.any():
            chosen = int(cand_idx[bif_mask][int(np.argmax(cand_rad[bif_mask]))])
        else:
            chosen = int(cand_idx[int(np.argmax(cand_rad))])
        is_root[:] = 0
        is_root[chosen] = 1
        log(f'  collapsed {len(cand_idx)} per-component roots → 1 global root '
            f'(vertex {chosen}, r={radii[chosen]:.2f} mm, deg={len(adj[chosen])})')

    # ─── Reporting ─────────────────────────────────────────────────────────
    max_g = int(gen.max())
    log(f'  generation distribution:')
    for g in range(max_g + 1):
        cnt = int((gen == g).sum())
        if cnt == 0: continue
        bar = '█' * (cnt // 50) if cnt > 50 else '·'
        log(f'    gen {g:2d} ({gen_name(g):16s}): {cnt:>5d}  {bar}')

    log('  mean radius per generation (should monotonically decrease):')
    for g in range(min(max_g + 1, 10)):
        cnt = int((gen == g).sum())
        if cnt == 0: continue
        r_mean = float(radii[gen == g].mean())
        r_max  = float(radii[gen == g].max())
        log(f'    gen {g} ({gen_name(g):16s}): n={cnt:>5d}  '
            f'mean_r={r_mean:.2f}  max_r={r_max:.2f} mm')

    # ─── Save (with inf sanitization) ─────────────────────────────────────
    dist_from_root = np.where(np.isfinite(dist_from_root), dist_from_root, -1.0).astype(np.float32)
    cl.point_data['Generation']             = gen
    cl.point_data['Is_Root']                = is_root.astype(np.int32)
    cl.point_data['Distance_From_Root_mm']  = dist_from_root
    gen_type_arr = np.array([gen_name(int(g)) for g in gen], dtype='U18')
    cl.point_data['Generation_Type']        = gen_type_arr
    cl.save(vtk_path, binary=False)
    log(f'  updated {vtk_path}')


if __name__ == '__main__':
    if not os.path.exists(AIR_VTK):
        print(f'  airway VTK not found at {AIR_VTK}; skipping airway generations.')
    else:
        process(AIR_VTK)
    print('\n[Done]')
