# Horton-Strahler order on the AIRWAY TEASAR centerline.
#
# Mirror of compute_strahler_order.py adapted for the airway tree.
# Requires that compute_airway_generations.py has already run, so the
# Is_Root field is present.
#
# Strahler rule (bottom-up, leaves first):
#   - leaves                                   → order 1
#   - vertex with at least 2 children of equal max order → max + 1
#   - vertex with children of different orders          → max child order
#
# Adds field:
#   Strahler_Order   int  (1 = leaf, N = trunk; -1 = unreachable from root)

import os, time
from collections import defaultdict, deque
import numpy as np
import pyvista as pv

CASE = os.environ.get('CASE_ID', 'NL001')
CASE_DIR = os.environ.get(
    'CASE_DIR',
    f'/path/to/av_phenotype/final_analysis/{CASE}/{CASE}')
AIR_VTK = f'{CASE_DIR}/airway_results/lung_airways_teasar.vtk'

t0 = time.time()
def log(msg): print(f'[{time.time()-t0:6.1f}s] {msg}', flush=True)


def process(vtk_path):
    log(f'\n══════════════ AIRWAY Strahler  ({CASE}) ══════════════')
    cl = pv.read(vtk_path)
    points = np.asarray(cl.points)
    n_v = points.shape[0]
    radii = np.asarray(cl.point_data['Radius']) if 'Radius' in cl.point_data else np.ones(n_v)
    if 'Is_Root' not in cl.point_data:
        log('  Is_Root field missing — run compute_airway_generations.py first. abort.')
        return
    is_root = np.asarray(cl.point_data['Is_Root']).astype(bool)
    lines = np.asarray(cl.lines).reshape(-1, 3)
    edges = lines[:, 1:3].astype(int)

    adj = defaultdict(list)
    for u, v in edges:
        adj[int(u)].append(int(v))
        adj[int(v)].append(int(u))

    # Find connected components, take largest as main
    visited = np.zeros(n_v, dtype=bool)
    comps = []
    for start in range(n_v):
        if visited[start]: continue
        c = []
        q = deque([start]); visited[start] = True
        while q:
            u = q.popleft(); c.append(u)
            for w in adj[u]:
                if not visited[w]:
                    visited[w] = True; q.append(w)
        comps.append(c)
    comps.sort(key=lambda c: -len(c))
    main = set(comps[0])
    log(f'  largest component: {len(main)} verts')

    roots_in_main = [int(i) for i in np.where(is_root)[0] if i in main]
    if not roots_in_main:
        log('  no root in main component, abort.')
        return
    root = roots_in_main[0]
    log(f'  root: vertex {root}, radius={radii[root]:.2f} mm')

    # BFS from root → predecessor tree
    pred = {root: -1}
    bfs_order = [root]
    q = deque([root])
    seen = {root}
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v in seen: continue
            seen.add(v); pred[v] = u
            bfs_order.append(v); q.append(v)
    log(f'  BFS reached {len(bfs_order)} verts')

    children = defaultdict(list)
    for v, par in pred.items():
        if par >= 0:
            children[par].append(v)

    # Strahler bottom-up
    strahler = np.full(n_v, -1, dtype=np.int32)
    for v in reversed(bfs_order):
        kids = children[v]
        if not kids:
            strahler[v] = 1
        else:
            kid_orders = [strahler[k] for k in kids]
            max_o = max(kid_orders)
            tied  = sum(1 for o in kid_orders if o == max_o)
            strahler[v] = max_o + 1 if tied >= 2 else max_o

    log(f'  Strahler order range: [{strahler[strahler >= 0].min()}, {strahler.max()}]')
    log('  distribution:')
    for s in range(1, int(strahler.max()) + 1):
        cnt = int((strahler == s).sum())
        if cnt == 0: continue
        bar = '█' * (cnt // 100) if cnt > 100 else '·'
        log(f'    Strahler {s:2d}: {cnt:>5d}  {bar}')

    log('  mean radius per Strahler order (should monotonically INCREASE toward root):')
    for s in range(1, int(strahler.max()) + 1):
        m = strahler == s
        if m.sum() == 0: continue
        log(f'    order {s}: n={int(m.sum()):>5d}  '
            f'mean_r={float(radii[m].mean()):5.2f}  max_r={float(radii[m].max()):5.2f} mm')

    cl.point_data['Strahler_Order'] = strahler
    cl.save(vtk_path, binary=False)
    log(f'  updated {vtk_path}')


if __name__ == '__main__':
    if not os.path.exists(AIR_VTK):
        print(f'  airway VTK not found at {AIR_VTK}; skipping airway Strahler.')
    else:
        process(AIR_VTK)
    print('\n[Done]')
