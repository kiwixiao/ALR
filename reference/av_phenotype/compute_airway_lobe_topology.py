# Topology-based airway lobe labeling on the TEASAR airway centerline.
#
# Method (Tschirren 2002 / Lo 2010 style):
#   1. Carina = global root (deg ≥ 3, highest radius)
#   2. Carina has 3 neighbors. Sort by mean_z of their subtrees:
#        highest mean_z → trachea trunk (going superior)
#        other two      → main bronchi (LMB + RMB)
#   3. Sort the two main bronchi by mean_x → side_A (smaller x), side_B
#      Disambiguate L vs R by sampling the totalseg lobe mask at side_A's
#      first vertex. If the mask is unavailable, default LPS convention
#      (smaller x = patient's right).
#   4. For each main bronchus:
#        Walk its deg-2 chain to the first deg≥3 bifurcation (gen-2 split)
#        Compare children by mean_z of subtrees:
#            higher z → "upper" daughter
#            lower  z → "other" daughter
#   5. LEFT side:    upper → LUL,  other → LLL
#      RIGHT side:   upper → RUL,  other → bronchus intermedius (BI)
#                    Walk BI to its first deg≥3 bifurcation (gen-3 split):
#                        higher z → RML,  lower z → RLL
#   6. Propagate the lobe label to every descendant via BFS through the subtree.
#   7. All vertices not assigned → 0 = "extra_lobar"
#      (trachea + main bronchi + BI proximal trunk).
#
# Adds two fields to the airway VTK:
#   Lobe_Topology       int  0 / 10 / 11 / 12 / 13 / 14
#   Lobe_Topology_Name  str  'extra_lobar' / 'LUL' / 'LLL' / 'RUL' / 'RML' / 'RLL'
#
# This is COMPLEMENTARY to the mask-derived `Lobe` field — both coexist on
# the VTK so the renderer can pick whichever is more reliable per case.

import os
from collections import defaultdict, deque
import numpy as np
import pyvista as pv

CASE = os.environ.get('CASE_ID', 'NL001')
ROOT = os.environ.get(
    'CASE_DIR',
    f'/path/to/av_phenotype/final_analysis/{CASE}/{CASE}')
AIR_VTK = f'{ROOT}/airway_results/lung_airways_teasar.vtk'

LOBE_CANDIDATES = [
    f'{ROOT}/lobes_ml.nii.gz',
    f'{ROOT}/totalseg_merged/lung_lobes_multilabel.nii.gz',
]

LBL_EXTRA = 0
LBL_LUL, LBL_LLL = 10, 11
LBL_RUL, LBL_RML, LBL_RLL = 12, 13, 14
NAMES = {0: 'extra_lobar', 10: 'LUL', 11: 'LLL',
         12: 'RUL', 13: 'RML', 14: 'RLL'}


def bfs_subtree(adj, start, blocked):
    """Vertices reachable from start without crossing into `blocked`."""
    seen = {start}
    q = deque([start])
    out = []
    while q:
        v = q.popleft(); out.append(v)
        for w in adj[v]:
            if w in seen or w in blocked: continue
            seen.add(w); q.append(w)
    return out


def walk_chain_to_bifurcation(adj, deg, start, blocked):
    """Walk from `start` along deg-2 chain (avoiding `blocked`) until the next
    deg≥3 vertex. Returns (path, bifurc_idx_or_None)."""
    path = [start]
    if deg[start] >= 3:
        return path, start          # already at a bifurcation
    cur = start
    prev = next(iter(blocked))      # the carina (only thing in blocked initially)
    while True:
        nxts = [w for w in adj[cur] if w != prev]
        if not nxts:
            return path, None       # dead-end
        prev, cur = cur, nxts[0]
        path.append(cur)
        if deg[cur] >= 3:
            return path, cur
        if deg[cur] == 1:
            return path, None       # hit a leaf without bifurcating


def main():
    print(f'\n══════════════ topology-based airway lobe label  ({CASE}) ══════════════')
    if not os.path.exists(AIR_VTK):
        print(f'  airway VTK not found at {AIR_VTK}; skipping.')
        return

    cl = pv.read(AIR_VTK)
    pts = np.asarray(cl.points)
    deg = np.asarray(cl.point_data['Degree'])
    is_root = np.asarray(cl.point_data['Is_Root']).astype(bool)
    edges = np.asarray(cl.lines).reshape(-1, 3)[:, 1:3].astype(int)
    n = len(pts)

    adj = defaultdict(list)
    for u, v in edges:
        adj[int(u)].append(int(v)); adj[int(v)].append(int(u))

    if not is_root.any():
        print('  no Is_Root vertex; abort.'); return
    carina = int(np.where(is_root)[0][0])
    print(f'  carina @ vertex {carina}, pos = {tuple(pts[carina].round(1))}')

    # ── Step 1: classify carina's 3 children by mean z of subtree ──────────
    children = adj[carina]
    if len(children) < 3:
        print(f'  carina has {len(children)} children, expected ≥ 3. abort.')
        return
    cinfo = []
    for c in children:
        sub = bfs_subtree(adj, c, {carina})
        mz = float(pts[sub].mean(axis=0)[2])
        mx = float(pts[sub].mean(axis=0)[0])
        cinfo.append({'first': c, 'subtree': sub, 'mean_z': mz, 'mean_x': mx,
                      'size': len(sub)})

    # Highest mean z = trachea trunk (going up)
    cinfo.sort(key=lambda d: -d['mean_z'])
    trachea = cinfo[0]
    main_bronchi = cinfo[1:3]
    print(f'  trachea trunk: subtree of {trachea["size"]} verts, '
          f'mean z = {trachea["mean_z"]:.1f}')

    # Two main bronchi sorted by x
    main_bronchi.sort(key=lambda d: d['mean_x'])
    side_A, side_B = main_bronchi
    print(f'  main bronchus A: {side_A["size"]} verts, mean (x,z) = '
          f'({side_A["mean_x"]:.1f}, {side_A["mean_z"]:.1f})')
    print(f'  main bronchus B: {side_B["size"]} verts, mean (x,z) = '
          f'({side_B["mean_x"]:.1f}, {side_B["mean_z"]:.1f})')

    # ── Step 2: decide which side is L vs R by sampling the lobe mask ─────
    side_A_is_left = None
    lobe_path = next((p for p in LOBE_CANDIDATES if os.path.exists(p)), None)
    if lobe_path is not None:
        import nibabel as nib
        img = nib.load(lobe_path)
        arr = img.get_fdata().astype(np.int16)
        inv = np.linalg.inv(img.affine)
        # Sample mask at all of side_A's subtree, see how many vote left vs right
        sub_pts = pts[side_A['subtree']]
        homog = np.c_[sub_pts, np.ones(len(sub_pts))]
        vox = np.round((inv @ homog.T).T[:, :3]).astype(np.int64)
        sx, sy, sz = arr.shape
        vox[:, 0] = np.clip(vox[:, 0], 0, sx - 1)
        vox[:, 1] = np.clip(vox[:, 1], 0, sy - 1)
        vox[:, 2] = np.clip(vox[:, 2], 0, sz - 1)
        labels_at_sub = arr[vox[:, 0], vox[:, 1], vox[:, 2]]
        n_L = int(((labels_at_sub == 10) | (labels_at_sub == 11)).sum())
        n_R = int(((labels_at_sub == 12) | (labels_at_sub == 13) |
                   (labels_at_sub == 14)).sum())
        side_A_is_left = (n_L > n_R)
        print(f'  L/R disambiguation by mask vote on side A: '
              f'L = {n_L},  R = {n_R}  →  side_A is '
              f'{"LEFT" if side_A_is_left else "RIGHT"}')
    else:
        # Default LPS convention: smaller x = patient's right
        side_A_is_left = False
        print('  no lobe mask; defaulting to LPS (side_A = right, side_B = left)')

    if side_A_is_left:
        left_side, right_side = side_A, side_B
    else:
        left_side, right_side = side_B, side_A

    # ── Step 2.5: tag trachea-trunk vertices ──────────────────────────────
    # The carina has three immediate children. BFS marks ALL three as gen 1,
    # but anatomically only the two main bronchi (LMB + RMB) are gen 1; the
    # third child + its deg-2 chain going SUPERIOR is the trachea trunk
    # (anatomically still gen 0 / "above the carina"). Tag those vertices so
    # downstream analyses using gen 1 = "main bronchi" can exclude them.
    is_trachea = np.zeros(n, dtype=np.int8)
    for v in trachea['subtree']:
        is_trachea[int(v)] = 1
    print(f'  trachea-trunk vertices flagged: {int(is_trachea.sum())} '
          f'(highest-mean-z child of carina + its deg-2 chain)')

    # ── Step 3: assign lobe labels ─────────────────────────────────────────
    lobe = np.zeros(n, dtype=np.int32)

    def label_subtree(start, blocked, label):
        for v in bfs_subtree(adj, start, blocked):
            lobe[v] = label

    def cascade(side, is_right):
        first = side['first']
        path, bif = walk_chain_to_bifurcation(adj, deg, first, {carina})
        if bif is None:
            print(f'    side has no gen-2 bifurcation; cannot label')
            return
        # Children of the gen-2 bifurcation (excluding the way we came from)
        prev_on_path = path[-2] if len(path) >= 2 else first
        bif_children = [w for w in adj[bif] if w != prev_on_path]
        kid_info = []
        for c in bif_children:
            sub = bfs_subtree(adj, c, {bif})
            mz = float(pts[sub].mean(axis=0)[2])
            kid_info.append((c, sub, mz))
        if len(kid_info) < 2:
            print(f'    gen-2 bifurcation has <2 daughters; cannot label')
            return
        kid_info.sort(key=lambda t: -t[2])     # by mean z descending
        upper_first, upper_sub, _ = kid_info[0]
        other_first, other_sub, _ = kid_info[1]

        if is_right:
            # Upper = RUL; other = BI → walk to next bifurcation
            label_subtree(upper_first, {bif}, LBL_RUL)
            print(f'    RUL = subtree of {len(upper_sub)} verts off gen-2 bifurc')
            bi_path, bi_bif = walk_chain_to_bifurcation(adj, deg, other_first, {bif})
            if bi_bif is None:
                print('    BI has no gen-3 bifurcation; labeling whole BI subtree as RLL')
                label_subtree(other_first, {bif}, LBL_RLL)
                return
            bi_prev = bi_path[-2] if len(bi_path) >= 2 else other_first
            bi_children = [w for w in adj[bi_bif] if w != bi_prev]
            bk_info = []
            for c in bi_children:
                sub = bfs_subtree(adj, c, {bi_bif})
                mz = float(pts[sub].mean(axis=0)[2])
                bk_info.append((c, sub, mz))
            if len(bk_info) < 2:
                label_subtree(other_first, {bif}, LBL_RLL)
                return
            bk_info.sort(key=lambda t: -t[2])
            rml_first, rml_sub, _ = bk_info[0]
            rll_first, rll_sub, _ = bk_info[1]
            label_subtree(rml_first, {bi_bif}, LBL_RML)
            label_subtree(rll_first, {bi_bif}, LBL_RLL)
            print(f'    RML = subtree of {len(rml_sub)} verts')
            print(f'    RLL = subtree of {len(rll_sub)} verts')
        else:
            label_subtree(upper_first, {bif}, LBL_LUL)
            label_subtree(other_first, {bif}, LBL_LLL)
            print(f'    LUL = subtree of {len(upper_sub)} verts')
            print(f'    LLL = subtree of {len(other_sub)} verts')

    print('\n  → labeling LEFT side cascade')
    cascade(left_side,  is_right=False)
    print('\n  → labeling RIGHT side cascade')
    cascade(right_side, is_right=True)

    # Distribution
    print('\n  per-vertex lobe distribution:')
    uniq, cnt = np.unique(lobe, return_counts=True)
    for L, c in zip(uniq, cnt):
        name = NAMES.get(int(L), f'unknown_{int(L)}')
        print(f'    {int(L):>3}  {name:>11s}: {int(c):>4d}  ({100*c/n:5.1f}%)')

    # Save
    cl.point_data['Lobe_Topology'] = lobe
    cl.point_data['Lobe_Topology_Name'] = np.array(
        [NAMES.get(int(L), f'unknown_{int(L)}') for L in lobe], dtype='U12')
    cl.point_data['Is_Trachea'] = is_trachea.astype(np.int32)
    cl.save(AIR_VTK, binary=False)
    print(f'\n  updated {AIR_VTK}  '
          f'(added Lobe_Topology + Lobe_Topology_Name + Is_Trachea)')


if __name__ == '__main__':
    main()
    print('\n[Done]')
