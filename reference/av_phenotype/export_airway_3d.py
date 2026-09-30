# Export the airway centerline of one case in three interactive 3-D formats:
#   - HTML  (vtk.js scene; rotate/zoom/pan in any browser; labels work)
#   - GLB   (color-coded geometry only; viewable in macOS Preview/Quick Look)
#   - 3D PDF (U3D embedded in PDF via LaTeX media9; viewable in Adobe Reader)
#
# Geometry includes:
#   - Tubes for the centerline, color-coded by Weibel Generation
#   - Big yellow sphere at the global root (Is_Root)
#   - At EVERY bifurcation (deg ≥ 3 vertex), a label of form "g{N}→{N+1}"
#     showing the parent gen and the daughter gen that opens at that
#     bifurcation. (Labels render in HTML and 3D PDF; GLB strips them.)

import os, subprocess, shutil, tempfile
import numpy as np
import pyvista as pv

CASE = os.environ.get('CASE_ID', 'NL001')
ROOT_DIR = os.environ.get(
    'CASE_DIR',
    f'/path/to/av_phenotype/final_analysis/{CASE}/{CASE}')
AIRWAY_DIR = f'{ROOT_DIR}/airway_results'
os.makedirs(AIRWAY_DIR, exist_ok=True)
AIR_VTK = f'{AIRWAY_DIR}/lung_airways_teasar.vtk'

OUT_HTML          = f'{AIRWAY_DIR}/{CASE}_airway_3d.html'           # primary (lobe-colored)
OUT_HTML_STRAHLER = f'{AIRWAY_DIR}/{CASE}_airway_3d_strahler.html'   # secondary (Strahler-colored)
OUT_GLB  = f'{AIRWAY_DIR}/{CASE}_airway_3d.glb'
OUT_PDF  = f'{AIRWAY_DIR}/{CASE}_airway_3d.pdf'

ROOT_SPHERE_MM = 12.0

# Plotly modebar config — high-res PNG snapshot via the camera button.
PLOTLY_CONFIG = {
    'displaylogo': False,
    'toImageButtonOptions': {
        'format': 'png',
        'filename': f'{CASE}_airway_3d',
        'scale': 3,            # ~300 DPI
        'width': 1280, 'height': 900,
    },
}

# Strahler-Order palette (1 = leaf cool, N = trunk warm).
STRAHLER_COLORS_AW = {
    1: '#3498db', 2: '#1abc9c', 3: '#27ae60',
    4: '#f1c40f', 5: '#e67e22', 6: '#e74c3c',
    7: '#9b59b6', 8: '#8e44ad',
}


# ─── Build one shared scene ────────────────────────────────────────────────
def build_scene():
    """Returns (tubes, root_sphere, labels_pos, labels_txt, gens_render).
    labels_pos / labels_txt are PER-BRANCH (one label per deg-2 chain),
    placed at each branch's midpoint vertex with text 'g{N}' showing the
    constant generation of that branch."""
    from collections import defaultdict, deque
    cl = pv.read(AIR_VTK)
    radii = np.asarray(cl.point_data['Radius'])
    gens  = np.asarray(cl.point_data['Generation'])
    deg   = np.asarray(cl.point_data['Degree'])
    is_root = np.asarray(cl.point_data['Is_Root']).astype(bool)
    pts   = np.asarray(cl.points)

    cl['Radius_clamped'] = np.clip(radii, 0.3, None)
    lines = np.asarray(cl.lines).reshape(-1, 3)[:, 1:3].astype(int)
    gens_per_edge = np.minimum(gens[lines[:, 0]], gens[lines[:, 1]])
    cl.cell_data['EdgeGen'] = gens_per_edge.astype(np.int32)
    tubes = cl.tube(radius=0.5, scalars='Radius_clamped',
                    radius_factor=4.0, n_sides=8)
    if 'EdgeGen' in tubes.cell_data:
        tubes.cell_data['Generation'] = tubes.cell_data['EdgeGen']
    gens_render = np.where(gens < 0, 0, gens).astype(np.int32)

    root_idx = int(np.where(is_root)[0][0]) if is_root.any() else int(np.argmax(radii))
    root_sphere = pv.Sphere(radius=ROOT_SPHERE_MM, center=tuple(pts[root_idx]))

    # ── Identify per-branch midpoints. A "branch" is a maximal contiguous
    # sequence of deg-2 vertices (with both endpoints adjacent to deg≠2
    # vertices). Plus any single edge directly connecting two deg≠2
    # vertices is its own degenerate branch.
    adj = defaultdict(list)
    for u, v in lines:
        adj[int(u)].append(int(v)); adj[int(v)].append(int(u))

    visited_v = set()
    labels_pos = []
    labels_txt = []
    # Walk every deg-2 vertex in a chain
    for start in range(len(pts)):
        if start in visited_v: continue
        if deg[start] != 2: continue
        chain = [start]; visited_v.add(start)
        # walk both directions
        for direction_start in adj[start]:
            cur, prev = direction_start, start
            while deg[cur] == 2 and cur not in visited_v:
                chain.append(cur); visited_v.add(cur)
                nxts = [n for n in adj[cur] if n != prev]
                if not nxts: break
                prev, cur = cur, nxts[0]
        # midpoint vertex (geometric middle of chain)
        mid = chain[len(chain) // 2]
        if gens[mid] < 0: continue
        labels_pos.append(pts[mid])
        labels_txt.append(f'g{int(gens[mid])}')
    # Also label any direct deg≠2 ↔ deg≠2 edges (rare, e.g., bifurcation
    # connecting straight to a leaf or to another bifurcation)
    for u, v in lines:
        if deg[int(u)] != 2 and deg[int(v)] != 2:
            g = int(min(gens[int(u)], gens[int(v)]))
            if g < 0: continue
            labels_pos.append(0.5 * (pts[int(u)] + pts[int(v)]))
            labels_txt.append(f'g{g}')
    print(f'  {len(labels_pos)} per-branch labels')
    return tubes, root_sphere, np.asarray(labels_pos), labels_txt, gens_render


# ─── HTML (Plotly 3D — colored by LOBE; gen labels billboarded) ────────────
def export_html(tubes_unused, root_sphere_unused, labels_pos, labels_txt, gens_render_unused):
    """Builds the airway scene in Plotly: one trace per LOBE so each lobar
    subtree wears a single color; gen-number labels remain placed at each
    branch midpoint as billboarded text."""
    import plotly.graph_objects as go
    cl = pv.read(AIR_VTK)
    pts = np.asarray(cl.points)
    radii = np.asarray(cl.point_data['Radius'])
    gens  = np.asarray(cl.point_data['Generation'])
    # Prefer the rule-based Lobe_Topology field (independent of lobe-mask
    # quality). Fall back to mask-derived Lobe, then to all-zeros.
    if 'Lobe_Topology' in cl.point_data:
        lobes = np.asarray(cl.point_data['Lobe_Topology'])
        lobe_source = 'topology'
    elif 'Lobe' in cl.point_data:
        lobes = np.asarray(cl.point_data['Lobe'])
        lobe_source = 'mask'
    else:
        lobes = np.zeros(len(pts), dtype=np.int32)
        lobe_source = 'none'
    is_root = np.asarray(cl.point_data['Is_Root']).astype(bool)
    edges = np.asarray(cl.lines).reshape(-1, 3)[:, 1:3].astype(int)

    # 5-lobe + extra-lobar palette (distinct hues, colorblind-aware)
    LOBE_COLORS = {
        0:  ('extra-lobar', '#7f8c8d'),  # gray (trachea + main bronchi)
        10: ('LUL',         '#3498db'),  # blue
        11: ('LLL',         '#1abc9c'),  # teal
        12: ('RUL',         '#e74c3c'),  # red
        13: ('RML',         '#e67e22'),  # orange
        14: ('RLL',         '#9b59b6'),  # purple
    }

    fig = go.Figure()
    for L, (name, color) in LOBE_COLORS.items():
        # Edges where the SMALLER-side gen vertex sits in this lobe
        # (matches the per-edge-color convention used in the static PNG).
        # Use min(lobe_u, lobe_v) so an edge straddling a lobe boundary
        # gets assigned to the lobe nearer the trunk.
        # Simpler: assign edge to lobe of either endpoint if both agree.
        m = (lobes[edges[:, 0]] == L) & (lobes[edges[:, 1]] == L)
        if not m.any(): continue
        sub = edges[m]
        xs, ys, zs = [], [], []
        for u, v in sub:
            xs += [pts[u, 0], pts[v, 0], None]
            ys += [pts[u, 1], pts[v, 1], None]
            zs += [pts[u, 2], pts[v, 2], None]
        # Tube width by median radius in this lobe (visually proportional)
        r_med = float(np.median(radii[lobes == L])) if (lobes == L).any() else 1.0
        width = max(2.0, min(8.0, r_med * 3.0))
        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs, mode='lines',
            line=dict(color=color, width=width),
            name=f'{name} (n={int((lobes == L).sum())})',
            hoverinfo='name', showlegend=True,
        ))

    # Edges crossing a lobe boundary: render in dark gray, no legend entry.
    m_cross = lobes[edges[:, 0]] != lobes[edges[:, 1]]
    if m_cross.any():
        sub = edges[m_cross]
        xs, ys, zs = [], [], []
        for u, v in sub:
            xs += [pts[u, 0], pts[v, 0], None]
            ys += [pts[u, 1], pts[v, 1], None]
            zs += [pts[u, 2], pts[v, 2], None]
        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs, mode='lines',
            line=dict(color='#34495e', width=2),
            name='lobe-boundary edges', hoverinfo='name',
            showlegend=True,
        ))

    # Root marker
    rt = np.where(is_root)[0]
    if len(rt) > 0:
        rp = pts[int(rt[0])]
        fig.add_trace(go.Scatter3d(
            x=[rp[0]], y=[rp[1]], z=[rp[2]], mode='markers',
            marker=dict(size=14, color='#f1c40f',
                        line=dict(color='black', width=1)),
            name='root (carina)', hoverinfo='name'))

    # Per-branch labels — mode='text' always billboards in Plotly 3D
    if len(labels_pos):
        lpos = np.asarray(labels_pos)
        fig.add_trace(go.Scatter3d(
            x=lpos[:, 0], y=lpos[:, 1], z=lpos[:, 2],
            mode='text', text=labels_txt,
            textposition='top center',
            textfont=dict(size=10, color='black', family='Arial'),
            name='gen labels', hoverinfo='text', showlegend=False,
        ))

    fig.update_layout(
        title=f'{CASE} — airway centerline by lobe ({lobe_source}-derived); '
              f'gen labels at branch midpoints',
        scene=dict(
            xaxis_title='x (mm)', yaxis_title='y (mm)', zaxis_title='z (mm)',
            aspectmode='data',
            bgcolor='white',
            xaxis=dict(showgrid=False, showbackground=False, zeroline=False),
            yaxis=dict(showgrid=False, showbackground=False, zeroline=False),
            zaxis=dict(showgrid=False, showbackground=False, zeroline=False),
            camera=dict(projection=dict(type='perspective')),
        ),
        paper_bgcolor='white', plot_bgcolor='white',
        margin=dict(l=0, r=0, t=40, b=0),
        height=900,
        # Three button rows:
        #  - projection: perspective vs orthographic
        #  - anatomic view: 3D / sagittal / coronal / axial
        #  - axes: show vs hide (when axes overlap at edge-on views)
        updatemenus=[
            dict(type='buttons', direction='right',
                 x=0.00, y=1.10, xanchor='left', yanchor='top', showactive=True,
                 buttons=[
                     dict(label='Perspective',
                          method='relayout',
                          args=[{'scene.camera.projection.type': 'perspective'}]),
                     dict(label='Orthographic',
                          method='relayout',
                          args=[{'scene.camera.projection.type': 'orthographic'}]),
                 ]),
            dict(type='buttons', direction='right',
                 x=0.00, y=1.04, xanchor='left', yanchor='top', showactive=True,
                 buttons=[
                     dict(label='3D',
                          method='relayout',
                          args=[{'scene.camera.eye': dict(x=1.4, y=1.4, z=1.0),
                                 'scene.camera.up':  dict(x=0,   y=0,   z=1)}]),
                     dict(label='Sagittal',
                          method='relayout',
                          args=[{'scene.camera.eye': dict(x=2.2, y=0,   z=0),
                                 'scene.camera.up':  dict(x=0,   y=0,   z=1)}]),
                     dict(label='Coronal',
                          method='relayout',
                          args=[{'scene.camera.eye': dict(x=0,   y=-2.2,z=0),
                                 'scene.camera.up':  dict(x=0,   y=0,   z=1)}]),
                     dict(label='Axial',
                          method='relayout',
                          args=[{'scene.camera.eye': dict(x=0,   y=0,   z=2.2),
                                 'scene.camera.up':  dict(x=0,   y=-1,  z=0)}]),
                 ]),
            dict(type='buttons', direction='right',
                 x=0.00, y=0.98, xanchor='left', yanchor='top', showactive=True,
                 buttons=[
                     dict(label='Axes ON',
                          method='relayout',
                          args=[{
                              'scene.xaxis.visible': True,
                              'scene.yaxis.visible': True,
                              'scene.zaxis.visible': True,
                              'scene.xaxis.title.text': 'x (mm)',
                              'scene.yaxis.title.text': 'y (mm)',
                              'scene.zaxis.title.text': 'z (mm)',
                          }]),
                     dict(label='Axes OFF',
                          method='relayout',
                          args=[{
                              'scene.xaxis.visible': False,
                              'scene.yaxis.visible': False,
                              'scene.zaxis.visible': False,
                              'scene.xaxis.title.text': '',
                              'scene.yaxis.title.text': '',
                              'scene.zaxis.title.text': '',
                          }]),
                 ]),
        ],
    )
    fig.write_html(OUT_HTML, include_plotlyjs='cdn', full_html=True,
                   config=PLOTLY_CONFIG)
    print(f'  wrote {OUT_HTML}  ({len(labels_txt)} billboarded gen labels)')

    # ── SECONDARY: Strahler-colored airway HTML ───────────────────────────
    if 'Strahler_Order' in cl.point_data:
        strahler = np.asarray(cl.point_data['Strahler_Order']).astype(np.int32)
        edges_arr = np.asarray(cl.lines).reshape(-1, 3)[:, 1:3].astype(int)
        radii_pts = np.asarray(cl.point_data['Radius'])
        s_levels = sorted(set(int(s) for s in np.unique(strahler) if s >= 1))
        fig2 = go.Figure()
        for s in s_levels:
            m = (strahler[edges_arr[:, 0]] == s) & (strahler[edges_arr[:, 1]] == s)
            if not m.any(): continue
            sub = edges_arr[m]
            xs, ys, zs = [], [], []
            for u, v in sub:
                xs += [pts[u, 0], pts[v, 0], None]
                ys += [pts[u, 1], pts[v, 1], None]
                zs += [pts[u, 2], pts[v, 2], None]
            col = STRAHLER_COLORS_AW.get(int(s), '#bdc3c7')
            r_med = float(np.median(radii_pts[strahler == s])) if (strahler == s).any() else 1.0
            width = max(2.0, min(8.0, r_med * 3.0))
            fig2.add_trace(go.Scatter3d(
                x=xs, y=ys, z=zs, mode='lines',
                line=dict(color=col, width=width),
                name=f'Strahler {s} (n={int((strahler == s).sum())})',
                hoverinfo='name'))
        # Root sphere
        rt = np.where(is_root)[0]
        if len(rt):
            rp = pts[int(rt[0])]
            fig2.add_trace(go.Scatter3d(
                x=[rp[0]], y=[rp[1]], z=[rp[2]], mode='markers',
                marker=dict(size=14, color='#f1c40f',
                            line=dict(color='black', width=1)),
                name='root', hoverinfo='name'))
        # Per-branch gen labels (same as primary)
        if len(labels_pos):
            lp = np.asarray(labels_pos)
            fig2.add_trace(go.Scatter3d(
                x=lp[:, 0], y=lp[:, 1], z=lp[:, 2],
                mode='text', text=labels_txt,
                textposition='top center',
                textfont=dict(size=10, color='black', family='Arial'),
                name='gen labels', hoverinfo='text', showlegend=False))
        fig2.update_layout(
            title=f'{CASE} — airway centerline by Strahler tier; '
                  f'gen labels at branch midpoints',
            scene=dict(
                xaxis_title='x (mm)', yaxis_title='y (mm)', zaxis_title='z (mm)',
                aspectmode='data',
                bgcolor='white',
                xaxis=dict(showgrid=False, showbackground=False, zeroline=False),
                yaxis=dict(showgrid=False, showbackground=False, zeroline=False),
                zaxis=dict(showgrid=False, showbackground=False, zeroline=False),
                camera=dict(projection=dict(type='perspective')),
            ),
            paper_bgcolor='white', plot_bgcolor='white',
            margin=dict(l=0, r=0, t=40, b=0), height=900,
            updatemenus=[
                dict(type='buttons', direction='right',
                     x=0.00, y=1.10, xanchor='left', yanchor='top', showactive=True,
                     buttons=[
                         dict(label='Perspective',
                              method='relayout',
                              args=[{'scene.camera.projection.type': 'perspective'}]),
                         dict(label='Orthographic',
                              method='relayout',
                              args=[{'scene.camera.projection.type': 'orthographic'}]),
                     ]),
                dict(type='buttons', direction='right',
                     x=0.00, y=1.04, xanchor='left', yanchor='top', showactive=True,
                     buttons=[
                         dict(label='3D',
                              method='relayout',
                              args=[{'scene.camera.eye': dict(x=1.4, y=1.4, z=1.0),
                                     'scene.camera.up':  dict(x=0,   y=0,   z=1)}]),
                         dict(label='Sagittal',
                              method='relayout',
                              args=[{'scene.camera.eye': dict(x=2.2, y=0,   z=0),
                                     'scene.camera.up':  dict(x=0,   y=0,   z=1)}]),
                         dict(label='Coronal',
                              method='relayout',
                              args=[{'scene.camera.eye': dict(x=0,   y=-2.2,z=0),
                                     'scene.camera.up':  dict(x=0,   y=0,   z=1)}]),
                         dict(label='Axial',
                              method='relayout',
                              args=[{'scene.camera.eye': dict(x=0,   y=0,   z=2.2),
                                     'scene.camera.up':  dict(x=0,   y=-1,  z=0)}]),
                     ]),
                dict(type='buttons', direction='right',
                     x=0.00, y=0.98, xanchor='left', yanchor='top', showactive=True,
                     buttons=[
                         dict(label='Axes ON',
                              method='relayout',
                              args=[{
                                  'scene.xaxis.visible': True,
                                  'scene.yaxis.visible': True,
                                  'scene.zaxis.visible': True,
                                  'scene.xaxis.title.text': 'x (mm)',
                                  'scene.yaxis.title.text': 'y (mm)',
                                  'scene.zaxis.title.text': 'z (mm)',
                              }]),
                         dict(label='Axes OFF',
                              method='relayout',
                              args=[{
                                  'scene.xaxis.visible': False,
                                  'scene.yaxis.visible': False,
                                  'scene.zaxis.visible': False,
                                  'scene.xaxis.title.text': '',
                                  'scene.yaxis.title.text': '',
                                  'scene.zaxis.title.text': '',
                              }]),
                     ]),
            ],
        )
        fig2.write_html(OUT_HTML_STRAHLER, include_plotlyjs='cdn',
                        full_html=True, config=PLOTLY_CONFIG)
        print(f'  wrote {OUT_HTML_STRAHLER}  (Strahler-tier colored)')


# ─── GLB (per-segment solid color + per-branch text-as-mesh labels) ────────
def export_glb(tubes, root_sphere, labels_pos, labels_txt):
    import trimesh, vtk
    import matplotlib.pyplot as plt

    # Tube surface — solid per-cell color via the EdgeGen scalar (already on cells)
    surf = tubes.extract_surface().triangulate()
    v = np.asarray(surf.points)
    faces = np.asarray(surf.faces).reshape(-1, 4)[:, 1:]
    cmap = plt.get_cmap('turbo')

    # Use cell-level colors and map to vertices via face lookup. To keep
    # segments solid (no Gouraud blending in viewers that interpolate
    # vertex colors), assign each FACE its cell color, then duplicate
    # vertices per face so each face's three verts share that color.
    g_per_face = np.asarray(surf['Generation'])
    n_gen = max(int(g_per_face.max()) + 1, 1)
    face_rgba = (cmap(g_per_face / n_gen) * 255).astype(np.uint8)

    # Duplicate verts per face → no shared verts, no color blending
    new_v = v[faces].reshape(-1, 3)             # 3 N x 3
    new_f = np.arange(len(new_v)).reshape(-1, 3)
    new_vc = np.repeat(face_rgba, 3, axis=0)
    tube_mesh = trimesh.Trimesh(vertices=new_v, faces=new_f,
                                vertex_colors=new_vc, process=False)

    # Root sphere
    rs = root_sphere.triangulate()
    rv = np.asarray(rs.points)
    rf = np.asarray(rs.faces).reshape(-1, 4)[:, 1:]
    sc = np.tile(np.array([241, 196, 15, 255], dtype=np.uint8), (len(rv), 1))
    sphere_mesh = trimesh.Trimesh(vertices=rv, faces=rf,
                                  vertex_colors=sc, process=False)

    # Text-as-mesh labels via vtkVectorText (each glyph rendered as triangles)
    # Then converted to trimesh and placed at each branch midpoint.
    TEXT_HEIGHT_MM = 4.0      # readable from default-zoom glTF preview
    TEXT_DEPTH_MM  = 0.5      # extrude slightly for solid look
    label_meshes = []
    for txt, pos in zip(labels_txt, labels_pos):
        src = vtk.vtkVectorText()
        src.SetText(txt)
        ext = vtk.vtkLinearExtrusionFilter()
        ext.SetInputConnection(src.GetOutputPort())
        ext.SetExtrusionTypeToVectorExtrusion()
        ext.SetVector(0.0, 0.0, 1.0)
        ext.SetScaleFactor(TEXT_DEPTH_MM)
        tri = vtk.vtkTriangleFilter()
        tri.SetInputConnection(ext.GetOutputPort())
        tri.Update()
        pd = pv.PolyData(tri.GetOutput())
        if pd.n_points == 0: continue
        pd = pd.scale(TEXT_HEIGHT_MM, inplace=False)
        bds = pd.bounds        # x_min,x_max,y_min,y_max,z_min,z_max
        cx = 0.5 * (bds[0] + bds[1]); cy = 0.5 * (bds[2] + bds[3])
        pd = pd.translate([-cx, -cy, 0.0], inplace=False)
        # Translate to branch position, slightly raised in y so labels sit
        # above the tube rather than inside it.
        pd = pd.translate([float(pos[0]),
                           float(pos[1]) - TEXT_HEIGHT_MM * 1.5,
                           float(pos[2])], inplace=False)
        pd = pd.triangulate()
        if pd.n_points == 0: continue
        tv = np.asarray(pd.points)
        tf = np.asarray(pd.faces).reshape(-1, 4)[:, 1:]
        tcol = np.tile(np.array([10, 10, 10, 255], dtype=np.uint8), (len(tv), 1))
        label_meshes.append(trimesh.Trimesh(vertices=tv, faces=tf,
                                            vertex_colors=tcol, process=False))

    scene = trimesh.Scene([tube_mesh, sphere_mesh] + label_meshes)
    scene.export(OUT_GLB)
    print(f'  wrote {OUT_GLB}  (with {len(label_meshes)} baked text labels)')


# ─── 3D PDF (U3D via PyMeshLab, embedded with LaTeX media9) ────────────────
def export_pdf(tubes, root_sphere):
    """Best-effort 3D PDF. If U3D export or LaTeX wrap fails, prints why."""
    import pymeshlab
    # Combine tubes + sphere into one OBJ for U3D conversion
    surf = tubes.extract_surface().triangulate() + root_sphere.triangulate()

    with tempfile.TemporaryDirectory() as tmp:
        obj_path = os.path.join(tmp, 'mesh.obj')
        u3d_path = os.path.join(tmp, 'mesh.u3d')
        surf.save(obj_path)

        try:
            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(obj_path)
            ms.save_current_mesh(u3d_path)
        except Exception as e:
            print(f'  U3D export FAILED: {type(e).__name__}: {e}')
            print('  3D-PDF requires MeshLab build with U3D plugin; recent macOS '
                  'pymeshlab wheels do not include it.')
            print('  Workaround: install Asymptote + LaTeX media9, or use '
                  'MeshLab 2016 manually for U3D conversion.')
            return False

        if not os.path.exists(u3d_path):
            print('  pymeshlab.save_current_mesh did not produce a .u3d file.')
            return False

        # LaTeX wrapper using media9 to embed the U3D.
        # Build via .replace() to avoid f-string brace-doubling issues.
        tex = (r"""
\documentclass{article}
\usepackage[a4paper, margin=1cm]{geometry}
\usepackage{media9}
\begin{document}
\begin{center}
{\Large @@CASE@@ --- airway centerline (interactive)}\\[2ex]
\textit{Open in Adobe Acrobat Reader; click on the figure; drag to rotate.}\\[1ex]
\includemedia[
    width=15cm, height=18cm,
    activate=onclick,
    3Dmenu
]{\fbox{Click to load 3D model}}{@@U3D@@}
\end{center}
\end{document}
""").replace('@@CASE@@', CASE).replace('@@U3D@@', u3d_path)
        tex_path = os.path.join(tmp, 'wrap.tex')
        with open(tex_path, 'w') as f: f.write(tex)
        try:
            r = subprocess.run(['pdflatex', '-interaction=nonstopmode',
                                '-output-directory', tmp, tex_path],
                               capture_output=True, timeout=60)
            if r.returncode != 0:
                print(f'  pdflatex failed (exit {r.returncode})')
                print(r.stdout.decode()[-1500:])
                return False
            shutil.copyfile(os.path.join(tmp, 'wrap.pdf'), OUT_PDF)
            print(f'  wrote {OUT_PDF}')
            return True
        except FileNotFoundError:
            print('  pdflatex not on PATH; skipping 3D PDF wrapper.')
            return False


def main():
    print(f'\n══════════════ 3-D HTML export  ({CASE}) ══════════════')
    if not os.path.exists(AIR_VTK):
        print(f'  airway VTK not found at {AIR_VTK}; skipping 3-D HTML export.')
        return
    tubes, root_sphere, labels_pos, labels_txt, gens_render = build_scene()
    print('\n[HTML]')
    export_html(tubes, root_sphere, labels_pos, labels_txt, gens_render)
    # GLB and 3D-PDF disabled per user request — Plotly HTML is the single
    # interactive output; labels billboard, no Adobe needed, opens in any
    # browser. Re-enable by uncommenting the calls if needed.
    # print('\n[GLB]'); export_glb(tubes, root_sphere, labels_pos, labels_txt)
    # print('\n[3D-PDF]'); export_pdf(tubes, root_sphere)


if __name__ == '__main__':
    main()
    print('\nDone.')
