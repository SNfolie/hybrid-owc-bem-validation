"""Mesh convergence of the isolated OWC (torus + skirt, piston, plate), Capytaine 3.0.

Same geometry and panel rules as owc_capy.py (final mesh: 0.25 m on the inner wall and skirt,
0.35 m on the rest of the torus, 0.3 m on the piston, 0.5 m on the plate, 96 azimuthal
divisions). Every panel size and the number of azimuthal divisions are scaled by s, so the
panel count goes as 1/s^2. For each mesh and each formulation (indirect, direct with the
Gauss-Legendre 2 rule) the script prints B/B_H of each heave mode (Haskind relation of the
paper), the piston added mass and the piston excitation, and writes everything to a CSV.

Run it in the folder that holds three_body.py, two_body.py and donut_mesh.py:
    python convergence_owc.py                       # s = 2, 1.41, 1, 0.71, 0.5 ; 5 frequencies
    python convergence_owc.py --scales 1,0.71 --freqs 0.19
The finest mesh (s = 0.5) has about 4 times the 14 496 panels of the final mesh.
"""
import argparse, csv, time
import numpy as np
import capytaine as cpt
import logging
from three_body import ring_lid_profile_fine, _discretize, _mesh_from
from two_body import disk_profile

logging.getLogger('capytaine').setLevel(logging.ERROR)
rho, g, hw = 1025.0, 9.80665, 45.0

ap = argparse.ArgumentParser()
ap.add_argument('--scales', default='2,1.41,1,0.71,0.5')
ap.add_argument('--freqs', default='0.10,0.15,0.19,0.25,0.30')
ap.add_argument('--methods', default='indirect,direct')
ap.add_argument('--out', default='convergence_owc.csv')
args = ap.parse_args()
scales = [float(x) for x in args.scales.split(',')]
freqs = [float(x) for x in args.freqs.split(',')]
methods = args.methods.split(',')


def owc_body(s):
    """Isolated OWC mesh with every panel size times s (s = 1: final mesh of the paper)."""
    nth = int(6 * round(96 / s / 6))                       # keep a multiple of 6
    ring = ring_lid_profile_fine(0.0, 0.35 * s, 0.2 * s, 0.25 * s, z0=0.0)
    pist = np.array(_discretize([((0.0, -0.2), (7.58, -0.2)), ((7.58, -0.2), (7.58, 0.0))], [0.3 * s, 0.1 * s]))
    mesh = _mesh_from([ring, pist, disk_profile(13.75, ds=0.5 * s)], nth)
    c = mesh.faces_centers
    r, z = np.hypot(c[:, 0], c[:, 1]), c[:, 2]
    disk = z < -20
    pis = (r < 7.63) & (z > -0.2 - 1e-6)
    rg = ~(disk | pis)
    ez = np.zeros_like(c); ez[:, 2] = 1.0
    return mesh, nth, {'disque': ez * disk[:, None], 'donut': ez * rg[:, None], 'piston': ez * pis[:, None]}


def cg(w, k):
    return 0.5 * w / k * (1 + 2 * k * hw / np.sinh(2 * k * hw))


rows = []
for s in scales:
    mesh0, nth, dofs = owc_body(s)
    for meth in methods:
        mesh = mesh0.with_quadrature('Gauss-Legendre 2') if meth == 'direct' else mesh0
        body = cpt.FloatingBody(mesh=mesh, dofs=dofs, name='owc')
        solver = cpt.BEMSolver(method=meth)
        D = list(body.dofs)
        t0 = time.time()
        for f in freqs:
            kw = dict(omega=2 * np.pi * f, water_depth=hw, rho=rho, g=g)
            rad = {d: solver.solve(cpt.RadiationProblem(body=body, radiating_dof=d, **kw), keep_details=False) for d in D}
            dif = solver.solve(cpt.DiffractionProblem(body=body, wave_direction=0.0, **kw), keep_details=False)
            FK = cpt.bem.airy_waves.froude_krylov_force(dif)
            k = rad[D[0]].wavenumber
            w = 2 * np.pi * f
            for d in D:
                F = dif.forces[d] + FK[d]
                BH = k / (4 * rho * g * cg(w, k)) * abs(F) ** 2
                B = rad[d].radiation_dampings[d]
                rows.append(dict(scale=s, nth=nth, panels=mesh0.nb_faces, method=meth, f_Hz=f, dof=d,
                                 A=rad[d].added_masses[d], B=B, B_H=BH, B_over_BH=B / BH, absF=abs(F)))
        dt = time.time() - t0
        sel = [r_ for r_ in rows if r_['scale'] == s and r_['method'] == meth and r_['dof'] == 'piston']
        print('s = %4.2f | %6d panels | %-8s | %6.0f s | piston B/B_H: %s' % (
            s, mesh0.nb_faces, meth, dt, '  '.join('%.3f@%.2fHz' % (r_['B_over_BH'], r_['f_Hz']) for r_ in sel)))
        with open(args.out, 'w', newline='') as fh:                # rewritten after each mesh
            wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
            wr.writeheader(); wr.writerows(rows)

print('\nPiston (lid) mode: B/B_H by mesh and formulation')
print('%6s %7s %-9s ' % ('s', 'panels', 'method') + ' '.join('%8.2f' % f for f in freqs))
for s in scales:
    for meth in methods:
        sel = {r_['f_Hz']: r_ for r_ in rows if r_['scale'] == s and r_['method'] == meth and r_['dof'] == 'piston'}
        if sel:
            n = next(iter(sel.values()))['panels']
            print('%6.2f %7d %-9s ' % (s, n, meth) + ' '.join('%8.3f' % sel[f]['B_over_BH'] for f in freqs))
print('\nCSV written to', args.out)
