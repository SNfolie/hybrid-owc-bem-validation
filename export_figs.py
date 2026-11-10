"""CSV data for the MATLAB figures 11-14 of the revision (drag on the plate + compressibility, nominal case).
fig11_drag.csv: regular waves, CWR of hybrid/OWC/floats (no drag, with compressibility; and drag at A = 0.37/0.75/1.38 m) and q.
fig12_drag.csv: RAOs of the hybrid at the optimum of the reference sea state, B_eq of that state.
fig13_drag.csv: mean power of the five configurations in the three sea states.
fig14_drag.csv: power of each float of the hybrid in the three sea states."""
import sys, json, csv
import numpy as np
from model import *
OUT = '../'   # the MATLAB folder
R = json.load(open('main.json')); G = json.load(open('regular.json'))
st = ['frequent', 'reference', 'energetic']; N = R['nominal']

# Fig. 11
T = np.array(G['T']); flo = np.array(G['flo'])
cols = ['T_s', 'CWR_flo', 'CWR_hyb_nodrag', 'CWR_owc_nodrag'] + ['CWR_hyb_A%03d' % round(100*A) for A in (0.37, 0.75, 1.38)] + \
       ['CWR_owc_A%03d' % round(100*A) for A in (0.37, 0.75, 1.38)] + ['q_A%03d' % round(100*A) for A in (0.37, 0.75, 1.38)] + ['q_nodrag']
with open(OUT + 'fig11_drag.csv', 'w', newline='') as fh:
    wr = csv.writer(fh); wr.writerow(cols)
    for i in range(T.size):
        row = [T[i], flo[i], G['hyb_nodrag_comp'][i], G['owc_nodrag_comp'][i]] + [G['hyb_A%.2f' % A][i] for A in (0.37, 0.75, 1.38)] + \
              [G['owc_A%.2f' % A][i] for A in (0.37, 0.75, 1.38)] + \
              [G['hyb_A%.2f' % A][i]/(G['owc_A%.2f' % A][i] + flo[i]) for A in (0.37, 0.75, 1.38)] + \
              [G['hyb_nodrag_comp'][i]/(G['owc_nodrag_comp'][i] + flo[i])]
        wr.writerow(['%.6g' % v for v in row])

# Fig. 12: RAOs at the reference optimum with the B_eq of the reference sea state
bl = load_blocs(); w = np.linspace(2*np.pi*0.05, 2*np.pi*0.35, 1501); k = wavenumber(w)
C = configs(hyd(bl, 'hyb'), hyd(bl, 'owc'), hyd(bl, 'flo'), w)
x = N['reference']; P = C['hyb']
Bp, Zt_ = pto(P, x['hyb_Bt'], x['hyb_Bth'], R['V_crest'])
q = solve(P, Bp, x['drag_hyb']['Beq'], -1j*w*zeta_plate(k))
with open(OUT + 'fig12_drag.csv', 'w', newline='') as fh:
    wr = csv.writer(fh); wr.writerow(['T_s', 'zp', 'zd', 'zw', 'zw_minus_zd'] + ['theta%d_deg' % a for a in range(1, 7)])
    for i in range(w.size):
        wr.writerow(['%.6g' % v for v in [2*np.pi/w[i], abs(q[0, i]), abs(q[1, i]), abs(q[2, i]), abs(q[2, i] - q[1, i])] +
                     [np.rad2deg(abs(q[3 + a, i])) for a in range(6)]])

# Fig. 13 and 14
with open(OUT + 'fig13_drag.csv', 'w', newline='') as fh:
    wr = csv.writer(fh); wr.writerow(['configuration'] + st)
    for c, lab in (('owc', 'Isolated OWC'), ('flo', 'Isolated floats'), ('opn', 'Floats on torus, open chamber'),
                   ('lck', 'Floats on torus, piston locked'), ('hyb', 'Hybrid')):
        wr.writerow([lab] + ['%.3f' % N[s][c] for s in st])
with open(OUT + 'fig14_drag.csv', 'w', newline='') as fh:
    wr = csv.writer(fh); wr.writerow(['float'] + st + ['OWC_share_' + s for s in st])
    for a in range(6):
        wr.writerow([a + 1] + ['%.3f' % N[s]['hyb_Pk'][a] for s in st] + ['%.3f' % N[s]['hyb_OWC_share'] for s in st])
print('written fig11-14 CSV to', OUT)
