"""Regular waves (Fig. 11 of the paper) with plate drag and compressibility at A = Hs/2 of the three sea
states, plus the no-drag curves. Writes regular.json and the CSV block for fig11 (curves_drag.csv)."""
import sys, json, time
import numpy as np
from model import *

t0 = time.time()
bl = load_blocs()
T = np.linspace(3.5, 20.0, 331); w = 2*np.pi/T[::-1]; k = wavenumber(w); cg = group_velocity(w, k)
Hh, Ho, Hf = hyd(bl, 'hyb'), hyd(bl, 'owc'), hyd(bl, 'flo')
C = configs(Hh, Ho, Hf, w)
tab = cd_table(); dr = Drag(k, tab)
V = np.pi*7.68**2*1.92
JA = 0.5*rho*g*cg                                     # flux per unit amplitude^2
out = dict(T=list(2*np.pi/w))
r = optimise_regular(C['flo'], 1.0); out['flo'] = list(r['P']/(JA*D))
for lab, drag, V0 in (('nodrag', None, np.inf), ('nodrag_comp', None, V)):
    for c in ('owc', 'hyb'):
        r = optimise_regular(C[c], 1.0, drag, V0); out['%s_%s' % (c, lab)] = list(r['P']/(JA*D))
    print(lab, '%.0f s' % (time.time() - t0)); sys.stdout.flush()
for A in (0.37, 0.75, 1.38):
    for c in ('owc', 'hyb'):
        r = optimise_regular(C[c], A, dr, V)
        out['%s_A%.2f' % (c, A)] = list(r['P']/(JA*D)); out['%s_A%.2f_KC' % (c, A)] = list(r['KC'])
        out['%s_A%.2f_conv' % (c, A)] = bool(np.all(r['conv']))
    print('A = %.2f m, %.0f s' % (A, time.time() - t0)); sys.stdout.flush()
json.dump(out, open('regular.json', 'w'))

# summary: peaks and q
Tm = np.array(out['T']); flo = np.array(out['flo'])
def peaks(y, lo, hi):
    m = (Tm >= lo) & (Tm <= hi); i = np.argmax(np.where(m, y, -1)); return Tm[i], y[i]
for key in ['hyb_nodrag', 'hyb_nodrag_comp', 'hyb_A0.37', 'hyb_A0.75', 'hyb_A1.38', 'owc_nodrag', 'owc_A0.75']:
    y = np.array(out[key])
    print('%-16s peak <7 s: %.3f at %.2f s | 10-14 s: %.3f at %.2f s' % ((key,) + peaks(y, 3.5, 7)[::-1] + peaks(y, 10, 14)[::-1]))
for A in (0.37, 0.75, 1.38):
    q = np.array(out['hyb_A%.2f' % A])/(np.array(out['owc_A%.2f' % A]) + flo)
    kc = np.array(out['hyb_A%.2f_KC' % A])
    print('A %.2f: q max %.2f at %.2f s, KC range %.3f-%.3f, conv %s' % (A, q.max(), Tm[np.argmax(q)], np.nanmin(kc), np.nanmax(kc), out['hyb_A%.2f_conv' % A]))
print('total %.0f s' % (time.time() - t0))
