import sys, time, numpy as np
sys.path.insert(0, '.')
from model import *
bl = load_blocs()
w = np.linspace(2*np.pi*0.05, 2*np.pi*0.35, 1501); dw = w[1] - w[0]
Hh, Ho, Hf = hyd(bl, 'hyb'), hyd(bl, 'owc'), hyd(bl, 'flo')
C = configs(Hh, Ho, Hf, w)
Hs = [0.74, 1.50, 2.76]; Te = [4.5, 6.1, 7.5]
t0 = time.time()
for c in ['owc', 'flo', 'opn', 'lck', 'hyb', 'owc_fixed', 'floats_fixed', 'floats_fixed_open', 'hyb_fixed']:
    print('%-18s' % c, ' '.join('%7.1f' % (optimise(C[c], pm_spec(w, Hs[s], Te[s]/0.857), dw)['P']/1e3) for s in range(3)), ' (%.0f s)' % (time.time() - t0))
print('J full (kW/m):', [round(flux_full(lambda W, s=s: pm_spec(W, Hs[s], Te[s]/0.857))/1e3, 3) for s in range(3)])
