"""Extra numbers for the revision text (nominal model: plate drag + compressibility, V0 = 356 m3):
(a) shares of the hybrid with the torus fixed; (b) equivalent cables with the whole torus buoyancy as pretension."""
import sys, json, numpy as np
from model import *
bl = load_blocs()
w = np.linspace(2*np.pi*0.05, 2*np.pi*0.35, 1501); dw = w[1] - w[0]; k = wavenumber(w)
Hh, Ho, Hf = hyd(bl, 'hyb'), hyd(bl, 'owc'), hyd(bl, 'flo')
C = configs(Hh, Ho, Hf, w)
Hs = [0.74, 1.50, 2.76]; Te = [4.5, 6.1, 7.5]
SPM = [pm_spec(w, Hs[s], Te[s]/0.857) for s in range(3)]
V = np.pi*7.68**2*1.92; dr = Drag(k, cd_table())
out = {}
# (a) hybrid with the torus fixed: shares
out['hyb_fixed'] = []
for s in range(3):
    r = optimise(C['hyb_fixed'], SPM[s], dw, None, V)
    out['hyb_fixed'].append(dict(P=r['P']/1e3, OWC=r['Pt']/1e3, floats=r['Pk'].sum()/1e3, Bt=r['Bt'], Bth=r['Bth'], edge=r['edge']))
    print('hyb_fixed', s, out['hyb_fixed'][-1])
# (b) pretension = whole buoyancy of the torus: torus model mass 0, plate + 214 t
Tb = rho*g*464.7; dT = Tb - 2.57e6
M2 = Hh['M'].copy(); M2[0, 0] += dT/g; M2[1, 1] -= dT/g
print('pretension %.3f MN, dm %.1f t, plate %.1f t, M22 %.1f t' % (Tb/1e6, dT/g/1e3, M2[0, 0]/1e3, M2[1, 1]/1e3))
e9 = np.zeros(9); e9[2] = 1; e9[1] = -1
P2 = submodel(Hh, w, et=e9, arms=range(3, 9), plate=0, M=M2)
out['cable_buoyancy'] = []
for s in range(3):
    S = SPM[s]; r = optimise(P2, S, dw, dr, V); q = r['q']
    e = q[1] - q[0]; m0e = np.sum(np.abs(e)**2*S)*dw; m2e = np.sum(w**2*np.abs(e)**2*S)*dw
    ps = float(np.exp(-(Tb/1e7)**2/(2*m0e)))
    out['cable_buoyancy'].append(dict(P=r['P']/1e3, p_slack=ps, per_hour=3600/(2*np.pi*np.sqrt(m0e/m2e))*ps, elong_sig=2*np.sqrt(m0e),
                                      KC=r['d']['KC'], Bt=r['Bt'], Bth=r['Bth']))
    print('cable_buoyancy', s, out['cable_buoyancy'][-1])
json.dump(out, open('extra.json', 'w'), indent=1, default=float)
