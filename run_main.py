"""All sea-state numbers of the revised paper: drag on the plate + air compressibility (nominal),
sensitivities, JONSWAP, 30 deg heading, Table 11 statistics, transfer. Writes main.json and prints a report."""
import sys, json, time
import numpy as np
from scipy.optimize import brentq
from model import *

t0 = time.time()
bl = load_blocs()
w = np.linspace(2*np.pi*0.05, 2*np.pi*0.35, 1501); dw = w[1] - w[0]; k = wavenumber(w)
Hh, Ho, Hf = hyd(bl, 'hyb'), hyd(bl, 'owc'), hyd(bl, 'flo')
C0 = configs(Hh, Ho, Hf, w, 0.0)
names = ['frequent', 'reference', 'energetic']
Hs = [0.74, 1.50, 2.76]; Te = [4.5, 6.1, 7.5]
SPM = [pm_spec(w, Hs[s], Te[s]/0.857) for s in range(3)]
J = [flux_full(lambda W, s=s: pm_spec(W, Hs[s], Te[s]/0.857))/1e3 for s in range(3)]
tab = cd_table()
V_crest = np.pi*7.68**2*1.92            # roof at the crest of the torus
V_zz = np.pi*7.68**2*7.09               # Zheng & Zhang convention: section x draft
Tc, kc = 2.57e6, 1.0e7
Lx, Rc, rh, zh, af = 6.92, 16.52, 9.60, 1.92, 4.0
phi = 2*np.pi*np.arange(6)/6; xk = Rc*np.cos(phi)

def case(name, dr, V0, C=C0, spectra=SPM, which=None):
    out = {}
    for c in (which or ['owc', 'flo', 'opn', 'lck', 'hyb', 'owc_fixed', 'floats_fixed', 'floats_fixed_open', 'hyb_fixed']):
        out[c] = []
        for s, S in enumerate(spectra):
            r = optimise(C[c], S, dw, dr, V0)
            out[c].append(r)
    print('[%s] done in %.0f s' % (name, time.time() - t0)); sys.stdout.flush()
    return out

def summary(R, spectra=SPM, Jv=J, C=C0, tag=''):
    rows = {}
    for s in range(len(spectra)):
        S = spectra[s]; x = {}
        for c in R: x[c] = R[c][s]['P']/1e3
        hy = R['hyb'][s]
        x['hyb_OWC_share'] = hy['Pt']/1e3; x['hyb_floats_share'] = hy['Pk'].sum()/1e3; x['hyb_Pk'] = list(hy['Pk']/1e3)
        x['hyb_Bt'] = hy['Bt']; x['hyb_Bth'] = hy['Bth']; x['edge'] = hy['edge']
        x['q'] = x['hyb']/(x['owc'] + x['flo'])
        if 'hyb_fixed' in R: x['q_fixed'] = x['hyb_fixed']/(x['owc_fixed'] + x['flo'])
        x['CWR_hyb'] = x['hyb']/(Jv[s]*D); x['CWR_owc'] = x['owc']/(Jv[s]*D); x['CWR_flo'] = x['flo']/(Jv[s]*D)
        for c in ('hyb', 'owc', 'opn', 'lck'):
            if c in R: x['drag_' + c] = {kk: R[c][s]['d'][kk] for kk in ('KC', 'Cd', 'Beq', 'Pdrag', 'conv')}
        th_h = np.rad2deg(2*np.sqrt(np.sum(np.abs(hy['q'][3:9])**2*S, axis=1)*dw))
        th_f = np.rad2deg(2*np.sqrt(np.sum(np.abs(R['flo'][s]['q'])**2*S, axis=1)*dw))
        x['theta_sig_hyb_max'] = float(th_h.max()); x['theta_sig_flo_max'] = float(th_f.max())
        rows[names[s] if len(spectra) == 3 else str(s)] = x
    return rows

# ---------------------------------------------------------------- nominal and sensitivities
dr_nom = Drag(k, tab, 1.0, 'Cd', 'rms')
R_nom = case('nominal: drag rms, Cd held, V0 crest', dr_nom, V_crest)
R_nodrag = case('compressibility only', None, V_crest, which=['owc', 'flo', 'opn', 'lck', 'hyb'])
R_nocomp = case('drag only', dr_nom, np.inf, which=['owc', 'flo', 'hyb', 'owc_fixed', 'hyb_fixed'])
R_sig = case('drag sig', Drag(k, tab, 1.0, 'Cd', 'sig'), V_crest, which=['owc', 'flo', 'opn', 'lck', 'hyb'])
R_cb = case('drag Cb held', Drag(k, tab, 1.0, 'Cb', 'rms'), V_crest, which=['owc', 'flo', 'opn', 'lck', 'hyb'])
R_vzz = case('V0 Zheng-Zhang', dr_nom, V_zz, which=['owc', 'flo', 'hyb', 'owc_fixed', 'hyb_fixed'])

res = dict(J=J, V_crest=V_crest, V_zz=V_zz, nominal=summary(R_nom), compressibility_only=summary(R_nodrag),
           drag_only=summary(R_nocomp), drag_sig=summary(R_sig), drag_Cb=summary(R_cb), V0_zz=summary(R_vzz))

# ---------------------------------------------------------------- Table 11 and strut statistics (nominal hybrid)
t1 = np.linspace(-np.pi/2, 0, 400)
pts = np.vstack([np.column_stack([Rc - af*np.cos(t1), -0.98 + af*np.sin(t1)]),
                 np.column_stack([np.full(300, Rc - af), np.linspace(-0.98, 1.92, 300)])])
def clr(th):
    xr = rh + np.cos(th)*(pts[:, 0] - rh) - np.sin(th)*(pts[:, 1] - zh)
    zr = zh + np.sin(th)*(pts[:, 0] - rh) + np.cos(th)*(pts[:, 1] - zh)
    return np.min(np.hypot(xr - 9.60, zr) - 1.92)
thc = brentq(clr, np.deg2rad(-89), 0, xtol=1e-15)
eta = np.exp(1j*np.outer(k, xk))
T11 = {}
for s in range(3):
    S = SPM[s]; q = R_nom['hyb'][s]['q']
    zf = q[1][:, None] + Lx*q[3:9].T; r = zf - eta
    m0 = np.sum(np.abs(r)**2*S[:, None], axis=0)*dw; wi = 3
    m0t = np.sum(np.abs(q[3:9].T)**2*S[:, None], axis=0)*dw; m2t = np.sum((w**2)[:, None]*np.abs(q[3:9].T)**2*S[:, None], axis=0)*dw
    ia = int(np.argmax(m0t)); tmax = np.sqrt(m0t[ia])*np.sqrt(2*np.log(1000)); Tz_arm = 2*np.pi*np.sqrt(m0t[ia]/m2t[ia])
    pc = np.exp(-thc**2/(2*m0t[ia]))
    e = q[1] - q[0]; m0e = np.sum(np.abs(e)**2*S)*dw; m2e = np.sum(w**2*np.abs(e)**2*S)*dw
    st = q[2] - q[1]; m0s = np.sum(np.abs(st)**2*S)*dw
    m0p = np.sum(np.abs(q[0])**2*S)*dw
    pslack = np.exp(-(Tc/kc)**2/(2*m0e)); Tz_e = 2*np.pi*np.sqrt(m0e/m2e)
    T11[names[s]] = dict(
        rel_sig=2*np.sqrt(m0[wi]), crest1000=np.sqrt(m0[wi])*np.sqrt(2*np.log(1000)),
        p_rise=float(np.exp(-0.98**2/(2*m0[wi]))), p_sink=float(np.exp(-1.92**2/(2*m0[wi]))),
        zf_sig_max=float((2*np.sqrt(np.sum(np.abs(zf)**2*S[:, None], axis=0)*dw)).max()),
        rot1000_deg=float(np.rad2deg(tmax)), clearance_1000=float(clr(-tmax)), p_contact=float(pc), Tz_arm=float(Tz_arm),
        contacts_per_year_bin=float(pc*0.051*365.25*86400/Tz_arm) if s == 2 else None,
        stroke_sig=2*np.sqrt(m0s), stroke1000=np.sqrt(2*m0s*np.log(1000)), plate_sig=2*np.sqrt(m0p),
        elong_sig=2*np.sqrt(m0e), load_sig_MN=kc*2*np.sqrt(m0e)/1e6, p_slack=float(pslack), slack_per_hour=float(3600/Tz_e*pslack),
        elong1000=np.sqrt(2*m0e*np.log(1000)))
res['T11'] = T11
# slack of the other floating configurations (nominal)
res['slack_other'] = {}
for c in ('owc', 'opn', 'lck'):
    res['slack_other'][c] = []
    for s in range(3):
        q = R_nom[c][s]['q']; S = SPM[s]
        e = q[1] - q[0]; m0e = np.sum(np.abs(e)**2*S)*dw
        res['slack_other'][c].append(float(np.exp(-(Tc/kc)**2/(2*m0e))))

# ---------------------------------------------------------------- transfer (energetic): hybrid at the dampings of the isolated parts
s = 2; S = SPM[s]
q, Pt, Pk, d = run_spectrum(C0['hyb'], S, dw, R_nom['owc'][s]['Bt'], R_nom['flo'][s]['Bth'], dr_nom, V_crest)
res['transfer_energetic'] = dict(OWC_share=Pt/1e3, floats_share=Pk.sum()/1e3, total=(Pt + Pk.sum())/1e3,
                                 Bt=R_nom['owc'][s]['Bt'], Bth=R_nom['flo'][s]['Bth'])

# ---------------------------------------------------------------- JONSWAP (gamma 2 and 3.3) and 30 deg heading
SJ, TpJ, JJ = {}, {}, {}
for gam in (2.0, 3.3):
    SJ[gam], TpJ[gam], JJ[gam] = [], [], []
    for s in range(3):
        f_, tp = jonswap(Hs[s], Te[s], gam); SJ[gam].append(f_(w)); TpJ[gam].append(tp); JJ[gam].append(flux_full(f_)/1e3)
for gam in (2.0, 3.3):
    Rj = case('JONSWAP %.1f' % gam, dr_nom, V_crest, spectra=SJ[gam], which=['owc', 'flo', 'hyb'])
    res['jonswap_%.1f' % gam] = dict(Tp=TpJ[gam], J=JJ[gam], rows=summary(Rj, SJ[gam], JJ[gam]))
try:
    C30 = configs(Hh, Ho, Hf, w, 30.0)
    R30 = case('heading 30', dr_nom, V_crest, C=C30, which=['owc', 'flo', 'hyb'])
    res['heading30'] = summary(R30, C=C30)
except AssertionError as ex:
    res['heading30'] = str(ex)

json.dump(res, open('main.json', 'w'), indent=1, default=float)
print('total %.0f s' % (time.time() - t0))
