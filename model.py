"""Frequency-domain model of the paper (Eq. 26 after renumbering) with two additions of 4/11:
- linearised Morison drag on the heave plate (Borgman in a sea state, Lorentz in regular waves),
  Cd(KC) from the solid plate of Medina-Manuel et al. (2025), Fig. 11, Cd = 1.8 pi Cb / KC;
- air compressibility in the chamber: the turbine damper B_t in series with the air spring
  K_p = gamma p_a S^2 / V0, i.e. B_t -> Z = B_t / (1 - i w B_t / K_p) on z_w - z_d.
Every configuration is a sub-model of the 9 (hybrid) or 3 (isolated OWC) generalised coordinates.
With drag off (scale 0) and V0 = inf the model returns the values of check_tables.py."""
import re, io, json
import numpy as np
from scipy.optimize import brentq

rho, g, h, D = 1025.0, 9.80665, 45.0, 41.04
Dp, zc = 27.5, -25.0
Ap = np.pi*Dp**2/4
S_pis = 180.5                     # piston area (Table 4)
gamma_pa = 1.4*101325.0           # isentropic bulk modulus of air (Pa)
MD = 'Donnees_modele_hybride_OWC_flotteurs.md'   # data file, in this folder
CB = 'cb_solid_fig11.json'   # digitised solid-plate curve of Medina Manuel et al. (2025), Fig. 11

# ------------------------------------------------------------------ data
def load_blocs(md=MD):
    txt = open(md, encoding='utf-8').read().replace('\r', '')
    return dict(re.findall(r'<!-- bloc: (\S+) -->\n```(?:csv|text)\n(.*?)\n```', txt, re.S))

def wavenumber(w):
    out = np.zeros_like(w)
    for n, wn in enumerate(w):
        kU = 2*(wn**2/g + wn/np.sqrt(g*h))
        out[n] = brentq(lambda x: wn**2 - g*x*np.tanh(x*h), 1e-10, kU, xtol=1e-16, rtol=8.9e-16, maxiter=500)
    return out

def group_velocity(w, k):
    return w/(2*k)*(1 + 2*k*h/np.sinh(2*k*h))

def hyd(blocs, config):
    csv = lambda nm: np.genfromtxt(io.StringIO(blocs[nm]), delimiter=',', names=True)
    mat = lambda nm: np.array([[float(v) for v in l.split(',')[1:]] for l in blocs[nm].strip().split('\n')[1:]])
    pre, Mn, Kn = {'hyb': ('hyb_d', 'M_hybride', 'K_hybride'), 'owc': ('owc_d', 'M_OWC_seul', 'K_OWC_seul'),
                   'flo': ('flo_d', 'M_hybride', 'K_hybride')}[config]
    AB, FT = csv(pre + '.AB'), csv(pre + '.F')
    key = lambda x: np.round(np.asarray(x)*1e9).astype(np.int64)
    f = np.unique(AB['f_Hz']); nf = f.size; fk = {kk: m for m, kk in enumerate(key(f))}
    n = int(max(AB['i'].max(), AB['j'].max())) + 1
    A = np.zeros((n, n, nf)); B = np.zeros((n, n, nf))
    for r in AB:
        m = fk[int(key(r['f_Hz']))]; A[int(r['i']), int(r['j']), m] = r['A']; B[int(r['i']), int(r['j']), m] = r['B']
    beta = np.unique(FT['beta_deg'])
    F = np.zeros((n, beta.size, nf), complex)
    for r in FT:
        F[int(r['i']), int(np.where(beta == r['beta_deg'])[0][0]), fk[int(key(r['f_Hz']))]] = r['Re_F'] + 1j*r['Im_F']
    M, K = mat(Mn), mat(Kn)
    if config == 'flo':
        M, K = M[3:9, 3:9], K[3:9, 3:9]
    return dict(config=config, n=n, omega=2*np.pi*f, beta=beta, A=A, B=B, F=F, M=M, K=K)

def interp_last(w0, X, w):
    sz = X.shape; X2 = X.reshape(-1, sz[-1])
    Y = np.array([np.interp(w, w0, x.real) + (1j*np.interp(w, w0, x.imag) if np.iscomplexobj(X) else 0) for x in X2])
    return Y.reshape(sz[:-1] + (w.size,))

# ------------------------------------------------------------------ sub-models
def submodel(H, w, keep=None, et=None, arms=(), plate=None, T=None, beta=0.0, M=None):
    """keep: kept coordinates (others fixed); T: constraint matrix (n x m) applied after keep (locked piston);
    et: turbine relative-motion vector in the final coordinates (None: no turbine); arms: arm indices;
    plate: index of z_p in the final coordinates (None: no plate drag)."""
    ib = int(np.argmin(abs(H['beta'] - beta)))
    assert abs(H['beta'][ib] - beta) < 1e-9, ('heading not in the data', beta)
    n0 = H['n']; keep = list(range(n0)) if keep is None else list(keep)
    Mm = H['M'] if M is None else M
    A = interp_last(H['omega'], H['A'][np.ix_(keep, keep)], w)
    B = interp_last(H['omega'], H['B'][np.ix_(keep, keep)], w)
    F = interp_last(H['omega'], H['F'][keep, ib, :], w)
    Z0 = H['K'][np.ix_(keep, keep)][:, :, None] - w**2*(Mm[np.ix_(keep, keep)][:, :, None] + A) - 1j*w*B
    if T is not None:
        Z0 = np.einsum('ai,abm,bj->ijm', T, Z0, T); F = T.T @ F
    n = Z0.shape[0]
    return dict(w=w, Zt=np.moveaxis(Z0, 2, 0).copy(), F=F, n=n, et=None if et is None else np.asarray(et, float),
                arms=list(arms), plate=plate)

def configs(Hh, Ho, Hf, w, beta=0.0):
    T = np.zeros((9, 8)); T[0, 0] = 1; T[1, 1] = 1; T[2, 1] = 1; T[3:9, 2:8] = np.eye(6)
    e9 = np.zeros(9); e9[2] = 1; e9[1] = -1
    e3 = np.zeros(3); e3[2] = 1; e3[1] = -1
    return {
        'owc': submodel(Ho, w, et=e3, plate=0, beta=beta if beta in Ho['beta'] else 0.0),   # axisymmetric
        'flo': submodel(Hf, w, arms=range(6), beta=beta),
        'opn': submodel(Hh, w, arms=range(3, 9), plate=0, beta=beta),
        'lck': submodel(Hh, w, T=T, arms=range(2, 8), plate=0, beta=beta),
        'hyb': submodel(Hh, w, et=e9, arms=range(3, 9), plate=0, beta=beta),
        'owc_fixed': submodel(Ho, w, keep=[2], et=[1.0], beta=beta if beta in Ho['beta'] else 0.0),
        'floats_fixed': submodel(Hh, w, keep=range(3, 9), arms=range(6), beta=beta),
        'floats_fixed_open': submodel(Hh, w, keep=range(2, 9), arms=range(1, 7), beta=beta),
        'hyb_fixed': submodel(Hh, w, keep=range(2, 9), et=[1.0] + [0]*6, arms=range(1, 7), beta=beta),
    }

# ------------------------------------------------------------------ drag law
def cd_table(path=CB):
    d = json.load(open(path)); KC = np.array(d['KC']); Cb = np.array(d['Cb'])
    o = np.argsort(KC); return KC[o], d['fconv']*Cb[o]/KC[o]

def cd_of(KC, tab, scale=1.0, hold='Cd'):
    KCt, Cdt = tab
    Cd = np.interp(np.clip(KC, KCt[0], KCt[-1]), KCt, Cdt)
    if hold == 'Cb':
        Cd = np.where(KC < KCt[0], Cdt[0]*KCt[0]/np.maximum(KC, 1e-12), Cd)
    return scale*Cd, (KC < KCt[0]) | (KC > KCt[-1])

def zeta_plate(k):
    return np.exp(k*zc)*(1 - np.exp(-2*k*(h + zc)))/(1 - np.exp(-2*k*h))

# ------------------------------------------------------------------ solve
def pto(P, Bt, Bth, V0=np.inf):
    """Frequency-dependent PTO impedance (nw x n x n) and the turbine impedance Z(w)."""
    w = P['w']; n = P['n']
    Zt_ = Bt/(1 - 1j*w*Bt/(gamma_pa*S_pis**2/V0)) if (P['et'] is not None and np.isfinite(V0)) else Bt*np.ones_like(w, complex)
    Bp = np.zeros((w.size, n, n), complex)
    if P['et'] is not None and Bt > 0:
        Bp += Zt_[:, None, None]*np.outer(P['et'], P['et'])[None]
    for i in P['arms']:
        Bp[:, i, i] += Bth
    return Bp, Zt_

def solve(P, Bp, Beq, wI):
    w = P['w']; Z = P['Zt'] - 1j*w[:, None, None]*Bp; F = P['F'].copy()
    if P['plate'] is not None:
        Bv = np.broadcast_to(np.asarray(Beq, float), w.shape)
        Z = Z.copy(); Z[:, P['plate'], P['plate']] -= 1j*w*Bv
        F[P['plate']] = F[P['plate']] + Bv*wI
    return np.linalg.solve(Z, F.T[:, :, None])[:, :, 0].T

def powers(P, q, Bt, Bth, Zt_):
    w = P['w']
    Pt = 0.5*w**2*np.real(Zt_)*np.abs(P['et'] @ q)**2 if (P['et'] is not None and Bt > 0) else np.zeros_like(w)
    Pk = 0.5*w**2*Bth*np.abs(q[P['arms']])**2 if P['arms'] else np.zeros((0, w.size))
    return Pt, Pk

class Drag:
    def __init__(self, k, tab, scale=1.0, hold='Cd', KCdef='rms'):
        self.k, self.tab, self.scale, self.hold, self.KCdef = k, tab, scale, hold, KCdef
        self.zt = zeta_plate(k)

def converge_spectrum(P, Bp, S, dw, dr, Beq0=0.0, relax=0.5, tol=1e-4, maxit=300):
    w = P['w']; wI = -1j*w*dr.zt; Beq = Beq0; conv = False
    for it in range(maxit):
        q = solve(P, Bp, Beq, wI)
        zr = q[P['plate']] - dr.zt
        szr = np.sqrt(np.sum(np.abs(zr)**2*S)*dw); sur = np.sqrt(np.sum(w**2*np.abs(zr)**2*S)*dw)
        KC = 2*np.pi*(np.sqrt(2) if dr.KCdef == 'rms' else 2.0)*szr/Dp
        Cd, out = cd_of(KC, dr.tab, dr.scale, dr.hold)
        Bn = 0.5*rho*float(Cd)*Ap*np.sqrt(8/np.pi)*sur
        if abs(Bn - Beq) <= tol*max(Bn, 1e-300):
            Beq = Bn; conv = True; break
        Beq = relax*Bn + (1 - relax)*Beq
    q = solve(P, Bp, Beq, wI)
    zr = q[P['plate']] - dr.zt
    szr = np.sqrt(np.sum(np.abs(zr)**2*S)*dw); sur = np.sqrt(np.sum(w**2*np.abs(zr)**2*S)*dw)
    KC = 2*np.pi*(np.sqrt(2) if dr.KCdef == 'rms' else 2.0)*szr/Dp
    Cd, out = cd_of(KC, dr.tab, dr.scale, dr.hold)
    return q, dict(Beq=Beq, KC=float(KC), Cd=float(Cd), out=bool(out), conv=conv, it=it, Pdrag=Beq*sur**2)

def run_spectrum(P, S, dw, Bt, Bth, dr=None, V0=np.inf, Beq0=0.0):
    Bp, Zt_ = pto(P, Bt, Bth, V0)
    if P['plate'] is None or dr is None or dr.scale == 0:
        q = solve(P, Bp, 0.0, np.zeros_like(P['w'])); d = dict(Beq=0.0, KC=np.nan, Cd=0.0, out=False, conv=True, Pdrag=0.0)
    else:
        q, d = converge_spectrum(P, Bp, S, dw, dr, Beq0)
    Pt, Pk = powers(P, q, Bt, Bth, Zt_)
    return q, 2*np.sum(Pt*S)*dw, 2*np.sum(Pk*S, axis=1)*dw, d

def grids(P, n2=46, n1=141):
    hasT, hasA = P['et'] is not None, bool(P['arms'])
    if hasT and hasA: return np.logspace(4, 7.5, n2), np.logspace(5.5, 9, n2)
    if hasT:          return np.logspace(4, 7.5, n1), np.array([0.0])
    return np.array([0.0]), np.logspace(5.5, 9, n1)

def optimise(P, S, dw, dr=None, V0=np.inf, gT=None, gA=None):
    if gT is None: gT, gA = grids(P)
    best = dict(P=-np.inf); Beq = 0.0; nonconv = 0
    for bt in gT:
        for ba in gA:
            q, Pt, Pk, d = run_spectrum(P, S, dw, bt, ba, dr, V0, Beq)
            Beq = d['Beq']; nonconv += not d['conv']
            tot = Pt + Pk.sum()
            if tot > best['P']:
                best = dict(P=tot, Pt=Pt, Pk=Pk, Bt=bt, Bth=ba, q=q, d=d)
    best['nonconv'] = nonconv
    best['edge'] = bool((gT.size > 1 and best['Bt'] in (gT[0], gT[-1])) or (gA.size > 1 and best['Bth'] in (gA[0], gA[-1])))
    return best

# ------------------------------------------------------------------ regular waves
def converge_regular(P, Bp, A, dr, Beq0=None, relax=0.5, tol=1e-4, maxit=300):
    w = P['w']; wI = -1j*w*dr.zt
    Beq = np.zeros_like(w) if Beq0 is None else Beq0.copy()
    for it in range(maxit):
        q = solve(P, Bp, Beq, wI)
        zr = A*(q[P['plate']] - dr.zt); KC = 2*np.pi*np.abs(zr)/Dp
        Cd, out = cd_of(KC, dr.tab, dr.scale, dr.hold)
        Bn = 0.5*rho*Cd*Ap*8/(3*np.pi)*w*np.abs(zr)
        conv = np.abs(Bn - Beq) <= tol*np.maximum(Bn, 1e-300)
        if conv.all(): Beq = Bn; break
        Beq = relax*Bn + (1 - relax)*Beq
    q = solve(P, Bp, Beq, wI); zr = A*(q[P['plate']] - dr.zt); KC = 2*np.pi*np.abs(zr)/Dp
    Cd, out = cd_of(KC, dr.tab, dr.scale, dr.hold)
    return q, dict(Beq=Beq, KC=KC, Cd=Cd, out=out, conv=conv)

def optimise_regular(P, A, dr=None, V0=np.inf, gT=None, gA=None):
    """Per frequency, best (B_t, B_theta) on the grid; power per unit amplitude^2 (W/m^2)."""
    if gT is None: gT, gA = grids(P)
    w = P['w']; best = dict(P=-np.inf*np.ones_like(w), Bt=np.zeros_like(w), Bth=np.zeros_like(w),
                            KC=np.full_like(w, np.nan), Cd=np.zeros_like(w), conv=np.ones(w.size, bool))
    Beq = None
    for bt in gT:
        for ba in gA:
            Bp, Zt_ = pto(P, bt, ba, V0)
            if P['plate'] is None or dr is None or dr.scale == 0:
                q = solve(P, Bp, 0.0, np.zeros_like(w)); d = None
            else:
                q, d = converge_regular(P, Bp, A, dr, Beq); Beq = d['Beq']
            Pt, Pk = powers(P, q, bt, ba, Zt_); Pw = Pt + Pk.sum(axis=0)
            m = Pw > best['P']
            best['P'][m] = Pw[m]; best['Bt'][m] = bt; best['Bth'][m] = ba
            if d is not None:
                best['KC'][m] = d['KC'][m]; best['Cd'][m] = d['Cd'][m]; best['conv'][m] = d['conv'][m]
    return best

# ------------------------------------------------------------------ spectra and flux
def pm_spec(w, Hs, Tp):
    f = w/(2*np.pi); fp = 1/Tp; Bs = 1.25*fp**4; As = Bs*Hs**2/4
    return As*f**-5*np.exp(-Bs*f**-4)/(2*np.pi)

def jonswap(Hs, Te, gam):
    Wg = np.linspace(0.02, 20, 400001); dWg = Wg[1] - Wg[0]
    shp = lambda W, wp: W**-5*np.exp(-1.25*(wp/W)**4)*gam**np.exp(-0.5*((W - wp)/((0.07 + 0.02*(W > wp))*wp))**2)
    TeOf = lambda wp: 2*np.pi*np.sum(shp(Wg, wp)/Wg)/np.sum(shp(Wg, wp))
    wp = brentq(lambda x: TeOf(x) - Te, 0.2, 5, xtol=1e-15, rtol=8.9e-16, maxiter=500)
    c = Hs**2/16/(np.sum(shp(Wg, wp))*dWg)
    return (lambda W: c*shp(W, wp)), 2*np.pi/wp

def flux_full(Sfun):
    wf = 2*np.pi*np.linspace(0.004, 5, 20001); dwf = wf[1] - wf[0]
    kf = wavenumber(wf); cgf = np.where(kf*h > 30, wf/(2*kf), group_velocity(wf, kf))
    return rho*g*np.sum(cgf*Sfun(wf))*dwf
