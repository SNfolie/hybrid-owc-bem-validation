"""Print every number the revised text needs from main.json (sea states) and regular.json (regular waves)."""
import json, numpy as np
R = json.load(open('main.json'))
N = R['nominal']; st = ['frequent', 'reference', 'energetic']
f1 = lambda v: '%.1f' % v
print('J full:', ['%.3f' % j for j in R['J']], '| V0 crest %.1f, ZZ %.1f m3' % (R['V_crest'], R['V_zz']))
print('\n=== Table 9 (kW): nominal (drag + compressibility) | no drag, compressibility | drag, no compressibility')
for c in ['owc', 'owc_fixed', 'flo', 'floats_fixed', 'floats_fixed_open', 'opn', 'lck', 'hyb_fixed', 'hyb']:
    a = [N[s][c] for s in st]
    b = [R['compressibility_only'][s].get(c, np.nan) for s in st]
    d = [R['drag_only'][s].get(c, np.nan) for s in st]
    print('%-18s %7.2f %7.2f %7.2f | %7.2f %7.2f %7.2f | %7.2f %7.2f %7.2f' % ((c,) + tuple(a) + tuple(b) + tuple(d)))
print('\n=== Table 8 and ratios (nominal)')
for s in st:
    x = N[s]
    print('%-9s hyb %.2f OWC %.2f flo %.2f | q %.3f q_fixed %.3f | CWR %.4f %.4f %.4f | shares OWC %.2f floats %.2f | Bt %.3g Bth %.3g edge %s'
          % (s, x['hyb'], x['owc'], x['flo'], x['q'], x['q_fixed'], x['CWR_hyb'], x['CWR_owc'], x['CWR_flo'],
             x['hyb_OWC_share'], x['hyb_floats_share'], x['hyb_Bt'], x['hyb_Bth'], x['edge']))
    Pk = x['hyb_Pk']
    print('          floats 1-6 %s | windward %.2f (%.0f%%) leeward %.2f (%.1f%%) | theta_sig max flo %.2f hyb %.2f (+%.0f%%)'
          % (' '.join('%.2f' % v for v in Pk), Pk[3], 100*Pk[3]/sum(Pk), Pk[0], 100*Pk[0]/sum(Pk), x['theta_sig_flo_max'], x['theta_sig_hyb_max'],
             100*(x['theta_sig_hyb_max']/x['theta_sig_flo_max'] - 1)))
    print('          hyb/open %.3f (+%.1f kW) hyb/lck %.3f | floating/fixed hybrid %.3f | floats beside fixed / isolated %.3f | floats on open floating / beside fixed open %.3f'
          % (x['hyb']/x['opn'], x['hyb'] - x['opn'], x['hyb']/x['lck'], x['hyb']/x['hyb_fixed'], x['floats_fixed']/x['flo'], x['opn']/x['floats_fixed_open']))
    print('          hyb/OWC %.2f | transfer OWC %+.2f floats %+.2f net %+.2f | OWC share of hybrid %.1f%%'
          % (x['hyb']/x['owc'], x['hyb_OWC_share'] - x['owc'], x['hyb_floats_share'] - x['flo'], x['hyb'] - x['owc'] - x['flo'], 100*x['hyb_OWC_share']/x['hyb']))
    for c in ('hyb', 'owc', 'opn', 'lck'):
        d = x['drag_' + c]
        print('          drag %-3s KC %.4f Cd %.2f Beq %.3g Pdrag %.2f kW conv %s' % (c, d['KC'], d['Cd'], d['Beq'], d['Pdrag']/1e3, d['conv']))
print('\n=== Effect of each addition on the hybrid (kW): no-drag no-comp (paper) -> comp only -> drag only -> both')
paper = {'hyb': [22.8, 123.4, 479.6], 'owc': [2.3, 30.7, 170.7]}
for c in ('hyb', 'owc'):
    for i, s in enumerate(st):
        print('%-4s %-9s %.1f -> %.2f -> %.2f -> %.2f' % (c, s, paper[c][i], R['compressibility_only'][s][c], R['drag_only'][s][c], N[s][c]))
print('\n=== Sensitivities of the hybrid (kW): nominal | KC sig | Cb held | V0 Zheng-Zhang')
for s in st:
    print('%-9s %.2f | %.2f | %.2f | %.2f ; OWC %.2f | %.2f | %.2f | %.2f' % (s, N[s]['hyb'], R['drag_sig'][s]['hyb'], R['drag_Cb'][s]['hyb'], R['V0_zz'][s]['hyb'],
          N[s]['owc'], R['drag_sig'][s]['owc'], R['drag_Cb'][s]['owc'], R['V0_zz'][s]['owc']))
print('\n=== Table 11 (nominal hybrid)')
for s in st:
    t = R['T11'][s]
    print('%-9s rel %.2f crest1000 %.2f rise %.2f%% sink %.3f%% | zf %.2f | rot1000 %.1f deg clear %.2f contact %.4f%% Tz %.1f s per yr %s'
          % (s, t['rel_sig'], t['crest1000'], 100*t['p_rise'], 100*t['p_sink'], t['zf_sig_max'], t['rot1000_deg'], t['clearance_1000'], 100*t['p_contact'], t['Tz_arm'],
             t['contacts_per_year_bin']))
    print('          stroke %.2f / %.2f | plate %.2f | elongation %.3f m, load %.2f MN, slack/compression %.4g per cycle (%.1f per hour), elong1000 %.2f m'
          % (t['stroke_sig'], t['stroke1000'], t['plate_sig'], t['elong_sig'], t['load_sig_MN'], t['p_slack'], t['slack_per_hour'], t['elong1000']))
print('slack in other configurations:', {c: ['%.3g' % v for v in R['slack_other'][c]] for c in R['slack_other']})
print('transfer energetic:', R['transfer_energetic'])
print('\n=== JONSWAP and heading')
for gam in ('2.0', '3.3'):
    J = R['jonswap_' + gam]
    for i, s in enumerate(st):
        x = J['rows'][s] if s in J['rows'] else J['rows'][str(i)]
        print('gamma %s %-9s Tp %.2f J %.2f | OWC %.2f floats %.2f hybrid %.2f (%+.1f%% vs PM) | q %.2f | CWR %.3f'
              % (gam, s, J['Tp'][i], J['J'][i], x['owc'], x['flo'], x['hyb'], 100*(x['hyb']/N[s]['hyb'] - 1), x['q'], x['CWR_hyb']))
H30 = R['heading30']
if isinstance(H30, dict):
    for s in st:
        x = H30[s]; print('30 deg %-9s hybrid %.2f (%+.1f%%) floats %.2f q %.2f' % (s, x['hyb'], 100*(x['hyb']/N[s]['hyb'] - 1), x['flo'], x['q']))
else:
    print('heading 30:', H30)
