import csv, numpy as np
rows = list(csv.DictReader(open('convergence_owc.csv')))
for r in rows:
    for k in ('scale', 'f_Hz', 'A', 'B', 'B_H', 'B_over_BH', 'absF'): r[k] = float(r[k])
    r['panels'] = int(r['panels'])
scales = sorted({r['scale'] for r in rows}, reverse=True); freqs = sorted({r['f_Hz'] for r in rows})
get = lambda s, m, d, f, key: next(r[key] for r in rows if r['scale'] == s and r['method'] == m and r['dof'] == d and r['f_Hz'] == f)
for m in ('indirect', 'direct'):
    print('\npiston B/B_H, %s' % m)
    for s in scales:
        n = next(r['panels'] for r in rows if r['scale'] == s)
        print('  s %.2f %6d panels: %s | deficit at 0.19 Hz %.3f, deficit/s %.3f' % (
            s, n, ' '.join('%.3f' % get(s, m, 'piston', f, 'B_over_BH') for f in freqs), 1 - get(s, m, 'piston', 0.19, 'B_over_BH'),
            (1 - get(s, m, 'piston', 0.19, 'B_over_BH'))/s))
fine = min(scales)
print('\nDirect formulation: change of A, B and |F| of each mode from s = 1 to the finest mesh (s = %.2f), %%' % fine)
for d in ('piston', 'donut', 'disque'):
    for key in ('A', 'B', 'absF'):
        v = [100*(get(fine, 'direct', d, f, key)/get(1.0, 'direct', d, f, key) - 1) for f in freqs]
        print('  %-6s %-4s %s' % (d, key, ' '.join('%+6.2f' % x for x in v)))
print('\nIndirect against direct on the finest mesh, piston A and |F| (%):')
for key in ('A', 'absF'):
    print('  %-4s %s' % (key, ' '.join('%+6.2f' % (100*(get(fine, 'indirect', 'piston', f, key)/get(fine, 'direct', 'piston', f, key) - 1)) for f in freqs)))
print('\ntorus and plate B/B_H, direct, s = 1 and finest:')
for d in ('donut', 'disque'):
    for s in (1.0, fine):
        print('  %-6s s %.2f %s' % (d, s, ' '.join('%.3f' % get(s, 'direct', d, f, 'B_over_BH') for f in freqs)))
