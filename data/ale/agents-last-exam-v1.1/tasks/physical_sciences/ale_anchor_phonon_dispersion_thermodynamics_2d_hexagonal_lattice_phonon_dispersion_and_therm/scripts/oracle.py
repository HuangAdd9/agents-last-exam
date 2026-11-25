"""Independent oracle: geometric bonds -> energy Gram matrix -> CDF DOS.

No code or imports from model.py/solver_template.py.  The candidate sums directed
force blocks; here distance shells are discovered geometrically and an undirected
bond-extension operator B gives D=B^H B.  DOS uses divided differences of the
triangle CDF; thermal observables use scalar quadrature and oscillator identities.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np


def geometry(cfg):
    a = float(cfg['a'])
    lattice = a * np.array([[1., 0.], [.5, np.sqrt(3.) / 2.]])
    basis = np.array([[0., 0.], [a / 2., a / np.sqrt(12.)]])
    reciprocal = 2. * np.pi * np.linalg.inv(lattice).T
    return lattice, basis, reciprocal


def bond_operator(cfg, q):
    """One row per undirected spring, weighted by sqrt(k) and inverse sqrt(m)."""
    lattice, basis, _ = geometry(cfg)
    masses = [cfg['m1'], cfg['m2']]
    rows, shells = [], [0, 0, 0]
    for source, target in ((0, 1), (0, 0), (1, 1)):
        for u in range(-2, 3):
            for v in range(-2, 3):
                if source == target and not (u > 0 or (u == 0 and v > 0)):
                    continue  # retain one orientation of each same-species bond
                displacement = np.array([u, v]) @ lattice + basis[target] - basis[source]
                length2 = float(displacement @ displacement)
                ratio = length2 / float(cfg['a']) ** 2
                if source != target and np.isclose(ratio, 1. / 3., atol=1e-13):
                    spring, shell = cfg['k1'], 0
                elif source != target and np.isclose(ratio, 4. / 3., atol=1e-13):
                    spring, shell = cfg['k3'], 2
                elif source == target and np.isclose(ratio, 1., atol=1e-13):
                    spring, shell = cfg['k2'], 1
                else:
                    continue
                shells[shell] += 1
                direction = displacement / np.sqrt(length2)
                row = np.zeros(4, dtype=complex)
                row[2*source:2*source+2] += direction / np.sqrt(masses[source])
                row[2*target:2*target+2] -= (direction / np.sqrt(masses[target])
                                                         * np.exp(1j * (q @ displacement)))
                rows.append(np.sqrt(spring) * row)
    assert shells == [3, 6, 3], shells
    return np.asarray(rows)


def dynamical_matrix(cfg, q):
    extension = bond_operator(cfg, np.asarray(q, dtype=float))
    return extension.conj().T @ extension


def eigenmodes(cfg, points):
    # Construct the extension rows once and recover their phase from geometry;
    # small problem sizes make direct Gram construction adequate for validation.
    values = np.array([np.linalg.eigvalsh(dynamical_matrix(cfg, q)) for q in points])
    if np.any(values < -1e-10):
        raise ValueError('unstable spring model')
    values[np.abs(values) < 1e-12] = 0.
    return np.sqrt(np.maximum(values, 0.)), values


def triangle_cdf_pdf(vertices, omega):
    """Divided difference of (omega-v)_+^2 and its analytic derivative.

    Confluent (two equal vertices) limits avoid division by a repeated knot.
    Use extended precision for cancellations in the upper CDF tail.  A flat
    triangle is a delta measure and is not representable as point density;
    the governed non-flat grids have no such triangles.
    """
    knots = np.sort(np.asarray(vertices, dtype=np.longdouble))
    t = np.asarray(omega, dtype=np.longdouble)
    lo, mid, hi = knots
    if hi - lo < 1e-12:
        # A measure-zero acoustic triangle contributes no sampled density.
        return np.zeros_like(t, dtype=float), np.zeros_like(t, dtype=float)
    inside = (t >= lo) & (t <= hi)
    cdf = np.zeros_like(t)
    pdf = np.zeros_like(t)
    if mid - lo < 1e-12:
        cdf[inside] = 1. - ((hi - t[inside]) / (hi - lo)) ** 2
        pdf[inside] = 2. * (hi - t[inside]) / (hi - lo) ** 2
    elif hi - mid < 1e-12:
        cdf[inside] = ((t[inside] - lo) / (hi - lo)) ** 2
        pdf[inside] = 2. * (t[inside] - lo) / (hi - lo) ** 2
    else:
        for index, knot in enumerate(knots):
            others = np.delete(knots, index)
            coefficient = 1. / np.prod(knot - others)
            ramp = np.maximum(t[inside] - knot, 0.)
            cdf[inside] += coefficient * ramp ** 2
            pdf[inside] += 2. * coefficient * ramp
    cdf[t > hi] = 1.
    return np.asarray(cdf, float), np.asarray(pdf, float)


def triangle_mesh(grid):
    """Two non-overlapping triangles per periodic cell, along 00--11."""
    n = grid.shape[0]
    values = []
    for i in range(n):
        for j in range(n):
            ni, nj = (i + 1) % n, (j + 1) % n
            for corners in (((i, j), (ni, j), (ni, nj)),
                            ((i, j), (ni, nj), (i, nj))):
                values.extend(np.stack([grid[p] for p in corners], axis=-1))
    return np.asarray(values)


def triangle_dos(cfg, grid):
    top = np.ceil(np.max(grid) * 10. - 1e-12) / 10.
    omega = np.linspace(0., top, int(cfg['omega_bins']))
    values = triangle_mesh(grid)
    density = np.sum([triangle_cdf_pdf(tri, omega)[1] for tri in values], axis=0)
    density /= 2. * grid.shape[0] ** 2
    # Explicit quadrature weights, independent of candidate's np.trapezoid.
    weights = trapezoid_weights(omega)
    density *= 4. / np.dot(weights, density)
    return omega, density


def trapezoid_weights(points):
    spacing = np.diff(points)
    weights = np.zeros(len(points))
    weights[:-1] += spacing / 2.
    weights[1:] += spacing / 2.
    return weights


def thermal_at_temperature(omega, density, temperature):
    weights = trapezoid_weights(omega) * density
    w = np.maximum(omega, 1e-14)
    half_x = w / (2. * temperature)
    # U=(w/2)coth(w/2T), C=(w/2T / sinh(w/2T))^2,
    # F=T log(2 sinh(w/2T)). Stable alternatives only at large x.
    energy = np.dot(weights, .5 * w / np.tanh(half_x))
    kernel = np.zeros_like(w)
    moderate = half_x < 350.
    kernel[moderate] = (half_x[moderate] / np.sinh(half_x[moderate])) ** 2
    cv = np.dot(weights, kernel)
    log_partition = np.empty_like(w)
    log_partition[moderate] = np.log(2. * np.sinh(half_x[moderate]))
    log_partition[~moderate] = half_x[~moderate] + np.log1p(-np.exp(-2. * half_x[~moderate]))
    free = temperature * np.dot(weights, log_partition)
    return energy, cv, free


def solve_reference(cfg):
    _, _, reciprocal = geometry(cfg)
    vertices = [np.zeros(2), reciprocal[0]/2., (2*reciprocal[0]+reciprocal[1])/3., np.zeros(2)]
    counts = [int(cfg[key]) for key in ('n_path_gm', 'n_path_mk', 'n_path_kg')]
    path = []
    for index, count in enumerate(counts):
        denominator = count - 1 if index == 2 else count
        path.extend(vertices[index] + (vertices[index+1]-vertices[index])*j/denominator for j in range(count))
    q = np.asarray(path)
    distance = np.r_[0., np.cumsum(np.sqrt(np.sum(np.diff(q, axis=0)**2, axis=1)))]
    freq, eigenvalues = eigenmodes(cfg, q)
    n = int(cfg['q_grid'])
    fractions = np.array([(i/n, j/n) for i in range(n) for j in range(n)])
    grid = eigenmodes(cfg, fractions @ reciprocal)[0].reshape(n, n, 4)
    omega, density = triangle_dos(cfg, grid)
    temperatures = np.geomspace(.02, 80., int(cfg['temperatures']))
    energy, cv, free = np.asarray([thermal_at_temperature(omega, density, t) for t in temperatures]).T
    # Chain reference obtained from its 2x2 Hermitian matrix (candidate uses formula).
    chain_q = np.linspace(0., np.pi, 121)
    chain = []
    k, m1, m2 = (float(cfg[s]) for s in ('chain_k', 'chain_m1', 'chain_m2'))
    for point in chain_q:
        coupling = -k * (1. + np.exp(1j * point)) / np.sqrt(m1*m2)
        chain.append(np.linalg.eigvalsh([[2*k/m1, coupling], [coupling.conjugate(), 2*k/m2]]))
    chain = np.asarray(chain)
    chain[np.abs(chain) < 1e-12] = 0.
    acoustic, optical = np.sqrt(np.maximum(chain, 0.)).T
    summary = {
        'tier1': {'optical_frequency_at_gamma': float(optical[0]), 'max_acoustic_at_gamma': float(acoustic[0]), 'num_q_points': 121},
        'tier2': {'num_q_points': len(q), 'num_branches': 4,
                  'freq_range': [float(freq[freq>1e-12].min()), float(freq.max())],
                  'negative_eigenvalue_count': int(np.sum(eigenvalues < -1e-10)),
                  'acoustic_at_gamma': list(map(float, freq[-1, :2]))},
        'tier3': {'omega_max': float(omega[-1]), 'dos_integral': float(trapezoid_weights(omega) @ density),
                  'cv_high_T': float(cv[-1]), 'cv_low_T': float(cv[0]), 'num_temperatures': len(temperatures),
                  'q_grid_size': [n, n]},
    }
    return {'diatomic': (chain_q, acoustic, optical), 'q_points': q, 'q_distance': distance, 'frequencies': freq,
            'omega_bins': omega, 'dos': density, 'temperatures': temperatures, 'energy': energy,
            'heat_capacity': cv, 'free_energy': free, 'results': summary}


def write_outputs(result, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    q, acoustic, optical = result['diatomic']
    np.savez_compressed(output/'diatomic_1d.npz', q_points=q, omega_acoustic=acoustic, omega_optical=optical)
    for name, keys in (
        ('dispersion_2d', ('q_points', 'q_distance', 'frequencies')),
        ('dos', ('omega_bins', 'dos')),
        ('thermodynamics', ('temperatures', 'energy', 'heat_capacity', 'free_energy')),
    ):
        np.savez_compressed(output/(name+'.npz'), **{k: result[k] for k in keys})
    (output/'results.json').write_text(json.dumps(result['results'], indent=2, sort_keys=True)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    write_outputs(solve_reference(json.loads(Path(args.config).read_text())), args.output_dir)
