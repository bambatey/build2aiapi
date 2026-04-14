"""SEA_Book ``sec4_plane_stress_rectangle_gauss.py`` matematiğinden altın çıktı.

Tek bir Q4 plane-stress elemanı için lokal K (8×8) matrisini hesaplar,
``plane_stress_q4_golden.json`` olarak yazar. Yeni port edilen
``elements/plane_stress_q4.py`` sınıfı bu referansla birebir uyuşmalı.

Test vakası: 1m × 0.5m dikdörtgen, E=210 GPa, ν=0.3, h=0.1m.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

INV = np.linalg.inv
DET = np.linalg.det


def SF(r, s):
    return 0.25 * np.asarray([(1 - r) * (1 - s),
                              (1 + r) * (1 - s),
                              (1 - r) * (1 + s),
                              (1 + r) * (1 + s)])


def dSF_dr(r, s):
    return 0.25 * np.asarray([[-1 + s, -1 + r],
                              [1 - s, -1 - r],
                              [-1 - s, 1 - r],
                              [1 + s, 1 + r]])


def IntegrateOn2DDomainWithGaussN2(h):
    total = 0
    p = [-1 / 3 ** 0.5, 1 / 3 ** 0.5]
    w = [1, 1]
    for i in range(2):
        for j in range(2):
            total = total + w[i] * w[j] * h(p[i], p[j])
    return total


def compute_K(nodes_xy, E, nu, h):
    """SEA_Book matematiğinin birebir aynısı."""
    XM = np.asarray([[nodes_xy[k][0] for k in range(4)],
                     [nodes_xy[k][1] for k in range(4)]])

    def JM(r, s):
        return XM @ dSF_dr(r, s)

    def dSF_Dx_T(r, s):
        return INV(JM(r, s)).T @ dSF_dr(r, s).T

    def BM(r, s):
        empty = np.zeros((3, 8))
        mat = dSF_Dx_T(r, s)
        empty[0, 0:4] = mat[0]
        empty[1, 4:8] = mat[1]
        empty[2, 0:4] = mat[1]
        empty[2, 4:8] = mat[0]
        return empty

    C = E / (1 - nu ** 2) * np.asarray([[1, nu, 0],
                                        [nu, 1, 0],
                                        [0, 0, 0.5 * (1 - nu)]])

    def dK(r, s):
        B = BM(r, s)
        J = DET(JM(r, s))
        return h * B.T @ C @ B * J

    return IntegrateOn2DDomainWithGaussN2(dK)


def main() -> None:
    # Test elemanı: 1m × 0.5m dikdörtgen
    nodes_xy = [
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 0.5],
        [1.0, 0.5],
    ]
    E = 210e6        # kN/m² (210 GPa)
    nu = 0.3
    h = 0.10         # m

    K = compute_K(nodes_xy, E, nu, h)

    payload = {
        "case": "Plane-stress Q4 — 1m × 0.5m, E=210e6 kN/m², ν=0.3, h=0.1m",
        "nodes_xy": nodes_xy,
        "material": {"E": E, "nu": nu, "h": h},
        "K_local": K.tolist(),
    }
    out = Path(__file__).parent / "plane_stress_q4_golden.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {out}")
    print(f"K shape: {K.shape}")
    print(f"K[0][0] = {K[0][0]:.6f}")


if __name__ == "__main__":
    main()
