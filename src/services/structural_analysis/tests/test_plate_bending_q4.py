"""PlateBendingQ4 element-level testleri.

Gerçek FEM literatüründe "rigid body + patch test" ile bir plate
elemanının doğruluğu doğrulanır:

1. K simetrik olmalı
2. Rigid body modları (w=c, w=αx + θ_y=-α, w=αy + θ_x=α) sıfır enerjili
3. Sabit eğrilik alanında rigidity-verilen eğilme momenti M = D × κ

Konvansiyon: θ_x = +∂w/∂y, θ_y = -∂w/∂x (Kirchhoff limit). Bu neden:
γ_xz = ∂w/∂x + θ_y = 0 ⇒ θ_y = -∂w/∂x.
"""

from __future__ import annotations

import numpy as np
import pytest

from services.structural_analysis.elements.plate_bending_q4 import (
    PlateBendingQ4,
)


def _square_plate(a: float = 1.0) -> PlateBendingQ4:
    """a × a kare plak — tensor sıra."""
    return PlateBendingQ4(
        nodes_local_xy=np.asarray([[0, 0], [a, 0], [0, a], [a, a]]),
        E=210e6, nu=0.3, thickness=0.1,
    )


def test_stiffness_symmetry():
    K = _square_plate().local_stiffness()
    np.testing.assert_allclose(K, K.T, rtol=1e-9, atol=1e-6)


def test_rigid_translation_z_zero_energy():
    """w = c sabiti, θ = 0 → eleman enerjisiz hareket eder."""
    K = _square_plate().local_stiffness()
    u = np.zeros(12)
    u[0::3] = 1.0       # tüm w = 1
    # θ_x, θ_y = 0
    energy = u @ K @ u
    assert abs(energy) < 1e-4


def test_rigid_rotation_about_y_zero_energy():
    """w = α×x, θ_y = -α, θ_x = 0 → rigid rotation ekseni y."""
    a = 1.0
    el = _square_plate(a)
    alpha = 1e-3
    # tensor sırası: n_ll (0,0), n_lr (a,0), n_ul (0,a), n_ur (a,a)
    coords = el.nodes_local_xy
    u = np.zeros(12)
    for i in range(4):
        x_i = coords[i, 0]
        u[3 * i] = alpha * x_i          # w
        u[3 * i + 1] = 0.0              # θ_x
        u[3 * i + 2] = -alpha           # θ_y = -∂w/∂x = -α
    K = el.local_stiffness()
    energy = u @ K @ u
    # Rigid body olduğu için enerji ≈ 0. Bilinear Q4'te küçük artık
    # olabilir — max diyagonale göre mertebe olarak ihmal edilebilir
    # olduğunu kontrol et.
    max_diag = np.max(np.abs(K.diagonal()))
    assert energy < max_diag * 1e-10


def test_rigid_rotation_about_x_zero_energy():
    """w = α×y, θ_x = α, θ_y = 0 → rigid rotation ekseni x."""
    a = 1.0
    el = _square_plate(a)
    alpha = 1e-3
    coords = el.nodes_local_xy
    u = np.zeros(12)
    for i in range(4):
        y_i = coords[i, 1]
        u[3 * i] = alpha * y_i
        u[3 * i + 1] = alpha            # θ_x = +∂w/∂y = +α
        u[3 * i + 2] = 0.0
    K = el.local_stiffness()
    energy = u @ K @ u
    max_diag = np.max(np.abs(K.diagonal()))
    assert energy < max_diag * 1e-10


def test_null_space_has_at_least_three_modes():
    """K'nın sıfır özdeğerine sahip en az 3 vektörü olmalı (3 rigid mode)."""
    K = _square_plate().local_stiffness()
    eigvals = np.linalg.eigvalsh(K)
    # En küçük 3 özdeğer sıfıra çok yakın olmalı
    near_zero = np.sum(np.abs(eigvals) < np.max(eigvals) * 1e-9)
    assert near_zero >= 3, f"Beklenen 3 rigid mode, bulundu {near_zero}"


def test_D_b_matrix_formula():
    """D_b = Eh³/(12(1-ν²)) × [[1,ν,0],[ν,1,0],[0,0,(1-ν)/2]]."""
    el = _square_plate()
    D = el.D_b()
    expected_coef = 210e6 * 0.1 ** 3 / (12 * (1 - 0.3 ** 2))
    assert D[0, 0] == pytest.approx(expected_coef)
    assert D[0, 1] == pytest.approx(expected_coef * 0.3)
    assert D[2, 2] == pytest.approx(expected_coef * 0.35)   # (1-0.3)/2
    # Simetri
    np.testing.assert_allclose(D, D.T)
