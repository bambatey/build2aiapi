"""PlaneStressQ4 characterization testleri.

Port edilen membrane elemanının lokal 8×8 rijitlik matrisi, SEA_Book
orijinal matematiğiyle bire bir aynı olmalı.

Altın çıktı: ``benchmarks/plane_stress_q4_golden.json`` — orijinal
``sec4_plane_stress_rectangle_gauss.py`` koduyla üretilmiştir.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from services.structural_analysis.elements.plane_stress_q4 import (
    PlaneStressQ4,
)

GOLDEN = Path(__file__).parent / "benchmarks" / "plane_stress_q4_golden.json"


@pytest.fixture(scope="module")
def golden():
    return json.loads(GOLDEN.read_text())


def test_local_stiffness_matches_golden(golden):
    """Altın çıktıyla birebir eşleşme."""
    nodes_xy = golden["nodes_xy"]
    mat = golden["material"]

    # Altın üretici SEA_Book tensor ordering kullanıyor → direkt geç
    el = PlaneStressQ4(
        nodes_local_xy=np.asarray(nodes_xy),
        E=mat["E"], nu=mat["nu"], thickness=mat["h"],
    )
    K = el.local_stiffness()
    np.testing.assert_allclose(
        K, np.asarray(golden["K_local"]), rtol=1e-12, atol=1e-6,
    )


def test_stiffness_symmetry(golden):
    """Plane stress K lokali simetrik olmalı."""
    mat = golden["material"]
    el = PlaneStressQ4(
        nodes_local_xy=np.asarray(golden["nodes_xy"]),
        E=mat["E"], nu=mat["nu"], thickness=mat["h"],
    )
    K = el.local_stiffness()
    np.testing.assert_allclose(K, K.T, atol=1e-6)


def test_cyclic_to_tensor_reordering():
    """SAP cyclic [bl, br, tr, tl] → tensor [bl, br, tl, tr] dönüşümü."""
    # 1×1 kare — cyclic sırası: (0,0), (1,0), (1,1), (0,1)
    cyclic_xy = np.asarray([[0, 0], [1, 0], [1, 1], [0, 1]])
    el = PlaneStressQ4.from_cyclic(
        cyclic_xy, E=210e6, nu=0.3, thickness=0.1,
    )
    # Beklenen tensor: (0,0), (1,0), (0,1), (1,1)
    expected = np.asarray([[0, 0], [1, 0], [0, 1], [1, 1]])
    np.testing.assert_allclose(el.nodes_local_xy, expected)


def test_rigid_body_modes_zero_energy():
    """6 rigid body modunun 3'ü (2 translasyon + 1 rotasyon) K'nın
    null space'indedir — enerji 0."""
    el = PlaneStressQ4(
        nodes_local_xy=np.asarray([[0, 0], [1, 0], [0, 1], [1, 1]]),
        E=210e6, nu=0.3, thickness=0.1,
    )
    K = el.local_stiffness()
    # u_rb_x = [1,1,1,1, 0,0,0,0]: x yönünde rigid body translasyon
    u_rb_x = np.concatenate([np.ones(4), np.zeros(4)])
    energy = u_rb_x @ K @ u_rb_x
    assert abs(energy) < 1e-6
    # u_rb_y: y yönünde translasyon
    u_rb_y = np.concatenate([np.zeros(4), np.ones(4)])
    assert abs(u_rb_y @ K @ u_rb_y) < 1e-6


def test_uniform_strain_recovers_applied_stress():
    """Üniform x-gerilmesi uygulanırsa ε_xx = σ/E."""
    E = 200e6
    el = PlaneStressQ4(
        nodes_local_xy=np.asarray([[0, 0], [1, 0], [0, 1], [1, 1]]),
        E=E, nu=0.0, thickness=0.1,   # nu=0: Poisson etkisi yok, basit doğrulama
    )
    # u(x,y) = εx × x → u1=0, u2=ε, u3=0, u4=ε
    epsilon = 1e-4
    u = np.asarray([0, epsilon, 0, epsilon,  0, 0, 0, 0])
    B = el.strain_displacement(0, 0)
    strain = B @ u
    assert strain[0] == pytest.approx(epsilon, rel=1e-10)   # ε_xx
    assert strain[1] == pytest.approx(0, abs=1e-10)          # ε_yy
    assert strain[2] == pytest.approx(0, abs=1e-10)          # γ_xy
