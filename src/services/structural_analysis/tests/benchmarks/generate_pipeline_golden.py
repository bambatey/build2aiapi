"""Tam pipeline (assemble → solve → recover) için altın çıktı üretici.

SEA_Book'un orijinal çözücüsünü (``bicg`` ile) aynı 3-düğümlü 2-elemanlı
test vakasında çalıştırır; sistem yer değiştirme (US) ve uç-kuvvet (PS)
vektörlerini JSON'a yazar.

DOF sıralaması: **partitioned** (serbestler [0..N-1], tutulular [N..M-1]).
Node id → code (DOF index) eşlemesi de kaydedilir ki test tarafı aynı
sırayı yeniden kursun.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

INV = np.linalg.inv

_materials: dict = {}
_sections: dict = {}
_nodes: dict = {}
_elements: dict = {}


class Material:
    def __init__(self, id, E, p):
        self.id, self.E, self.p = id, E, p
        self.G = 0.5 * E / (1 + p)
        _materials[id] = self

    def props(self):
        return [self.E, self.p, self.G]


class Section:
    def __init__(self, id, A, Ix, Iy, Iz, Iyz):
        self.id, self.A, self.Ix, self.Iy, self.Iz, self.Iyz = id, A, Ix, Iy, Iz, Iyz
        _sections[id] = self

    def props(self):
        return [self.A, self.Ix, self.Iy, self.Iz, self.Iyz]


class NodeFrame3D:
    def __init__(self, id, X, Y, Z):
        self.id, self.X, self.Y, self.Z = id, X, Y, Z
        self.rest = [0] * 6
        self.disp = [0] * 6
        self.code = [-1] * 6
        self.EulerZYX = [0] * 3
        self.GlobalForces = [0] * 6
        _nodes[id] = self

    def getNodeLokalAxesTransformation(self):
        ezd, eyd, exd = self.EulerZYX
        rx, ry, rz = np.pi / 180 * np.asarray([exd, eyd, ezd])
        cx, cy, cz = np.cos([rx, ry, rz])
        sx, sy, sz = np.sin([rx, ry, rz])
        return np.asarray(
            [
                [cy * cz, cz * sx * sy - cx * sz, sx * sz + cx * cz * sy],
                [cy * sz, cx * cz + sx * sy * sz, cx * sy * sz - cz * sx],
                [-sy, cy * sx, cx * cy],
            ]
        )

    @property
    def force(self):
        Px, Py, Pz, Mx, My, Mz = self.GlobalForces
        ET_INV = INV(self.getNodeLokalAxesTransformation())
        p = ET_INV @ [Px, Py, Pz]
        m = ET_INV @ [Mx, My, Mz]
        return [*p, *m]


class ElementFrame3D:
    def __init__(self, id, conn, mat, sec, omega=0, q=None):
        self.id = id
        self.conn = [_nodes[i] for i in conn]
        self.mat = _materials[mat]
        self.sec = _sections[sec]
        self.omega = np.pi / 180 * omega
        self.q = q if q is not None else [0] * 6
        _elements[id] = self

    def code(self):
        n1, n2 = self.conn
        return n1.code + n2.code

    def nx_ny_nz_L(self):
        n1, n2 = self.conn
        Lx, Ly, Lz = n2.X - n1.X, n2.Y - n1.Y, n2.Z - n1.Z
        L = (Lx ** 2 + Ly ** 2 + Lz ** 2) ** 0.5
        return Lx / L, Ly / L, Lz / L, L

    def TLG(self):
        omg = self.omega
        TOMG = np.asarray(
            [
                [1, 0, 0],
                [0, np.cos(omg), np.sin(omg)],
                [0, -np.sin(omg), np.cos(omg)],
            ]
        )
        nx, ny, nz, _ = self.nx_ny_nz_L()
        if np.abs(1 - nz ** 2) < 0.001:
            TALFA = np.asarray([[nx, ny, nz], [1, 0, 0], [0, nz, -ny]])
        else:
            a = 1 / (1 - nz ** 2)
            TALFA = np.asarray(
                [[nx, ny, nz], [-a * nx * nz, -a * ny * nz, 1], [a * ny, -a * nx, 0]]
            )
        TBETA1 = self.conn[0].getNodeLokalAxesTransformation()
        TBETA2 = self.conn[1].getNodeLokalAxesTransformation()
        T = np.identity(12)
        T[0:3, 0:3] = TOMG @ TALFA @ TBETA1
        T[3:6, 3:6] = TOMG @ TALFA @ TBETA1
        T[6:9, 6:9] = TOMG @ TALFA @ TBETA2
        T[9:12, 9:12] = TOMG @ TALFA @ TBETA2
        return T

    def K_Local(self):
        E, _, G = self.mat.props()
        A, Ix, Iy, Iz, Iyz = self.sec.props()
        EA, GIx, EIy, EIz, EIyz = E * A, G * Ix, E * Iy, E * Iz, E * Iyz
        _, _, _, L = self.nx_ny_nz_L()
        L2, L3 = L ** 2, L ** 3
        ku = EA / L
        ktx = GIx / L
        k1z, k1y, k1yz = 2 * EIz / L, 2 * EIy / L, 2 * EIyz / L
        k2z, k2y, k2yz = 6 * EIz / L2, 6 * EIy / L2, 6 * EIyz / L2
        k3z, k3y, k3yz = 12 * EIz / L3, 12 * EIy / L3, 12 * EIyz / L3
        return np.asarray(
            [
                [ku, 0, 0, 0, 0, 0, -ku, 0, 0, 0, 0, 0],
                [0, k3z, k3yz, 0, -k2yz, k2z, 0, -k3z, -k3yz, 0, -k2yz, k2z],
                [0, k3yz, k3y, 0, -k2y, k2yz, 0, -k3yz, -k3y, 0, -k2y, k2yz],
                [0, 0, 0, ktx, 0, 0, 0, 0, 0, -ktx, 0, 0],
                [0, -k2yz, -k2y, 0, 2 * k1y, -2 * k1yz, 0, k2yz, k2y, 0, k1y, -k1yz],
                [0, k2z, k2yz, 0, -2 * k1yz, 2 * k1z, 0, -k2z, -k2yz, 0, -k1yz, k1z],
                [-ku, 0, 0, 0, 0, 0, ku, 0, 0, 0, 0, 0],
                [0, -k3z, -k3yz, 0, k2yz, -k2z, 0, k3z, k3yz, 0, k2yz, -k2z],
                [0, -k3yz, -k3y, 0, k2y, -k2yz, 0, k3yz, k3y, 0, k2y, -k2yz],
                [0, 0, 0, -ktx, 0, 0, 0, 0, 0, ktx, 0, 0],
                [0, -k2yz, -k2y, 0, k1y, -k1yz, 0, k2yz, k2y, 0, 2 * k1y, -2 * k1yz],
                [0, k2z, k2yz, 0, -k1yz, k1z, 0, -k2z, -k2yz, 0, -2 * k1yz, 2 * k1z],
            ]
        )

    def K(self):
        T = self.TLG()
        return INV(T) @ self.K_Local() @ T

    def q_Local(self):
        qx, qy, qz, mx, my, mz = self.q
        _, _, _, L = self.nx_ny_nz_L()
        return np.asarray(
            [
                0.5 * qx * L, 0.5 * qy * L - mz, 0.5 * qz * L + my,
                0.5 * mx * L, -qz * L ** 2 / 12, qy * L ** 2 / 12,
                0.5 * qx * L, 0.5 * qy * L + mz, 0.5 * qz * L - my,
                0.5 * mx * L, qz * L ** 2 / 12, -qy * L ** 2 / 12,
            ]
        )

    def B(self):
        T = self.TLG()
        return INV(T) @ self.q_Local()


def solve(nodes, elements):
    """SEA_Book/solver/__init__.py'nin birebir aynısı."""
    M = 0
    for _, n in nodes.items():
        for i, r in enumerate(n.rest):
            if r == 0:
                n.code[i] = M
                M += 1
    N = M
    for _, n in nodes.items():
        for i, r in enumerate(n.rest):
            if r == 1:
                n.code[i] = M
                M += 1

    US = np.zeros(M)
    PS = np.zeros(M)
    RHS = np.zeros(M)
    rows, cols, data = [], [], []
    for _, e in elements.items():
        code = e.code()
        Ke = e.K()
        rows += np.repeat(code, len(code)).tolist()
        cols += code * len(code)
        data += Ke.flatten().tolist()
    KS = sp.coo_matrix((data, (rows, cols)), shape=(M, M), dtype=float).tocsc()
    for _, e in elements.items():
        RHS[e.code()] += e.B()
    for _, n in nodes.items():
        US[n.code] = n.disp
        PS[n.code] = n.force

    K11 = KS[0:N, 0:N]
    K12 = KS[0:N, N:M]
    K21 = KS[N:M, 0:N]
    K22 = KS[N:M, N:M]
    U2 = US[N:M]
    P1 = PS[0:N]
    RHS1 = RHS[0:N]
    RHS2 = RHS[N:M]
    for i in range(N):
        if abs(K11[i, i]) < 1e-10:
            K11[i, i] = 1e-5
    U1 = spla.bicg(K11, P1 + RHS1 - K12 @ U2, rtol=1e-9)[0]
    P2 = K21 @ U1 + K22 @ U2 - RHS2
    US_out = np.concatenate((U1, U2))
    PS_out = np.concatenate((P1, P2))
    return US_out, PS_out, N, M


def build_and_solve():
    Material(id="steel", E=210e6, p=0.3)
    Section(
        id="L", A=0.002364, Ix=0.00000002805048,
        Iy=0.00000458124835, Iz=0.00001597484835, Iyz=0.00000501624365,
    )
    NodeFrame3D(id="A", X=-3, Y=-3, Z=0)
    NodeFrame3D(id="B", X=3, Y=3, Z=0)
    NodeFrame3D(id="C", X=3, Y=-3, Z=3)
    ElementFrame3D(id=1, conn=["A", "C"], mat="steel", sec="L", omega=20)
    ElementFrame3D(id=2, conn=["B", "C"], mat="steel", sec="L")
    _nodes["C"].EulerZYX = [30, 20, 10]
    _nodes["A"].rest = [1, 1, 1, 1, 1, 1]
    _nodes["B"].rest = [1, 1, 1, 1, 1, 1]
    _nodes["C"].GlobalForces = [0, 0, -10, 0, 0, 0]
    return solve(_nodes, _elements)


def main() -> None:
    US, PS, N, M = build_and_solve()
    payload = {
        "case": "sec5_frame_3d minimal — 3 düğüm, 2 eleman, C düğümünde -10 kN Z yönünde",
        "N_free": int(N),
        "M_total": int(M),
        "node_codes": {id_: n.code for id_, n in _nodes.items()},
        "US": US.tolist(),
        "PS": PS.tolist(),
    }
    out = Path(__file__).parent / "pipeline_golden.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {out}  (N={N}, M={M})")
    print("US (ilk 6):", US[:6])
    print("PS (son 12 — mesnet reaksiyonları):", PS[-12:])


if __name__ == "__main__":
    main()
