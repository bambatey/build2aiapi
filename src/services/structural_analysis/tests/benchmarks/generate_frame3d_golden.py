"""SEA_Book `sec5_frame_3d.py` matematiğinden altın çıktı üretir.

Bu script elle bir kez çalıştırılır: ``python generate_frame3d_golden.py``.
Orijinal ``ElementFrame3D`` sınıfının lokal/global rijitlik matrislerini ve
yayılı yük vektörlerini JSON olarak ``frame3d_golden.json``'a yazar.

Sonraki portlarda bu dosyadaki değerler karşılaştırma referansıdır.
Kaynak koda göre BIREBIR kopyalanmıştır; refactor edilmemiştir.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

INV = np.linalg.inv

_materials: dict[str, "Material"] = {}
_sections: dict[str, "Section"] = {}
_nodes: dict[str, "NodeFrame3D"] = {}


class Material:
    def __init__(self, id, E, p):
        self.id = id
        self.E = E
        self.p = p
        self.G = 0.5 * E / (1 + p)
        _materials[id] = self

    def props(self):
        return [self.E, self.p, self.G]


class Section:
    def __init__(self, id, A, Ix, Iy, Iz, Iyz):
        self.id = id
        self.A = A
        self.Ix = Ix
        self.Iy = Iy
        self.Iz = Iz
        self.Iyz = Iyz
        _sections[id] = self

    def props(self):
        return [self.A, self.Ix, self.Iy, self.Iz, self.Iyz]


class NodeFrame3D:
    def __init__(self, id, X, Y, Z):
        self.id = id
        self.X, self.Y, self.Z = X, Y, Z
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


class ElementFrame3D:
    def __init__(self, id, conn, mat, sec, omega=0, q=None):
        self.id = id
        self.conn = [_nodes[i] for i in conn]
        self.mat = _materials[mat]
        self.sec = _sections[sec]
        self.omega = np.pi / 180 * omega
        self.q = q if q is not None else [0] * 6

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
        nx, ny, nz, L = self.nx_ny_nz_L()
        if np.abs(1 - nz ** 2) < 0.001:
            TALFA = np.asarray([[nx, ny, nz], [1, 0, 0], [0, nz, -ny]])
        else:
            a = 1 / (1 - nz ** 2)
            TALFA = np.asarray(
                [
                    [nx, ny, nz],
                    [-a * nx * nz, -a * ny * nz, 1],
                    [a * ny, -a * nx, 0],
                ]
            )
        n1, n2 = self.conn
        TBETA1 = n1.getNodeLokalAxesTransformation()
        TBETA2 = n2.getNodeLokalAxesTransformation()
        _TLG = np.identity(12)
        _TLG[0:3, 0:3] = TOMG @ TALFA @ TBETA1
        _TLG[3:6, 3:6] = TOMG @ TALFA @ TBETA1
        _TLG[6:9, 6:9] = TOMG @ TALFA @ TBETA2
        _TLG[9:12, 9:12] = TOMG @ TALFA @ TBETA2
        return _TLG

    def K_Local(self):
        E, p, G = self.mat.props()
        A, Ix, Iy, Iz, Iyz = self.sec.props()
        EA, GIx, EIy, EIz, EIyz = E * A, G * Ix, E * Iy, E * Iz, E * Iyz
        nx, ny, nz, L = self.nx_ny_nz_L()
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
        nx, ny, nz, L = self.nx_ny_nz_L()
        return np.asarray(
            [
                0.5 * qx * L,
                0.5 * qy * L - mz,
                0.5 * qz * L + my,
                0.5 * mx * L,
                -qz * L ** 2 / 12,
                qy * L ** 2 / 12,
                0.5 * qx * L,
                0.5 * qy * L + mz,
                0.5 * qz * L - my,
                0.5 * mx * L,
                qz * L ** 2 / 12,
                -qy * L ** 2 / 12,
            ]
        )

    def B(self):
        T = self.TLG()
        return INV(T) @ self.q_Local()


def build_test_case() -> list[ElementFrame3D]:
    """sec5_frame_3d.py'deki aynı 2 elemanlı örnek."""
    Material(id="steel", E=210e6, p=0.3)
    Section(
        id="L",
        A=0.002364,
        Ix=0.00000002805048,
        Iy=0.00000458124835,
        Iz=0.00001597484835,
        Iyz=0.00000501624365,
    )
    NodeFrame3D(id="A", X=-3, Y=-3, Z=0)
    NodeFrame3D(id="B", X=3, Y=3, Z=0)
    NodeFrame3D(id="C", X=3, Y=-3, Z=3)
    e1 = ElementFrame3D(id=1, conn=["A", "C"], mat="steel", sec="L", omega=20)
    # Yayılı yük olayını da tetikleyelim ki q_Local/B karşılaştırılsın
    e2 = ElementFrame3D(id=2, conn=["B", "C"], mat="steel", sec="L", q=[1, 2, 3, 0.1, 0.2, 0.3])
    _nodes["C"].EulerZYX = [30, 20, 10]
    _nodes["A"].rest = [1, 1, 1, 1, 1, 1]
    _nodes["B"].rest = [1, 1, 1, 1, 1, 1]
    return [e1, e2]


def main() -> None:
    elements = build_test_case()
    payload = {
        "material": {"id": "steel", "E": 210e6, "p": 0.3},
        "section": {
            "id": "L",
            "A": 0.002364,
            "Ix": 0.00000002805048,
            "Iy": 0.00000458124835,
            "Iz": 0.00001597484835,
            "Iyz": 0.00000501624365,
        },
        "nodes": {
            id_: {
                "X": n.X,
                "Y": n.Y,
                "Z": n.Z,
                "EulerZYX": n.EulerZYX,
            }
            for id_, n in _nodes.items()
        },
        "elements": [],
    }
    for e in elements:
        payload["elements"].append(
            {
                "id": e.id,
                "conn": [n.id for n in e.conn],
                "omega_deg": float(np.degrees(e.omega)),
                "q": e.q,
                "L": e.nx_ny_nz_L()[3],
                "direction_cosines": list(e.nx_ny_nz_L()[:3]),
                "TLG": e.TLG().tolist(),
                "K_Local": e.K_Local().tolist(),
                "K": e.K().tolist(),
                "q_Local": e.q_Local().tolist(),
                "B": e.B().tolist(),
            }
        )

    out = Path(__file__).parent / "frame3d_golden.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
