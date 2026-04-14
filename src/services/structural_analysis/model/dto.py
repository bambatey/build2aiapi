"""Pydantic veri aktarım nesneleri (DTO).

METHOD.md §3'te tanımlanan şemaların ilk sürümü. Faz 1 için gerekli alanlar
tam, gelişmiş alanlar (shell, spektrum, vb.) yer tutucu olarak mevcut.
"""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field

from .enums import ElementType, LoadType


class UnitsDTO(BaseModel):
    """Model birim sistemi. SAP2000 .s2k genelde kN-m-C ya da N-mm-C kullanır."""

    length: Literal["m", "mm", "cm", "in", "ft"] = "m"
    force: Literal["N", "kN", "lb", "kip"] = "kN"
    temperature: Literal["C", "F"] = "C"


class NodeDTO(BaseModel):
    id: int
    x: float
    y: float
    z: float
    # [ux, uy, uz, rx, ry, rz] — True = tutulu, False = serbest
    restraints: list[bool] = Field(default_factory=lambda: [False] * 6)
    # Mesnet çökmesi / öteleme {"ux": 0.005} gibi
    settlements: dict[str, float] | None = None
    # Frame düğümü için lokal eksen Euler açıları [betaZ, betaY, betaX] (derece)
    euler_zyx: tuple[float, float, float] = (0.0, 0.0, 0.0)


class MaterialDTO(BaseModel):
    id: str
    E: float                          # Elastisite modülü
    nu: float                         # Poisson oranı
    rho: float = 0.0                  # Yoğunluk (öz ağırlık için)
    alpha: float = 0.0                # Isıl genleşme katsayısı


class FrameSectionDTO(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    kind: Literal["frame"] = "frame"
    A: float                          # Kesit alanı
    Iy: float = 0.0                   # y ekseni etrafı eğilme atalet
    Iz: float = 0.0                   # z ekseni etrafı eğilme atalet
    J: float = 0.0                    # Burulma atalet momenti (Ix)
    Iyz: float = 0.0                  # Asimetrik kesit çarpım atalet


class ShellSectionDTO(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    kind: Literal["shell"] = "shell"
    thickness: float


SectionDTO = Annotated[
    Union[FrameSectionDTO, ShellSectionDTO],
    Field(discriminator="kind"),
]


class FrameElementDTO(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int
    kind: Literal["frame"] = "frame"
    type: ElementType                 # frame_2d, frame_3d, truss_2d, truss_3d
    nodes: list[int]                  # [start, end]
    section_id: str
    material_id: str
    # Mafsal tanımları: {"start": ["mz"], "end": []}
    hinges: dict[str, list[str]] | None = None
    # Kesit duruş açısı (omega, derece) — frame_3d için
    local_axis_angle: float = 0.0


class ShellElementDTO(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int
    kind: Literal["shell"] = "shell"
    type: ElementType                 # plane_stress_q4, shell_dkq, ...
    nodes: list[int]                  # Q4 için 4, Q9 için 9, T3 için 3
    section_id: str
    material_id: str


ElementDTO = Annotated[
    Union[FrameElementDTO, ShellElementDTO],
    Field(discriminator="kind"),
]


class PointLoadDTO(BaseModel):
    node_id: int
    # [Fx, Fy, Fz, Mx, My, Mz] global eksende
    values: list[float] = Field(default_factory=lambda: [0.0] * 6)


class DistributedLoadDTO(BaseModel):
    """Yayılı çizgi yükü — SAP2000 FRAME LOADS - DISTRIBUTED satırına birebir karşılık.

    Üniform yük için ``magnitude_a == magnitude_b``. Trapez için farklı.
    Yönü eksen sistemiyle birlikte tutulur; yerel/global dönüşüm assembler
    aşamasında yapılır.
    """

    element_id: int
    coord_sys: Literal["local", "global"] = "global"
    # SAP: Gravity (global -Z), X/Y/Z global, 1/2/3 lokal eksen
    direction: Literal["gravity", "x", "y", "z", "local_1", "local_2", "local_3"] = "gravity"
    kind: Literal["force", "moment"] = "force"
    magnitude_a: float = 0.0
    magnitude_b: float = 0.0
    rel_dist_a: float = 0.0
    rel_dist_b: float = 1.0


class LoadCaseDTO(BaseModel):
    id: str
    type: LoadType = LoadType.OTHER
    point_loads: list[PointLoadDTO] = Field(default_factory=list)
    distributed_loads: list[DistributedLoadDTO] = Field(default_factory=list)
    self_weight_factor: float = 0.0   # 1.0 = tam öz ağırlık dahil


class CombinationDTO(BaseModel):
    id: str
    factors: dict[str, float]         # {"DEAD": 1.4, "LIVE": 1.6}


class ModelDTO(BaseModel):
    """Parser/mutator çıktısı — analiz motorunun tek kanonik girdisi.

    Frame ve shell elemanları ayrı sözlüklerde tutulur — SAP2000 her iki
    elemanı da birbirinden bağımsız id dizisinde numaralandırır (ör. aynı
    dosyada Frame=14 ve Area=14 birlikte olabilir).
    """

    model_config = ConfigDict(extra="forbid")

    units: UnitsDTO = Field(default_factory=UnitsDTO)
    nodes: dict[int, NodeDTO] = Field(default_factory=dict)
    frame_elements: dict[int, FrameElementDTO] = Field(default_factory=dict)
    shell_elements: dict[int, ShellElementDTO] = Field(default_factory=dict)
    materials: dict[str, MaterialDTO] = Field(default_factory=dict)
    sections: dict[str, SectionDTO] = Field(default_factory=dict)
    load_cases: dict[str, LoadCaseDTO] = Field(default_factory=dict)
    combinations: list[CombinationDTO] = Field(default_factory=list)
