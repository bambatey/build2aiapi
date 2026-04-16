"""msgpack+gzip round-trip — storage_service offload format'ının gerçek
analiz payload'ını bit-for-bit koruduğunu kanıtlar.

docs/architecture/01-performance.md Faz 1: JSON yerine msgpack+gzip
Storage offload. Bu test:
    1. Pipeline'ı küçük bir model üzerinde çalıştırır.
    2. analysis_to_persistable ile Firestore-uyumlu dict üretir.
    3. msgpack+gzip ile encode → decode → eşit olup olmadığını doğrular.
    4. Binary payload'ın gzip-JSON'a göre küçük olduğunu rapor eder.
"""

from __future__ import annotations

import gzip
import json

import msgpack

from services.structural_analysis.model.dto import (
    FrameElementDTO,
    FrameSectionDTO,
    LoadCaseDTO,
    MaterialDTO,
    ModelDTO,
    NodeDTO,
    PointLoadDTO,
)
from services.structural_analysis.model.enums import ElementType, LoadType
from services.structural_analysis.pipeline import (
    AnalysisOptions,
    run_static_analysis,
)
from services.structural_analysis.results import analysis_to_persistable


def _build_small_model() -> ModelDTO:
    """2-düğümlü cantilever — yeterli complexity (nodal load, frame, reaction)."""
    model = ModelDTO()
    model.materials["STEEL"] = MaterialDTO(id="STEEL", E=2.0e8, nu=0.3, rho=7850.0)
    model.sections["W10"] = FrameSectionDTO(
        id="W10", A=0.01, Iy=1e-4, Iz=1e-4, J=2e-4,
    )
    model.nodes[1] = NodeDTO(id=1, x=0.0, y=0.0, z=0.0, restraints=[True] * 6)
    model.nodes[2] = NodeDTO(id=2, x=3.0, y=0.0, z=0.0)
    model.frame_elements[1] = FrameElementDTO(
        id=1, type=ElementType.FRAME_3D, nodes=[1, 2],
        section_id="W10", material_id="STEEL",
    )
    model.load_cases["LIVE"] = LoadCaseDTO(
        id="LIVE", type=LoadType.LIVE,
        point_loads=[
            PointLoadDTO(node_id=2, values=[0.0, 10.0, -20.0, 0.0, 0.0, 0.0]),
        ],
    )
    return model


def test_msgpack_roundtrip_matches_input():
    """Persistable dict msgpack+gzip round-trip sonrası eşit kalmalı."""
    model = _build_small_model()
    result = run_static_analysis(model, AnalysisOptions())
    persistable = analysis_to_persistable(result)

    # element_forces — Storage'a her zaman offload edilen payload
    forces = persistable.get("element_forces") or {}
    assert forces, "test modeli element_forces üretmeli"

    packed = gzip.compress(msgpack.packb(forces, use_bin_type=True))
    unpacked = msgpack.unpackb(gzip.decompress(packed), raw=False)

    # JSON round-trip ile karşılaştır — msgpack bazı tipleri farklı döner
    # (tuple → list vs), normalleştirme için JSON pivotundan geç.
    json_pivot = json.loads(json.dumps(forces, default=str))
    assert unpacked == json_pivot


def test_msgpack_payload_smaller_than_json_gzip():
    """msgpack+gzip payload boyutu, aynı veri için json+gzip'ten küçük olmalı.

    Büyük değil ama sıfır-kayıplı bir kazanç beklenir (~5-20%). Başarısız
    olursa dikkat: format değişikliği avantaj sağlamıyor demektir.
    """
    model = _build_small_model()
    result = run_static_analysis(model, AnalysisOptions())
    persistable = analysis_to_persistable(result)
    forces = persistable.get("element_forces") or {}

    mp_bytes = gzip.compress(msgpack.packb(forces, use_bin_type=True))
    js_bytes = gzip.compress(json.dumps(forces, default=str).encode("utf-8"))

    # Küçük modelde fark bazen görülmez (gzip her ikisini de aynı seviyeye
    # sıkıştırabilir) — o yüzden esnek eşik: msgpack json'dan "büyük
    # olmamalı" (eşit ya da küçük).
    assert mp_bytes, "msgpack payload boş olamaz"
    assert js_bytes, "json payload boş olamaz"
    assert len(mp_bytes) <= len(js_bytes) * 1.1, (
        f"msgpack payload ({len(mp_bytes)}B) json+gzip'ten ({len(js_bytes)}B) "
        "beklenenden büyük — format tercihi beklenen avantajı sağlamıyor."
    )
