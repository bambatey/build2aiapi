"""Üretilen .s2k dosyalarını doğrula — parser açabilmeli, en az 1 frame
ve 1 düğüm olmalı, ankastre düğümler sistemdeki bir düğüme işaret etmeli.

Generator çıkışı validator'dan geçer; geçmezse çağıran bilgilendirilir.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from services.structural_analysis.parser import parse_s2k


class GenerationError(RuntimeError):
    """Üretim veya validasyon hatası."""


@dataclass
class ValidationReport:
    ok: bool
    n_nodes: int = 0
    n_frames: int = 0
    n_load_cases: int = 0
    n_materials: int = 0
    n_sections: int = 0
    issues: list[str] = field(default_factory=list)


def validate_generated_model(s2k_text: str) -> ValidationReport:
    """Üretilen s2k metnini parseS2K ile aç ve temel sağlama kontrolleri yap.

    Başarısız olursa ``GenerationError`` fırlatır. Başarılıysa ``ValidationReport``
    döner (telemetri/log için).
    """
    try:
        model = parse_s2k(s2k_text)
    except Exception as exc:
        raise GenerationError(f"Üretilen dosya parse edilemedi: {exc}") from exc

    issues: list[str] = []
    if not model.nodes:
        issues.append("Hiç düğüm üretilmedi.")
    if not model.frame_elements:
        issues.append("Hiç frame elemanı üretilmedi.")
    if not model.materials:
        issues.append("Malzeme tanımı yok.")
    if not model.sections:
        issues.append("Kesit tanımı yok.")
    if not model.load_cases:
        issues.append("Yük durumu tanımı yok.")

    # En az bir ankastre düğüm olmalı (yoksa model rijit cisim hareketi yapar)
    restrained = [n for n in model.nodes.values() if any(n.restraints)]
    if not restrained:
        issues.append("Mesnet (restraint) atanmış düğüm yok.")

    # Frame'lerin joint'leri var mı
    for el in model.frame_elements.values():
        for nid in el.nodes:
            if nid not in model.nodes:
                issues.append(f"Frame #{el.id} joint {nid}'e işaret ediyor ama düğüm yok.")
                break

    ok = not issues
    report = ValidationReport(
        ok=ok,
        n_nodes=len(model.nodes),
        n_frames=len(model.frame_elements),
        n_load_cases=len(model.load_cases),
        n_materials=len(model.materials),
        n_sections=len(model.sections),
        issues=issues,
    )
    if not ok:
        raise GenerationError(
            "Üretilen model tutarsız:\n- " + "\n- ".join(issues)
        )
    return report
