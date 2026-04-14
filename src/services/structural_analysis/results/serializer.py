"""Pipeline çıktısını (``AnalysisResult``) Firestore'a uygun JSON'a çevir.

Tek-doküman stratejisi: küçük/orta modeller için tüm analiz sonucu tek
Firestore dokümanında yaşar. Büyük modeller (>1MB) için ileride
Storage'a gzip JSON olarak ayırma desteği eklenir.
"""

from __future__ import annotations

from typing import Any

from ..pipeline import AnalysisResult, CaseResult


def analysis_to_persistable(result: AnalysisResult) -> dict[str, Any]:
    """Analiz sonucunu Firestore'a yazılabilir sözlüğe çevir."""
    return {
        "summary": result.summary,
        "cases": {
            case_id: _case_to_persistable(case)
            for case_id, case in result.cases.items()
            if case_id != "_empty"
        },
    }


def case_summary_dict(case: CaseResult) -> dict[str, Any]:
    """Tek bir yük durumu için özet — hızlı list endpoint'leri için."""
    max_disp = 0.0
    for d in case.displacements.values():
        for v in d.values():
            if abs(v) > max_disp:
                max_disp = abs(v)
    return {
        "case_id": case.case_id,
        "max_abs_displacement": max_disp,
        "n_nodes_with_reaction": len(case.reactions),
    }


def case_displacements_dict(case: CaseResult) -> list[dict[str, Any]]:
    """Her düğüm için yer değiştirme kaydı — NodeDisplacementDTO uyumlu."""
    return [
        {
            "node_id": nid,
            "load_case": case.case_id,
            **disp,
        }
        for nid, disp in sorted(case.displacements.items())
    ]


def case_reactions_dict(case: CaseResult) -> list[dict[str, Any]]:
    """Her mesnet için reaksiyon kaydı — ReactionDTO uyumlu."""
    return [
        {
            "node_id": nid,
            "load_case": case.case_id,
            **react,
        }
        for nid, react in sorted(case.reactions.items())
    ]


def _case_to_persistable(case: CaseResult) -> dict[str, Any]:
    return {
        "case_id": case.case_id,
        "displacements": case_displacements_dict(case),
        "reactions": case_reactions_dict(case),
        "summary": case_summary_dict(case),
    }
