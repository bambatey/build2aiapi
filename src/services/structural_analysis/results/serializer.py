"""Pipeline çıktısını (``AnalysisResult``) Firestore'a uygun JSON'a çevir.

Tek-doküman stratejisi: küçük/orta modeller için tüm analiz sonucu tek
Firestore dokümanında yaşar. Büyük modeller (>1MB) için ileride
Storage'a gzip JSON olarak ayırma desteği eklenir.

NaN / inf değerleri JSON'a yazılmadan önce 0.0'a sanitize edilir
(aksi halde FastAPI JSON encoder 500 atar). NaN üretimi genelde
singular K matrisi ya da eksik eleman verisi göstergesidir — uyarı
loglanır.
"""

from __future__ import annotations

import logging
import math
from typing import Any

from ..pipeline import AnalysisResult, CaseResult

logger = logging.getLogger(__name__)


def analysis_to_persistable(result: AnalysisResult) -> dict[str, Any]:
    """Analiz sonucunu Firestore'a yazılabilir sözlüğe çevir."""
    return {
        "summary": result.summary,
        "cases": {
            case_id: _case_to_persistable(case)
            for case_id, case in result.cases.items()
            if case_id != "_empty"
        },
        "modes": [_mode_to_persistable(m) for m in result.modes],
    }


def _mode_to_persistable(mode) -> dict[str, Any]:
    return {
        "mode_no": mode.mode_no,
        "period": _safe(mode.period),
        "frequency": _safe(mode.frequency),
        "angular_frequency": _safe(mode.angular_frequency),
        "mass_participation": {
            k: _safe(v) for k, v in (mode.mass_participation or {}).items()
        },
        # Mod şekli tablosu (her düğüm için) — UI görselleştirmesi için
        "shape": [
            {"node_id": nid, **{k: _safe(v) for k, v in disp.items()}}
            for nid, disp in sorted(mode.shape.items())
        ],
    }


def case_summary_dict(case: CaseResult) -> dict[str, Any]:
    """Tek bir yük durumu için özet — hızlı list endpoint'leri için."""
    max_disp = 0.0
    for d in case.displacements.values():
        for v in d.values():
            vv = _safe(v)
            if abs(vv) > max_disp:
                max_disp = abs(vv)
    return {
        "case_id": case.case_id,
        "max_abs_displacement": max_disp,
        "n_nodes_with_reaction": len(case.reactions),
    }


def case_displacements_dict(case: CaseResult) -> list[dict[str, Any]]:
    """Her düğüm için yer değiştirme kaydı — NodeDisplacementDTO uyumlu."""
    out = []
    nan_count = 0
    for nid, disp in sorted(case.displacements.items()):
        clean = {}
        for k, v in disp.items():
            cv, was_bad = _sanitize(v)
            clean[k] = cv
            if was_bad:
                nan_count += 1
        out.append({"node_id": nid, "load_case": case.case_id, **clean})
    if nan_count:
        logger.warning(
            "Case %s: %d yer değiştirme değeri NaN/inf — 0.0'a düşürüldü "
            "(model sağlaması gerekli)",
            case.case_id, nan_count,
        )
    return out


def case_reactions_dict(case: CaseResult) -> list[dict[str, Any]]:
    """Her mesnet için reaksiyon kaydı — ReactionDTO uyumlu."""
    out = []
    nan_count = 0
    for nid, react in sorted(case.reactions.items()):
        clean = {}
        for k, v in react.items():
            cv, was_bad = _sanitize(v)
            clean[k] = cv
            if was_bad:
                nan_count += 1
        out.append({"node_id": nid, "load_case": case.case_id, **clean})
    if nan_count:
        logger.warning(
            "Case %s: %d reaksiyon değeri NaN/inf — 0.0'a düşürüldü",
            case.case_id, nan_count,
        )
    return out


def _sanitize(v: float) -> tuple[float, bool]:
    """NaN/inf → 0.0. İkinci dönüş değeri temizlenme olup olmadığıdır."""
    if isinstance(v, float) and not math.isfinite(v):
        return 0.0, True
    return float(v), False


def _safe(v: float) -> float:
    return _sanitize(v)[0]


def _case_to_persistable(case: CaseResult) -> dict[str, Any]:
    return {
        "case_id": case.case_id,
        "kind": getattr(case, "kind", "case"),
        "displacements": case_displacements_dict(case),
        "reactions": case_reactions_dict(case),
        "summary": case_summary_dict(case),
    }
