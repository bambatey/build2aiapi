"""Statik lineer analiz pipeline orchestrator.

Kaba akış (METHOD.md §5):

    parse → validate → dof_numbering → assemble(K) → assemble(F)
    → solve(static) → recover(displacements, reactions) → AnalysisResult

Modal/spektrum/kombinasyonlar ileri fazlarda eklenecek.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .assembly import assemble_load_vectors, assemble_stiffness, number_dofs
from .model.dto import ModelDTO
from .parser import parse_s2k
from .recovery import node_displacements, node_reactions
from .solver import StaticSolution, solve_static


@dataclass
class CaseResult:
    """Tek bir yük durumunun sonucu."""

    case_id: str
    displacements: dict[int, dict[str, float]]
    reactions: dict[int, dict[str, float]]
    raw: StaticSolution


@dataclass
class AnalysisResult:
    """Tüm yük durumları için toplu sonuç."""

    model: ModelDTO
    cases: dict[str, CaseResult] = field(default_factory=dict)
    summary: dict[str, float] = field(default_factory=dict)


def run_static_analysis(model: ModelDTO) -> AnalysisResult:
    """Bir ``ModelDTO`` üzerinden tüm yük durumlarını statik çöz."""
    dof_map = number_dofs(model)
    K = assemble_stiffness(model, dof_map)
    load_vectors = assemble_load_vectors(model, dof_map)

    result = AnalysisResult(model=model)
    max_disp = 0.0
    for case_id, (PS, RHS, US) in load_vectors.items():
        sol = solve_static(case_id, K, PS, RHS, US, dof_map)
        disp = node_displacements(sol.U, dof_map)
        reacts = node_reactions(sol.P, dof_map, model)
        result.cases[case_id] = CaseResult(
            case_id=case_id, displacements=disp, reactions=reacts, raw=sol
        )
        case_max = float(np.max(np.abs(sol.U))) if sol.U.size else 0.0
        max_disp = max(max_disp, case_max)

    result.summary = {
        "n_nodes": len(model.nodes),
        "n_frame_elements": len(model.frame_elements),
        "n_shell_elements": len(model.shell_elements),
        "n_dofs_free": dof_map.n_free,
        "n_dofs_total": dof_map.n_total,
        "n_load_cases": len([c for c in result.cases if c != "_empty"]),
        "max_displacement": max_disp,
    }
    return result


def run_from_s2k(text: str) -> AnalysisResult:
    """Yardımcı: .s2k metninden ModelDTO parse et ve çöz."""
    return run_static_analysis(parse_s2k(text))
