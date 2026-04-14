"""Statik lineer analiz pipeline orchestrator.

Kaba akış (METHOD.md §5):

    parse → validate → dof_numbering → assemble(K, M?) → assemble(F)
    → solve(static | modal) → combinations → recover → AnalysisResult

Desteklenenler:
- Statik lineer çözüm (yük durumu bazında)
- Yük durumu filtreleme (``options.selected_load_cases``)
- Yük kombinasyonları — lineer süperpozisyon
- Modal analiz (lumped mass + eigsh)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np

from .assembly import (
    assemble_load_vectors,
    assemble_stiffness,
    build_diaphragm_transform,
    number_dofs,
)
from .model.dto import CombinationDTO, ModelDTO
from .parser import parse_s2k
from .recovery import node_displacements, node_reactions
from .solver import StaticSolution, solve_static
from .validation import validate_model

logger = logging.getLogger(__name__)


@dataclass
class SpectrumOptions:
    """Response spectrum parametreleri (TBDY 2018)."""

    Ss: float = 1.0           # Kısa periyot harita ivmesi (g)
    S1: float = 0.3           # 1-sn harita ivmesi (g)
    soil: str = "ZC"          # ZA..ZE
    R: float = 4.0            # Davranış katsayısı
    I: float = 1.0            # Önem katsayısı
    run_x: bool = True
    run_y: bool = True


@dataclass
class AnalysisOptions:
    """Pipeline çalıştırıcıya verilen seçenekler."""

    # None = hepsi, boş liste = hiçbiri
    selected_load_cases: Optional[list[str]] = None
    selected_combinations: Optional[list[str]] = None
    run_modal: bool = False
    modal_n_modes: int = 12
    run_response_spectrum: bool = False
    spectrum: SpectrumOptions | None = None


@dataclass
class CaseResult:
    """Tek bir yük durumu ya da kombinasyonun sonucu."""

    case_id: str
    displacements: dict[int, dict[str, float]]
    reactions: dict[int, dict[str, float]]
    raw: StaticSolution
    kind: str = "case"     # "case" | "combination"


@dataclass
class ModeResult:
    mode_no: int
    period: float
    frequency: float
    angular_frequency: float
    shape: dict[int, dict[str, float]]   # node_id → {ux, uy, uz, rx, ry, rz}


@dataclass
class AnalysisResult:
    model: ModelDTO
    cases: dict[str, CaseResult] = field(default_factory=dict)
    modes: list[ModeResult] = field(default_factory=list)
    summary: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


# --------------------------------------------------------------------- main
def run_static_analysis(
    model: ModelDTO, options: AnalysisOptions | None = None
) -> AnalysisResult:
    """Statik + (opsiyonel) kombinasyon + (opsiyonel) modal."""
    options = options or AnalysisOptions()

    # Çözüm öncesi sağlama
    report = validate_model(model)
    warnings_list: list[str] = []
    for issue in report.issues:
        msg = f"[{issue.code}] {issue.message}"
        if issue.severity == "error":
            logger.error(msg)
        else:
            logger.warning(msg)
        warnings_list.append(msg)
    if report.has_errors():
        logger.error(
            "Model sağlama: %d hata — çözüm denenecek ama sonuçlar "
            "singular K yüzünden NaN içerebilir.",
            len(report.errors),
        )

    dof_map = number_dofs(model)
    K = assemble_stiffness(model, dof_map)
    load_vectors = assemble_load_vectors(model, dof_map)
    diaphragm_transform = build_diaphragm_transform(model, dof_map)
    if diaphragm_transform:
        logger.info(
            "Rijit diyafram: %d slave DOF elenecek (reduced N_free=%d, N_total=%d).",
            diaphragm_transform.n_slaves_eliminated,
            diaphragm_transform.n_free_reduced,
            diaphragm_transform.n_total_reduced,
        )

    # Hangi yük durumlarını çözeceğiz?
    available_cases = {cid for cid in load_vectors if cid != "_empty"}
    selected_cases = _resolve_selection(
        all_ids=available_cases,
        selection=options.selected_load_cases,
    )
    # Kombinasyonlar için referans edilen base case'ler her zaman çözülmeli
    selected_combos = _resolve_selection(
        all_ids=[c.id for c in model.combinations],
        selection=options.selected_combinations,
    )
    referenced_by_combos: set[str] = set()
    unknown_refs: dict[str, list[str]] = {}   # combo_id → [missing base case ids]
    for c in model.combinations:
        if c.id not in selected_combos:
            continue
        for case_id in c.factors:
            if case_id in available_cases:
                referenced_by_combos.add(case_id)
            else:
                # Nested kombinasyon ya da tanımsız referans — pipeline'da çözülemez
                unknown_refs.setdefault(c.id, []).append(case_id)
    for combo_id, missing in unknown_refs.items():
        msg = (
            f"[combo_unknown_refs] Kombinasyon {combo_id!r}: {len(missing)} base case "
            f"referansı bulunamadı (iç içe kombinasyon olabilir): {missing[:5]}"
        )
        logger.warning(msg)
        warnings_list.append(msg)
    # SADECE gerçekten var olan base case'leri çöz — combinasyon adları buraya düşmesin
    cases_to_solve = (set(selected_cases) | referenced_by_combos) & available_cases

    result = AnalysisResult(model=model)
    max_disp = 0.0

    # --- Base yük durumları
    for case_id in cases_to_solve:
        PS, RHS, US = load_vectors[case_id]
        sol = solve_static(case_id, K, PS, RHS, US, dof_map, diaphragm_transform)
        disp = node_displacements(sol.U, dof_map)
        reacts = node_reactions(sol.P, dof_map, model)
        result.cases[case_id] = CaseResult(
            case_id=case_id, displacements=disp, reactions=reacts,
            raw=sol, kind="case",
        )
        max_disp = max(max_disp, _finite_max_abs(sol.U))

    # --- Kombinasyonlar: lineer süperpozisyon
    for combo in model.combinations:
        if combo.id not in selected_combos:
            continue
        combo_result = _combine(combo, result.cases, dof_map, model)
        if combo_result is not None:
            result.cases[combo.id] = combo_result
            max_disp = max(max_disp, _finite_max_abs(combo_result.raw.U))

    # --- Modal analiz (opsiyonel ya da RS istendiyse zorunlu)
    periods: list[float] = []
    needs_modal = options.run_modal or options.run_response_spectrum
    M_matrix = None
    if needs_modal:
        from .assembly.mass_assembler import assemble_mass
        from .solver.modal_solver import solve_modal

        try:
            # MASS SOURCE yük bazlıysa, referans load pattern'lerin çözülmüş
            # olmasına gerek yok — yalnızca F vektörleri lazım. load_vectors
            # zaten tüm pattern'ler için oluşturuldu.
            M_matrix = assemble_mass(model, dof_map, load_vectors=load_vectors)
            result.modes = solve_modal(
                K, M_matrix, dof_map, options.modal_n_modes,
            )
            periods = [m.period for m in result.modes]
        except Exception as exc:
            logger.exception("Modal analiz başarısız: %s", exc)
            warnings_list.append(f"[modal_failed] Modal analiz başarısız: {exc}")

    # --- Response spectrum (opsiyonel)
    if options.run_response_spectrum and result.modes and M_matrix is not None:
        from .solver.response_spectrum import solve_response_spectrum
        from .solver.static_solver import StaticSolution
        from .spectra.tbdy_2018 import TBDY2018Spectrum

        spec_opts = options.spectrum or SpectrumOptions()
        spectrum = TBDY2018Spectrum(
            Ss=spec_opts.Ss, S1=spec_opts.S1, soil=spec_opts.soil,  # type: ignore[arg-type]
            R=spec_opts.R, I=spec_opts.I,
        )
        directions: list[tuple[str, str]] = []
        if spec_opts.run_x:
            directions.append(("x", "EQX_RS"))
        if spec_opts.run_y:
            directions.append(("y", "EQY_RS"))
        for dir_code, case_id in directions:
            try:
                rs = solve_response_spectrum(
                    K, M_matrix, result.modes, dof_map, spectrum, dir_code,
                )
                # SRSS sonucundan reaksiyon = K·U (RHS yok)
                P = K @ rs.U
                raw = StaticSolution(case_id=case_id, U=rs.U, P=P)
                disp = node_displacements(rs.U, dof_map)
                reacts = node_reactions(P, dof_map, model)
                result.cases[case_id] = CaseResult(
                    case_id=case_id, displacements=disp, reactions=reacts,
                    raw=raw, kind="response_spectrum",
                )
                max_disp = max(max_disp, _finite_max_abs(rs.U))
            except Exception as exc:
                logger.exception("RS başarısız (%s): %s", dir_code, exc)
                warnings_list.append(f"[rs_failed_{dir_code}] {exc}")

    # --- Özet
    n_case_selected = sum(
        1 for cid in result.cases
        if result.cases[cid].kind == "case" and cid in selected_cases
    )
    n_combo_selected = sum(
        1 for cid in result.cases
        if result.cases[cid].kind == "combination"
    )
    n_rs = sum(
        1 for c in result.cases.values() if c.kind == "response_spectrum"
    )
    result.summary = {
        "n_nodes": len(model.nodes),
        "n_frame_elements": len(model.frame_elements),
        "n_shell_elements": len(model.shell_elements),
        "n_dofs_free": dof_map.n_free,
        "n_dofs_total": dof_map.n_total,
        "n_load_cases": n_case_selected,
        "n_combinations": n_combo_selected,
        "n_response_spectrum": n_rs,
        "n_modes": len(result.modes),
        "fundamental_period": periods[0] if periods else 0.0,
        "max_displacement": max_disp,
    }
    result.warnings = warnings_list
    return result


def run_from_s2k(text: str, options: AnalysisOptions | None = None) -> AnalysisResult:
    return run_static_analysis(parse_s2k(text), options)


# ------------------------------------------------------------------- helpers
def _resolve_selection(
    all_ids: Iterable[str], selection: list[str] | None
) -> set[str]:
    all_set = set(all_ids)
    if selection is None:
        return all_set
    return {s for s in selection if s in all_set}


def _finite_max_abs(u: np.ndarray) -> float:
    if u.size == 0:
        return 0.0
    finite = u[np.isfinite(u)]
    return float(np.max(np.abs(finite))) if finite.size else 0.0


def _combine(
    combo: CombinationDTO,
    base_cases: dict[str, CaseResult],
    dof_map,
    model: ModelDTO,
) -> CaseResult | None:
    """Lineer süperpozisyon: U_combo = Σ factor × U_case."""
    referenced = list(combo.factors.items())
    if not referenced:
        return None
    # Eksik base case → uyar, yine de geri kalanla hesapla
    valid = [(cid, f) for cid, f in referenced if cid in base_cases]
    if not valid:
        logger.warning(
            "Kombinasyon %s: referans edilen yük durumları çözülmemiş, atlanıyor.",
            combo.id,
        )
        return None

    first_U = base_cases[valid[0][0]].raw.U
    U_combo = np.zeros_like(first_U)
    P_combo = np.zeros_like(first_U)
    for cid, factor in valid:
        sol = base_cases[cid].raw
        U_combo = U_combo + factor * sol.U
        P_combo = P_combo + factor * sol.P

    raw = StaticSolution(case_id=combo.id, U=U_combo, P=P_combo)
    disp = node_displacements(U_combo, dof_map)
    reacts = node_reactions(P_combo, dof_map, model)
    return CaseResult(
        case_id=combo.id, displacements=disp, reactions=reacts,
        raw=raw, kind="combination",
    )
