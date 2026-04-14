"""Model sağlaması — çözüm öncesi bozuk veriyi tespit edip raporlar.

Amaç: singular K'ya yol açan yaygın durumları (sıfır rijitlik, bağlantısız
düğüm, eksik kesit/malzeme) pipeline çalışmadan önce yakalamak ve hangi
elemanların problemli olduğunu net olarak bildirmek.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..model.dto import FrameSectionDTO, ModelDTO


@dataclass
class ValidationIssue:
    severity: str            # "warning" | "error"
    code: str                # makine okunabilir kod
    message: str             # insan okur
    element_id: int | None = None
    node_id: int | None = None


@dataclass
class ValidationReport:
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    def has_errors(self) -> bool:
        return any(i.severity == "error" for i in self.issues)


def validate_model(model: ModelDTO) -> ValidationReport:
    """Modeli sağla ve bulunan sorunları ``ValidationReport`` olarak döndür."""
    report = ValidationReport()

    _check_global_model_integrity(model, report)
    _check_materials(model, report)
    _check_sections(model, report)
    _check_frame_elements(model, report)
    _check_node_connectivity(model, report)

    return report


def _check_global_model_integrity(model: ModelDTO, report: ValidationReport) -> None:
    """SAP'tan eksik export edilen modellerde en kritik 2 eksiği yakala."""
    total_restrained_dofs = sum(
        sum(1 for r in n.restraints if r) for n in model.nodes.values()
    )
    if model.nodes and total_restrained_dofs == 0:
        report.issues.append(
            ValidationIssue(
                "error", "no_restraints",
                f"Modelde {len(model.nodes)} düğüm var ama hiç mesnet tanımlı değil. "
                "SAP dosyasında 'JOINT RESTRAINT ASSIGNMENTS' tablosu eksik. "
                "SAP'ta File → Export → SAP2000 .s2k File ile tüm tabloları "
                "seçerek yeniden export edin.",
            )
        )

    frames_without_section = sum(
        1 for el in model.frame_elements.values() if not el.section_id
    )
    if model.frame_elements and frames_without_section == len(model.frame_elements):
        report.issues.append(
            ValidationIssue(
                "error", "no_section_assignments",
                f"{frames_without_section} frame'in kesit ataması yok. "
                "SAP dosyasında 'FRAME SECTION ASSIGNMENTS' tablosu eksik. "
                "Bu elemanlar K matrisine katkı sağlayamaz.",
            )
        )
    elif frames_without_section > 0:
        report.issues.append(
            ValidationIssue(
                "warning", "partial_section_assignments",
                f"{frames_without_section}/{len(model.frame_elements)} frame'e "
                "kesit atanmamış — bu elemanlar K'ya dahil edilmeyecek.",
            )
        )


def _check_materials(model: ModelDTO, report: ValidationReport) -> None:
    if not model.materials:
        report.issues.append(
            ValidationIssue("error", "no_materials", "Modelde hiç malzeme yok.")
        )
        return
    for mid, mat in model.materials.items():
        if not math.isfinite(mat.E) or mat.E <= 0:
            report.issues.append(
                ValidationIssue(
                    "error", "material_zero_E",
                    f"Malzeme {mid!r}: E={mat.E} (≤0 veya geçersiz). "
                    "Bu malzemeye bağlı tüm elemanlar singular K üretir.",
                )
            )
        if mat.nu < 0 or mat.nu >= 0.5:
            report.issues.append(
                ValidationIssue(
                    "warning", "material_bad_nu",
                    f"Malzeme {mid!r}: ν={mat.nu} fiziksel aralık [0, 0.5) dışında.",
                )
            )


def _check_sections(model: ModelDTO, report: ValidationReport) -> None:
    for sid, sec in model.sections.items():
        if not isinstance(sec, FrameSectionDTO):
            continue
        if sec.A <= 0:
            report.issues.append(
                ValidationIssue(
                    "error", "section_zero_A",
                    f"Kesit {sid!r}: A={sec.A} (eksen kuvveti singular olur).",
                )
            )
        # J=0 burulma rijitliği sıfır → singular. Iy/Iz=0 eğilme singular.
        if sec.J <= 0:
            report.issues.append(
                ValidationIssue(
                    "error", "section_zero_J",
                    f"Kesit {sid!r}: J={sec.J} (burulma rijitliği sıfır).",
                )
            )
        if sec.Iy <= 0:
            report.issues.append(
                ValidationIssue(
                    "error", "section_zero_Iy",
                    f"Kesit {sid!r}: Iy={sec.Iy} (minor eksen eğilme rijitliği sıfır).",
                )
            )
        if sec.Iz <= 0:
            report.issues.append(
                ValidationIssue(
                    "error", "section_zero_Iz",
                    f"Kesit {sid!r}: Iz={sec.Iz} (major eksen eğilme rijitliği sıfır).",
                )
            )


def _check_frame_elements(model: ModelDTO, report: ValidationReport) -> None:
    for eid, el in model.frame_elements.items():
        # Bağlanan iki düğüm de modelde var mı?
        missing_nodes = [n for n in el.nodes if n not in model.nodes]
        if missing_nodes:
            report.issues.append(
                ValidationIssue(
                    "error", "element_missing_node",
                    f"Frame {eid}: tanımsız düğüm referansı {missing_nodes}.",
                    element_id=eid,
                )
            )
            continue
        # Sıfır uzunluk
        n1, n2 = model.nodes[el.nodes[0]], model.nodes[el.nodes[1]]
        L = math.sqrt((n2.x - n1.x) ** 2 + (n2.y - n1.y) ** 2 + (n2.z - n1.z) ** 2)
        if L < 1e-9:
            report.issues.append(
                ValidationIssue(
                    "error", "element_zero_length",
                    f"Frame {eid}: düğüm {el.nodes[0]}↔{el.nodes[1]} arasında L≈0.",
                    element_id=eid,
                )
            )
        # Kesit / malzeme atanmış mı?
        if not el.section_id:
            report.issues.append(
                ValidationIssue(
                    "error", "element_no_section",
                    f"Frame {eid}: kesit atanmamış.",
                    element_id=eid,
                )
            )
        elif el.section_id not in model.sections:
            report.issues.append(
                ValidationIssue(
                    "error", "element_unknown_section",
                    f"Frame {eid}: tanımsız kesit {el.section_id!r}.",
                    element_id=eid,
                )
            )
        if not el.material_id:
            report.issues.append(
                ValidationIssue(
                    "error", "element_no_material",
                    f"Frame {eid}: malzeme atanmamış.",
                    element_id=eid,
                )
            )
        elif el.material_id not in model.materials:
            report.issues.append(
                ValidationIssue(
                    "error", "element_unknown_material",
                    f"Frame {eid}: tanımsız malzeme {el.material_id!r}.",
                    element_id=eid,
                )
            )


def _check_node_connectivity(model: ModelDTO, report: ValidationReport) -> None:
    """Bağlantısız veya sadece-shell'e-bağlı düğümleri yakala.

    Motorun şu anki sürümü shell elemanlarını K'ya KATMAZ. Dolayısıyla
    sadece shell'e bağlı olan bir düğüm — frame ile desteklenmiyorsa —
    efektif olarak bağlantısızdır, K matrisi singular olur.
    """
    frame_referenced: set[int] = set()
    for el in model.frame_elements.values():
        frame_referenced.update(el.nodes)

    shell_referenced: set[int] = set()
    for el in model.shell_elements.values():
        shell_referenced.update(el.nodes)

    shell_only_count = 0
    for nid, node in model.nodes.items():
        is_restrained = any(node.restraints)
        if nid in frame_referenced or is_restrained:
            continue
        if nid in shell_referenced:
            shell_only_count += 1
            continue
        report.issues.append(
            ValidationIssue(
                "error", "node_disconnected",
                f"Düğüm {nid}: hiçbir elemana bağlı değil ve tutulu değil "
                "— K matrisinde sıfır satır oluşur.",
                node_id=nid,
            )
        )
    if shell_only_count > 0:
        report.issues.append(
            ValidationIssue(
                "error", "shell_only_nodes",
                f"{shell_only_count} düğüm yalnızca kabuk (shell) elemanlarına bağlı. "
                "Motorun şu anki sürümü shell rijitliğini hesaba katmıyor → bu "
                "düğümler K matrisinde bağlantısız kalır, sistem singular olur. "
                "Çözümler: (1) shell kapsamı sonraki iterasyon, "
                "(2) ilgili shell düğümlerini frame ile bağlayın, "
                "(3) shell'siz bir analiz modeli kullanın.",
            )
        )
