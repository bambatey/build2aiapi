"""SAP2000 .s2k parser.

MVP kapsam:
    - Birim sistemi, malzemeler, kesitler, düğümler, mesnetler
    - Çerçeve/kabuk elemanları + kesit atamaları
    - Yük desenleri, düğüm yükleri, yayılı frame yükleri, kombinasyonlar
    - Frame releases (mafsallar) — FRAME RELEASE ASSIGNMENTS 1
    - Rijit diyafram — CONSTRAINT DEFINITIONS - DIAPHRAGM + JOINT CONSTRAINT
    - Area → frame yük aktarımı — AREA LOADS - UNIFORM TO FRAME

Henüz eklenmemiş: response spectrum fonksiyonları, time history, nonlineer
case tanımları — motor tarafından da kullanılmayan tablolar.
"""

from __future__ import annotations

import math
import re

from ..exceptions import ParseError
from ..model.dto import (
    AreaUniformLoadDTO,
    CombinationDTO,
    DiaphragmDTO,
    DistributedLoadDTO,
    FrameElementDTO,
    FrameSectionDTO,
    LoadCaseDTO,
    MassSourceDTO,
    MassSourcePatternDTO,
    MaterialDTO,
    ModelDTO,
    NodeDTO,
    PointLoadDTO,
    ShellElementDTO,
    ShellSectionDTO,
    UnitsDTO,
)
from ..model.enums import ElementType, LoadType
from .tables import extract_tables
from .tokens import to_float, to_int

_JOINT_TABLES = ("JOINT COORDINATES", "POINT COORDINATES")
_FRAME_TABLES = ("CONNECTIVITY - FRAME", "CONNECTIVITY - LINE")

_UNITS_RE = re.compile(r'"?\s*([A-Za-z]+)\s*,\s*([A-Za-z]+)\s*,\s*([A-Za-z]+)\s*"?')

_FORCE_UNIT_MAP = {"N": "N", "KN": "kN", "LB": "lb", "KIP": "kip"}
_LENGTH_UNIT_MAP = {"M": "m", "MM": "mm", "CM": "cm", "IN": "in", "FT": "ft"}
_TEMP_UNIT_MAP = {"C": "C", "F": "F"}

_LOAD_TYPE_MAP = {
    "DEAD": LoadType.DEAD,
    "LIVE": LoadType.LIVE,
    "WIND": LoadType.WIND,
    "SNOW": LoadType.SNOW,
    "QUAKE": LoadType.EARTHQUAKE_X,   # yön sonradan adından çıkarılır
    "TEMPERATURE": LoadType.TEMPERATURE,
}

_SAP_DIR_MAP = {
    "GRAVITY": "gravity",
    "X": "x",
    "Y": "y",
    "Z": "z",
    "1": "local_1",
    "2": "local_2",
    "3": "local_3",
}


class S2KParser:
    """Durumsuz parser. ``parse(text)`` → ``ModelDTO``."""

    def parse(self, text: str) -> ModelDTO:
        if not text or not text.strip():
            raise ParseError(".s2k içeriği boş.")

        tables = extract_tables(text)

        model = ModelDTO()
        self._parse_units(tables, model)
        self._parse_materials(tables, model)
        self._parse_frame_sections(tables, model)
        self._parse_shell_sections(tables, model)
        self._parse_nodes(tables, model)
        self._parse_restraints(tables, model)
        self._parse_frame_elements(tables, model)
        self._parse_frame_assignments(tables, model)
        self._parse_frame_releases(tables, model)
        self._parse_shell_elements(tables, model)
        self._parse_shell_assignments(tables, model)
        self._parse_diaphragms(tables, model)
        self._parse_load_patterns(tables, model)
        self._parse_joint_loads(tables, model)
        self._parse_distributed_loads(tables, model)
        self._parse_area_uniform_loads(tables, model)
        self._parse_combinations(tables, model)
        self._parse_mass_source(tables, model)
        return model

    # ----------------------------------------------------- MASS SOURCE
    def _parse_mass_source(self, tables, model: ModelDTO) -> None:
        """SAP MASS SOURCE tablosu — kütle kaynağı.

        Format (tipik TBDY):
            MassSource=MS1  Elements=No  Masses=No  Loads=Yes  IsDefault=Yes
                            LoadPat=G  Multiplier=1
            MassSource=MS1                          LoadPat=Q  Multiplier=0.3

        İlk satırda flag'ler, sonraki satırlarda load pattern eklemeleri.
        """
        rows = tables.get("MASS SOURCE", [])
        if not rows:
            return
        # Birden fazla mass source olabilir — IsDefault olanı seç, yoksa ilk
        by_name: dict[str, dict] = {}
        for row in rows:
            name = row.get("MassSource") or "default"
            entry = by_name.setdefault(name, {
                "from_elements": False,
                "from_masses": False,
                "from_loads": False,
                "is_default": False,
                "patterns": [],
            })
            # Flag alanları yalnızca ilk satırda bulunur
            if "Elements" in row:
                entry["from_elements"] = _yn(row.get("Elements"))
            if "Masses" in row:
                entry["from_masses"] = _yn(row.get("Masses"))
            if "Loads" in row:
                entry["from_loads"] = _yn(row.get("Loads"))
            if "IsDefault" in row:
                entry["is_default"] = _yn(row.get("IsDefault"))
            load_pat = row.get("LoadPat")
            mult = to_float(row.get("Multiplier"))
            if load_pat and math.isfinite(mult):
                entry["patterns"].append(
                    MassSourcePatternDTO(load_pat=load_pat, multiplier=mult)
                )

        # Default olanı seç (yoksa ilk)
        chosen_name: str | None = None
        for name, ent in by_name.items():
            if ent["is_default"]:
                chosen_name = name
                break
        if chosen_name is None and by_name:
            chosen_name = next(iter(by_name))
        if chosen_name is None:
            return
        ent = by_name[chosen_name]
        model.mass_source = MassSourceDTO(
            name=chosen_name,
            from_elements=ent["from_elements"],
            from_masses=ent["from_masses"],
            from_loads=ent["from_loads"],
            is_default=ent["is_default"],
            load_patterns=ent["patterns"],
        )

    # ---------------------------------------------------- frame releases
    def _parse_frame_releases(self, tables, model: ModelDTO) -> None:
        """SAP FRAME RELEASE ASSIGNMENTS 1 - GENERAL → element.hinges.

        Alan adları: PI/V2I/V3I/TI/M2I/M3I ve ...J (uç J için). Değer "Yes"
        ise o DOF mafsallı (serbest bırakılmış).
        """
        tags = [("PI", "p"), ("V2I", "v2"), ("V3I", "v3"),
                ("TI", "t"), ("M2I", "m2"), ("M3I", "m3")]
        tags_j = [("PJ", "p"), ("V2J", "v2"), ("V3J", "v3"),
                  ("TJ", "t"), ("M2J", "m2"), ("M3J", "m3")]
        for row in tables.get("FRAME RELEASE ASSIGNMENTS 1 - GENERAL", []):
            fid = to_int(row.get("Frame"))
            if fid is None or fid not in model.frame_elements:
                continue
            start = [tag for col, tag in tags
                     if (row.get(col) or "").strip().lower() == "yes"]
            end = [tag for col, tag in tags_j
                   if (row.get(col) or "").strip().lower() == "yes"]
            if start or end:
                model.frame_elements[fid].hinges = {"start": start, "end": end}

    # ----------------------------------------------------------- diyafram
    def _parse_diaphragms(self, tables, model: ModelDTO) -> None:
        """CONSTRAINT DEFINITIONS - DIAPHRAGM + JOINT CONSTRAINT ASSIGNMENTS."""
        defs: dict[str, dict[str, str]] = {}
        for row in tables.get("CONSTRAINT DEFINITIONS - DIAPHRAGM", []):
            name = row.get("Name")
            if name:
                defs[name] = {"axis": (row.get("Axis") or "Z").upper()}

        assignments: dict[str, list[int]] = {}
        for row in tables.get("JOINT CONSTRAINT ASSIGNMENTS", []):
            if (row.get("Type") or "").strip().lower() != "diaphragm":
                continue
            name = row.get("Constraint")
            jid = to_int(row.get("Joint"))
            if name and jid is not None and jid in model.nodes:
                assignments.setdefault(name, []).append(jid)

        out: list[DiaphragmDTO] = []
        for name, joints in assignments.items():
            axis = defs.get(name, {}).get("axis", "Z")
            if axis not in ("X", "Y", "Z"):
                axis = "Z"
            out.append(DiaphragmDTO(name=name, axis=axis, joints=joints))
        model.diaphragms = out

    # ----------------------------------------------- area → frame loads
    def _parse_area_uniform_loads(self, tables, model: ModelDTO) -> None:
        """AREA LOADS - UNIFORM TO FRAME.

        Her satır: döşeme üstünde belirli yoğunlukta yük (kN/m²). Biz
        ``model.area_uniform_loads`` listesine ham kayıt olarak koyuyoruz;
        gerçek frame'e dağıtım assembler aşamasında yapılır.
        """
        out: list[AreaUniformLoadDTO] = []
        for row in tables.get("AREA LOADS - UNIFORM TO FRAME", []):
            aid = to_int(row.get("Area"))
            pat = row.get("LoadPat")
            val = to_float(row.get("UnifLoad"))
            if aid is None or not pat or not math.isfinite(val):
                continue
            direction = (row.get("Dir") or "Gravity").strip().lower()
            dist_type = (row.get("DistType") or "Two way").strip().lower()
            if dist_type not in ("one way", "two way"):
                dist_type = "two way"
            out.append(AreaUniformLoadDTO(
                area_id=aid,
                load_pat=pat,
                direction=direction if direction in ("gravity", "x", "y", "z") else "gravity",
                magnitude=val,
                dist_type=dist_type,  # type: ignore[arg-type]
            ))
        model.area_uniform_loads = out

    # -------------------------------------------------------- units / materials
    def _parse_units(self, tables, model: ModelDTO) -> None:
        rows = tables.get("PROGRAM CONTROL", [])
        if not rows:
            return
        raw = rows[0].get("CurrUnits", "")
        m = _UNITS_RE.search(raw)
        if not m:
            return
        force_s, length_s, temp_s = m.group(1).upper(), m.group(2).upper(), m.group(3).upper()
        model.units = UnitsDTO(
            force=_FORCE_UNIT_MAP.get(force_s, "kN"),
            length=_LENGTH_UNIT_MAP.get(length_s, "m"),
            temperature=_TEMP_UNIT_MAP.get(temp_s, "C"),
        )

    def _parse_materials(self, tables, model: ModelDTO) -> None:
        mech = {
            row.get("Material", ""): row
            for row in tables.get("MATERIAL PROPERTIES 02 - BASIC MECHANICAL PROPERTIES", [])
        }
        for row in tables.get("MATERIAL PROPERTIES 01 - GENERAL", []):
            mid = row.get("Material", "")
            if not mid:
                continue
            mech_row = mech.get(mid, {})
            E = to_float(mech_row.get("E1"))
            nu = to_float(mech_row.get("U12"))
            unit_mass = to_float(mech_row.get("UnitMass"))
            alpha = to_float(mech_row.get("A1"))
            model.materials[mid] = MaterialDTO(
                id=mid,
                E=E if math.isfinite(E) else 0.0,
                nu=nu if math.isfinite(nu) else 0.0,
                rho=unit_mass if math.isfinite(unit_mass) else 0.0,
                alpha=alpha if math.isfinite(alpha) else 0.0,
            )

    def _parse_frame_sections(self, tables, model: ModelDTO) -> None:
        for row in tables.get("FRAME SECTION PROPERTIES 01 - GENERAL", []):
            sid = row.get("SectionName", "")
            if not sid:
                continue
            # SAP: I33 = eğilme etrafı 3-3 (kuvvetli), I22 = 2-2 (zayıf), TorsConst = J
            model.sections[sid] = FrameSectionDTO(
                id=sid,
                A=_safe(row.get("Area")),
                Iy=_safe(row.get("I22")),
                Iz=_safe(row.get("I33")),
                J=_safe(row.get("TorsConst")),
                Iyz=_safe(row.get("I23")),
            )

    def _parse_shell_sections(self, tables, model: ModelDTO) -> None:
        for row in tables.get("AREA SECTION PROPERTIES", []):
            sid = row.get("Section", "")
            if not sid:
                continue
            thickness = _safe(row.get("Thickness"))
            model.sections[sid] = ShellSectionDTO(id=sid, thickness=thickness)

    # ----------------------------------------------------------- nodes / dofs
    def _parse_nodes(self, tables, model: ModelDTO) -> None:
        rows = _first_table(tables, _JOINT_TABLES)
        for row in rows:
            node_id = to_int(row.get("Joint") or row.get("Point"))
            if node_id is None:
                continue
            x = to_float(row.get("GlobalX") or row.get("XorR") or row.get("X"))
            y = to_float(row.get("GlobalY") or row.get("Y"))
            z = to_float(row.get("GlobalZ") or row.get("Z"))
            if not all(math.isfinite(v) for v in (x, y, z)):
                continue
            model.nodes[node_id] = NodeDTO(id=node_id, x=x, y=y, z=z)

    def _parse_restraints(self, tables, model: ModelDTO) -> None:
        for row in tables.get("JOINT RESTRAINT ASSIGNMENTS", []):
            node_id = to_int(row.get("Joint"))
            if node_id is None or node_id not in model.nodes:
                continue
            keys = ("U1", "U2", "U3", "R1", "R2", "R3")
            flags = [row.get(k, "No").strip().lower() in ("yes", "true", "1") for k in keys]
            model.nodes[node_id].restraints = flags

    # ---------------------------------------------------------------- frames
    def _parse_frame_elements(self, tables, model: ModelDTO) -> None:
        rows = _first_table(tables, _FRAME_TABLES)
        for idx, row in enumerate(rows, start=1):
            raw_id = row.get("Frame") or row.get("Line")
            el_id = to_int(raw_id) or idx
            i = to_int(row.get("JointI") or row.get("Joint1") or row.get("iJoint"))
            j = to_int(row.get("JointJ") or row.get("Joint2") or row.get("jJoint"))
            if i is None or j is None:
                continue
            if i not in model.nodes or j not in model.nodes:
                continue
            model.frame_elements[el_id] = FrameElementDTO(
                id=el_id,
                type=ElementType.FRAME_3D,
                nodes=[i, j],
                section_id="",
                material_id="",
            )

    def _parse_frame_assignments(self, tables, model: ModelDTO) -> None:
        for row in tables.get("FRAME SECTION ASSIGNMENTS", []):
            fid = to_int(row.get("Frame"))
            if fid is None or fid not in model.frame_elements:
                continue
            el = model.frame_elements[fid]
            if not isinstance(el, FrameElementDTO):
                continue
            section_id = row.get("AnalSect") or row.get("DesignSect") or ""
            el.section_id = section_id
            sec = model.sections.get(section_id)
            if isinstance(sec, FrameSectionDTO):
                # Kesitin bağlı olduğu malzemeyi kesit tablosundan arka yoldan çek.
                # FRAME SECTION PROPERTIES 01 satırındaki Material sütunundan geldi.
                pass
        # Malzeme id'sini FRAME SECTION PROPERTIES 01 → kesit → malzeme bağı
        # üzerinden doldur:
        section_to_material = {
            row.get("SectionName", ""): row.get("Material", "")
            for row in tables.get("FRAME SECTION PROPERTIES 01 - GENERAL", [])
            if row.get("SectionName")
        }
        for el in model.frame_elements.values():
            if isinstance(el, FrameElementDTO) and el.section_id:
                el.material_id = section_to_material.get(el.section_id, "")

    # ----------------------------------------------------------------- shells
    def _parse_shell_elements(self, tables, model: ModelDTO) -> None:
        for row in tables.get("CONNECTIVITY - AREA", []):
            aid = to_int(row.get("Area"))
            if aid is None:
                continue
            n = to_int(row.get("NumJoints")) or 0
            joints: list[int] = []
            for k in range(1, n + 1):
                jv = to_int(row.get(f"Joint{k}"))
                if jv is not None and jv in model.nodes:
                    joints.append(jv)
            if len(joints) < 3:
                continue
            model.shell_elements[aid] = ShellElementDTO(
                id=aid,
                type=ElementType.PLANE_STRESS_Q4 if len(joints) == 4 else ElementType.PLANE_STRESS_Q9,
                nodes=joints,
                section_id="",
                material_id="",
            )

    def _parse_shell_assignments(self, tables, model: ModelDTO) -> None:
        shell_sections = {
            sid: s for sid, s in model.sections.items() if isinstance(s, ShellSectionDTO)
        }
        shell_to_material = {
            row.get("Section", ""): row.get("Material", "")
            for row in tables.get("AREA SECTION PROPERTIES", [])
            if row.get("Section")
        }
        # SAP bazen Section=None yazar. Tek bir kabuk kesiti tanımlıysa o varsayılır.
        default_section = (
            next(iter(shell_sections)) if len(shell_sections) == 1 else None
        )
        for row in tables.get("AREA SECTION ASSIGNMENTS", []):
            aid = to_int(row.get("Area"))
            if aid is None or aid not in model.shell_elements:
                continue
            el = model.shell_elements[aid]
            raw_section = (row.get("Section") or "").strip()
            section_id = raw_section if raw_section and raw_section != "None" else ""
            if not section_id and default_section is not None:
                section_id = default_section
            el.section_id = section_id
            el.material_id = shell_to_material.get(section_id, "")

    # ------------------------------------------------------------------ loads
    def _parse_load_patterns(self, tables, model: ModelDTO) -> None:
        for row in tables.get("LOAD PATTERN DEFINITIONS", []):
            pat = row.get("LoadPat")
            if not pat:
                continue
            design = (row.get("DesignType") or "OTHER").upper()
            lt = _LOAD_TYPE_MAP.get(design, LoadType.OTHER)
            # EQX/EQY yön ayrımı: pattern adından çıkar
            if lt == LoadType.EARTHQUAKE_X and pat.upper().endswith("Y"):
                lt = LoadType.EARTHQUAKE_Y
            swm = to_float(row.get("SelfWtMult"))
            model.load_cases[pat] = LoadCaseDTO(
                id=pat,
                type=lt,
                self_weight_factor=swm if math.isfinite(swm) else 0.0,
            )

    def _parse_joint_loads(self, tables, model: ModelDTO) -> None:
        # Manuel düğüm yükleri (JOINT LOADS - FORCE)
        for row in tables.get("JOINT LOADS - FORCE", []):
            pat = row.get("LoadPat")
            nid = to_int(row.get("Joint"))
            if pat not in model.load_cases or nid is None:
                continue
            values = [
                _safe(row.get("F1")),
                _safe(row.get("F2")),
                _safe(row.get("F3")),
                _safe(row.get("M1")),
                _safe(row.get("M2")),
                _safe(row.get("M3")),
            ]
            model.load_cases[pat].point_loads.append(
                PointLoadDTO(node_id=nid, values=values)
            )

        # Otomatik deprem yükleri (AUTO SEISMIC LOADS TO JOINTS)
        # SAP eşdeğer deprem yükü hesaplayıp düğümlere dağıtıyor; biz
        # bunları o pattern'in point_loads'u gibi ekliyoruz.
        # Format: LoadPat=EXp Joint=N FX/FY/FZ/MX/MY/MZ
        for row in tables.get("AUTO SEISMIC LOADS TO JOINTS", []):
            pat = row.get("LoadPat")
            nid = to_int(row.get("Joint"))
            if pat not in model.load_cases or nid is None:
                continue
            values = [
                _safe(row.get("FX")),
                _safe(row.get("FY")),
                _safe(row.get("FZ")),
                _safe(row.get("MX")),
                _safe(row.get("MY")),
                _safe(row.get("MZ")),
            ]
            # En az bir bileşen sıfırdan farklı olmalı
            if not any(v != 0.0 for v in values):
                continue
            model.load_cases[pat].point_loads.append(
                PointLoadDTO(node_id=nid, values=values)
            )

        # Otomatik rüzgar yükleri (AUTO WIND LOADS TO JOINTS) — aynı yapı
        for row in tables.get("AUTO WIND LOADS TO JOINTS", []):
            pat = row.get("LoadPat")
            nid = to_int(row.get("Joint"))
            if pat not in model.load_cases or nid is None:
                continue
            values = [
                _safe(row.get("FX")),
                _safe(row.get("FY")),
                _safe(row.get("FZ")),
                _safe(row.get("MX")),
                _safe(row.get("MY")),
                _safe(row.get("MZ")),
            ]
            if not any(v != 0.0 for v in values):
                continue
            model.load_cases[pat].point_loads.append(
                PointLoadDTO(node_id=nid, values=values)
            )

    def _parse_distributed_loads(self, tables, model: ModelDTO) -> None:
        for row in tables.get("FRAME LOADS - DISTRIBUTED", []):
            pat = row.get("LoadPat")
            fid = to_int(row.get("Frame"))
            if pat not in model.load_cases or fid is None:
                continue
            if fid not in model.frame_elements:
                continue
            dir_raw = (row.get("Dir") or "Gravity").strip().upper()
            direction = _SAP_DIR_MAP.get(dir_raw, "gravity")
            coord_sys = "local" if direction.startswith("local_") else "global"
            kind = "moment" if (row.get("Type") or "").strip().lower() == "moment" else "force"
            load = DistributedLoadDTO(
                element_id=fid,
                coord_sys=coord_sys,
                direction=direction,
                kind=kind,
                magnitude_a=_safe(row.get("FOverLA") or row.get("MOverLA")),
                magnitude_b=_safe(row.get("FOverLB") or row.get("MOverLB")),
                rel_dist_a=_safe(row.get("RelDistA"), default=0.0),
                rel_dist_b=_safe(row.get("RelDistB"), default=1.0),
            )
            model.load_cases[pat].distributed_loads.append(load)

    def _parse_combinations(self, tables, model: ModelDTO) -> None:
        """SAP COMBINATION DEFINITIONS → CombinationDTO.

        Tüm CaseType'lar kabul edilir (Linear Static + NonLin Static +
        Response Spectrum vb.). Referans edilen case ismi ya doğrudan bir
        load pattern'dir ya da CASE - STATIC 1 - LOAD ASSIGNMENTS tablosunda
        tanımlı bir türetilmiş case'dir. İkinci durumda recursive expansion
        ile base pattern'lara indirgenir.

        Not: Modal/Response Spectrum case'lerine referans veren kombinasyon
        satırları atlanır (lineer süperpozisyon statik case'ler arasında
        anlamlıdır; RS sonuçları kendi yük durumu olarak zaten ayrıdır).
        """
        # 1) Her case'in (pattern → scale) haritasını çıkar
        case_definitions: dict[str, dict[str, float]] = {}
        for row in tables.get("CASE - STATIC 1 - LOAD ASSIGNMENTS", []):
            case_name = row.get("Case")
            load_type = (row.get("LoadType") or "").strip().lower()
            load_name = row.get("LoadName")
            sf = to_float(row.get("LoadSF"))
            if not case_name or not load_name or not math.isfinite(sf):
                continue
            # Load type "Load pattern" veya "Load case" olabilir
            if "pattern" in load_type or "case" in load_type:
                case_definitions.setdefault(case_name, {})[load_name] = (
                    case_definitions.get(case_name, {}).get(load_name, 0.0) + sf
                )

        # 2) Kombinasyonları topla (satır başına: ComboName + CaseName + SF)
        raw_combos: dict[str, dict[str, float]] = {}
        skipped_reasons: dict[str, str] = {}
        for row in tables.get("COMBINATION DEFINITIONS", []):
            name = row.get("ComboName")
            case = row.get("CaseName")
            if not name or not case:
                continue
            case_type = (row.get("CaseType") or "").strip().lower()
            # Modal / spektrum / time history gibi tipleri atla — süperpozisyon
            # anlamlı değil
            if case_type in ("modal", "response spectrum", "time history",
                             "response combo", "moving load"):
                skipped_reasons.setdefault(name, f"tip={case_type}")
                continue
            factor = to_float(row.get("ScaleFactor"))
            if not math.isfinite(factor):
                continue
            raw_combos.setdefault(name, {})[case] = (
                raw_combos.get(name, {}).get(case, 0.0) + factor
            )

        # 3) Her kombinasyonun factor dict'ini base pattern'lara expand et
        base_patterns = set(model.load_cases.keys())

        def resolve(
            name: str, depth: int = 0, stack: tuple[str, ...] = (),
        ) -> dict[str, float]:
            if name in base_patterns:
                return {name: 1.0}
            if depth > 10 or name in stack:
                return {}
            expansion = case_definitions.get(name)
            if expansion is None:
                return {}    # unknown — drop silently
            result: dict[str, float] = {}
            for inner, inner_sf in expansion.items():
                sub = resolve(inner, depth + 1, stack + (name,))
                for bp, bsf in sub.items():
                    result[bp] = result.get(bp, 0.0) + inner_sf * bsf
            return result

        combos_final: list[CombinationDTO] = []
        for name, factors in raw_combos.items():
            resolved: dict[str, float] = {}
            unresolved: list[str] = []
            for case_name, sf in factors.items():
                expanded = resolve(case_name)
                if not expanded:
                    unresolved.append(case_name)
                    continue
                for bp, bsf in expanded.items():
                    resolved[bp] = resolved.get(bp, 0.0) + sf * bsf
            # Sıfıra eşit faktörleri temizle (birbirini götüren terimler)
            resolved = {k: v for k, v in resolved.items() if abs(v) > 1e-12}
            if resolved:
                combos_final.append(CombinationDTO(id=name, factors=resolved))
        model.combinations = combos_final


def parse_s2k(text: str) -> ModelDTO:
    """Kolaylık fonksiyonu — ``S2KParser().parse(text)`` ile aynı."""
    return S2KParser().parse(text)


# --------------------------------------------------------------------- helpers
def _yn(v: str | None) -> bool:
    return bool(v) and v.strip().lower() in ("yes", "true", "1")


def _first_table(tables, names: tuple[str, ...]) -> list[dict[str, str]]:
    for name in names:
        if name in tables:
            return tables[name]
    return []


def _safe(v: str | None, default: float = 0.0) -> float:
    x = to_float(v)
    return x if math.isfinite(x) else default
