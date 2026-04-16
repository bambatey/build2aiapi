"""Mevcut .s2k dosyasında tablo bazında düzenleme.

AI tam dosya regenerate etmek yerine küçük structured edit emirleri
("kat ekle", "kolon büyüt", "beton sınıfını C35 yap", "yükü artır")
yollar; bu modül o emirleri tam dosya çıktısına çevirir.

Akış: extract_tables → manipulate dict → serialize_tables.
Hem üretici çıktıları hem de SAP'tan gelen orjinal dosyalar editlenebilir.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from services.structural_analysis.parser.tables import extract_tables
from services.structural_analysis.parser.tokens import to_float, to_int

from .sections_rc import concrete_default, rect_section, square_column
from .serializer import serialize_tables

logger = logging.getLogger(__name__)


class EditError(RuntimeError):
    pass


# ---------------------------------------------------------------- helpers
def _row_get(row: dict, *keys: str, default: str = "") -> str:
    """Birden fazla key dene (Joint vs Point gibi alternatifler)."""
    for k in keys:
        v = row.get(k)
        if v is not None and v != "":
            return v
    return default


def _max_int(rows: list[dict], keys: tuple[str, ...]) -> int:
    """Satırlardaki belirtilen anahtarların max int değerini bul (boşsa 0)."""
    m = 0
    for r in rows:
        v = to_int(_row_get(r, *keys))
        if v is not None and v > m:
            m = v
    return m


def _fmt(v: float, prec: int = 6) -> str:
    s = f"{v:.{prec}f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s if s else "0"


# ====================================================================== ADD STORY
def add_stories(
    s2k_text: str,
    n: int = 1,
    story_h: float | None = None,
) -> tuple[str, dict]:
    """En üst kata ``n`` adet yeni kat ekler.

    Üst kattaki düğümlerin x,y koordinatları yeni katlara kopyalanır,
    yeni z = z_top + i × story_h. Her yeni kat için:
      - yeni düğümler
      - yeni kolonlar (önceki kat ↔ yeni kat, aynı x,y)
      - yeni X-kirişler ve Y-kirişler (üst kattan kopyala)
      - frame section assignments (kolon → COL, kiriş → BEAM ya da mevcut)
      - distributed loads (üst katın yüklerini klonla)

    Returns:
        (new_s2k_text, info_dict)
    """
    tables = extract_tables(s2k_text)

    # ----- Mevcut düğümler ve üst seviye
    joint_rows = tables.get("JOINT COORDINATES") or tables.get("POINT COORDINATES") or []
    if not joint_rows:
        raise EditError("JOINT COORDINATES tablosu bulunamadı.")

    parsed_nodes: list[dict] = []
    for r in joint_rows:
        nid = to_int(_row_get(r, "Joint", "Point"))
        x = to_float(_row_get(r, "GlobalX", "XorR", "X"))
        y = to_float(_row_get(r, "GlobalY", "Y"))
        z = to_float(_row_get(r, "GlobalZ", "Z"))
        if nid is None:
            continue
        parsed_nodes.append({"id": nid, "x": x, "y": y, "z": z, "raw": r})

    if not parsed_nodes:
        raise EditError("Düğümler ayrıştırılamadı.")

    z_top = max(p["z"] for p in parsed_nodes)
    top_nodes = [p for p in parsed_nodes if abs(p["z"] - z_top) < 1e-6]

    # Kat yüksekliğini tahmin et: ikinci üst seviye - üst seviye
    inferred_h = story_h
    if inferred_h is None:
        zs = sorted({round(p["z"], 6) for p in parsed_nodes}, reverse=True)
        if len(zs) >= 2:
            inferred_h = zs[0] - zs[1]
        else:
            inferred_h = 3.0

    # ----- Mevcut frame'ler — üst kattaki kirişleri bul (kolon değil)
    frame_rows = tables.get("CONNECTIVITY - FRAME") or tables.get("CONNECTIVITY - LINE") or []
    parsed_frames: list[dict] = []
    for r in frame_rows:
        fid = to_int(_row_get(r, "Frame", "Line"))
        i = to_int(_row_get(r, "JointI", "Joint1", "iJoint"))
        j = to_int(_row_get(r, "JointJ", "Joint2", "jJoint"))
        if fid is None or i is None or j is None:
            continue
        parsed_frames.append({"id": fid, "i": i, "j": j, "raw": r})

    nodes_by_id = {p["id"]: p for p in parsed_nodes}

    # Üst kat kirişleri: her iki ucu da z_top'ta
    top_beams = []
    for f in parsed_frames:
        ni = nodes_by_id.get(f["i"])
        nj = nodes_by_id.get(f["j"])
        if ni and nj and abs(ni["z"] - z_top) < 1e-6 and abs(nj["z"] - z_top) < 1e-6:
            top_beams.append(f)

    # ----- Yeni id sayaçları
    next_node_id = max(p["id"] for p in parsed_nodes) + 1
    next_frame_id = _max_int(frame_rows, ("Frame", "Line")) + 1

    # ----- Frame section assignments — section adlarını cache et
    fsa_rows = tables.get("FRAME SECTION ASSIGNMENTS") or []
    sec_by_frame = {to_int(_row_get(r, "Frame")): _row_get(r, "AnalSect", "DesignSect")
                    for r in fsa_rows}
    # Default kolon ve kiriş kesiti tahmini
    col_section = "COL"
    beam_section = "BEAM"
    for f in parsed_frames:
        ni = nodes_by_id.get(f["i"])
        nj = nodes_by_id.get(f["j"])
        if not ni or not nj:
            continue
        if abs(ni["x"] - nj["x"]) < 1e-6 and abs(ni["y"] - nj["y"]) < 1e-6:
            sec = sec_by_frame.get(f["id"])
            if sec:
                col_section = sec
                break
    if top_beams:
        sec = sec_by_frame.get(top_beams[0]["id"])
        if sec:
            beam_section = sec

    # ----- Mevcut yayılı yükler — üst kat kirişleri için patternleri çıkar
    dist_rows = tables.get("FRAME LOADS - DISTRIBUTED") or []
    top_beam_loads_template: list[dict] = []  # (load_pat, fover) çiftleri
    seen_pat: set[str] = set()
    top_beam_ids = {b["id"] for b in top_beams}
    for r in dist_rows:
        fid = to_int(_row_get(r, "Frame"))
        if fid in top_beam_ids:
            pat = _row_get(r, "LoadPat")
            key = (pat, _row_get(r, "Dir"))
            if key in seen_pat:
                continue
            seen_pat.add(key)
            top_beam_loads_template.append(dict(r))  # kopya — frame'i sonra düzenliyeceğiz

    # ============ Şimdi yeni kat(lar)ı ekle
    new_joint_rows = []
    new_frame_rows = []
    new_fsa_rows = []
    new_dist_rows = []
    added_nodes_by_xy_z: dict[tuple[float, float, float], int] = {}
    # Önceki seviyeyi başlangıç olarak top_nodes (id, x, y) eşle
    prev_level_id_for_xy: dict[tuple[float, float], int] = {
        (round(p["x"], 6), round(p["y"], 6)): p["id"] for p in top_nodes
    }

    for k in range(1, n + 1):
        z_new = z_top + k * inferred_h
        # 1) Yeni düğümler (üst kattaki x,y kombinasyonları)
        new_level_id_for_xy: dict[tuple[float, float], int] = {}
        for p in top_nodes:
            new_id = next_node_id
            next_node_id += 1
            new_joint_rows.append({
                "Joint": str(new_id),
                "CoordSys": "GLOBAL",
                "CoordType": "Cartesian",
                "XorR": _fmt(p["x"]),
                "Y": _fmt(p["y"]),
                "Z": _fmt(z_new),
                "SpecialJt": "No",
                "GlobalX": _fmt(p["x"]),
                "GlobalY": _fmt(p["y"]),
                "GlobalZ": _fmt(z_new),
            })
            new_level_id_for_xy[(round(p["x"], 6), round(p["y"], 6))] = new_id
            added_nodes_by_xy_z[(round(p["x"], 6), round(p["y"], 6), z_new)] = new_id

        # 2) Yeni kolonlar — alt seviyeden yeni seviyeye
        for xy, new_id in new_level_id_for_xy.items():
            below_id = prev_level_id_for_xy[xy]
            fid = next_frame_id
            next_frame_id += 1
            new_frame_rows.append({
                "Frame": str(fid),
                "JointI": str(below_id),
                "JointJ": str(new_id),
                "IsCurved": "No",
            })
            new_fsa_rows.append({
                "Frame": str(fid),
                "SectionType": "Frame",
                "AutoSelect": "N.A.",
                "AnalSect": col_section,
                "DesignSect": col_section,
                "MatProp": "Default",
            })

        # 3) Yeni kirişler — üst kat geometrisini klonla
        new_beam_id_map: dict[int, int] = {}  # eski beam id → yeni beam id
        for old_beam in top_beams:
            old_i = nodes_by_id[old_beam["i"]]
            old_j = nodes_by_id[old_beam["j"]]
            new_i = new_level_id_for_xy.get((round(old_i["x"], 6), round(old_i["y"], 6)))
            new_j = new_level_id_for_xy.get((round(old_j["x"], 6), round(old_j["y"], 6)))
            if new_i is None or new_j is None:
                continue
            fid = next_frame_id
            next_frame_id += 1
            new_beam_id_map[old_beam["id"]] = fid
            new_frame_rows.append({
                "Frame": str(fid),
                "JointI": str(new_i),
                "JointJ": str(new_j),
                "IsCurved": "No",
            })
            new_fsa_rows.append({
                "Frame": str(fid),
                "SectionType": "Frame",
                "AutoSelect": "N.A.",
                "AnalSect": beam_section,
                "DesignSect": beam_section,
                "MatProp": "Default",
            })

        # 4) Yayılı yükler — üst katın yük şablonunu yeni kirişlere uygula
        for old_beam_id, new_beam_id in new_beam_id_map.items():
            for tpl in top_beam_loads_template:
                row = dict(tpl)
                row["Frame"] = str(new_beam_id)
                new_dist_rows.append(row)

        # Bir sonraki iterasyon için seviye bilgisini güncelle
        prev_level_id_for_xy = new_level_id_for_xy

    # Tabloları güncelle
    target_joint_table = "JOINT COORDINATES" if "JOINT COORDINATES" in tables else "POINT COORDINATES"
    tables.setdefault(target_joint_table, []).extend(new_joint_rows)

    target_frame_table = "CONNECTIVITY - FRAME" if "CONNECTIVITY - FRAME" in tables else "CONNECTIVITY - LINE"
    tables.setdefault(target_frame_table, []).extend(new_frame_rows)

    tables.setdefault("FRAME SECTION ASSIGNMENTS", []).extend(new_fsa_rows)
    if new_dist_rows:
        tables.setdefault("FRAME LOADS - DISTRIBUTED", []).extend(new_dist_rows)

    info = {
        "added_stories": n,
        "added_nodes": len(new_joint_rows),
        "added_frames": len(new_frame_rows),
        "story_height": inferred_h,
        "z_top_before": z_top,
        "z_top_after": z_top + n * inferred_h,
    }
    logger.info("add_stories: %s", info)
    return serialize_tables(tables), info


# ============================================================ CHANGE CONCRETE GRADE
def change_concrete_grade(s2k_text: str, new_fck_mpa: int) -> tuple[str, dict]:
    """Tüm Concrete materyallerinin sınıfını yeni fck'a günceller.

    MATERIAL PROPERTIES 01'de Type=Concrete olanların Grade'ini değiştirir,
    MATERIAL PROPERTIES 02'de E1, UnitMass, UnitWeight değerlerini günceller.
    Materyal id'si korunur (kesit atamaları bozulmasın).
    """
    tables = extract_tables(s2k_text)
    new_concrete = concrete_default(new_fck_mpa)

    mat01 = tables.get("MATERIAL PROPERTIES 01 - GENERAL", [])
    mat02 = tables.get("MATERIAL PROPERTIES 02 - BASIC MECHANICAL PROPERTIES", [])
    if not mat01:
        raise EditError("MATERIAL PROPERTIES 01 tablosu yok.")

    updated_ids: list[str] = []
    for row in mat01:
        if (row.get("Type") or "").strip().lower() == "concrete":
            row["Grade"] = f"f'c {new_fck_mpa} MPa"
            updated_ids.append(row.get("Material", ""))

    for row in mat02:
        if row.get("Material") in updated_ids:
            row["E1"] = _fmt(new_concrete.E)
            row["UnitMass"] = _fmt(new_concrete.unit_mass)
            row["UnitWeight"] = _fmt(new_concrete.unit_weight)

    info = {"updated_materials": updated_ids, "new_fck_mpa": new_fck_mpa}
    return serialize_tables(tables), info


# ============================================================== CHANGE SECTION SIZE
def change_section_size(
    s2k_text: str,
    section_id: str | None = None,    # belirtilmezse "COL" ya da kategoriye göre
    kind: str | None = None,          # "column" | "beam"
    t2: float | None = None,
    t3: float | None = None,
) -> tuple[str, dict]:
    """Belirli bir kesitin t2,t3 boyutlarını değiştirir; A/I/J yeniden hesaplanır.

    Eğer ``section_id`` verilmezse:
      kind='column' → section_id="COL" (üretici defaultu)
      kind='beam'   → section_id="BEAM"
    """
    tables = extract_tables(s2k_text)
    sec_rows = tables.get("FRAME SECTION PROPERTIES 01 - GENERAL", [])
    if not sec_rows:
        raise EditError("FRAME SECTION PROPERTIES tablosu yok.")

    if section_id is None:
        section_id = "COL" if kind == "column" else "BEAM"

    target = next(
        (r for r in sec_rows if r.get("SectionName") == section_id),
        None,
    )
    if target is None:
        raise EditError(f"Kesit '{section_id}' bulunamadı.")

    # Mevcut t2/t3'ten geri kazan (verilmediyse korur)
    cur_t2 = to_float(target.get("t2")) or 0.0
    cur_t3 = to_float(target.get("t3")) or 0.0
    new_t2 = t2 if t2 is not None else cur_t2
    new_t3 = t3 if t3 is not None else cur_t3
    if new_t2 <= 0 or new_t3 <= 0:
        raise EditError("Kesit boyutları geçersiz.")

    material_id = target.get("Material", "")
    if kind == "column" or new_t2 == new_t3:
        sec = square_column(section_id, material_id, new_t2)
    else:
        sec = rect_section(section_id, material_id, new_t2, new_t3)

    target.update({
        "t2": _fmt(sec.t2),
        "t3": _fmt(sec.t3),
        "Area": _fmt(sec.A),
        "TorsConst": _fmt(sec.J),
        "I33": _fmt(sec.I33),
        "I22": _fmt(sec.I22),
        "AS2": _fmt(sec.A * (5.0 / 6.0)),
        "AS3": _fmt(sec.A * (5.0 / 6.0)),
        "S33": _fmt(sec.I33 / max(sec.t3 / 2.0, 1e-9)),
        "S22": _fmt(sec.I22 / max(sec.t2 / 2.0, 1e-9)),
        "Z33": _fmt(sec.t2 * (sec.t3 ** 2) / 4.0),
        "Z22": _fmt(sec.t3 * (sec.t2 ** 2) / 4.0),
        "R33": _fmt((sec.I33 / max(sec.A, 1e-9)) ** 0.5),
        "R22": _fmt((sec.I22 / max(sec.A, 1e-9)) ** 0.5),
    })

    info = {"section": section_id, "t2": new_t2, "t3": new_t3}
    return serialize_tables(tables), info


# ================================================================ CHANGE LOADS
def change_beam_loads(
    s2k_text: str,
    load_pattern: str,
    new_q_knm: float,
) -> tuple[str, dict]:
    """Tüm kirişlerdeki belirli pattern'in yayılı yük şiddetini değiştirir."""
    tables = extract_tables(s2k_text)
    rows = tables.get("FRAME LOADS - DISTRIBUTED", [])
    n_updated = 0
    for r in rows:
        if r.get("LoadPat") == load_pattern:
            r["FOverLA"] = _fmt(new_q_knm)
            r["FOverLB"] = _fmt(new_q_knm)
            n_updated += 1
    info = {"load_pattern": load_pattern, "new_q_knm": new_q_knm, "updated": n_updated}
    return serialize_tables(tables), info
