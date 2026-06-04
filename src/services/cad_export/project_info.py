"""Antette gösterilecek proje meta bilgisi.

Frontend kullanıcının doldurduğu form bu DTO'ya map olur. Eksik alanlar antet
çizilirken görünmez (boş satır yerine gizlenir).
"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field


class ProjectInfo(BaseModel):
    """İller Bankası tip şartnamesi (statik-betonarme) antet alanları."""

    # Zorunlu (boş gelirse stub değer)
    project_name: str = "İsimsiz Proje"
    sheet_kind: str = "KALIP PLANI"             # KALIP / KOLON APLIKASYON / TEMEL ...
    scale: str = "1/50"                          # TR konvansiyonel ölçek
    drawing_date: date = Field(default_factory=date.today)

    # Lokasyon (belediye onayı için zorunlu)
    city: str = ""                               # İl
    district: str = ""                           # İlçe
    municipality: str = ""                       # Belediye
    neighborhood: str = ""                       # Mahalle
    ada: str = ""                                # Ada no
    parsel: str = ""                             # Parsel no

    # Mühendis bilgisi
    engineer_name: str = ""
    engineer_chamber_no: str = ""                # İMO oda sicil no
    engineer_itb_no: str = ""                    # İTB (iş tasdik belge) no

    # Yapısal parametreler (TBDY 2018 referansı, antet alt bandı)
    building_importance: str = ""                # I = 1.0
    building_behavior_R: str = ""                # R = 4 / 8 vb.
    seismic_zone: str = ""                       # DTS / Yer hareketi seviyesi
    soil_class: str = ""                         # ZA..ZE
    concrete_class: str = "C30/37"
    steel_class: str = "B500C"

    # Bürodan opsiyonel
    firm_name: str = ""
    firm_logo_text: str = ""                     # Logo yerine metin (DXF embed kompleks)
    sheet_size: str = "A3"                       # A3 / A2
