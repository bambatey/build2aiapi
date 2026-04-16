"""
DSPy Signature — Yapısal mühendislik AI chat analizi.
AI sadece değişen TABLE bloklarını üretir, backend orijinal dosyaya merge eder.
"""
from typing import Optional
import dspy
from pydantic import BaseModel, Field

dspy.configure_cache(enable_memory_cache=False, enable_disk_cache=False)


class Inputs(BaseModel):
    user_message: str = Field(description="Kullanıcının sorusu")
    s2k_file: str = Field(description=".s2k dosya içeriği")
    s2k_file_updated_content: Optional[str] = Field(default=None, description="Eğer dosyada değişiklik yapıldıysa, tüm .s2k dosya içeriği. Değişiklik yoksa None.")
    updated_s2k_file: Optional[str] = Field(default=None, description="Eğer dosyada değişiklik yapıldıysa, tüm .s2k dosya içeriği. Değişiklik yoksa None.") 
    canvas_picture: Optional[str] = Field(default=None, description="Eğer kullanıcı çizim yaptıysa, çizimin base64 formatında stringi. Çizim yoksa None.")



class Outputs(BaseModel):
    chat_response: str = Field(description="Kullanıcının sorusuna detaylı yanıt. Türkçe, markdown formatında.")
    updated_tables: str = Field(default="", description="Dosyada değişiklik varsa, SADECE değişen veya eklenen TABLE bloklarını yaz. Her tablo 'TABLE: \"TABLO ADI\"' ile başlayıp satırlarla devam eder. Birden fazla tablo olabilir. Değişmeyen tabloları YAZMA. Değişiklik yoksa boş bırak. Örnek format:\nTABLE:  \"JOINT COORDINATES\"\n   Joint=1   CoordSys=GLOBAL   XorR=0   Y=0   Z=0\n\nTABLE:  \"CONNECTIVITY - FRAME\"\n   Frame=1   JointI=1   JointJ=2")
    change_summary: str = Field(default="", description="Dosyada yapılan değişikliklerin kısa özeti. Değişiklik yoksa boş.")


class StructuralChatSignature(dspy.Signature):
    """Sen yapısal mühendislik AI asistanısın. SAP2000/ETABS .s2k dosyalarını analiz ve düzenleme yaparsın.

    Dosya düzenlemesi istenirse:
    - updated_tables alanına SADECE değişen veya yeni eklenen TABLE bloklarını yaz
    - Değişmeyen tabloları YAZMA — backend otomatik merge edecek
    - TABLE formatı: TABLE:  "TABLO_ADI" ardından her satır 3 boşluk girintili Key=Value
    - Joint key'leri: JointI, JointJ (CONNECTIVITY tablolarında)
    - Koordinat key'leri: XorR, Y, Z (JOINT COORDINATES'da)
    - Yanıtlarını Türkçe ver."""

    Inputs = Inputs
    Outputs = Outputs


