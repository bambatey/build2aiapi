"""
DSPy Signature — Yapısal mühendislik AI chat analizi.
Kullanıcı sorusuna göre yapılandırılmış yanıt üretir.
"""
from typing import Optional
import dspy
from pydantic import BaseModel, Field

dspy.configure_cache(enable_memory_cache=False, enable_disk_cache=False)


class CodeReference(BaseModel):
    """Yönetmelik referansı"""
    code_name: str = Field(description="Yönetmelik adı (örn: TBDY 2018, ASCE 7-22)")
    section: str = Field(description="İlgili madde/bölüm numarası")
    description: str = Field(description="Maddenin kısa açıklaması")


class FileDiff(BaseModel):
    """Dosya değişiklik önerisi"""
    line_number: int = Field(description="Değiştirilecek satır numarası")
    old_value: str = Field(description="Mevcut değer")
    new_value: str = Field(description="Önerilen yeni değer")
    reason: str = Field(description="Değişiklik nedeni")


class StructuralChatResponse(BaseModel):
    """Yapısal mühendislik chat yanıtı — yapılandırılmış çıktı"""
    answer: str = Field(description="Kullanıcının sorusuna detaylı yanıt. Markdown formatında, Türkçe.")
    analysis_type: str = Field(description="Analiz türü: 'load_check', 'section_optimization', 'seismic_analysis', 'code_compliance', 'model_summary', 'displacement_check', 'material_info', 'general'")
    findings: list[str] = Field(default_factory=list, description="Tespit edilen bulgular listesi")
    recommendations: list[str] = Field(default_factory=list, description="Öneriler listesi")
    code_references: list[CodeReference] = Field(default_factory=list, description="İlgili yönetmelik referansları")
    suggested_changes: list[FileDiff] = Field(default_factory=list, description="Dosyada önerilen değişiklikler (varsa)")
    confidence: float = Field(ge=0.0, le=1.0, default=0.8, description="Yanıtın güven skoru (0.0-1.0)")
    warning: Optional[str] = Field(default=None, description="Varsa uyarı mesajı")


class StructuralChatSignature(dspy.Signature):
    """
    Sen StructAI, yapısal mühendislik konusunda uzmanlaşmış bir AI asistanısın.

    Görevlerin:
    - SAP2000, ETABS, RISA-3D, STAAD Pro model dosyalarını (.s2k, .e2k, .r3d, .std) analiz etmek
    - TBDY 2018, ASCE 7-22, Eurocode 8, ACI 318 yönetmeliklerine göre kontrol yapmak
    - Kesit optimizasyonu, deprem analizi, yük kontrolleri yapmak
    - Yapısal mühendislik sorularını detaylı yanıtlamak
    - Dosya içeriği verilmişse, dosyadaki parametreleri analiz edip somut önerilerde bulunmak

    Yanıtlarını Türkçe ver. Teknik terimleri doğru kullan.
    Eğer dosyada değişiklik öneriyorsan, suggested_changes field'ını doldur.
    """

    conversation_history: str = dspy.InputField(description="Önceki mesajlar (User/Assistant formatında)")
    user_question: str = dspy.InputField(description="Kullanıcının son sorusu")
    file_content: str = dspy.InputField(description="Aktif .s2k/.e2k dosya içeriği (yoksa 'Dosya yüklenmemiş')")

    response: StructuralChatResponse = dspy.OutputField(description="Yapılandırılmış analiz yanıtı")
