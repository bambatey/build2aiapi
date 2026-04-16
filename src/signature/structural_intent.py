"""DSPy Signature — Yapısal chat ajanı: intent sınıflandırma + parametre çıkarımı.

Eski yaklaşım: AI ham .s2k yazmaya çalışıyordu, bozuk çıktılar geliyordu.
Yeni yaklaşım: AI sadece niyeti + parametreleri çıkarır; gerçek .s2k üretimi
``services.structural_generator`` tarafında deterministik olarak yapılır.

Mevcut intent'ler:
  - create_rc_frame  : Yeni RC çerçeve modeli oluştur (params: bays, stories, ...)
  - edit_model       : Mevcut modelde değişiklik iste (delta tablo — TBD)
  - query            : Sadece soru/analiz yorumu, dosya değişmez

Analiz context'i: eğer projede analiz çıktısı varsa, özet AI'a
"analysis_summary" olarak verilir. AI sorular cevaplarken bu veriye erişir.
"""

from __future__ import annotations

from typing import Literal, Optional

import dspy
from pydantic import BaseModel, Field

dspy.configure_cache(enable_memory_cache=False, enable_disk_cache=False)


# ------------------------------------------------------------------- intent params
class RCFrameParamsDto(BaseModel):
    """Kullanıcının isteğinden çıkarılan RC çerçeve parametreleri.

    Belirtilmeyen alanlar backend'de SABİT default'a düşer. Açıklamalardaki
    parantez içindeki değer GERÇEK default — yanlış değer söyleme!
    Kullanıcı "3x4 2 katlı" derse: bays_x=3, bays_y=4, stories=2.
    Diğerlerini None bırak; backend default'ları uygular.
    """

    bays_x: Optional[int] = Field(default=None, description="X yönü açıklık sayısı (default: 3)")
    bays_y: Optional[int] = Field(default=None, description="Y yönü açıklık sayısı (default: 4)")
    stories: Optional[int] = Field(default=None, description="Kat sayısı (default: 2)")
    bay_dx_m: Optional[float] = Field(default=None, description="X açıklık uzunluğu m (default: 6.0)")
    bay_dy_m: Optional[float] = Field(default=None, description="Y açıklık uzunluğu m (default: 5.0)")
    story_h_m: Optional[float] = Field(default=None, description="Kat yüksekliği m (default: 3.0)")
    col_size_m: Optional[float] = Field(default=None, description="Kare kolon ebadı m (default: 0.40 → 40x40)")
    beam_w_m: Optional[float] = Field(default=None, description="Kiriş genişliği m (default: 0.30 → 30 cm)")
    beam_h_m: Optional[float] = Field(default=None, description="Kiriş yüksekliği m (default: 0.60 → 60 cm)")
    fck_mpa: Optional[int] = Field(default=None, description="Beton sınıfı fck MPa (default: 30 → C30/37). Geçerli: 20,25,30,35,40,45,50")
    dead_q_knm: Optional[float] = Field(default=None, description="Kiriş yayılı sabit yükü kN/m (default: 5.0)")
    live_q_knm: Optional[float] = Field(default=None, description="Kiriş yayılı hareketli yükü kN/m (default: 2.0)")


# --------------------------------------------------------------------- signature
class StructuralIntent(BaseModel):
    """AI'ın kullanıcı mesajından çıkardığı niyet + parametreler."""

    action: Literal["create_rc_frame", "edit_model", "query"] = Field(
        description=(
            "Kullanıcının asıl niyeti. "
            "'create_rc_frame': sıfırdan betonarme çerçeve bina oluşturmak istiyor. "
            "'edit_model': mevcut modelde değişiklik istiyor (kesit büyüt, kat ekle, vs). "
            "'query': soru soruyor ya da yorum istiyor; dosya değişmemeli."
        )
    )
    rc_frame_params: Optional[RCFrameParamsDto] = Field(
        default=None,
        description="action='create_rc_frame' ise parametreler. Belirtilmeyenler default olur.",
    )
    edit_description: Optional[str] = Field(
        default=None,
        description="action='edit_model' ise, yapılacak değişikliğin kısa Türkçe tarifi.",
    )


class StructuralChatV2(dspy.Signature):
    """Sen yapısal mühendislik AI asistanısın. Build2AI içinde SAP2000 .s2k modelleri oluşturup düzenliyorsun.

    Görevin: kullanıcının mesajından NİYET ve PARAMETRE'leri çıkar — modelin
    kendisini SEN yazmıyorsun, yalnızca ne yapılması gerektiğini söylüyorsun.
    Backend sana bağlı olarak deterministik bir üretici çağıracak.

    Karar kuralları:
    1. Kullanıcı "yeni bina", "oluştur", "modelle", "çerçeve yap", "3x4 2 katlı",
       "RC frame", "betonarme çerçeve" gibi yaratma komutları veriyorsa →
       action="create_rc_frame". Cümledeki sayıları parametrelere çevir.
    2. Kullanıcı mevcut modeli değiştirmek istiyorsa (ör. "kolonları 50x50 yap",
       "bir kat daha ekle", "beton sınıfını C35 yap", "kirişleri güçlendir") →
       action="edit_model" ve edit_description ile kısa açıklama.
    3. Kullanıcı soru soruyorsa (ör. "bu modelde max yer değiştirme kaç?",
       "yönetmeliğe uygun mu?", "deprem kontrolü yap") → action="query".

    Parametre çıkarımı (create_rc_frame):
    - "3x4" → bays_x=3, bays_y=4
    - "2 katlı" → stories=2
    - "6m açıklık" veya "6'ya 5 açıklık" → bay_dx_m=6 (ya da bay_dy)
    - "kat yüksekliği 3m" → story_h_m=3
    - "40x40 kolon" → col_size_m=0.40
    - "30x60 kiriş" → beam_w_m=0.30, beam_h_m=0.60
    - "C35 beton" → fck_mpa=35
    - "ölü yük 6" → dead_q_knm=6, "canlı yük 3" → live_q_knm=3
    - Belirtilmemiş alanlar → None bırak (default değer backend'de seçilir)

    GERÇEK DEFAULT DEĞERLER (chat_response'da bu değerleri söyle, uydurma!):
    - bays_x=3, bays_y=4, stories=2
    - bay_dx_m=6.0, bay_dy_m=5.0, story_h_m=3.0
    - col_size_m=0.40 (40x40 kolon)
    - beam_w_m=0.30, beam_h_m=0.60 (30x60 kiriş)
    - fck_mpa=30 (C30/37 beton)
    - dead_q_knm=5.0, live_q_knm=2.0

    chat_response: Kullanıcıya Türkçe açıklama yap. Eğer create_rc_frame ise:
    - Kullanıcının BELİRTTİĞİ değerleri yaz
    - Belirtmediği alanlar için yukarıdaki GERÇEK default değerleri yaz
    - Asla başka değer (3.5m, C25 vb) UYDURMA
    Edit veya query ise uygun Türkçe yanıt ver. Markdown kullan.

    Analiz sonucu varsa (analysis_summary) referans alarak yanıt ver.
    """

    user_message: str = dspy.InputField(description="Kullanıcının sorusu/komutu")
    current_file: str = dspy.InputField(
        description=".s2k dosya içeriği. Boş veya 'Dosya yok' ise yeni proje.",
    )
    analysis_summary: str = dspy.InputField(
        description=(
            "Son analiz çıktısının özet metni (Türkçe). 'Analiz yapılmamış' olabilir. "
            "AI sorulara bu özeti referans göstererek cevap verebilir."
        ),
    )

    intent: StructuralIntent = dspy.OutputField(
        description="Kullanıcının niyeti ve ilgili parametreler."
    )
    chat_response: str = dspy.OutputField(
        description="Kullanıcıya Türkçe, markdown formatında yanıt."
    )
