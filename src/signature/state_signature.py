from typing import List
import dspy
from pydantic import BaseModel, Field

dspy.configure_cache(enable_memory_cache=False, enable_disk_cache=False)

class StateAnalysisDto(BaseModel):
    """State analiz sonuçları için Pydantic model"""

    sub_tag_key: str = Field(description="Alt state için tag key")

    tagReasoning: str = Field(
        description="Bu alt state'e neden eşleştirildiğinin açıklaması"
    )

    confidenceScore: float = Field(
        ge=0.0, le=1.0, description="Bu alt state'in ne kadar güvenilir olduğunun skoru (0.0-1.0 arası)"
    )


class StateDto(BaseModel):
    main_state: str = Field(description="Ana state")
    sub_state: str = Field(description="Alt state")
    sub_state_description: str = Field(description="Alt state açıklaması")
    sub_tag_key: str = Field(description="Alt state için tag key")

class MessagesToTags(BaseModel):
    role: str = Field(description="Mesajın rolü")
    content: str = Field(description="Mesajın içeriği")

class MessagesAlreadyTagged(MessagesToTags, StateAnalysisDto):
    pass

class StateSignature(dspy.Signature):
    """
    Sen ödeal için geliştirilmiş ai sales agentin lead ile yapılan konuşmalarını verlien etiket yapısına göre analiz eden bir uzmansın.
    """

    company_info: str = dspy.InputField(description="Ödeal şirketi hakkında bilgi")
    project_info: str = dspy.InputField(description="Proje hakkında bilgi")

    messages_to_tags: List[MessagesToTags] = dspy.InputField(
        description="Chatbot veya kullanıcı mesajı"
    )

    messages_already_tagged: List[str] = dspy.InputField(
        description="Önceki analizlerden taglenmiş mesajlar"
    )

    states: List[StateDto] = dspy.InputField(
        description="Mevcut tüm state kategorileri ve açıklamalarının listesi"
    )

    state_analysis: List[StateAnalysisDto] = dspy.OutputField(
        description="Her mesaj için state analiz sonuçları. Analiz edilecek mesaj sayısı kadar sonuç döndürüp her mesaj için ilgili tag'in üretilmesi gerekiyor."
    )
