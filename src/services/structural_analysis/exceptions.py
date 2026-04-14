class StructuralAnalysisError(Exception):
    """Yapısal analiz motoru kök hata sınıfı."""


class ParseError(StructuralAnalysisError):
    """.s2k veya benzeri girdi dosyası ayrıştırılırken oluşan hata."""


class ValidationError(StructuralAnalysisError):
    """Model bütünlük/geometri/yük tutarlılık hatası."""


class AssemblyError(StructuralAnalysisError):
    """Rijitlik, kütle veya yük vektörü birleştirme hatası."""


class SolverError(StructuralAnalysisError):
    """K·U = F çözümü veya öz değer analizi sırasında oluşan hata."""
