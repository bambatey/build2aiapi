"""Yapısal model üreticiler — chat intent'ine göre parametrik .s2k üretir.

Temel prensip: AI ham .s2k metni yazmaz; kullanıcı niyetini parametre'ye
çevirir, bu modül deterministik olarak doğru .s2k üretir. Validator her
çıktının ``parseS2K`` ile açılabildiğini garanti eder.
"""

from .rc_frame import RCFrameParams, generate_rc_frame
from .validators import GenerationError, validate_generated_model

__all__ = [
    "RCFrameParams",
    "generate_rc_frame",
    "validate_generated_model",
    "GenerationError",
]
