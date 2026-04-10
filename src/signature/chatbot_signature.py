from typing import List
import dspy
from pydantic import BaseModel, Field

dspy.configure_cache(enable_memory_cache=False, enable_disk_cache=False)


class InputField:
    company_info: str = Field(description="Ödeal şirketi hakkında bilgi")
    project_info: str = Field(description="Proje hakkında bilgi")

    user_message: str = Field(description="Alt state için tag key")
