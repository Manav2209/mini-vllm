from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    prompt: str = Field(min_length=1)
    max_tokens: int = Field(default=32, ge=1)

    @field_validator("prompt")
    @classmethod
    def nonempty(cls, value):
        if not value.strip():
            raise ValueError("Prompt must not be empty")
        return value


class GenerateResponse(BaseModel):
    request_id: str
    text: str
    generated_tokens: int
    finish_reason: Literal["length", "eos", "cancelled"]
