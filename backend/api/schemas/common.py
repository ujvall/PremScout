"""Common response schemas for the PremAI API."""

from typing import Generic, TypeVar, Optional, Dict, Any
from pydantic import BaseModel, Field

T = TypeVar("T")


class APIResponse(BaseModel, Generic[T]):
    """Standardized API response envelope for React frontend consumption."""
    success: bool = Field(True, description="Indicates whether the request completed successfully")
    data: Optional[T] = Field(None, description="Payload data returned by the operation")
    error: Optional[str] = Field(None, description="Error message if the operation failed")


class RootResponse(BaseModel):
    """API root service discovery response."""
    name: str = "PremAI Football Intelligence API"
    status: str = "running"
    version: str = "1.0.0"
    docs_url: str = "/docs"


class HealthResponse(BaseModel):
    """System health check reporting availability of all serialized ML model artifacts."""
    status: str = Field(..., description="Overall status: 'healthy' or 'degraded'")
    models: Dict[str, bool] = Field(..., description="Availability status per ML module")
    data_as_of: Optional[str] = Field("2026-09-20", description="Latest available data timestamp")
    current_data_layer: Optional[bool] = Field(True, description="Whether current data layer is active")

