"""
Database Schemas for Tradie Scheduler SaaS

Each Pydantic model represents a collection in MongoDB.
Collection name is the lowercase class name (e.g., Organization -> "organization").
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Literal
from datetime import datetime

# Core entities

class Organization(BaseModel):
    name: str
    email: str
    slug: str = Field(..., description="Public booking slug, unique per organization")
    phone: Optional[str] = None
    timezone: str = Field("UTC", description="IANA timezone, e.g. Australia/Sydney")

class Worker(BaseModel):
    organization_id: str
    name: str
    phone: Optional[str] = None
    capacity: int = Field(1, ge=1, description="Concurrent jobs they can handle")
    active: bool = True

class ServiceType(BaseModel):
    organization_id: str
    name: str
    description: Optional[str] = None
    duration_minutes: int = Field(..., ge=15, le=8*60)
    price: float = Field(0, ge=0)
    requires_deposit: bool = False
    deposit_amount: float = Field(0, ge=0)
    is_quote_only: bool = Field(False, description="If true, creates a quote appointment type")

AppointmentStatus = Literal[
    "tentative", "confirmed", "deposit_paid", "paid", "cancelled", "missed"
]

class Appointment(BaseModel):
    organization_id: str
    service_id: str
    worker_id: Optional[str] = None
    customer_name: str
    customer_email: Optional[str] = None
    customer_phone: Optional[str] = None
    start_time: datetime
    end_time: datetime
    status: AppointmentStatus = "tentative"
    notes: Optional[str] = None
    public_code: Optional[str] = Field(None, description="Token customers can use to manage booking")

class AvailabilityRule(BaseModel):
    organization_id: str
    rule_type: Literal["work_hours", "block_date", "block_weekday"]
    # work hours: weekday, start, end (HH:MM)
    weekday: Optional[int] = Field(None, ge=0, le=6)
    start: Optional[str] = Field(None, description="Start time HH:MM")
    end: Optional[str] = Field(None, description="End time HH:MM")
    # block_date: specific date YYYY-MM-DD
    date: Optional[str] = None
    reason: Optional[str] = None
    service_ids: Optional[List[str]] = None

class Subscription(BaseModel):
    organization_id: str
    plan: Literal["starter", "pro", "team"] = "starter"
    status: Literal["trialing", "active", "past_due", "canceled"] = "trialing"
    renews_at: Optional[datetime] = None

class Payment(BaseModel):
    appointment_id: str
    amount: float
    currency: str = "AUD"
    status: Literal["unpaid", "deposit_paid", "paid", "refunded"] = "unpaid"
    intent_id: Optional[str] = None

# These schemas are used by the database viewer and for validation.
