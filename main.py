import os
from datetime import datetime, timedelta
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from bson import ObjectId

from database import db, create_document, get_documents
from schemas import Organization, Worker, ServiceType, Appointment, AvailabilityRule, Subscription, Payment

app = FastAPI(title="Tradie Scheduler API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Helpers

def to_object_id(id_str: str) -> ObjectId:
    try:
        return ObjectId(id_str)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid id")


def with_id(doc):
    if not doc:
        return doc
    doc["id"] = str(doc.pop("_id"))
    return doc


# Public booking link resolution
@app.get("/public/{slug}")
def get_public_org(slug: str):
    org = db["organization"].find_one({"slug": slug})
    if not org:
        raise HTTPException(status_code=404, detail="Organization not found")
    services = list(db["servicetype"].find({"organization_id": str(org["_id"])}))
    return {
        "organization": {"id": str(org["_id"]), "name": org["name"], "slug": org["slug"]},
        "services": [{"id": str(s["_id"]), "name": s["name"], "duration_minutes": s["duration_minutes"], "price": s.get("price", 0), "is_quote_only": s.get("is_quote_only", False)} for s in services]
    }


class CreateAppointmentRequest(BaseModel):
    organization_id: str
    service_id: str
    customer_name: str
    customer_email: Optional[str] = None
    customer_phone: Optional[str] = None
    start_time: datetime
    notes: Optional[str] = None


@app.post("/public/book")
def public_book(req: CreateAppointmentRequest):
    # basic validation: org and service exist
    org = db["organization"].find_one({"_id": to_object_id(req.organization_id)})
    if not org:
        raise HTTPException(status_code=404, detail="Organization not found")
    svc = db["servicetype"].find_one({"_id": to_object_id(req.service_id)})
    if not svc:
        raise HTTPException(status_code=404, detail="Service not found")

    duration = int(svc.get("duration_minutes", 60))
    start = req.start_time
    end = start + timedelta(minutes=duration)

    # check conflicts for worker-agnostic booking (later assign)
    conflict = db["appointment"].find_one({
        "organization_id": req.organization_id,
        "start_time": {"$lt": end},
        "end_time": {"$gt": start},
        "status": {"$in": ["tentative", "confirmed", "deposit_paid", "paid"]}
    })
    if conflict:
        raise HTTPException(status_code=409, detail="Time slot unavailable")

    # generate public code
    public_code = os.urandom(4).hex()

    appt = Appointment(
        organization_id=req.organization_id,
        service_id=req.service_id,
        customer_name=req.customer_name,
        customer_email=req.customer_email,
        customer_phone=req.customer_phone,
        start_time=start,
        end_time=end,
        status="tentative",
        notes=req.notes,
        public_code=public_code
    )

    appt_id = create_document("appointment", appt)
    return {"appointment_id": appt_id, "public_code": public_code, "status": "tentative"}


class ModifyAppointmentRequest(BaseModel):
    appointment_id: str
    public_code: str
    start_time: Optional[datetime] = None
    cancel: Optional[bool] = False


@app.post("/public/modify")
def public_modify(req: ModifyAppointmentRequest):
    appt = db["appointment"].find_one({"_id": to_object_id(req.appointment_id), "public_code": req.public_code})
    if not appt:
        raise HTTPException(status_code=404, detail="Appointment not found")
    if req.cancel:
        db["appointment"].update_one({"_id": appt["_id"]}, {"$set": {"status": "cancelled", "updated_at": datetime.utcnow()}})
        return {"status": "cancelled"}
    if req.start_time:
        svc = db["servicetype"].find_one({"_id": to_object_id(appt["service_id"])})
        duration = int(svc.get("duration_minutes", 60))
        start = req.start_time
        end = start + timedelta(minutes=duration)
        conflict = db["appointment"].find_one({
            "_id": {"$ne": appt["_id"]},
            "organization_id": appt["organization_id"],
            "start_time": {"$lt": end},
            "end_time": {"$gt": start},
            "status": {"$in": ["tentative", "confirmed", "deposit_paid", "paid"]}
        })
        if conflict:
            raise HTTPException(status_code=409, detail="Time slot unavailable")
        db["appointment"].update_one({"_id": appt["_id"]}, {"$set": {"start_time": start, "end_time": end, "updated_at": datetime.utcnow()}})
    return {"status": "updated"}


# Dashboard endpoints
class ConfirmRequest(BaseModel):
    appointment_id: str


@app.post("/dashboard/confirm")
def confirm_appointment(req: ConfirmRequest):
    appt = db["appointment"].find_one({"_id": to_object_id(req.appointment_id)})
    if not appt:
        raise HTTPException(status_code=404, detail="Appointment not found")
    db["appointment"].update_one({"_id": appt["_id"]}, {"$set": {"status": "confirmed", "updated_at": datetime.utcnow()}})
    return {"status": "confirmed"}


@app.get("/dashboard/appointments")
def list_appointments(organization_id: str, status: Optional[str] = None):
    q = {"organization_id": organization_id}
    if status:
        q["status"] = status
    items = [with_id(a) for a in db["appointment"].find(q).sort("start_time", 1)]
    return {"items": items}


class ServiceRequest(BaseModel):
    organization_id: str
    name: str
    duration_minutes: int
    price: float = 0
    is_quote_only: bool = False
    requires_deposit: bool = False
    deposit_amount: float = 0


@app.post("/dashboard/service")
def create_service(req: ServiceRequest):
    service = ServiceType(**req.model_dump())
    sid = create_document("servicetype", service)
    return {"service_id": sid}


class OrgRequest(BaseModel):
    name: str
    email: str
    slug: str


@app.post("/dashboard/org")
def create_org(req: OrgRequest):
    existing = db["organization"].find_one({"slug": req.slug})
    if existing:
        raise HTTPException(status_code=409, detail="Slug already in use")
    org = Organization(name=req.name, email=req.email, slug=req.slug)
    oid = create_document("organization", org)
    return {"organization_id": oid}


# Availability rules
class AvailabilityRequest(BaseModel):
    organization_id: str
    rule_type: str
    weekday: Optional[int] = None
    start: Optional[str] = None
    end: Optional[str] = None
    date: Optional[str] = None
    reason: Optional[str] = None
    service_ids: Optional[List[str]] = None


@app.post("/dashboard/availability")
def add_availability(req: AvailabilityRequest):
    rule = AvailabilityRule(**req.model_dump())
    rid = create_document("availabilityrule", rule)
    return {"rule_id": rid}


@app.get("/dashboard/availability")
def get_availability(organization_id: str):
    rules = [with_id(r) for r in db["availabilityrule"].find({"organization_id": organization_id})]
    return {"rules": rules}


# Basic payment intent mock (replace with Stripe later)
class PaymentIntentRequest(BaseModel):
    appointment_id: str
    amount: float


@app.post("/payments/create-intent")
def create_payment_intent(req: PaymentIntentRequest):
    # In real world integrate Stripe and return client secret
    pay = Payment(appointment_id=req.appointment_id, amount=req.amount, status="unpaid")
    pid = create_document("payment", pay)
    return {"payment_id": pid, "client_secret": "mock_secret"}


@app.get("/test")
def test_database():
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }

    try:
        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Configured"
            response["database_name"] = db.name if hasattr(db, 'name') else "✅ Connected"
            response["connection_status"] = "Connected"
            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️  Connected but Error: {str(e)[:50]}"
        else:
            response["database"] = "⚠️  Available but not initialized"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:50]}"

    import os as _os
    response["database_url"] = "✅ Set" if _os.getenv("DATABASE_URL") else "❌ Not Set"
    response["database_name"] = "✅ Set" if _os.getenv("DATABASE_NAME") else "❌ Not Set"
    return response


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
