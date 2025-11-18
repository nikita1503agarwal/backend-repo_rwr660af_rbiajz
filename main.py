import os
from datetime import datetime, timedelta, time, date
from typing import List, Optional, Dict, Tuple
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from bson import ObjectId
from zoneinfo import ZoneInfo

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


ACTIVE_APPT_STATUSES = ["tentative", "confirmed", "deposit_paid", "paid"]


def parse_hhmm(s: str) -> time:
    h, m = s.split(":")
    return time(int(h), int(m))


def get_org_timezone(org_doc) -> ZoneInfo:
    tz = (org_doc.get("timezone") if isinstance(org_doc, dict) else getattr(org_doc, "timezone", None)) or "UTC"
    try:
        return ZoneInfo(tz)
    except Exception:
        return ZoneInfo("UTC")


def load_availability_rules(organization_id: str) -> List[dict]:
    return list(db["availabilityrule"].find({"organization_id": organization_id}))


def build_daily_work_windows(
    org: dict,
    rules: List[dict],
    service_id: str
) -> Dict[int, List[Tuple[time, time]]]:
    """
    Returns mapping weekday (0=Mon) -> list of (start_time, end_time) windows in local org time.
    If no work_hours rules exist, defaults to Mon-Fri 09:00-17:00.
    Respects service-specific rules (service_ids). If any service-specific work_hours rules exist
    for the given service, use only those for that weekday; otherwise fall back to global rules.
    """
    # Separate rules
    work_rules = [r for r in rules if r.get("rule_type") == "work_hours"]
    svc_specific = [r for r in work_rules if r.get("service_ids")]  # any with list
    svc_rules = [r for r in svc_specific if service_id in (r.get("service_ids") or [])]

    result: Dict[int, List[Tuple[time, time]]] = {i: [] for i in range(7)}

    def add_rule_to_result(r):
        wd = r.get("weekday")
        if wd is None:
            return
        try:
            st = parse_hhmm(r.get("start"))
            en = parse_hhmm(r.get("end"))
            if st < en:
                result[int(wd)].append((st, en))
        except Exception:
            pass

    if svc_rules:
        for r in svc_rules:
            add_rule_to_result(r)
    else:
        # use global work rules (no service_ids)
        global_work = [r for r in work_rules if not r.get("service_ids")]
        if global_work:
            for r in global_work:
                add_rule_to_result(r)
        else:
            # default Mon-Fri 09:00-17:00
            for wd in range(0, 5):
                result[wd].append((time(9, 0), time(17, 0)))

    # Sort windows
    for wd in result:
        result[wd].sort()
    return result


def is_blocked_day(dt_local: date, rules: List[dict], service_id: str) -> bool:
    ds = dt_local.isoformat()  # YYYY-MM-DD
    weekday = dt_local.weekday()

    # Rule: block_date
    for r in rules:
        if r.get("rule_type") == "block_date" and r.get("date") == ds:
            # if service_ids exists, only block for those services; otherwise global
            svc_ids = r.get("service_ids")
            if not svc_ids or service_id in svc_ids:
                return True

    # Rule: block_weekday
    for r in rules:
        if r.get("rule_type") == "block_weekday" and r.get("weekday") == weekday:
            svc_ids = r.get("service_ids")
            if not svc_ids or service_id in svc_ids:
                return True

    return False


def existing_conflict(organization_id: str, start_utc: datetime, end_utc: datetime, exclude_appt_id: Optional[ObjectId] = None) -> bool:
    q = {
        "organization_id": organization_id,
        "start_time": {"$lt": end_utc},
        "end_time": {"$gt": start_utc},
        "status": {"$in": ACTIVE_APPT_STATUSES}
    }
    if exclude_appt_id:
        q["_id"] = {"$ne": exclude_appt_id}
    conflict = db["appointment"].find_one(q)
    return bool(conflict)


def generate_slots(
    organization_id: str,
    service_id: str,
    start_date_local: date,
    days: int,
) -> List[Dict[str, str]]:
    org = db["organization"].find_one({"_id": to_object_id(organization_id)})
    if not org:
        raise HTTPException(status_code=404, detail="Organization not found")
    svc = db["servicetype"].find_one({"_id": to_object_id(service_id)})
    if not svc:
        raise HTTPException(status_code=404, detail="Service not found")

    tz = get_org_timezone(org)
    duration = int(svc.get("duration_minutes", 60))
    rules = load_availability_rules(organization_id)
    windows_by_wd = build_daily_work_windows(org, rules, service_id)

    out: List[Dict[str, str]] = []

    for d in range(days):
        day_local = start_date_local + timedelta(days=d)
        weekday = day_local.weekday()
        if is_blocked_day(day_local, rules, service_id):
            continue
        windows = windows_by_wd.get(weekday, [])
        if not windows:
            continue
        for st_t, en_t in windows:
            # iterate over the window in duration increments
            # Localize to org tz and then convert to UTC for conflict checks/return
            cur_local_dt = datetime.combine(day_local, st_t)
            end_window_local_dt = datetime.combine(day_local, en_t)
            while cur_local_dt + timedelta(minutes=duration) <= end_window_local_dt:
                start_local = cur_local_dt.replace(tzinfo=tz)
                end_local = (cur_local_dt + timedelta(minutes=duration)).replace(tzinfo=tz)
                start_utc = start_local.astimezone(ZoneInfo("UTC"))
                end_utc = end_local.astimezone(ZoneInfo("UTC"))
                if not existing_conflict(organization_id, start_utc, end_utc):
                    out.append({
                        "start": start_utc.isoformat(),
                        "end": end_utc.isoformat(),
                    })
                cur_local_dt += timedelta(minutes=duration)

    return out


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


@app.get("/public/slots")
def public_slots(organization_id: str, service_id: str, date_str: str, days: int = 1):
    try:
        start_date_local = datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format, expected YYYY-MM-DD")
    days = max(1, min(days, 14))
    slots = generate_slots(organization_id, service_id, start_date_local, days)
    return {"slots": slots}


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

    # interpret provided start_time as UTC (ISO assumed). Ensure it's within availability windows
    start_utc = req.start_time if req.start_time.tzinfo else req.start_time.replace(tzinfo=ZoneInfo("UTC"))
    end_utc = start_utc + timedelta(minutes=duration)

    # Validate against availability engine
    rules = load_availability_rules(req.organization_id)
    tz = get_org_timezone(org)
    start_local = start_utc.astimezone(tz)
    day_local = start_local.date()

    if is_blocked_day(day_local, rules, req.service_id):
        raise HTTPException(status_code=409, detail="Selected day is unavailable")

    windows_by_wd = build_daily_work_windows(org, rules, req.service_id)
    weekday = day_local.weekday()
    windows = windows_by_wd.get(weekday, [])
    within_window = False
    for st_t, en_t in windows:
        win_start = datetime.combine(day_local, st_t).replace(tzinfo=tz)
        win_end = datetime.combine(day_local, en_t).replace(tzinfo=tz)
        if start_local >= win_start and (start_local + timedelta(minutes=duration)) <= win_end:
            within_window = True
            break
    if not within_window:
        raise HTTPException(status_code=409, detail="Selected time outside working hours")

    # check appointment conflicts
    if existing_conflict(req.organization_id, start_utc, end_utc):
        raise HTTPException(status_code=409, detail="Time slot unavailable")

    # generate public code
    public_code = os.urandom(4).hex()

    appt = Appointment(
        organization_id=req.organization_id,
        service_id=req.service_id,
        customer_name=req.customer_name,
        customer_email=req.customer_email,
        customer_phone=req.customer_phone,
        start_time=start_utc,
        end_time=end_utc,
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
        org = db["organization"].find_one({"_id": to_object_id(appt["organization_id"])})
        duration = int(svc.get("duration_minutes", 60))
        start_utc = req.start_time if req.start_time.tzinfo else req.start_time.replace(tzinfo=ZoneInfo("UTC"))
        end_utc = start_utc + timedelta(minutes=duration)

        # availability validation
        rules = load_availability_rules(appt["organization_id"]) 
        tz = get_org_timezone(org)
        start_local = start_utc.astimezone(tz)
        day_local = start_local.date()
        if is_blocked_day(day_local, rules, appt["service_id"]):
            raise HTTPException(status_code=409, detail="Selected day is unavailable")
        windows_by_wd = build_daily_work_windows(org, rules, appt["service_id"]) 
        weekday = day_local.weekday()
        windows = windows_by_wd.get(weekday, [])
        within_window = False
        for st_t, en_t in windows:
            win_start = datetime.combine(day_local, st_t).replace(tzinfo=tz)
            win_end = datetime.combine(day_local, en_t).replace(tzinfo=tz)
            if start_local >= win_start and (start_local + timedelta(minutes=duration)) <= win_end:
                within_window = True
                break
        if not within_window:
            raise HTTPException(status_code=409, detail="Selected time outside working hours")

        conflict = existing_conflict(appt["organization_id"], start_utc, end_utc, exclude_appt_id=appt["_id"]) 
        if conflict:
            raise HTTPException(status_code=409, detail="Time slot unavailable")
        db["appointment"].update_one({"_id": appt["_id"]}, {"$set": {"start_time": start_utc, "end_time": end_utc, "updated_at": datetime.utcnow()}})
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
