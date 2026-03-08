from __future__ import annotations

import os
import random
import re
import csv
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional, Set
from copy import deepcopy

# 1) CONSISTENT DATA

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXCEL_INPUT_PATH = os.path.join(BASE_DIR, "data", "from_data_ship")
XML_NS = {
    "a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}

BUSY_DAYS_BY_SHIP: Dict[str, Set[int]] = {}
SHOW_EXCEL_DEBUG_OUTPUT = False
EXCEL_PREVIEW_ROWS = 5
ASSUME_ALL_SHIPS_AVAILABLE_AT_START = True
DEPOT_MIN_SAILING_DAYS = 5
COMPACT_CONSOLE_OUTPUT = True
MERGE_WAIT_LOOKAHEAD_DAYS = 10

# 2) MODELS

@dataclass(frozen=True)
class Location:
    loc: str
    region: str
    product: str
    qty: int
    demand_day: int

@dataclass(frozen=True)
class Ship:
    ship_id: str
    location: str
    ownership: str
    capacity: int
    fixed_daily_cost: int

@dataclass
class Depot:
    depot: str
    region: str
    max_cap: int
    min_cap: int
    daily_decay: int
    initial_level: int = 0
    daily_consumption_by_day: Optional[Dict[int, int]] = None
    max_by_product: Optional[Dict[str, int]] = None
    min_by_product: Optional[Dict[str, int]] = None
    initial_by_product: Optional[Dict[str, int]] = None
    daily_consumption_by_day_product: Optional[Dict[int, Dict[str, int]]] = None
    
@dataclass(frozen=True)
class Route:
    route_id: str
    ship_id: str
    start_port: str
    end_port: str
    stops: Tuple[str, ...]
    stop_day_offsets: Tuple[int, ...]
    duration: int
    cost: int
    region: str
    demand_tons: int
    is_depot_delivery: bool = False 

Solution = Dict[str, Tuple[str, ...]]

# --- NEW EVENT-DRIVEN PLANNING MODELS ---

@dataclass
class Job: # Müşteri veya depo işi olabilir, paketleme ve atama adımlarında ortak model olarak kullanacağız
    job_id: str
    job_type: str  # "customer" or "depot"
    location: str
    qty: int
    release_day: int
    deadline: int
    region: str
    meta: Optional[Dict[str, Any]] = None


@dataclass
class ShipState: # Geminin gün bazlı meşguliyet durumunu tutar (planlama sırasında güncellenir)
    ship_id: str
    busy_intervals: List[Tuple[int, int]]  # (start_day, end_day)


@dataclass
class PlanRecord: # Günlük planlama döngüsünde her gün için oluşturulacak log kaydı modeli
    day: int
    window_start: int # Planlama penceresi başlangıcı (örneğin, gün 1'de pencere 1-10, gün 2'de 2-11 gibi kayar)
    window_end: int # Planlama penceresi sonu
    selected_jobs: List[str] # Bu gün değerlendirmeye alınan işler (job_id listesi)
    package_previews: List[str] # Bu gün oluşturulan paketlerin kısa tanımları (örneğin, "M1+M5", "DEPOT_D_L1", vb.)
    unassigned_count: int # Bu gün atanmayan ve queue'da bekleyen iş sayısı


# Global queue (silmek yok!)
UNASSIGNED_QUEUE: List[Job] = []

# =========================================================
# DEPOT RISK FUNCTION (ADIM 3)
# =========================================================

def compute_depot_risk(day: int, depot: Depot) -> Optional[str]:
    day_consumption = get_depot_daily_consumption(depot, day)
    if day_consumption <= 0:# tüketim olmayan depolar riskli değil, onları atla
        return None

    projected_level = depot.initial_level - day_consumption * day # gün sonunda projekte edilen seviye (yaklaşık)
    days_to_min = (projected_level - depot.min_cap) / day_consumption # min seviyeye kaç gün kaldı?

    if days_to_min <= 5:
        return "CRITICAL"
    elif days_to_min <= 7: # 5-7 gün arası uyarı, 7 gün ve üzeri güvenli kabul edebiliriz
        return "WARNING"
    else:
        return None


def get_depot_daily_consumption_by_product(depot: Depot, day: int) -> Dict[str, int]:
    if depot.daily_consumption_by_day_product:
        if day in depot.daily_consumption_by_day_product:
            return {
                p: max(0, safe_int(v, 0))
                for p, v in depot.daily_consumption_by_day_product[day].items()
            }
        last_known_day = max(depot.daily_consumption_by_day_product.keys())
        if last_known_day in depot.daily_consumption_by_day_product:
            return {
                p: max(0, safe_int(v, 0))
                for p, v in depot.daily_consumption_by_day_product[last_known_day].items()
            }
    # Fallback: eski toplam tüketimi tek pseudo ürün olarak döndür
    total = max(0, safe_int(depot.daily_decay, 0))
    return {"TOTAL": total} if total > 0 else {}


def get_depot_daily_consumption(depot: Depot, day: int) -> int:
    by_product = get_depot_daily_consumption_by_product(depot, day)
    if by_product:
        return sum(max(0, safe_int(v, 0)) for v in by_product.values())
    if depot.daily_consumption_by_day:
        if day in depot.daily_consumption_by_day:
            return max(0, depot.daily_consumption_by_day[day])
        last_known_day = max(depot.daily_consumption_by_day.keys())
        if last_known_day in depot.daily_consumption_by_day:
            return max(0, depot.daily_consumption_by_day[last_known_day])
    return max(0, depot.daily_decay)
    
## Adim 2: Müşteri işleri oluştur (artık her pozitif talep AYRI bir job)
def build_customer_jobs(
    locs: Dict[str, Location],
    demand_rows: List[Dict[str, str]],
    tolerance: int = 2,
) -> Dict[str, Job]:
    jobs: Dict[str, Job] = {}
    if not demand_rows:
        return jobs

    customer_cols = [c for c in demand_rows[0].keys() if normalize_code(c).startswith("M")]
    customer_cols = sorted(customer_cols, key=lambda x: safe_int(re.sub(r"[^0-9]", "", x), 999))

    # Her pozitif hücreyi ayrı bir talep işi yap:
    # örn. M6 gün10 -> CUST_M6_D10, M6 gün19 -> CUST_M6_D19
    for row in demand_rows:
        day = safe_int(row.get("Gun", row.get("Gün", "")), 0)
        if day <= 0:
            continue
        for raw_c in customer_cols:
            qty = safe_int(row.get(raw_c, 0), 0)
            if qty <= 0:
                continue
            cid = normalize_code(raw_c)
            loc = locs.get(cid)
            if loc is None:
                continue
            # Savunma: talebi olmayan / aktif olmayan musteriden job üretme.
            if loc.qty <= 0:
                continue

            base_job_id = f"CUST_{cid}_D{day}"
            job_id = base_job_id
            suffix = 2
            while job_id in jobs:
                job_id = f"{base_job_id}_{suffix}"
                suffix += 1
            jobs[job_id] = Job(
                job_id=job_id,
                job_type="customer",
                location=cid,
                qty=qty,
                release_day=max(1, day - tolerance),
                deadline=min(35, day + tolerance),
                region=loc.region,
                meta={"source_day": day},
            )
    return jobs
# Adim 2: ShipState map'i oluştur (gemi bazında meşgul günlerden blok aralığına çevir)
def build_ship_state_map(ships: Dict[str, Ship], busy_days: Dict[str, Set[int]]) -> Dict[str, ShipState]: 
    state_map: Dict[str, ShipState] = {} # Her gemi için, busy_days'den gün setini alıp blok aralıklarına çeviriyoruz (örneğin, busy_days[G1] = {1,2,3,5,6} -> busy_intervals=[(1,3), (5,6)]). Bu blok aralıkları planlama sırasında geminin uygunluğunu kontrol etmek için kullanılacak.
    for sid in ships:
        days = sorted(busy_days.get(sid, set()))
        intervals: List[Tuple[int, int]] = [] 
        if days: 
            s = e = days[0] # ilk busy günü başlat ilk günü hem başlangıç hem bitiş kabul ediyoruz
            for d in days[1:]: # sonraki busy günlerine bak, eğer ardışık ise bitişi güncelle, değilse mevcut bloğu kaydet ve yeni bloğa başla
                if d == e + 1:
                    e = d
                else:
                    intervals.append((s, e))
                    s = e = d
            intervals.append((s, e))
        state_map[sid] = ShipState(ship_id=sid, busy_intervals=intervals)
    return state_map

# Adim 3: Depo riskine göre kritik depo işleri oluştur (release=bugün, deadline=5 gün sonrası, qty=hedef seviye - projekte seviye)
def build_depot_risk_jobs(
    day: int,
    depots: Dict[str, Depot],
    min_sailing_days: int = DEPOT_MIN_SAILING_DAYS,
    planning_horizon_days: int = 10,
) -> List[Job]:
    risk_jobs: List[Job] = [] #riskli depolar için oluşturulan işleri eklicez
    for depot in depots.values():
        day_consumption = get_depot_daily_consumption(depot, day)
        if day_consumption <= 0: #bu durumda depo stok tüketmiyor riskli değil
            continue
        projected_level = depot.initial_level - day_consumption * max(0, day - 1)
        threshold = depot.min_cap + day_consumption * min_sailing_days
        if projected_level <= threshold: # depo seviyesi kritik eşiğin altına düşüyor, riskli iş oluştur
            target_level = depot.max_cap
            risk_jobs.append( #depo işi oluşturuluyor 
                Job(
                    job_id=f"DEPOT_{depot.depot}_D{day}",
                    job_type="depot",
                    location=depot.depot,
                    qty=max(1, target_level - projected_level), # hedef seviyeyi max_cap al: gemi bırakınca depo max'e kadar yükselebilir
                    release_day=day,
                    deadline=min(35, day + planning_horizon_days - 1),  # bir sonraki 10 günlük plan penceresinde zorunlu iş
                    region=depot.region,
                )
            )
    return risk_jobs

def build_daily_critical_jobs( # Adim 4: Günlük kritik iş listesi oluştur (depo risk işleri + deadline yaklaşan müşteri işleri + queue'dan pencereye düşen işler)
    day: int,
    window_start: int,
    window_end: int,
    unassigned_queue: List[Job], #bu günkü pencereye düşen ve henüz atanmayan müşteri işleri (queue'dan geri alınacak)
    depot_risk_jobs: List[Job], #bu gün için oluşturulan kritik depo işleri (Adım 3'te hesaplanan)
) -> List[Job]:
    window_customer_jobs = [ #deadline yaklaşan müşteri işleri (release ve deadline pencere içinde olanlar) + queue'dan pencereye düşen işler (deadline pencere içinde olanlar)
        j for j in unassigned_queue 
        if j.job_type == "customer" #sadece müşteri işleri değerlendirilecek, depo işleri zaten risk listesinde var
        and not (j.deadline < window_start or j.release_day > window_end) #deadline pencere içinde olanlar (örneğin, pencere=[16,25], job deadline=18, release=16 -> dahil; job deadline=15 -> hariç; job release=26 -> hariç)
    ]
    window_customer_jobs.sort(key=lambda j: (j.deadline, -j.qty)) #deadline'a göre sırala, deadline küçük olan önce gelir, deadline aynıysa qty büyük olan önce gelir (örneğin, job1: deadline=18, qty=10; job2: deadline=18, qty=20 -> job2 önce gelir)
    return depot_risk_jobs + window_customer_jobs #önce depo işleri sonra müşteri işleri, her iki grup da deadline'a göre sıralanmış olarak

def jobs_window_overlap(a: Job, b: Job) -> bool:
    return not (a.deadline < b.release_day or b.deadline < a.release_day)


def _job_demand_day(job: Job) -> int:
    m = re.search(r"_D(\d+)(?:_|$)", job.job_id)
    if m:
        return safe_int(m.group(1), 0)
    if job.meta and "demand_day" in job.meta:
        return safe_int(job.meta["demand_day"], 0)
    return max(0, job.release_day)


def _has_delivery_pair_with_gap(a: Job, b: Job, min_gap: int = 1, max_gap: int = 3) -> bool:
    # Erken talep olani once teslim varsayimiyla iki teslim arasi 1..3 gun kurali
    da = _job_demand_day(a)
    db = _job_demand_day(b)
    first, second = (a, b) if da <= db else (b, a)
    for d1 in range(first.release_day, first.deadline + 1):
        for d2 in range(second.release_day, second.deadline + 1):
            if min_gap <= (d2 - d1) <= max_gap:
                return True
    return False


def can_merge_customer_jobs(a: Job, b: Job) -> bool:
    # Is kurali:
    # 1) talep gunleri arasi fark max 7
    # 2) iki teslim arasi transit 1..3 gun bulunabilmeli
    if a.job_type != "customer" or b.job_type != "customer":
        return False
    if abs(_job_demand_day(a) - _job_demand_day(b)) > 7:
        return False
    return _has_delivery_pair_with_gap(a, b, min_gap=1, max_gap=3)

# Adim 5: Paketleme (seed job etrafında, aynı bölgeden ve pencere örtüşen müşteri işleri öncelikli, sonra diğer bölgeden pencere örtüşen işler)
def build_job_package(seed_job: Job, candidate_jobs: List[Job]) -> List[Job]:
    # Adim 5: Local merge -> Inventory trigger -> Cross-city merge
    package = [seed_job] 
    if seed_job.job_type != "customer": # eğer seed job depo işi ise, sadece tek başına değerlendirelim, merge yapmayalım (çünkü depo işleri zaten risk bazlı kritik işler, onları merge yaparak müşteri işlerinin önüne atmak istemeyiz)
        return package

    selected_customer: Optional[Job] = None

    same_region = [
        j for j in candidate_jobs
        if j.job_id != seed_job.job_id
        and j.job_type == "customer"
        and j.region == seed_job.region
        and can_merge_customer_jobs(seed_job, j)
    ]
    same_region.sort(key=lambda j: (j.deadline, -j.qty)) #aynı bölgeden ve pencere örtüşen müşteri işleri öncelikli, deadline küçük olan önce gelir, deadline aynıysa qty büyük olan önce gelir
    if same_region:
        selected_customer = same_region[0]
    else:
        cross_city = [ 
            j for j in candidate_jobs # diğer bölgeden pencere örtüşen işler, aynı bölgeden uygun iş yoksa bunları değerlendirelim, deadline küçük olan önce gelir, deadline aynıysa qty büyük olan önce gelir
            if j.job_id != seed_job.job_id
            and j.job_type == "customer"
            and can_merge_customer_jobs(seed_job, j)
        ]
        cross_city.sort(key=lambda j: (j.deadline, -j.qty)) 
        if cross_city:
            selected_customer = cross_city[0]

    if selected_customer is not None:
        package.append(selected_customer)

    # Inventory trigger:
    # Tek müşteri paketi varsa ve kritik depo işi varsa, boş kapasiteyi depoya yönlendirebilmek için
    # aynı bölgeden kritik depoyu pakete eklemeyi dene. Bunu tek müşteri ile sınırlıyoruz;
    # çünkü üretilen depo rotaları en güvenli olarak "müşteri + depo" veya "direkt depo" desenini destekliyor.
    if len(package) == 1:
        depot_candidates = [
            j for j in candidate_jobs
            if j.job_type == "depot"
            and j.location not in {p.location for p in package}
            and j.region == seed_job.region
        ]
        depot_candidates.sort(key=lambda j: (j.deadline, -j.qty))
        if depot_candidates:
            package.append(depot_candidates[0])
    return package

def _interval_free(intervals: List[Tuple[int, int]], start_day: int, end_day: int) -> bool:
    for s, e in intervals:
        if not (end_day < s or start_day > e):
            return False
    return True

def _add_busy_interval(intervals: List[Tuple[int, int]], start_day: int, end_day: int) -> List[Tuple[int, int]]:
    merged: List[Tuple[int, int]] = []
    for s, e in sorted(intervals + [(start_day, end_day)]):
        if not merged or s > merged[-1][1] + 1:
            merged.append((s, e))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
    return merged

def _package_targets(package: List[Job]) -> Tuple[List[str], List[str]]:
    customer_targets = [j.location for j in package if j.job_type == "customer"]
    depot_targets = [j.location for j in package if j.job_type == "depot"]
    return customer_targets, depot_targets


def _route_matches_job_windows(route: Route, package: List[Job], start_day: int) -> bool:
    stop_offset_map = {
        stop: offset
        for stop, offset in zip(route.stops, route.stop_day_offsets)
    }
    customer_arrivals: List[Tuple[Job, int]] = []
    for job in package:
        offset = stop_offset_map.get(job.location)
        if offset is None:
            continue
        arrival_day = start_day + offset - 1
        if arrival_day < job.release_day or arrival_day > job.deadline:
            return False
        if job.job_type == "customer":
            customer_arrivals.append((job, arrival_day))

    if len(customer_arrivals) >= 2:
        customer_arrivals.sort(key=lambda x: x[1])
        # bu modelde en fazla 2 musterili paket kullaniyoruz; ilk ikiye kural uygula
        (j1, d1), (j2, d2) = customer_arrivals[0], customer_arrivals[1]
        if not (1 <= (d2 - d1) <= 3):
            return False
        if abs(_job_demand_day(j1) - _job_demand_day(j2)) > 7:
            return False
    return True

def find_feasible_ship_options_for_package(
    package: List[Job],
    routes: Dict[str, Route],
    ships: Dict[str, Ship],
    ship_state_map: Dict[str, ShipState],
    window_start: int,
    window_end: int,
) -> List[Dict[str, Any]]:
    # STEP 7: Feasibility filter (kapasite + meşguliyet + rota uyumu)
    customer_targets, depot_targets = _package_targets(package) #müşteri ve depo hedeflerini paket içinden çıkarıyoruz, bu hedefler rotada olmalı
    if not customer_targets and not depot_targets: 
        return []

    total_customer_qty = sum(j.qty for j in package if j.job_type == "customer") #paket içindeki müşterilerin toplam talebini toplar 
    release_day = min((j.release_day for j in package), default=window_start)
    deadline_day = max((j.deadline for j in package), default=window_end)
    earliest_start = max(window_start, release_day)
    latest_start = min(window_end, deadline_day)
    if earliest_start > latest_start:
        return []

    feasible: List[Dict[str, Any]] = []
    for r in routes.values():
        stop_set = set(r.stops)
        # Paketin tum musterileri/depolari rotada olmali
        if any(t not in stop_set for t in customer_targets):
            continue
        if any(d not in stop_set for d in depot_targets):
            continue

        ship = ships.get(r.ship_id)
        if ship is None:
            continue
        if total_customer_qty > ship.capacity:
            continue

        state = ship_state_map.get(r.ship_id)
        if state is None:
            continue

        # Pencere içinde uygun bir start var mı bak
        route_latest_start = min(latest_start, window_end - r.duration + 1)
        route_earliest_start = earliest_start
        if route_earliest_start > route_latest_start:
            continue

        chosen_start = None
        chosen_end = None
        # Backward scheduling: en geç feasible slotu seç
        for start_day in range(route_latest_start, route_earliest_start - 1, -1):
            end_day = start_day + r.duration - 1
            if end_day > 35:
                continue
            if _interval_free(state.busy_intervals, start_day, end_day):
                if not _route_matches_job_windows(r, package, start_day):
                    continue
                chosen_start = start_day
                chosen_end = end_day
                break
        if chosen_start is None:
            continue

        feasible.append({
            "ship_id": r.ship_id,
            "route_id": r.route_id,
            "start_day": chosen_start,
            "end_day": chosen_end,
            "cost": r.cost,
            "duration": r.duration,
            "ship_capacity": ship.capacity,
            "customer_stop_count": len([s for s in r.stops if s.startswith("M")]),
        })

    feasible.sort(key=lambda x: (x["cost"], x["duration"]))
    return feasible

def select_ship_for_package(
    feasible_ship_options: List[Dict[str, Any]],
    mode: str = "best_fit",
) -> Optional[Dict[str, Any]]:
    # STEP 8: Gemiyi ata (simdilik "en uygun"; sonra "en ucuz")
    if not feasible_ship_options:
        return None

    if mode == "lowest_cost":
        ranked = sorted(
            feasible_ship_options,
            key=lambda x: (x["cost"], x["duration"], x["start_day"])
        )
        return ranked[0]

    if mode == "max_fill":
        ranked = sorted(
            feasible_ship_options,
            key=lambda x: (-x.get("ship_capacity", 0), x["cost"], x["duration"], x["start_day"])
        )
        return ranked[0]

    if mode == "merge_first":
        ranked = sorted(
            feasible_ship_options,
            key=lambda x: (-x.get("customer_stop_count", 0), x["cost"], x["duration"], x["end_day"])
        )
        return ranked[0]

    if mode == "urgency_fit":
        ranked = sorted(
            feasible_ship_options,
            key=lambda x: (x["end_day"], x["duration"], x["start_day"], x["ship_id"], x["route_id"])
        )
        return ranked[0]

    # default = best_fit:
    # pencereye en iyi oturan (erken bitebilen), sonra kısa süre, sonra düşük maliyet
    ranked = sorted(
        feasible_ship_options,
        key=lambda x: (x["end_day"], x["duration"], x["start_day"], x["ship_id"], x["route_id"])
    )
    return ranked[0]
# Adim 2-3-4-5-6-7-8: Günlük döngü + pencere + depo risk + kritik iş + paket + queue + gemi filtresi + gemi seçimi.
def run_rolling_horizon_day_loop_step15( 
    jobs_by_id: Dict[str, Job], 
    depots: Dict[str, Depot],
    ships: Dict[str, Ship],
    routes: Dict[str, Route],
    busy_days: Dict[str, Set[int]],
    ship_selection_mode: str = "best_fit", #gemiyi seçme modu
    horizon_len: int = 10, # rolling horizon penceresi uzunlugu (gun)
    total_days: int = 35,
) -> Tuple[List[Dict], List[Job], Dict[str, ShipState]]: 
    """STEP 2-3-4-5-6-7-8: Günlük döngü + pencere + depo risk + kritik iş + paket + queue + gemi filtresi + gemi seçimi.
#yani şu an bu iş şu gemiye sığar mı diye bakar planlama ön izlmesi
    
    (Gerçek atama/takvim bloklama Step 8-9'da tamamlanacak. Burada feasibility preview var.)
    """

    # ShipState'leri üret (meşgul günlerden blok aralığı çıkar)
    ship_state_map = build_ship_state_map(ships, busy_days) #çakışma kontrolünü hızlı yapmak, bu rota şu aralığa sığar mı


    # Depo seviyeleri (simülasyon için), her depo için mevcut stok seviyesi başlatılıyor
    depot_levels_by_product: Dict[str, Dict[str, int]] = {}
    for d_id, d_obj in depots.items():
        init_map = dict(d_obj.initial_by_product or {})
        max_map = dict(d_obj.max_by_product or {})
        if not init_map and max_map:
            init_map = dict(max_map)
        if not init_map:
            init_map = {"TOTAL": int(d_obj.initial_level if d_obj.initial_level > 0 else d_obj.max_cap)}
        # Negatif/boş değerleri temizle
        depot_levels_by_product[d_id] = {k: max(0, safe_int(v, 0)) for k, v in init_map.items()}

    depot_levels: Dict[str, int] = {
        d_id: sum(levels.values()) for d_id, levels in depot_levels_by_product.items()
    }

    # STEP 1: unassigned_queue (silmek yok, burada beklet)
    unassigned_queue: List[Job] = [] # atama yapılamayan müşteri işleri burada bekleyecek, her gün pencereye düşenleri değerlendirirken bu kuyruktan da bakacağız
    unassigned_ids: Set[str] = set() 
    assigned_customer_ids: Set[str] = set() #müşteri işleri tekrar tekrar atanmasın diye bir müşteri işi bi kere servis edilisn

    plan_logs: List[Dict] = [] 

    # Pre: müşteri işleri ,hem müşteri hem depo işleri olabilir
    all_customer_jobs = [j for j in jobs_by_id.values() if j.job_type == "customer"]

#her gün yeniden planla, önümüzdeki 10 günü 
    for t in range(1, total_days + 1): # rolling horizon'un her gününde, o günün penceresini belirle, depo riskini hesapla, kritik işleri seç, paketleme ve atama adımlarını uygula, günlük log'u oluştur. Pencere sonunda atanmayan işler unassigned_queue'ya eklenir ve sonraki gün değerlendirilir.
        window_start = t
        window_end = min(total_days, t + horizon_len - 1)
        window = (window_start, window_end)
        endgame_direct_mode = (t >= 30) # Ay sonuna yaklaştıkça merge yapmadan direkt atama moduna geçelim, kalan gün azaldıkça paketleme esnekliği de azalır, mümkün olduğunca tek işleri atamaya çalışalım

        # -----------------------
        # STEP 3: Depo risk (ilk öncelik)
        # -----------------------
        critical_depot_jobs: List[Job] = [] #kritik depo işleri
        critical_depot_context: Dict[str, Dict[str, float]] = {} # kritik tetik anı (teslimat oncesi) bilgileri
        depot_risk_snapshot: Dict[str, Dict[str, float]] = {} #raporlama her depo için o anki stok seviyesi min stok min seviyeye kaç gün kaladığı gibi değerler saklıyor

        for d_id, depot in depots.items(): 
            # günlük tüketim (ürün bazlı)
            cons_by_product = get_depot_daily_consumption_by_product(depot, t)
            if d_id not in depot_levels_by_product:
                depot_levels_by_product[d_id] = {}
            for p, cons in cons_by_product.items():
                cur = depot_levels_by_product[d_id].get(p, 0)
                depot_levels_by_product[d_id][p] = max(0, cur - max(0, safe_int(cons, 0)))
            depot_levels[d_id] = sum(depot_levels_by_product[d_id].values())

            # Ürün bazlı eşik:
            # threshold_p = safety_p + min_seyir_gunu * tuketim_p
            min_by_product = depot.min_by_product or {}
            threshold_by_product: Dict[str, int] = {}
            for p in set(list(depot_levels_by_product[d_id].keys()) + list(cons_by_product.keys()) + list(min_by_product.keys())):
                cons_p = max(0, safe_int(cons_by_product.get(p, 0), 0))
                min_p = max(0, safe_int(min_by_product.get(p, 0), 0))
                threshold_by_product[p] = min_p + DEPOT_MIN_SAILING_DAYS * cons_p

            threshold = sum(threshold_by_product.values())
            days_to_min = 10**9
            day_consumption = sum(max(0, safe_int(v, 0)) for v in cons_by_product.values())
            if day_consumption > 0:
                days_to_min = (depot_levels[d_id] - depot.min_cap) / day_consumption

            depot_risk_snapshot[d_id] = { #rapor yazdırılır
                "level": float(depot_levels[d_id]),
                "min": float(depot.min_cap),
                "threshold": float(threshold),
                "days_to_min": float(days_to_min),
            }

            # Depo seviyesi eşiğe yaklaşınca zorunlu depo işi:
            # örn. 5500 + 200 = 5700
            critical_products: List[str] = []
            for p, th_p in threshold_by_product.items():
                cur_p = depot_levels_by_product[d_id].get(p, 0)
                if cur_p <= th_p:
                    critical_products.append(p)
            if critical_products:
                target_level = depots[d_id].max_cap
                refill_by_product: Dict[str, int] = {}
                max_by_product = depots[d_id].max_by_product or {}
                critical_gap_total = 0
                for p in critical_products:
                    critical_gap_total += max(0, threshold_by_product.get(p, 0) - depot_levels_by_product[d_id].get(p, 0))
                for p in set(list(max_by_product.keys()) + list(depot_levels_by_product[d_id].keys())):
                    tgt = max(0, safe_int(max_by_product.get(p, 0), 0))
                    cur = max(0, safe_int(depot_levels_by_product[d_id].get(p, 0), 0))
                    gap = max(0, tgt - cur)
                    if gap > 0:
                        refill_by_product[p] = gap
                refill_qty = max(1, sum(refill_by_product.values()))
                critical_depot_context[d_id] = {
                    "trigger_level": float(depot_levels[d_id]),
                    "threshold": float(threshold),
                    "day_consumption": float(day_consumption),
                    "refill_qty": float(refill_qty),
                    "days_to_min": float(days_to_min),
                    "critical_products": ", ".join(sorted(critical_products)),
                    "threshold_by_product": ", ".join(
                        f"{p}:{threshold_by_product.get(p, 0)}"
                        for p in sorted(threshold_by_product.keys())
                    ),
                    "critical_gap_total": float(critical_gap_total),
                    "level_by_product": ", ".join(
                        f"{p}:{depot_levels_by_product[d_id].get(p, 0)}"
                        for p in sorted(depot_levels_by_product[d_id].keys())
                    ),
                    "refill_by_product": ", ".join(
                        f"{p}:{v}" for p, v in sorted(refill_by_product.items())
                    ),
                }
                critical_depot_jobs.append(
                    Job(
                        job_id=f"DEPOT_{d_id}_D{t}",
                        job_type="depot",
                        location=d_id,
                        qty=refill_qty,  # kritik depoyu tekrar max seviyeye yaklaştıracak miktar
                        release_day=t,
                        deadline=window_end,
                        region=depot.region,
                        meta={
                            "depot_level": depot_levels[d_id],
                            "days_to_min": days_to_min,
                            "threshold": threshold,
                            "refill_qty": refill_qty,
                            "critical_products": critical_products,
                            "refill_by_product": refill_by_product,
                        },
                    )
                )

        # -----------------------
        # STEP 4: Kritik müşteri işleri + queue geri alma
        # Kural: deadline yaklaşanı öne al (deadline küçük olan önce)
        # Pencere dışı işleri bu gün değerlendirme.
        # -----------------------
        active_jobs: List[Job] = []

        # Müşteri işleri: deadline pencere içinde olanları değerlendir
        for j in all_customer_jobs:
            if j.job_id in assigned_customer_ids: #bu müşteri işi zaten atandıysa, tekrar değerlendirme
                continue  # artik atandi
            if j.job_id in unassigned_ids: #bu müşteri işi zaten queue'da bekliyorsa, onu da pencereye düşür, ama duplicate yapma
                continue  # zaten kuyrukta, duplicate yapma
            if j.deadline < window_start: 
                continue  # geçmiş
            if j.deadline > window_end:
                continue  # pencere dışı
            active_jobs.append(j)

        # Queue'dan pencereye düşenleri geri al (duplicate olmasın)
        if unassigned_queue:
            still_waiting: List[Job] = []
            for q in unassigned_queue:
                if q.deadline < window_start: # geçmiş, bu işi artık değerlendirme, ama kuyruktan da çıkarma (silmek yok, raporlama için lazım olabilir)
                    # kaçırıldı: yine de kuyrukta tut (silmek yok)
                    still_waiting.append(q)
                    continue
                if q.deadline > window_end: # pencere dışı, bu işi bu gün değerlendirme, ama kuyrukta tutmaya devam et (silmek yok)
                    still_waiting.append(q)
                    continue
                active_jobs.append(q) #bu iş artık pencereye düştü, aktif işlere ekle
                unassigned_ids.discard(q.job_id) #bu iş artık pencereye düştü, unassigned_ids'den çıkar (çünkü artık aktif işlerde, kuyrukta değil)
            unassigned_queue = still_waiting

        # Depo + müşteri birlikte önceliklendir:
        # Kural: aynı gün müşteri deadline daha acilse depodan önce ele alınır.
        active_jobs = critical_depot_jobs + active_jobs

        def _job_urgency_key(job: Job) -> Tuple[int, int, int, str]:
            # Küçük değer = daha acil.
            if job.job_type == "customer":
                slack = max(0, safe_int(job.deadline, window_end) - window_start)
                # Müşteri işi release günü henüz gelmediyse biraz geri it.
                not_released_penalty = 1 if safe_int(job.release_day, 0) > window_start else 0
                return (slack, not_released_penalty, 0, job.job_id)
            # Depo işi için urgency: min seviyeye kalan gün.
            d2m = 999
            if isinstance(job.meta, dict):
                d2m = safe_int(job.meta.get("days_to_min", 999), 999)
            return (max(0, d2m), 0, 1, job.job_id)

        active_jobs.sort(key=_job_urgency_key)

        # -----------------------
        # STEP 5-6-7-8-9 (+ STEP 10 endgame mode)
        # Step 6: Paket yok/uygun değilse queue'da beklet
        # Step 7: Uygun gemileri feasibility filtresi ile çıkar
        # Step 10: Ay sonu (t>=30) birlestirme azalt, direkt atama fallback dene
        # -----------------------
        chosen_jobs = [j.job_id for j in active_jobs]
        packages_preview: List[Dict] = []
        step6_no_package = 0
        step6_waiting = 0
        step7_feasible_hits = 0
        step8_selected = 0
        step9_assigned = 0
        step9_depot_asg = 0
        step9_depot_ton = 0
        step10_direct_mode_hits = 0
        day_assignments: List[Dict[str, Any]] = []

        for job in active_jobs:
            if job.job_type == "customer" and job.job_id in assigned_customer_ids:
                continue

            force_direct_mode = (
                job.job_type == "customer"
                and (job.deadline - window_start) <= 1
            )
            precomputed_feasible_options: Optional[List[Dict[str, Any]]] = None
            pkg: List[Job]

            # MERGE-FIRST (hard rule):
            # Musteri isi icin, tekli feasible olsa bile once 2'li feasible paket dene.
            # Tekli atama sadece 2'li bulunamazsa fallback.
            if (
                job.job_type == "customer"
                and not endgame_direct_mode
                and not force_direct_mode
            ):
                candidate_customers = [
                    c for c in active_jobs
                    if c.job_type == "customer"
                    and c.job_id != job.job_id
                    and c.job_id not in assigned_customer_ids
                    and can_merge_customer_jobs(job, c)
                ]
                candidate_customers.sort(
                    key=lambda c: (
                        0 if c.region == job.region else 1,
                        c.deadline,
                        -c.qty,
                    )
                )

                best_merge_pkg: Optional[List[Job]] = None
                best_merge_options: Optional[List[Dict[str, Any]]] = None
                for cand in candidate_customers:
                    trial_pkg = [job, cand]
                    trial_options = find_feasible_ship_options_for_package(
                        package=trial_pkg,
                        routes=routes,
                        ships=ships,
                        ship_state_map=ship_state_map,
                        window_start=window_start,
                        window_end=window_end,
                    )
                    if not trial_options:
                        continue
                    if best_merge_options is None:
                        best_merge_pkg = trial_pkg
                        best_merge_options = trial_options
                        continue
                    # Aciliyet tie-break:
                    # 1) ayni region, 2) daha yakin deadline, 3) daha erken bitebilen rota
                    current_best_second = best_merge_pkg[1]
                    cand_score = (
                        0 if cand.region == job.region else 1,
                        cand.deadline,
                        trial_options[0]["end_day"],
                    )
                    best_score = (
                        0 if current_best_second.region == job.region else 1,
                        current_best_second.deadline,
                        best_merge_options[0]["end_day"],
                    )
                    if cand_score < best_score:
                        best_merge_pkg = trial_pkg
                        best_merge_options = trial_options

                if best_merge_pkg is not None and best_merge_options is not None:
                    pkg = best_merge_pkg
                    precomputed_feasible_options = best_merge_options
                else:
                    pkg = [job]  # 2'li bulunamazsa tekli fallback
            else:
                pkg = [job] if (endgame_direct_mode or force_direct_mode) else build_job_package(job, active_jobs)

            pkg = [
                p for p in pkg
                if not (p.job_type == "customer" and p.job_id in assigned_customer_ids)
            ]

            # Erken tekli atamayi onle:
            # Eger is acil degilse ve ileride merge olasiligi varsa,
            # bugun tekli atama yapma, kuyruga birak.
            if (
                len(pkg) == 1
                and pkg[0].job_type == "customer"
                and not endgame_direct_mode
                and not force_direct_mode
            ):
                slack_days = max(0, safe_int(job.deadline, window_end) - window_start)
                # Kisa erteleme: onumuzdeki 10 gunde acilacak bir adayla merge feasible ise bekle.
                lookahead_start = min(total_days, window_start + 1)
                lookahead_end = min(total_days, window_start + MERGE_WAIT_LOOKAHEAD_DAYS)
                future_merge_candidate_exists = False
                if slack_days >= 2:
                    for c in all_customer_jobs:
                        if c.job_id == job.job_id or c.job_id in assigned_customer_ids:
                            continue
                        if not can_merge_customer_jobs(job, c):
                            continue
                        if not (lookahead_start <= c.release_day <= lookahead_end):
                            continue
                        for d in range(c.release_day, lookahead_end + 1):
                            trial_window_start = d
                            trial_window_end = min(total_days, trial_window_start + horizon_len - 1)
                            trial = find_feasible_ship_options_for_package(
                                package=[job, c],
                                routes=routes,
                                ships=ships,
                                ship_state_map=ship_state_map,
                                window_start=trial_window_start,
                                window_end=trial_window_end,
                            )
                            if trial:
                                future_merge_candidate_exists = True
                                break
                        if future_merge_candidate_exists:
                            break

                if future_merge_candidate_exists:
                    step6_waiting += 1
                    if job.job_id not in unassigned_ids:
                        unassigned_queue.append(job)
                        unassigned_ids.add(job.job_id)
                    packages_preview.append({
                        "job": job.job_id,
                        "package": pkg,
                        "decision_order": ["JOB", "PACKAGE", "SHIP", "CALENDAR"],
                        "feasible_ship_count": 0,
                        "top_ship_options": [],
                        "selected_ship_option": None,
                        "endgame_direct_mode": endgame_direct_mode,
                        "defer_reason": "wait_for_merge",
                    })
                    continue

            if not pkg:
                step6_no_package += 1
                if job.job_id not in unassigned_ids:
                    unassigned_queue.append(job)
                    unassigned_ids.add(job.job_id)
                continue

            if precomputed_feasible_options is None:
                feasible_ship_options = find_feasible_ship_options_for_package(
                    package=pkg,
                    routes=routes,
                    ships=ships,
                    ship_state_map=ship_state_map,
                    window_start=window_start,
                    window_end=window_end,
                )
            else:
                feasible_ship_options = precomputed_feasible_options
            if feasible_ship_options:
                step7_feasible_hits += 1
            else:
                step6_waiting += 1
                # Paket başarısızsa tekli müşteri fallback
                if (
                    len(pkg) > 1
                    and job.job_type == "customer"
                ):
                    direct_pkg = [job]
                    if job.job_id in assigned_customer_ids:
                        continue
                    feasible_ship_options = find_feasible_ship_options_for_package(
                        package=direct_pkg,
                        routes=routes,
                        ships=ships,
                        ship_state_map=ship_state_map,
                        window_start=window_start,
                        window_end=window_end,
                    )
                    if feasible_ship_options:
                        pkg = direct_pkg
                        step7_feasible_hits += 1
                        if endgame_direct_mode or force_direct_mode:
                            step10_direct_mode_hits += 1

            selection_mode = "urgency_fit"
            if any(p.job_type == "depot" for p in pkg):
                selection_mode = "max_fill"
            selected_ship_option = select_ship_for_package(
                feasible_ship_options,
                mode=selection_mode,
            )
            if selected_ship_option is not None:
                step8_selected += 1
                if endgame_direct_mode and len(pkg) == 1:
                    step10_direct_mode_hits += 1

            packages_preview.append({
                "job": job.job_id,
                "package": pkg,
                "decision_order": ["JOB", "PACKAGE", "SHIP", "CALENDAR"],
                "feasible_ship_count": len(feasible_ship_options),
                "top_ship_options": feasible_ship_options[:3],
                "selected_ship_option": selected_ship_option,
                "endgame_direct_mode": endgame_direct_mode,
            })

            # STEP 9: Takvimi blokla, atanan işleri listeden düş
            if selected_ship_option is not None:
                sid = selected_ship_option["ship_id"]
                start_day = selected_ship_option["start_day"]
                end_day = selected_ship_option["end_day"]
                ship = ships.get(sid)
                selected_route = routes.get(selected_ship_option["route_id"])
                ship_state_map[sid].busy_intervals = _add_busy_interval(
                    ship_state_map[sid].busy_intervals, start_day, end_day
                )

                depot_delivery_debug: List[str] = []
                if ship is not None:
                    customer_jobs_in_pkg = [p for p in pkg if p.job_type == "customer"]
                    depot_jobs_in_pkg = [p for p in pkg if p.job_type == "depot"]
                    spare_capacity = max(0, ship.capacity - sum(p.qty for p in customer_jobs_in_pkg))

                    route_depot_stops: List[str] = []
                    if selected_route is not None:
                        route_depot_stops = [s for s in selected_route.stops if s in depots]

                    depot_targets: List[Tuple[str, int]] = []
                    seen_depots: Set[str] = set()
                    for depot_job in depot_jobs_in_pkg:
                        if depot_job.location in seen_depots:
                            continue
                        depot_targets.append((depot_job.location, max(0, depot_job.qty)))
                        seen_depots.add(depot_job.location)
                    for depot_stop in route_depot_stops:
                        if depot_stop in seen_depots:
                            continue
                        depot_targets.append((depot_stop, spare_capacity if customer_jobs_in_pkg else ship.capacity))
                        seen_depots.add(depot_stop)

                    for depot_loc, desired_amount in depot_targets:
                        depot_obj = depots.get(depot_loc)
                        if depot_obj is None:
                            continue

                        target_level = depot_obj.max_cap
                        headroom = max(0, target_level - depot_levels.get(depot_loc, 0))
                        capacity_for_delivery = spare_capacity if customer_jobs_in_pkg else ship.capacity
                        delivered_amount = min(desired_amount, headroom, capacity_for_delivery)
                        if delivered_amount <= 0:
                            continue

                        # Teslimati urun bazli dagit: en buyuk acik hangi urundeyse once ona ver.
                        max_by_product = depot_obj.max_by_product or {"TOTAL": target_level}
                        if depot_loc not in depot_levels_by_product:
                            depot_levels_by_product[depot_loc] = {"TOTAL": depot_levels.get(depot_loc, 0)}
                        deficits: List[Tuple[str, int]] = []
                        for p, pmax in max_by_product.items():
                            cur = depot_levels_by_product[depot_loc].get(p, 0)
                            gap = max(0, max(0, safe_int(pmax, 0)) - cur)
                            if gap > 0:
                                deficits.append((p, gap))
                        deficits.sort(key=lambda x: x[1], reverse=True)
                        remaining_to_put = int(delivered_amount)
                        delivered_by_product: Dict[str, int] = {}
                        for p, gap in deficits:
                            if remaining_to_put <= 0:
                                break
                            put = min(gap, remaining_to_put)
                            if put <= 0:
                                continue
                            depot_levels_by_product[depot_loc][p] = depot_levels_by_product[depot_loc].get(p, 0) + put
                            delivered_by_product[p] = delivered_by_product.get(p, 0) + put
                            remaining_to_put -= put

                        delivered_total = sum(delivered_by_product.values())
                        if delivered_total <= 0:
                            continue
                        depot_levels[depot_loc] = sum(depot_levels_by_product[depot_loc].values())
                        product_txt = ",".join(f"{k}:{v}" for k, v in sorted(delivered_by_product.items()))
                        depot_delivery_debug.append(f"{depot_loc}:{delivered_total}[{product_txt}]")

                        if depot_loc in depot_risk_snapshot:
                            updated_level = float(depot_levels[depot_loc])
                            depot_risk_snapshot[depot_loc]["level"] = updated_level
                            next_day_consumption = get_depot_daily_consumption(depot_obj, t)
                            if next_day_consumption > 0:
                                depot_risk_snapshot[depot_loc]["days_to_min"] = float(
                                    (updated_level - depot_obj.min_cap) / next_day_consumption
                                )

                        if customer_jobs_in_pkg:
                            spare_capacity = max(0, spare_capacity - delivered_total)

                assigned_now_ids = [
                    p.job_id for p in pkg
                    if p.job_type == "customer"
                ]
                for aid in assigned_now_ids:
                    assigned_customer_ids.add(aid)
                    if aid in unassigned_ids:
                        unassigned_ids.discard(aid)
                        unassigned_queue = [q for q in unassigned_queue if q.job_id != aid]
                step9_assigned += len(assigned_now_ids)
                if depot_delivery_debug:
                    step9_depot_asg += 1
                    for dep in depot_delivery_debug:
                        qty_txt = str(dep).split(":", 1)[1] if ":" in str(dep) else "0"
                        qty_match = re.match(r"(\d+)", qty_txt)
                        if qty_match:
                            step9_depot_ton += safe_int(qty_match.group(1), 0)
                day_assignments.append({
                    "ship_id": sid,
                    "route_id": selected_ship_option["route_id"],
                    "start_day": start_day,
                    "end_day": end_day,
                    "jobs": assigned_now_ids,
                    "depot_deliveries": depot_delivery_debug,
                })
                continue

            # Step 6-8 sonucu atama yoksa queue'da beklet
            if job.job_type == "customer" and job.job_id not in unassigned_ids:
                unassigned_queue.append(job)
                unassigned_ids.add(job.job_id)

        # -----------------------
        # STEP 11: Günlük log (zorunlu)
        # -----------------------
        if COMPACT_CONSOLE_OUTPUT:
            print(
                f"[DAY {t:02d}] win=[{window_start},{window_end}] "
                f"mode={'ENDGAME' if endgame_direct_mode else 'OPT'} "
                f"crit={len(critical_depot_jobs)} act={len(active_jobs)} "
                f"asg={step9_assigned} dep_asg={step9_depot_asg} dep_ton={step9_depot_ton} q={len(unassigned_queue)}"
            )
        else:
            print(
                f"[DAY {t:02d}] window=[{window_start},{window_end}] "
                f"mode={'ENDGAME_DIRECT' if endgame_direct_mode else 'OPTIMIZE'} "
                f"critical={len(critical_depot_jobs)} active={len(active_jobs)} "
                f"queue={len(unassigned_queue)} assigned={step9_assigned} "
                f"depot_assigned={step9_depot_asg} depot_ton={step9_depot_ton} "
                f"feasible={step7_feasible_hits} selected={step8_selected}"
            )
            if day_assignments:
                assignment_parts: List[str] = []
                for a in day_assignments[:3]:
                    jobs_txt = ",".join(a.get("jobs", []))
                    depot_txt = ""
                    if a.get("depot_deliveries"):
                        depot_txt = " +DEPOT"
                    assignment_parts.append(
                        f"{a.get('ship_id')}:{a.get('route_id')}[{a.get('start_day')}-{a.get('end_day')}] -> {jobs_txt}{depot_txt}"
                    )
                print("  atamalar:", " | ".join(assignment_parts))

        plan_logs.append({
            "day": t,
            "window": window,
            "mode": ("ENDGAME_DIRECT" if endgame_direct_mode else "OPTIMIZE"),
            "critical_depot_jobs": [j.job_id for j in critical_depot_jobs],
            "critical_depot_context": critical_depot_context,
            "active_jobs": [j.job_id for j in active_jobs],
            "chosen_jobs": chosen_jobs,
            "packages_preview": packages_preview,
            "unassigned_queue_size": len(unassigned_queue),
            "step6_no_package": step6_no_package,
            "step6_waiting": step6_waiting,
            "step7_feasible_hits": step7_feasible_hits,
            "step8_selected": step8_selected,
            "step9_assigned": step9_assigned,
            "step9_depot_assigned": step9_depot_asg,
            "step9_depot_ton": step9_depot_ton,
            "step10_direct_mode_hits": step10_direct_mode_hits,
            "assignments": day_assignments,
            "depot_risk": depot_risk_snapshot,
        })

    # -----------------------
    # FINAL CLEANUP PASS:
    # Kalan müşteri işlerini tekli ve direkt modda son kez dene.
    # Amaç: queue'da kalmış servis edilmemiş işleri zorlayarak tamamlamak.
    # -----------------------
    cleanup_assignments: List[Dict[str, Any]] = []
    remaining_customer_jobs = sorted(
        [j for j in all_customer_jobs if j.job_id not in assigned_customer_ids],
        key=lambda j: (j.deadline, j.release_day, j.job_id),
    )

    for job in remaining_customer_jobs:
        cleanup_window_start = max(1, job.release_day)
        cleanup_window_end = min(total_days, job.deadline)
        if cleanup_window_start > cleanup_window_end:
            continue

        feasible_ship_options = find_feasible_ship_options_for_package(
            package=[job],
            routes=routes,
            ships=ships,
            ship_state_map=ship_state_map,
            window_start=cleanup_window_start,
            window_end=cleanup_window_end,
        )
        if not feasible_ship_options:
            continue

        selected_ship_option = select_ship_for_package(
            feasible_ship_options,
            mode="urgency_fit",
        )
        if selected_ship_option is None:
            continue

        sid = selected_ship_option["ship_id"]
        start_day = selected_ship_option["start_day"]
        end_day = selected_ship_option["end_day"]
        ship_state_map[sid].busy_intervals = _add_busy_interval(
            ship_state_map[sid].busy_intervals, start_day, end_day
        )

        assigned_customer_ids.add(job.job_id)
        if job.job_id in unassigned_ids:
            unassigned_ids.discard(job.job_id)
            unassigned_queue = [q for q in unassigned_queue if q.job_id != job.job_id]

        cleanup_assignments.append({
            "ship_id": sid,
            "route_id": selected_ship_option["route_id"],
            "start_day": start_day,
            "end_day": end_day,
            "jobs": [job.job_id],
        })

    print(
        f"[CLEANUP] assigned={sum(len(a.get('jobs', [])) for a in cleanup_assignments)} "
        f"remaining_queue={len(unassigned_queue)}"
    )
    if cleanup_assignments:
        assignment_parts: List[str] = []
        for a in cleanup_assignments[:5]:
            jobs_txt = ",".join(a.get("jobs", []))
            assignment_parts.append(
                f"{a.get('ship_id')}:{a.get('route_id')}[{a.get('start_day')}-{a.get('end_day')}] -> {jobs_txt}"
            )
        print("  atamalar:", " | ".join(assignment_parts))
        plan_logs.append({
            "day": total_days + 1,
            "window": (total_days + 1, total_days + 1),
            "mode": "CLEANUP",
            "critical_depot_jobs": [],
            "active_jobs": [j.job_id for j in remaining_customer_jobs],
            "chosen_jobs": [j.job_id for j in remaining_customer_jobs],
            "packages_preview": [],
            "unassigned_queue_size": len(unassigned_queue),
            "step6_no_package": 0,
            "step6_waiting": 0,
            "step7_feasible_hits": len(cleanup_assignments),
            "step8_selected": len(cleanup_assignments),
            "step9_assigned": sum(len(a.get("jobs", [])) for a in cleanup_assignments),
            "step10_direct_mode_hits": len(cleanup_assignments),
            "assignments": cleanup_assignments,
            "depot_risk": {},
        })

    return plan_logs, unassigned_queue, ship_state_map
    
# 3) PARSING

def normalize_code(raw: str) -> str:
    return str(raw or "").strip().upper()

def normalize_depot_code(raw: str) -> str:
    # Kullanici talebine gore depo kodlari musteriyle ayni namespace'te kalacak.
    # Ornek: M8 hem musteri hem depo kimligi olarak ayni string'i kullanir.
    return normalize_code(raw)

def humanize_code(code: str) -> str:
    v = normalize_code(code)
    m = re.fullmatch(r"M(\d+)", v)
    if m:
        return f"Müşteri{m.group(1)}"
    g = re.fullmatch(r"G(\d+)", v)
    if g:
        return f"Gemi{g.group(1)}"
    l = re.fullmatch(r"L(\d+)", v)
    if l:
        return f"Liman{l.group(1)}"
    d = re.fullmatch(r"D_L(\d+)", v)
    if d:
        return f"DepoLiman{d.group(1)}"
    return code

def safe_int(value: object, default: int = 0) -> int:
    s = str(value or "").strip().replace(",", ".")
    if not s:
        return default
    try:
        return int(round(float(s)))
    except ValueError:
        return default

def _xlsx_shared_strings(book: zipfile.ZipFile) -> List[str]:
    if "xl/sharedStrings.xml" not in book.namelist():
        return []
    root = ET.fromstring(book.read("xl/sharedStrings.xml"))
    items: List[str] = []
    for si in root.findall("a:si", XML_NS):
        parts = [t.text or "" for t in si.findall(".//a:t", XML_NS)]
        items.append("".join(parts))
    return items

def _xlsx_sheet_paths(book: zipfile.ZipFile) -> Dict[str, str]:
    wb = ET.fromstring(book.read("xl/workbook.xml"))
    rels = ET.fromstring(book.read("xl/_rels/workbook.xml.rels"))
    rel_map = {r.attrib["Id"]: r.attrib["Target"] for r in rels}
    paths: Dict[str, str] = {}
    for sheet in wb.findall("a:sheets/a:sheet", XML_NS):
        rid = sheet.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]
        paths[sheet.attrib["name"]] = "xl/" + rel_map[rid]
    return paths

def _xlsx_rows_as_dicts(path: str) -> Dict[str, List[Dict[str, str]]]:
    # Yeni veri kaynagi: data/from_data_ship klasorundeki CSV dosyalari
    if os.path.isdir(path):
        out: Dict[str, List[Dict[str, str]]] = {}
        csv_files = [f for f in os.listdir(path) if f.lower().endswith(".csv")]
        if not csv_files:
            return out

        sheet_name_map = {
            "musteri_talepleri": "Musteri_Talepleri",
            "kapasite": "Kapasite",
            "mesguliyetler": "Mesguliyetler",
            "gun_maliyet": "Gun_Maliyet",
            "max_stock": "Max Stock",
            "initial_stock": "Initial Stock",
            "safety_stock": "Safety Stock",
            "tuketim": "Tuketim",
            "rafineri_talebi": "Rafineri Talebi",
        }

        for fname in csv_files:
            raw_key = os.path.splitext(fname)[0]
            key = normalize_code(raw_key.lower())
            sheet_name = sheet_name_map.get(key, raw_key)
            fpath = os.path.join(path, fname)

            rows: List[Dict[str, str]] = []
            with open(fpath, "r", encoding="utf-8-sig", newline="") as f:
                reader = csv.reader(f, delimiter=";")
                matrix = [[(c or "").strip() for c in r] for r in reader]

            # Tamamen bos satirlari at
            matrix = [r for r in matrix if any(c.strip() for c in r)]
            if not matrix:
                out[sheet_name] = []
                continue

            headers = [h.strip() for h in matrix[0]]
            # Trailing bos kolonlari temizle (CSV'lerde ";;" var)
            while headers and headers[-1] == "":
                headers.pop()
            if not headers:
                out[sheet_name] = []
                continue

            for r in matrix[1:]:
                if len(r) < len(headers):
                    r = r + [""] * (len(headers) - len(r))
                else:
                    r = r[:len(headers)]
                rec = {headers[i] if headers[i] else f"COL_{i+1}": r[i] for i in range(len(headers))}
                if any(str(v).strip() for v in rec.values()):
                    rows.append(rec)

            out[sheet_name] = rows

            # Kodda hem bosluklu hem alti-cizgili isimler kullanildigi icin alias ekle
            if " " in sheet_name:
                out[sheet_name.replace(" ", "_")] = rows
            if "_" in sheet_name:
                out[sheet_name.replace("_", " ")] = rows

        return out

    out: Dict[str, List[Dict[str, str]]] = {}
    with zipfile.ZipFile(path) as book:
        sst = _xlsx_shared_strings(book)
        sheet_paths = _xlsx_sheet_paths(book)

        for name, sheet_path in sheet_paths.items():
            root = ET.fromstring(book.read(sheet_path))
            rows = root.findall(".//a:sheetData/a:row", XML_NS)
            if not rows:
                out[name] = []
                continue

            matrix: List[List[str]] = []
            max_col = 0
            for row in rows:
                row_cells: Dict[int, str] = {}
                for cell in row.findall("a:c", XML_NS):
                    ref = cell.attrib.get("r", "")
                    col_txt = "".join(ch for ch in ref if ch.isalpha())
                    col_num = 0
                    for ch in col_txt:
                        col_num = col_num * 26 + (ord(ch.upper()) - 64)
                    if col_num <= 0:
                        continue
                    max_col = max(max_col, col_num)
                    t = cell.attrib.get("t")
                    v = cell.find("a:v", XML_NS)
                    value = ""
                    if v is not None and v.text is not None:
                        value = v.text
                        if t == "s":
                            idx = int(value)
                            value = sst[idx] if 0 <= idx < len(sst) else ""
                    row_cells[col_num] = value
                row_values = [row_cells.get(i, "").strip() for i in range(1, max_col + 1)]
                matrix.append(row_values)

            if not matrix:
                out[name] = []
                continue
            headers = [h.strip() or f"COL_{i+1}" for i, h in enumerate(matrix[0])]
            data: List[Dict[str, str]] = []
            for r in matrix[1:]:
                record: Dict[str, str] = {}
                for i, h in enumerate(headers):
                    record[h] = r[i] if i < len(r) else ""
                if any(v.strip() for v in record.values()):
                    data.append(record)
            out[name] = data
    return out

def load_from_excel(path: str) -> Tuple[Dict[str, Location], Dict[str, Ship], Dict[str, Depot], Dict[str, Route], Dict[str, Set[int]]]:
    sheets = _xlsx_rows_as_dicts(path)

    raw_demand = sheets.get("Musteri_Talepleri", [])
    if not raw_demand:
        raise ValueError("Musteri_Talepleri sayfasi bos veya bulunamadi.")

    customer_cols = [c for c in raw_demand[0].keys() if normalize_code(c).startswith("M")]
    customer_cols = sorted(customer_cols, key=lambda x: safe_int(re.sub(r"[^0-9]", "", x), 999))
    if not customer_cols:
        raise ValueError("Musteri_Talepleri sayfasinda musteri kolonu bulunamadi.")

    customers_by_half = max(1, len(customer_cols) // 2)
    locs: Dict[str, Location] = {}
    for idx, raw_c in enumerate(customer_cols):
        c = normalize_code(raw_c)
        day_values: List[Tuple[int, int]] = []
        for row in raw_demand:
            day = safe_int(row.get("Gun", row.get("Gün", "")), 0)
            qty = safe_int(row.get(raw_c, "0"), 0)
            if day > 0:
                day_values.append((day, qty))
        non_zero = [(d, q) for d, q in day_values if q > 0]
        demand_day = non_zero[0][0] if non_zero else 0
        qty = max((q for _, q in non_zero), default=0)
        region = "North" if idx < customers_by_half else "South"
        locs[c] = Location(loc=c, region=region, product="mixed", qty=qty, demand_day=demand_day)

    raw_ships = sheets.get("Kapasite", [])
    if not raw_ships:
        raise ValueError("Kapasite sayfasi bos veya bulunamadi.")
    ships: Dict[str, Ship] = {}
    for row in raw_ships:
        sid = normalize_code(row.get("Gemi", ""))
        ship_loc = normalize_code(row.get("Lokasyon", ""))
        cap = safe_int(row.get("Kapasite", "0"), 0)
        if not sid or cap <= 0:
            continue
        ships[sid] = Ship(
            ship_id=sid,
            location=ship_loc,
            ownership="owned",
            capacity=cap,
            fixed_daily_cost=0,
        )
    if not ships:
        raise ValueError("Kapasite sayfasindan gecerli gemi okunamadi.")

    raw_refinery = sheets.get("Rafineri Talebi", [])
    port_demand: Dict[str, int] = {}
    for row in raw_refinery:
        p = normalize_code(row.get("Varis", ""))
        if p.startswith("L"):
            port_demand[p] = port_demand.get(p, 0) + max(0, safe_int(row.get("Talep", "0"), 0))

    ports: List[str] = []
    for row in raw_ships:
        p = normalize_code(row.get("Lokasyon", ""))
        if p.startswith("L") and p not in ports:
            ports.append(p)
    if not ports:
        ports = ["L1", "L2"]


    # --- DEPOTS (from Excel) ---
    # Expected sheets: "Max Stock", "Safety Stock", "Initial Stock", "Tuketim"
    depots: Dict[str, Depot] = {}

    def _read_depot_table(sheet_name: str) -> Dict[str, Dict[str, int]]:
        """Return {depot_id: {product: value}} from a table like:
           index/depot column + product columns (e.g., Ü1, Ü2)."""
        tbl = sheets.get(sheet_name, [])
        if not tbl:
            return {}
        headers = list(tbl[0].keys())
        depot_col = headers[0]
        product_cols = [h for h in headers[1:] if str(h).strip()]
        out_map: Dict[str, Dict[str, int]] = {}
        for row in tbl:
            did = normalize_depot_code(row.get(depot_col, ""))
            if not did:
                continue
            prod_map: Dict[str, int] = {}
            for pc in product_cols:
                prod_map[str(pc).strip()] = max(0, safe_int(row.get(pc, 0), 0))
            out_map[did] = prod_map
        return out_map

    max_stock = _read_depot_table("Max Stock")
    safety_stock = _read_depot_table("Safety Stock")
    initial_stock = _read_depot_table("Initial Stock")

    # Daily consumption: Tuketim tablosunu gerçek gün bazlı oku.
    daily_decay_by_depot: Dict[str, int] = {}
    daily_consumption_by_depot: Dict[str, Dict[int, int]] = {}
    daily_consumption_by_depot_product: Dict[str, Dict[int, Dict[str, int]]] = {}
    raw_cons = sheets.get("Tuketim", [])
    if raw_cons:
        headers = list(raw_cons[0].keys())
        depot_col = headers[0]
        day_key = None
        for k in headers:
            if normalize_code(k) in {"GUN", "GÜN", "DAY"}:
                day_key = k
                break
        product_cols = [h for h in headers if h not in {depot_col, day_key}]

        for row in raw_cons:
            did = normalize_depot_code(row.get(depot_col, ""))
            day = safe_int(row.get(day_key or "Gun", 0), 0)
            if not did or day <= 0:
                continue
            per_product: Dict[str, int] = {}
            total_consumption = 0
            for pc in product_cols:
                pv = max(0, safe_int(row.get(pc, 0), 0))
                per_product[str(pc).strip()] = pv
                total_consumption += pv
            daily_consumption_by_depot.setdefault(did, {})[day] = total_consumption
            daily_consumption_by_depot_product.setdefault(did, {})[day] = per_product

        for did, per_day in daily_consumption_by_depot.items():
            non_zero_values = [v for v in per_day.values() if v > 0]
            daily_decay_by_depot[did] = non_zero_values[0] if non_zero_values else 0

    if max_stock:
        for did, prod_map in max_stock.items():
            max_cap_total = sum(prod_map.values())
            min_cap_total = sum(safety_stock.get(did, {}).values()) if safety_stock else 0
            init_total = sum(initial_stock.get(did, {}).values()) if initial_stock else 0
            depots[did] = Depot(
                depot=did,
                region="UNKNOWN",
                max_cap=max_cap_total,
                min_cap=min_cap_total,
                daily_decay=daily_decay_by_depot.get(did, 0),
                initial_level=init_total if init_total > 0 else max_cap_total,
                daily_consumption_by_day=daily_consumption_by_depot.get(did, {}),
                max_by_product=prod_map,
                min_by_product=safety_stock.get(did, {}) if safety_stock else {},
                initial_by_product=initial_stock.get(did, {}) if initial_stock else {},
                daily_consumption_by_day_product=daily_consumption_by_depot_product.get(did, {}),
            )
    else:
        # Fallback: create depots from ports list (old behavior)
        for idx, p in enumerate(ports):
            depot_id = f"D_{p}"
            region = "North" if idx == 0 else "South"
            max_cap = max(10000, port_demand.get(p, 0) + 5000)
            depots[depot_id] = Depot(
                depot=depot_id,
                region=region,
                max_cap=max_cap,
                min_cap=0,
                daily_decay=0,
                initial_level=max_cap,
                max_by_product={"TOTAL": max_cap},
                min_by_product={"TOTAL": 0},
                initial_by_product={"TOTAL": max_cap},
                daily_consumption_by_day_product={},
            )

    raw_routes = sheets.get("Gun_Maliyet", [])
    if not raw_routes:
        raise ValueError("Gun_Maliyet sayfasi bos veya bulunamadi.")
    routes: Dict[str, Route] = {}
    route_no = 1
    for row in raw_routes:
        if safe_int(row.get("Mumkun", "0"), 0) != 1:
            continue
        sid = normalize_code(row.get("g", ""))
        if sid not in ships:
            continue
        start_port = normalize_code(row.get("i", "L1")) or "L1"
        end_port = normalize_code(row.get("o", start_port)) or start_port

        stop_candidates = [normalize_code(row.get("j", "")), normalize_code(row.get("k", ""))]
        raw_stop_days = [
            safe_int(row.get("Gun1", "0"), 0),
            safe_int(row.get("Gun2", "0"), 0),
        ]
        stops: List[str] = []
        stop_raw_days: List[int] = []
        for idx, s in enumerate(stop_candidates):
            if s.startswith("M") and s in locs and s not in stops:
                stops.append(s)
                stop_raw_days.append(raw_stop_days[idx] if idx < len(raw_stop_days) else 0)
        if not stops:
            continue

        day_points = [
            safe_int(row.get("Gun1", "0"), 0),
            safe_int(row.get("Gun2", "0"), 0),
            safe_int(row.get("Gun3", "0"), 0),
        ]
        day_points = [d for d in day_points if d > 0]
        if day_points:
            duration = max(1, max(day_points) - min(day_points) + 1)
            route_day_zero = min(day_points)
        else:
            duration = 1
            route_day_zero = 1
        cost = max(1, safe_int(row.get("Maliyet", "1"), 1))
        stop_day_offsets = tuple(
            max(1, raw_day - route_day_zero + 1) if raw_day > 0 else 1
            for raw_day in stop_raw_days
        )
        if stop_day_offsets:
            # Güvence: son durak günü rota süresini aşamaz.
            duration = max(duration, max(stop_day_offsets))

        stop_regions = {locs[s].region for s in stops}
        region = stop_regions.pop() if len(stop_regions) == 1 else "MIXED"
        demand_tons = sum(locs[s].qty for s in stops)
        # Route ID'yi Gun_Maliyet'teki "se" kolonu ile birebir eslestir.
        se_val = safe_int(row.get("se", "0"), 0)
        if se_val > 0:
            rid = f"X{se_val}"
        else:
            rid = f"X{route_no}"
            route_no += 1
        if rid in routes:
            # Olasi cakismanin ustune yazma; deterministik suffix ekle.
            suffix = 2
            while f"{rid}_{suffix}" in routes:
                suffix += 1
            rid = f"{rid}_{suffix}"

        routes[rid] = Route(
            route_id=rid,
            ship_id=sid,
            start_port=start_port,
            end_port=end_port,
            stops=tuple(stops),
            stop_day_offsets=stop_day_offsets,
            duration=duration,
            cost=cost,
            region=region,
            demand_tons=demand_tons,
            is_depot_delivery=False,
        )
    if not routes:
        raise ValueError("Gun_Maliyet sayfasindan gecerli rota okunamadi.")

    busy_days: Dict[str, Set[int]] = {}
    if not ASSUME_ALL_SHIPS_AVAILABLE_AT_START:
        raw_busy = sheets.get("Mesguliyetler", [])
        for row in raw_busy:
            day = safe_int(row.get("Gun", row.get("Gün", "")), 0)
            if day <= 0:
                continue
            for col, value in row.items():
                sid = normalize_code(col)
                if sid.startswith("G") and safe_int(value, 0) == 1:
                    busy_days.setdefault(sid, set()).add(day)

    return locs, ships, depots, routes, busy_days


def print_excel_data_preview(
    path: str,
    locs: Dict[str, Location],
    ships: Dict[str, Ship],
    depots: Dict[str, Depot],
    routes: Dict[str, Route],
    busy_days: Dict[str, Set[int]],
    max_rows: int = 5,
) -> None:
    sheets = _xlsx_rows_as_dicts(path)
    ordered_sheets = [
        "Musteri_Talepleri",
        "Kapasite",
        "Mesguliyetler",
        "Gun_Maliyet",
        "Max Stock",
        "Initial Stock",
        "Safety Stock",
        "Tuketim",
        "Rafineri Talebi",
    ]

    print("\n=== EXCEL RAW PREVIEW ===")
    for sheet_name in ordered_sheets:
        rows = sheets.get(sheet_name, [])
        print(f"\n[{sheet_name}] satir={len(rows)}")
        if not rows:
            continue
        columns = list(rows[0].keys())
        print("kolonlar:", columns)
        for idx, row in enumerate(rows[:max_rows], start=1):
            compact = {k: v for k, v in row.items() if str(v).strip()}
            print(f"  {idx}. {compact}")

    print("\n=== PARSED SUMMARY ===")
    print(f"Musteri sayisi: {len(locs)}")
    for k in sorted(locs.keys(), key=lambda x: safe_int(re.sub(r'[^0-9]', '', x), 999))[:max_rows]:
        v = locs[k]
        print(f"  {k}: qty={v.qty}, demand_day={v.demand_day}, region={v.region}")

    print(f"\nGemi sayisi: {len(ships)}")
    for k in sorted(ships.keys(), key=lambda x: safe_int(re.sub(r'[^0-9]', '', x), 999))[:max_rows]:
        v = ships[k]
        print(f"  {k}: capacity={v.capacity}")

    print(f"\nDepo sayisi: {len(depots)}")
    for k in sorted(depots.keys())[:max_rows]:
        v = depots[k]
        print(f"  {k}: max_cap={v.max_cap}, min_cap={v.min_cap}, initial={v.initial_level}")

    print(f"\nRota sayisi: {len(routes)}")
    for rid, route in list(routes.items())[:max_rows]:
        print(
            f"  {rid}: ship={route.ship_id}, stops={route.stops}, "
            f"duration={route.duration}, cost={route.cost}, demand={route.demand_tons}"
        )

    print(f"\nMesgul gemi sayisi: {len(busy_days)}")
    for sid in sorted(busy_days.keys(), key=lambda x: safe_int(re.sub(r'[^0-9]', '', x), 999))[:max_rows]:
        print(f"  {sid}: days={sorted(busy_days[sid])}")


def print_all_customer_demands_from_excel(path: str) -> None:
    """Musteri_Talepleri sayfasindaki tum pozitif talepleri tek tek bas."""
    sheets = _xlsx_rows_as_dicts(path)
    raw_demand = sheets.get("Musteri_Talepleri", [])
    if not raw_demand:
        print("\n=== MUSTERI TALEPLERI (tek tek) ===")
        print("  Musteri_Talepleri sayfasi bulunamadi / bos.")
        return

    customer_cols = [c for c in raw_demand[0].keys() if normalize_code(c).startswith("M")]
    customer_cols = sorted(customer_cols, key=lambda x: safe_int(re.sub(r"[^0-9]", "", x), 999))

    print("\n=== MUSTERI TALEPLERI (tek tek, >0) ===")
    all_items: List[Tuple[int, str, int]] = []
    for row in raw_demand:
        day = safe_int(row.get("Gun", row.get("Gün", "")), 0)
        if day <= 0:
            continue
        for raw_c in customer_cols:
            qty = safe_int(row.get(raw_c, 0), 0)
            if qty <= 0:
                continue
            cid = normalize_code(raw_c)
            all_items.append((day, cid, qty))

    if not all_items:
        print("  Pozitif talep bulunamadi.")
        return

    all_items.sort(key=lambda x: (x[0], safe_int(re.sub(r"[^0-9]", "", x[1]), 999)))
    print("Tek tek talep listesi:")
    for idx, (day, cid, qty) in enumerate(all_items, start=1):
        print(f"  {idx:02d}. Gun {day:02d} | {cid} ({humanize_code(cid)}) | talep={qty}")

    print(f"Toplam talep satiri (pozitif hucre): {len(all_items)}")

    # Müşteri bazlı adet kontrolü (örn. M6 icin 2 talep var mi)
    count_by_customer: Dict[str, int] = {}
    grouped_items: Dict[str, List[Tuple[int, int]]] = {}
    for _, cid, _ in all_items:
        count_by_customer[cid] = count_by_customer.get(cid, 0) + 1
    for day, cid, qty in all_items:
        grouped_items.setdefault(cid, []).append((day, qty))
    print("Talep adedi (musteri bazli):")
    for cid in sorted(count_by_customer.keys(), key=lambda x: safe_int(re.sub(r"[^0-9]", "", x), 999)):
        print(f"  {cid}: {count_by_customer[cid]} satir")
        details = [f"Gun {day:02d}={qty}" for day, qty in grouped_items.get(cid, [])]
        if details:
            print(f"    detay: {', '.join(details)}")


def print_all_ship_capacities(ships: Dict[str, Ship]) -> None:
    """Kapasite sayfasindan parse edilen tum gemi kapasitelerini bas."""
    print("\n=== GEMI KAPASITELERI ===")
    if not ships:
        print("  Gemi verisi bulunamadi.")
        return

    for sid in sorted(ships.keys(), key=lambda x: safe_int(re.sub(r"[^0-9]", "", x), 999)):
        ship = ships[sid]
        print(
            f"  {sid} ({humanize_code(sid)}) | kapasite={ship.capacity} | "
            f"lokasyon={ship.location} | "
            f"sahiplik={ship.ownership} | fixed_daily_cost={ship.fixed_daily_cost}"
        )


def print_all_ship_busy_days(busy_days: Dict[str, Set[int]]) -> None:
    """Mesguliyetler sayfasindan parse edilen gemi bazli mesgul gunleri bas."""
    print("\n=== GEMI MESGULIYETLERI ===")
    if not busy_days:
        print("  Mesguliyet verisi bulunamadi.")
        return

    for sid in sorted(busy_days.keys(), key=lambda x: safe_int(re.sub(r"[^0-9]", "", x), 999)):
        days = sorted(busy_days.get(sid, set()))
        if not days:
            continue
        day_text = ", ".join(str(d) for d in days)
        print(f"  {sid} ({humanize_code(sid)}) | gunler=[{day_text}]")


def print_stock_tables_from_excel(path: str) -> None:
    """Max Stock / Initial Stock / Safety Stock tablolarini ham haliyle bas."""
    sheets = _xlsx_rows_as_dicts(path)
    target_sheets = ["Max Stock", "Initial Stock", "Safety Stock"]

    for sheet_name in target_sheets:
        rows = sheets.get(sheet_name, [])
        print(f"\n=== {sheet_name.upper()} ===")
        if not rows:
            print("  Veri bulunamadi.")
            continue

        headers = list(rows[0].keys())
        print("\t".join(headers))
        for row in rows:
            values = [str(row.get(h, "")).strip() for h in headers]
            print("\t".join(values))


def read_depot_table_from_excel(path: str, sheet_name: str) -> Dict[str, Dict[str, int]]:
    """Tek bir stok tablosunu {depo: {urun: deger}} olarak oku."""
    sheets = _xlsx_rows_as_dicts(path)
    rows = sheets.get(sheet_name, [])
    if not rows:
        return {}

    headers = list(rows[0].keys())
    depot_col = headers[0]
    product_cols = [h for h in headers[1:] if str(h).strip()]
    out_map: Dict[str, Dict[str, int]] = {}
    for row in rows:
        did = normalize_depot_code(row.get(depot_col, ""))
        if not did:
            continue
        prod_map: Dict[str, int] = {}
        for pc in product_cols:
            prod_map[str(pc).strip()] = max(0, safe_int(row.get(pc, 0), 0))
        out_map[did] = prod_map
    return out_map


def print_tuketim_table_from_excel(path: str) -> None:
    """Tuketim tablosunu ham haliyle bas."""
    sheets = _xlsx_rows_as_dicts(path)
    rows = sheets.get("Tuketim", [])

    print("\n=== TUKETIM ===")
    if not rows:
        print("  Veri bulunamadi.")
        return

    headers = list(rows[0].keys())
    print("\t".join(headers))
    for row in rows:
        values = [str(row.get(h, "")).strip() for h in headers]
        print("\t".join(values))


def print_rafineri_talebi_from_excel(path: str) -> None:
    """Rafineri Talebi tablosunu ham haliyle bas."""
    sheets = _xlsx_rows_as_dicts(path)
    rows = sheets.get("Rafineri Talebi", [])

    print("\n=== RAFINERI TALEBI ===")
    if not rows:
        print("  Veri bulunamadi.")
        return

    headers = list(rows[0].keys())
    print("\t".join(headers))
    for row in rows:
        values = [str(row.get(h, "")).strip() for h in headers]
        print("\t".join(values))


def print_all_route_costs(routes: Dict[str, Route]) -> None:
    """Gun_Maliyet sayfasindan parse edilen rotalari ve maliyetlerini bas."""
    print("\n=== GUN_MALIYET / ROTALAR ===")
    if not routes:
        print("  Rota verisi bulunamadi.")
        return

    print(f"Toplam rota sayisi: {len(routes)}")

    routes_by_ship: Dict[str, List[Route]] = {}
    for route in routes.values():
        routes_by_ship.setdefault(route.ship_id, []).append(route)

    print("Gemi bazli rota adedi:")
    for sid in sorted(routes_by_ship.keys(), key=lambda x: safe_int(re.sub(r"[^0-9]", "", x), 999)):
        print(f"  {sid} ({humanize_code(sid)}): {len(routes_by_ship[sid])} rota")

    print("Ilk rota ornekleri:")
    ordered_routes = sorted(
        routes.values(),
        key=lambda r: safe_int(re.sub(r"[^0-9]", "", r.route_id), 999),
    )
    for route in ordered_routes[:20]:
        stops_text = ", ".join(route.stops)
        print(
            f"  {route.route_id} | gemi={route.ship_id} | baslangic={route.start_port} | "
            f"bitis={route.end_port} | stops=[{stops_text}] | sure={route.duration} | "
            f"maliyet={route.cost}"
        )


def print_raw_gun_maliyet_from_excel(path: str) -> None:
    """Gun_Maliyet sayfasini ham haliyle, tablo sirasiyla yazdir."""
    sheets = _xlsx_rows_as_dicts(path)
    raw_rows = sheets.get("Gun_Maliyet", [])

    print("\n=== GUN_MALIYET (HAM SATIRLAR) ===")
    if not raw_rows:
        print("  Gun_Maliyet sayfasi bulunamadi / bos.")
        return

    print(f"Toplam ham satir sayisi: {len(raw_rows)}")
    print("g\ti\tj\tk\to\tse\tGun1\tGun2\tGun3\tSure\tMaliyet\tMumkun")

    for idx, row in enumerate(raw_rows, start=1):
        gemi = normalize_code(row.get("g", ""))
        baslangic = normalize_code(row.get("i", ""))
        durak1 = normalize_code(row.get("j", ""))
        durak2 = normalize_code(row.get("k", ""))
        bitis = normalize_code(row.get("o", ""))
        se = safe_int(row.get("se", 0), 0)
        gun1 = safe_int(row.get("Gun1", 0), 0)
        gun2 = safe_int(row.get("Gun2", 0), 0)
        gun3 = safe_int(row.get("Gun3", 0), 0)
        maliyet = safe_int(row.get("Maliyet", 0), 0)
        mumkun = safe_int(row.get("Mumkun", 0), 0)

        gun_points = [g for g in [gun1, gun2, gun3] if g > 0]
        sure = (max(gun_points) - min(gun_points) + 1) if gun_points else 0

        print(
            f"{gemi}\t{baslangic}\t{durak1}\t{durak2}\t{bitis}\t{se}\t"
            f"{gun1}\t{gun2}\t{gun3}\t{sure}\t{maliyet}\t{mumkun}"
        )


def export_raw_gun_maliyet_to_tsv(path: str, out_path: str) -> None:
    """Gun_Maliyet sayfasini ham haliyle TSV dosyasina yaz."""
    sheets = _xlsx_rows_as_dicts(path)
    raw_rows = sheets.get("Gun_Maliyet", [])

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("g\ti\tj\tk\to\tse\tGun1\tGun2\tGun3\tSure\tMaliyet\tMumkun\n")
        for row in raw_rows:
            gemi = normalize_code(row.get("g", ""))
            baslangic = normalize_code(row.get("i", ""))
            durak1 = normalize_code(row.get("j", ""))
            durak2 = normalize_code(row.get("k", ""))
            bitis = normalize_code(row.get("o", ""))
            se = safe_int(row.get("se", 0), 0)
            gun1 = safe_int(row.get("Gun1", 0), 0)
            gun2 = safe_int(row.get("Gun2", 0), 0)
            gun3 = safe_int(row.get("Gun3", 0), 0)
            maliyet = safe_int(row.get("Maliyet", 0), 0)
            mumkun = safe_int(row.get("Mumkun", 0), 0)

            gun_points = [g for g in [gun1, gun2, gun3] if g > 0]
            sure = (max(gun_points) - min(gun_points) + 1) if gun_points else 0

            f.write(
                f"{gemi}\t{baslangic}\t{durak1}\t{durak2}\t{bitis}\t{se}\t"
                f"{gun1}\t{gun2}\t{gun3}\t{sure}\t{maliyet}\t{mumkun}\n"
            )

       
# 3.5) DYNAMIC DEPOT ROUTE GENERATION

def generate_depot_routes(
    ships: Dict[str, Ship],
    routes: Dict[str, Route], 
    depots: Dict[str, Depot],
    locs: Dict[str, Location]
) -> Dict[str, Route]:
    new_routes = {}
    
    # 1. Karma Rotalar
    for rid, r in routes.items():
        if r.region == "MIXED" or len(r.stops) >= 2:
            continue
        
        ship = ships[r.ship_id]
        if r.demand_tons >= ship.capacity:
            continue
        
        target_depots = [d for d in depots.values() if d.region == r.region]
        if not target_depots:
            target_depots = list(depots.values())
            
        for d in target_depots:
            new_id = f"{rid}_{d.depot}"
            new_stops = r.stops + (d.depot,)
            
            new_cost = int(r.cost * 1.2)
            last_offset = max(r.stop_day_offsets) if r.stop_day_offsets else r.duration
            new_stop_offset = last_offset + 1
            new_duration = max(r.duration + 1, new_stop_offset)
            
            new_routes[new_id] = Route(
                route_id=new_id,
                ship_id=r.ship_id,
                start_port=r.start_port,
                end_port=r.end_port,
                stops=new_stops,
                stop_day_offsets=r.stop_day_offsets + (new_stop_offset,),
                duration=new_duration,
                cost=new_cost,
                region=r.region,
                demand_tons=r.demand_tons,
                is_depot_delivery=True
            )
            
    # 2. Direkt Depo Rotaları
    for sid, ship in ships.items():
        for d in depots.values():
            route_id = f"DIRECT_{sid}_{d.depot}"
            cost = 45000
            duration = 5
            
            new_routes[route_id] = Route(
                route_id=route_id,
                ship_id=sid,
                start_port="P1",
                end_port="P1",
                stops=(d.depot,),
                stop_day_offsets=(duration,),
                duration=duration,
                cost=cost,
                region=d.region,
                demand_tons=0,
                is_depot_delivery=True
            )

    return new_routes


def generate_synthetic_merge_routes(
    ships: Dict[str, Ship],
    routes: Dict[str, Route],
    locs: Dict[str, Location],
    max_duration: int = 7,
    min_gap_days: int = 1,
    max_gap_days: int = 3,
) -> Dict[str, Route]:
    """
    Kullanıcı kuralına göre sentetik iki-müşteri rota üretir:
    - Aynı gemi
    - Müşteri A -> Müşteri B arası teslim farkı 1..3 gün
    - Toplam rota süresi <= 7 gün
    """
    new_routes: Dict[str, Route] = {}

    # Her gemi + müşteri için en ucuz tek-müşteri rota referansı
    single_by_ship_loc: Dict[Tuple[str, str], Route] = {}
    for r in routes.values():
        if r.is_depot_delivery:
            continue
        cust_stops = [s for s in r.stops if s.startswith("M")]
        if len(cust_stops) != 1:
            continue
        loc = cust_stops[0]
        key = (r.ship_id, loc)
        cur = single_by_ship_loc.get(key)
        if cur is None or (r.cost, r.duration) < (cur.cost, cur.duration):
            single_by_ship_loc[key] = r

    customer_ids = sorted(locs.keys(), key=lambda x: safe_int(re.sub(r"[^0-9]", "", x), 999))
    for sid in ships.keys():
        for a in customer_ids:
            ra = single_by_ship_loc.get((sid, a))
            if ra is None:
                continue
            for b in customer_ids:
                if a == b:
                    continue
                rb = single_by_ship_loc.get((sid, b))
                if rb is None:
                    continue

                # Kural: talep günleri arası çok uzaksa sentetik rota üretme.
                if abs(locs[a].demand_day - locs[b].demand_day) > 7:
                    continue

                for gap in range(min_gap_days, max_gap_days + 1):
                    second_offset = 1 + gap
                    # 7 gunu asmayan sentetik duration (konservatif)
                    duration = max(second_offset, min(max_duration, max(ra.duration, rb.duration) + gap))
                    if duration > max_duration:
                        continue

                    rid = f"SYN_{sid}_{a}_{b}_G{gap}"
                    if rid in routes or rid in new_routes:
                        continue

                    stop_regions = {locs[a].region, locs[b].region}
                    region = stop_regions.pop() if len(stop_regions) == 1 else "MIXED"
                    est_cost = int(max(1, 0.9 * (ra.cost + rb.cost)))

                    new_routes[rid] = Route(
                        route_id=rid,
                        ship_id=sid,
                        start_port=ra.start_port,
                        end_port=rb.end_port,
                        stops=(a, b),
                        stop_day_offsets=(1, second_offset),
                        duration=duration,
                        cost=est_cost,
                        region=region,
                        demand_tons=locs[a].qty + locs[b].qty,
                        is_depot_delivery=False,
                    )

    return new_routes


# 4) CONSTRAINT CONFIG

@dataclass
class ConstraintConfig:
    enforce_duration_le_7: bool = True
    enforce_max_2_stops: bool = True
    enforce_ship_route_region_match: bool = False
    enforce_capacity: bool = True
    enforce_total_duration_le_35: bool = True
    enforce_unique_route_globally: bool = True
    enforce_cover_all_locations: bool = True
    enforce_time_window_pm2: bool = True
    include_inventory_penalty: bool = False 
    enforce_depot_min_level: bool = True 


# 5) TIME WINDOWS + SCHEDULING

def feasible_service_days_for_route(route: Route, locs: Dict[str, Location]) -> List[int]:
    lo, hi = 1, 35
    has_customer = False
    for s in route.stops:
        if s in locs:
            d = locs[s].demand_day
            lo = max(lo, d - 2)
            hi = min(hi, d + 2)
            has_customer = True
            
    if not has_customer:
        return list(range(1, 36 - route.duration))

    if lo > hi:
        return []
    return list(range(lo, hi + 1))

def schedule_trips_backtracking( #sıralama
    route_ids: Tuple[str, ...],
    ship: Ship,
    routes: Dict[str, Route],
    locs: Dict[str, Location],
    prefer_latest: bool = True,
) -> Optional[Dict[str, int]]:
    items = []
    for rid in route_ids:
        r = routes[rid]
        days = feasible_service_days_for_route(r, locs)
        if not days:
            return None
        lo, hi = days[0], days[-1]
        items.append((rid, lo, hi, r.duration))

    items.sort(key=lambda x: ((x[2] - x[1]), x[2]))
    chosen: Dict[str, int] = {}

    busy_days = BUSY_DAYS_BY_SHIP.get(ship.ship_id, set())

    if prefer_latest:
        rev_items = list(reversed(items))

        def dfs_latest(idx: int, latest_end: int) -> bool:
            if idx == len(rev_items):
                return True
            rid, lo, hi, dur = rev_items[idx]
            start_max = min(hi, latest_end - dur + 1)
            for start in range(start_max, lo - 1, -1):
                end_day = start + dur - 1
                if end_day > 35:
                    continue
                blocked = any(d in busy_days for d in range(start, end_day + 1))
                if blocked:
                    continue
                chosen[rid] = start
                if dfs_latest(idx + 1, start - 1):
                    return True
                del chosen[rid]
            return False

        return chosen if dfs_latest(0, 35) else None

    def dfs_earliest(idx: int, current_time: int) -> bool:
        if idx == len(items):
            return True
        rid, lo, hi, dur = items[idx]
        start_min = max(current_time, lo)
        for start in range(start_min, hi + 1):
            end_day = start + dur - 1
            if end_day > 35:
                continue
            blocked = any(d in busy_days for d in range(start, end_day + 1))
            if blocked:
                continue
            chosen[rid] = start
            if dfs_earliest(idx + 1, end_day + 1):
                return True
            del chosen[rid]
        return False

    return chosen if dfs_earliest(0, 1) else None


# 7) EVALUATION

BIG_M = 10_000_000

def evaluate(
    sol: Solution,
    ships: Dict[str, Ship],
    routes: Dict[str, Route],
    locs: Dict[str, Location],
    depots: Dict[str, Depot],
    cfg: ConstraintConfig,
    rental_income_if_idle_ge_5: int = 0, # #5 günden fazla boşta kalırsa günlük kira geliri
    idle_penalty_if_idle_lt_5: int = 100,#5 günden az boşta kalırsa günlük ceza
) -> Tuple[int, Dict]:

    violations: List[str] = []
    total_cost = 0
    served: Set[str] = set()
    used_routes: Set[str] = set()
    ship_schedules: Dict[str, Dict[str, int]] = {}

    # 1. Gemi ve Rota Kontrolleri
    for ship_id, trip_list in sol.items():
        ship = ships[ship_id]
        total_duration = 0

        for rid in trip_list: #rota var mı 
            if rid not in routes:
                violations.append(f"{ship_id}: route {rid} not found")
                continue

            r = routes[rid] #rota aynı gemiye mi ait
            if r.ship_id != ship_id:
                violations.append(f"{ship_id}: picked route {rid} belongs to {r.ship_id}")

            if cfg.enforce_unique_route_globally:
                if rid in used_routes:
                    violations.append(f"route used multiple times: {rid}") #aynı rota 2 gemiye verilmez
                used_routes.add(rid)

            if cfg.enforce_duration_le_7 and r.duration > 7:
                violations.append(f"{ship_id}/{rid}: duration {r.duration} > 7")

            if cfg.enforce_max_2_stops and len(r.stops) > 2:
                violations.append(f"{ship_id}/{rid}: stops {len(r.stops)} > 2")

            if r.region == "MIXED":
                violations.append(f"{ship_id}/{rid}: mixed route region")

            if cfg.enforce_capacity and r.demand_tons > ship.capacity:
                violations.append(f"{ship_id}/{rid}: demand {r.demand_tons} > cap {ship.capacity}")

            total_duration += r.duration
            total_cost += r.cost + ship.fixed_daily_cost * r.duration

            # Hizmet edilen noktalar
            for s in r.stops:
                if s in locs:
                    served.add(s)

        if cfg.enforce_total_duration_le_35 and total_duration > 35:
            violations.append(f"{ship_id}: total_duration {total_duration} > 35")

        idle = 35 - total_duration #boşta geçen gün sayısı 
        if idle <= 5:
            total_cost += idle_penalty_if_idle_lt_5 * idle
        else:
            total_cost -= rental_income_if_idle_ge_5 * idle

        if cfg.enforce_time_window_pm2: 
            sched = schedule_trips_backtracking(trip_list, ship, routes, locs, prefer_latest=True)
            if sched is None:
                violations.append(f"{ship_id}: cannot schedule trips within ±2 windows")
            else:
                ship_schedules[ship_id] = sched

    # 2. Müşteri Kapsama Kontrolü
    if cfg.enforce_cover_all_locations: #tüm 24 lokasyonun kapsanması
        missing = sorted(set(locs.keys()) - served) 
        for m in missing:
            violations.append(f"missing location: {m}")

    # 3. Depo Envanter Simülasyonu
    inv_debug = {}
    if cfg.enforce_depot_min_level:
        current_inv = {d: (depot.initial_level if getattr(depot, 'initial_level', 0) > 0 else depot.max_cap) for d, depot in depots.items()}
        deliveries = {d: {day: 0 for day in range(1, 37)} for d in depots}
        
        for sid, sched in ship_schedules.items():
            for rid, start_day in sched.items():
                r = routes[rid]
                if r.is_depot_delivery:
                    amount = max(0, ships[sid].capacity - r.demand_tons)
                    target_depot = None
                    for s in r.stops:
                        if s in depots:
                            target_depot = s
                            break
                    if target_depot:
                        arrival_day = start_day + r.duration - 1 
                        if arrival_day <= 35:
                            deliveries[target_depot][arrival_day] += amount

        for d_id, depot in depots.items():
            inv_history = []
            
            for day in range(1, 36):
                current_inv[d_id] -= depot.daily_decay
                daily_supply = deliveries[d_id][day]
                if daily_supply > 0:
                    current_inv[d_id] += daily_supply
                    if current_inv[d_id] > depot.max_cap:
                        current_inv[d_id] = depot.max_cap
                
                if current_inv[d_id] < depot.min_cap:
                    violations.append(f"Depot {d_id} underflow on day {day} (Level: {current_inv[d_id]})")
                
                inv_history.append(current_inv[d_id])
            
            inv_debug[d_id] = inv_history

    objective = total_cost + BIG_M * len(violations)
    
    debug = {
        "total_cost": total_cost,
        "violations_count": len(violations),
        "violations": violations[:10],
        "served_count": len(served),
        "ship_schedules": ship_schedules,
        "depot_levels": inv_debug
    }
    return objective, debug


# 8) TABU SEARCH
#rotaları gemi bazında grupla
def routes_by_ship(routes: Dict[str, Route]) -> Dict[str, List[str]]:
    d: Dict[str, List[str]] = {}
    for r in routes.values():
        d.setdefault(r.ship_id, []).append(r.route_id)
    return d
#lokasyon
def route_covers(routes: Dict[str, Route]) -> Dict[str, Set[str]]:
    return {rid: set(routes[rid].stops) for rid in routes.keys()}

def depot_risk_by_region(day: int, depots: Dict[str, Depot]) -> Dict[str, float]:
    risk: Dict[str, float] = {}
    for d in depots.values():
        if d.daily_decay <= 0:
            continue
        projected = d.max_cap - d.daily_decay * day
        days_to_min = (projected - d.min_cap) / d.daily_decay
        # <=5 gun kaldiysa risk hizla artsin
        local_risk = max(0.0, 6.0 - days_to_min)
        if local_risk <= 0:
            continue
        risk[d.region] = max(risk.get(d.region, 0.0), local_risk)
    return risk

def customer_urgency_score(loc: Location, day: int, region_risk: Dict[str, float]) -> float:
    slack = (loc.demand_day + 2) - day
    overdue = max(0, day - (loc.demand_day + 2))
    near_window = max(0, 6 - max(0, slack))
    qty_bonus = min(20.0, loc.qty / 1000.0)
    depot_bonus = 8.0 * region_risk.get(loc.region, 0.0)
    return 100.0 * overdue + 12.0 * near_window + qty_bonus + depot_bonus

def build_event_driven_order(
    locs: Dict[str, Location],
    depots: Dict[str, Depot],
    uncovered: Set[str],
) -> List[str]:
    # 1) Event-driven simulation: gun gun ilerleyip acil musterileri one al.
    ordered: List[str] = []
    seen: Set[str] = set()
    for day in range(1, 36):
        if len(seen) == len(uncovered):
            break
        region_risk = depot_risk_by_region(day, depots)
        day_candidates = sorted(
            (lid for lid in uncovered if lid not in seen),
            key=lambda lid: (
                -customer_urgency_score(locs[lid], day, region_risk),
                abs(locs[lid].demand_day - day),
            ),
        )
        for lid in day_candidates:
            if lid not in seen:
                seen.add(lid)
                ordered.append(lid)
    for lid in sorted(uncovered, key=lambda x: int(re.sub(r"[^0-9]", "", x) or "0")):
        if lid not in seen:
            ordered.append(lid)
    return ordered

#başlangıç çözümü oluşturma
def build_initial_solution_greedy_randomized(
    ships: Dict[str, Ship],
    routes: Dict[str, Route],
    depots: Dict[str, Depot],
    locs: Dict[str, Location],
    cfg: ConstraintConfig,
    rcl_size: int = 5 #en ucuz 5 rotadan rastgele seçim yap
) -> Solution:
    # Event-driven simulation + time-window feasibility + backward scheduling
    customer_routes = {rid: r for rid, r in routes.items() if not r.is_depot_delivery}
    covers = route_covers(customer_routes)
    routes_by_loc: Dict[str, List[str]] = {
        loc: sorted(
            [rid for rid, cv in covers.items() if loc in cv],
            key=lambda rid: customer_routes[rid].cost
        )
        for loc in locs.keys()
    }
    
    sol: Solution = {sid: tuple() for sid in ships.keys()}
    used_routes: Set[str] = set()
    uncovered: Set[str] = set(locs.keys())
#tek tek kısıtlar kontrol edilerek rota ekleme fonk
    def can_add_trip(ship_id: str, rid: str) -> bool:
        r = routes[rid]
        ship = ships[ship_id]
        if r.ship_id != ship_id: return False
        if r.region == "MIXED": return False
        if cfg.enforce_capacity and r.demand_tons > ship.capacity: return False
        if cfg.enforce_duration_le_7 and r.duration > 7: return False
        if cfg.enforce_unique_route_globally and rid in used_routes: return False
        
        current = sol[ship_id]
        new_list = current + (rid,)
        total_dur = sum(routes[x].duration for x in new_list)
        if cfg.enforce_total_duration_le_35 and total_dur > 35: return False

        # 2) Time window scheduling: her rota pencere icine dusebilmeli.
        if cfg.enforce_time_window_pm2:
            for x in new_list:
                if not feasible_service_days_for_route(routes[x], locs):
                    return False

        # 3) Backward scheduling: rotalari en gec yerlestirip paketlenebilirlik kontrolu.
        if cfg.enforce_time_window_pm2:
            if schedule_trips_backtracking(new_list, ship, routes, locs, prefer_latest=True) is None:
                return False
        return True

    # 1) Event-driven simulation sonucunda aciliyet sirasi
    priority_order = build_event_driven_order(locs, depots, uncovered)

    # 4) Greedy: uygun rotalari maliyete gore sirala, en ucuz k icinden rastgele sec.
    for target_loc in priority_order:
        if target_loc not in uncovered:
            continue
        candidates: List[Tuple[str, str]] = []
        for rid in routes_by_loc.get(target_loc, []):
            if rid in used_routes:
                continue
            sid = routes[rid].ship_id
            if can_add_trip(sid, rid):
                candidates.append((rid, sid))
        if not candidates:
            continue
        top_candidates = candidates[:rcl_size]
        chosen_rid, chosen_sid = random.choice(top_candidates)
        sol[chosen_sid] = sol[chosen_sid] + (chosen_rid,)
        used_routes.add(chosen_rid)
        uncovered -= covers[chosen_rid]

    # 2) Tamamlama passi: kalan musteriler icin en dar opsiyonlardan doldur.
    max_rounds = max(200, len(locs) * 30)
    for _ in range(max_rounds):
        if not uncovered:
            break
        progress = False
        hardest_first = sorted(uncovered, key=lambda l: len(routes_by_loc.get(l, [])))
        for target_loc in hardest_first:
            candidates: List[Tuple[str, str]] = []
            for rid in routes_by_loc.get(target_loc, []):
                if rid in used_routes:
                    continue
                sid = routes[rid].ship_id
                if can_add_trip(sid, rid):
                    candidates.append((rid, sid))
            if not candidates:
                continue
            top_candidates = candidates[:rcl_size]
            chosen_rid, chosen_sid = random.choice(top_candidates)
            sol[chosen_sid] = sol[chosen_sid] + (chosen_rid,)
            used_routes.add(chosen_rid)
            uncovered -= covers[chosen_rid]
            progress = True
            if not uncovered:
                break
        if not progress:
            break

    return sol

def tabu_search_multi_trip(
    ships: Dict[str, Ship],
    routes: Dict[str, Route],
    locs: Dict[str, Location],
    depots: Dict[str, Depot],
    cfg: ConstraintConfig,
    max_iters: int = 1500,
    neighborhood_size: int = 240,
    tabu_tenure: int = 24,
    seed: int = 11,
    max_trips_per_ship: int = 9,
    initial_solution: Optional[Solution] = None,
):
    random.seed(seed)
    by_ship = routes_by_ship(routes)
    ship_ids = list(ships.keys())

    if initial_solution is None:
        current = build_initial_solution_greedy_randomized(ships, routes, depots, locs, cfg, rcl_size=5)
    else:
        current = dict(initial_solution)
    best = dict(current)

    best_obj, best_dbg = evaluate(best, ships, routes, locs, depots, cfg)
    cur_obj = best_obj
    
    # Initial Solution Log 
    print(f"Initial Random Solution Cost: {best_obj:,}")

    tabu: Dict[Tuple[str, str, str], int] = {}

    def decay_tabu(): ##Her iterasyonda tabu sürelerini 1 azaltıyor. Sıfıra düşünce tabu listesinden kaldırılıyor.
        for k in list(tabu.keys()):
            tabu[k] -= 1
            if tabu[k] <= 0: del tabu[k]
    
    def add_tabu(sid, action, rid): tabu[(sid, action, rid)] = tabu_tenure #yeni bir hamleyi tabu listsine ekleme
    def is_tabu(sid, action, rid): return (sid, action, rid) in tabu

    for it in range(1, max_iters + 1):
        decay_tabu()
        best_cand, best_cand_obj, best_cand_dbg, best_move = None, None, None, None

        for _ in range(neighborhood_size): #240 komşu çözüm dene
            action = random.choice(["add", "remove", "swap_order", "shift_ship"])
            sid = random.choice(ship_ids)
            trips = list(current[sid])
            cand = dict(current)
            move_info = None

            if action == "add": #rota ekleme
                if len(trips) >= max_trips_per_ship: continue
                available = by_ship.get(sid, [])
                if not available: continue
                rid = random.choice(available)
                if rid in trips: continue
                cand[sid] = tuple(trips + [rid])
                move_info = (sid, "add", rid)

            elif action == "remove": #rota çıkarma
                if not trips: continue
                rid = random.choice(trips)
                new_trips = trips[:]
                new_trips.remove(rid)
                cand[sid] = tuple(new_trips)
                move_info = (sid, "remove", rid)

            elif action == "swap_order": #rota sırasını değiştirme
                if len(trips) < 2: continue
                i, j = random.sample(range(len(trips)), 2)
                trips[i], trips[j] = trips[j], trips[i]
                cand[sid] = tuple(trips)
                move_info = (sid, "swap_order", "ORDER")
            
            elif action == "shift_ship": #rota başka bir gemiye kaydırma
                if not trips: continue
                rid_from = random.choice(trips)
                sid_to = random.choice(ship_ids)
                if sid_to == sid: continue
                
                r_from = routes[rid_from]
                candidates_to = []
                for pot_rid in by_ship.get(sid_to, []):
                    r_pot = routes[pot_rid]
                    if set(r_pot.stops) == set(r_from.stops) and r_pot.is_depot_delivery == r_from.is_depot_delivery:
                        candidates_to.append(pot_rid)
                
                if not candidates_to: continue #aynı rotayı başka gemide bulamadı iptal
                rid_to = candidates_to[0]
                
                in_use = False
                for s_chk, t_chk in cand.items():
                    if rid_to in t_chk: in_use = True; break
                if in_use: continue

                new_trips_from = list(cand[sid])
                new_trips_from.remove(rid_from) #ilk gemiden rota çıkar 
                cand[sid] = tuple(new_trips_from)
                
                new_trips_to = list(cand[sid_to])
                new_trips_to.append(rid_to)
                cand[sid_to] = tuple(new_trips_to)
                move_info = (sid, "shift_to_" + sid_to, rid_from)

            if move_info is None: continue 
            
            cand_obj, cand_dbg = evaluate(cand, ships, routes, locs, depots, cfg)
            if cfg.enforce_cover_all_locations and cand_dbg["served_count"] < len(locs):
                continue
            
            if is_tabu(*move_info): #tabu- konrolü , maliyet mevcut en iyi çözümden daha iyi değilse atla
                if cand_obj >= best_obj: continue 
            
            if best_cand is None or cand_obj < best_cand_obj: #yeni en iyi komşu çözüm
                best_cand, best_cand_obj, best_cand_dbg, best_move = cand, cand_obj, cand_dbg, move_info

        if best_cand is None: continue

        current = best_cand
        cur_obj = best_cand_obj
        if best_move: add_tabu(*best_move)

        if cur_obj < best_obj:
            best = dict(current)
            best_obj, best_dbg = cur_obj, best_cand_dbg

        # LOGGING: Periyot 120
        if it % 120 == 0:
            print(
                f"iter={it:4d} best_obj={best_obj:,} cur_obj={cur_obj:,} "
                f"viol={best_dbg['violations_count']} served={best_dbg['served_count']}/{len(locs)}"
            )

    return best, best_obj, best_dbg


# 9) PRINTING

def pretty_print_solution(sol: Solution, routes: Dict[str, Route], dbg: Dict, total_locations: int):
    print("\nBEST OBJ:", f"{dbg['total_cost']:,}")
    print("violations:", dbg["violations_count"])
    print("served:", f"{dbg['served_count']}/{total_locations}")
    
    print("\n=== SOLUTION (Ship -> trips) ===")
    sched = dbg.get("ship_schedules", {})
    for sid in sorted(sol.keys(), key=lambda x: int(x[1:])):
        trips = sol[sid]
        if not trips:
            print(f"{humanize_code(sid)}: []")
            continue
        parts = []
        for rid in trips:
            r = routes[rid]
            day = sched.get(sid, {}).get(rid, None)
            extra = " [DEPOT]" if r.is_depot_delivery else ""
            stops_txt = tuple(humanize_code(s) for s in r.stops)
            parts.append(f"{rid}(stops={stops_txt},cost={r.cost},day={day}{extra})")
        print(f"{humanize_code(sid)}: " + "  ".join(parts))


# 10) RUN

def main():

    global BUSY_DAYS_BY_SHIP

    if not os.path.exists(EXCEL_INPUT_PATH):
        raise FileNotFoundError(
            f"Veri klasoru bulunamadi: {EXCEL_INPUT_PATH}. "
            "Bu uygulama sadece data/from_data_ship verileri ile calisacak sekilde ayarlandi."
        )

    locs, ships, depots, base_routes, BUSY_DAYS_BY_SHIP = load_from_excel(EXCEL_INPUT_PATH)
    if not COMPACT_CONSOLE_OUTPUT:
        print("DEPOT IDS:", sorted(depots.keys()))
        print(f"[DATA] Veri klasoru yüklendi: {EXCEL_INPUT_PATH}")
        print(f"[DATA] Musteri={len(locs)} Gemi={len(ships)} Rota={len(base_routes)}")
    if SHOW_EXCEL_DEBUG_OUTPUT:
        print_excel_data_preview(
            EXCEL_INPUT_PATH,
            locs,
            ships,
            depots,
            base_routes,
            BUSY_DAYS_BY_SHIP,
            max_rows=EXCEL_PREVIEW_ROWS,
        )
        print_all_customer_demands_from_excel(EXCEL_INPUT_PATH)
        print_all_ship_capacities(ships)
        print_all_ship_busy_days(BUSY_DAYS_BY_SHIP)
        print_stock_tables_from_excel(EXCEL_INPUT_PATH)
        print_tuketim_table_from_excel(EXCEL_INPUT_PATH)
        print_rafineri_talebi_from_excel(EXCEL_INPUT_PATH)
        return
    # =========================================================
    # ROLLING HORIZON PREVIEW (STEP 1-7)
    # =========================================================
    raw_sheets = _xlsx_rows_as_dicts(EXCEL_INPUT_PATH)
    raw_demand = raw_sheets.get("Musteri_Talepleri", [])
    jobs_by_id = build_customer_jobs(locs, raw_demand, tolerance=2)

    job_count_by_customer: Dict[str, int] = {}
    job_ids_by_customer: Dict[str, List[str]] = {}
    for job in jobs_by_id.values():
        if job.job_type != "customer":
            continue
        job_count_by_customer[job.location] = job_count_by_customer.get(job.location, 0) + 1
        job_ids_by_customer.setdefault(job.location, []).append(job.job_id)

    # Sadece Gun_maliyet tablosundaki rotalar kullanilsin.
    planning_routes = dict(base_routes)

    plan_logs, unassigned_queue, ship_state_map = run_rolling_horizon_day_loop_step15(
        jobs_by_id=jobs_by_id,
        depots=depots,
        ships=ships,
        routes=planning_routes,
        busy_days=BUSY_DAYS_BY_SHIP,
        ship_selection_mode="best_fit",
        horizon_len=10,
        total_days=35,
    )

    total_assigned_jobs = sum(rec.get("step9_assigned", 0) for rec in plan_logs)
    print("\n=== ROLLING HORIZON SUMMARY (Event-driven + Time Window + Backward) ===")
    print(f"Toplam customer job: {len(jobs_by_id)}")
    print(f"Toplam atanan customer job: {total_assigned_jobs}")
    print(f"Kalan queue: {len(unassigned_queue)}")

    # =========================================================
    # STEP 12: Hoca ciktilari (feasible plan ozeti)
    # 1) Gun bazli ozet zaten STEP 11 loglarinda basiliyor.
    # 2) Gemi takvimleri
    # 3) Servis edilen isler listesi
    # 4) Depo stok tablo ozeti
    # =========================================================
    if not COMPACT_CONSOLE_OUTPUT:
        print("\n=== STEP 12 OUTPUTS ===")
        print("\n[2] Gemi Takvimleri (tum gemiler)")
        for sid in sorted(ship_state_map.keys(), key=lambda x: safe_int(re.sub(r'[^0-9]', '', x), 999)):
            print(f"  {sid}: busy={ship_state_map[sid].busy_intervals}")

    print("\n[3] Servis Edilen Isler")
    served_records: List[Dict[str, Any]] = []
    for rec in plan_logs:
        day = rec.get("day")
        for a in rec.get("assignments", []):
            customer_jobs = a.get("jobs", [])
            depot_deliveries = a.get("depot_deliveries", [])
            if not customer_jobs and not depot_deliveries:
                continue
            served_records.append({
                "jobs": customer_jobs,
                "depot_deliveries": depot_deliveries,
                "planned_day": a.get("start_day"),
                "end_day": a.get("end_day"),
                "ship_id": a.get("ship_id"),
                "route_id": a.get("route_id"),
                "logged_on_day": day,
            })
    if served_records:
        served_records.sort(key=lambda r: (safe_int(r["planned_day"], 999), str(r["route_id"])))
        for r in served_records:
            customer_parts: List[str] = []
            for jid in r.get("jobs", []):
                job = jobs_by_id.get(jid)
                if job:
                    customer_parts.append(f"{jid}({humanize_code(job.location)})")
                else:
                    customer_parts.append(jid)
            payload = " + ".join(customer_parts) if customer_parts else "DEPOT_ONLY"
            if r.get("depot_deliveries"):
                payload += " + DEPOT"
            print(
                f"  {payload} -> start={r['planned_day']} end={r['end_day']} "
                f"ship={r['ship_id']} route={r['route_id']}"
            )
    else:
        print("  (servis edilen is yok)")

    if COMPACT_CONSOLE_OUTPUT:
        print("\n[2] Gemi Kullanimi")
        usage_count: Dict[str, int] = {sid: 0 for sid in ships.keys()}
        for r in served_records:
            sid = str(r.get("ship_id", "")).strip()
            if sid in usage_count:
                usage_count[sid] += 1
        used = [sid for sid, c in sorted(usage_count.items(), key=lambda kv: (safe_int(re.sub(r'[^0-9]', '', kv[0]), 999), kv[0])) if c > 0]
        idle = [sid for sid, c in sorted(usage_count.items(), key=lambda kv: (safe_int(re.sub(r'[^0-9]', '', kv[0]), 999), kv[0])) if c == 0]
        print(f"  Kullanilan ({len(used)}): " + (", ".join(f"{sid}({usage_count[sid]})" for sid in used) if used else "-"))
        print(f"  Bos ({len(idle)}): " + (", ".join(idle) if idle else "-"))
        print("  Is Bazli:")
        for sid in sorted(ship_state_map.keys(), key=lambda x: safe_int(re.sub(r'[^0-9]', '', x), 999)):
            ship_records = [r for r in served_records if str(r.get("ship_id", "")) == sid]
            ship_records.sort(key=lambda r: (safe_int(r.get("planned_day", 999), 999), safe_int(r.get("end_day", 999), 999)))
            if not ship_records:
                print(f"    {sid}: -")
                continue
            print(f"    {sid}:")
            for r in ship_records:
                customer_parts: List[str] = []
                for jid in r.get("jobs", []):
                    job = jobs_by_id.get(jid)
                    if job:
                        customer_parts.append(f"{jid}({humanize_code(job.location)})")
                    else:
                        customer_parts.append(jid)
                payload = " + ".join(customer_parts) if customer_parts else "DEPOT_ONLY"
                if r.get("depot_deliveries"):
                    payload += " + DEPOT"
                print(
                    f"      ({r['planned_day']},{r['end_day']}) {payload} route={r['route_id']}"
                )

    if not COMPACT_CONSOLE_OUTPUT:
        print("\n[4] Depo Stok Tablosu (gun sonu seviyeleri)")
    depot_ids_sorted = sorted(depots.keys(), key=lambda x: safe_int(re.sub(r'[^0-9]', '', x), 999))
    max_stock_products = read_depot_table_from_excel(EXCEL_INPUT_PATH, "Max Stock")
    initial_stock_products = read_depot_table_from_excel(EXCEL_INPUT_PATH, "Initial Stock")
    safety_stock_products = read_depot_table_from_excel(EXCEL_INPUT_PATH, "Safety Stock")
    if (not COMPACT_CONSOLE_OUTPUT) and depot_ids_sorted:
        for d_id in depot_ids_sorted:
            level_values: List[int] = []
            breaches: List[int] = []
            for rec in plan_logs:
                risk = rec.get("depot_risk", {}).get(d_id, {})
                level = risk.get("level")
                if level is None:
                    continue
                level_int = int(round(level))
                level_values.append(level_int)
                minv = risk.get("min")
                if minv is not None and level < minv:
                    breaches.append(rec.get("day"))

            if level_values:
                end_level = level_values[-1]
                min_seen = min(level_values)
                product_max_map = max_stock_products.get(d_id, {})
                product_init_map = initial_stock_products.get(d_id, {})
                product_min_map = safety_stock_products.get(d_id, {})
                max_text = ", ".join(f"{k}:{v}" for k, v in product_max_map.items()) if product_max_map else "?"
                init_text = ", ".join(f"{k}:{v}" for k, v in product_init_map.items()) if product_init_map else "?"
                min_text = ", ".join(f"{k}:{v}" for k, v in product_min_map.items()) if product_min_map else "?"
                print(
                    f"  {d_id} | max={max_text} | initial={init_text} | min_limit={min_text} | "
                    f"sim_total_bitis={end_level} | sim_total_en_dusuk={min_seen} | "
                    f"min_alti_gunler={breaches if breaches else 'yok'}"
                )
    elif depot_ids_sorted:
        print("\n[4] Depo Ozeti")
        for d_id in depot_ids_sorted:
            level_values: List[int] = []
            for rec in plan_logs:
                risk = rec.get("depot_risk", {}).get(d_id, {})
                level = risk.get("level")
                if level is not None:
                    level_values.append(int(round(level)))
            if not level_values:
                continue
            end_level = level_values[-1]
            min_seen = min(level_values)
            print(f"  {d_id}: son={end_level} en_dusuk={min_seen}")

    if not COMPACT_CONSOLE_OUTPUT:
        print("\n[5] Depo Kritik Olaylari (gun -> seviye/esik -> atama detayi)")
    critical_days_found = False
    for rec in plan_logs:
        day = rec.get("day")
        if not isinstance(day, int) or day < 1 or day > 35:
            continue
        critical_jobs = rec.get("critical_depot_jobs", [])
        if not critical_jobs:
            continue
        critical_days_found = True
        critical_depots: List[str] = []
        for jid in critical_jobs:
            m = re.match(r"DEPOT_([A-Z0-9_]+)_D\d+$", str(jid))
            if m:
                critical_depots.append(m.group(1))
        critical_depots = sorted(set(critical_depots), key=lambda x: safe_int(re.sub(r'[^0-9]', '', x), 999))
        window = rec.get("window", (day, day))
        if isinstance(window, (tuple, list)) and len(window) == 2:
            w_start, w_end = window[0], window[1]
        else:
            w_start, w_end = day, day

        risk_map = rec.get("depot_risk", {}) or {}
        critical_ctx = rec.get("critical_depot_context", {}) or {}
        if not COMPACT_CONSOLE_OUTPUT:
            print(f"  D{day:02d} window=[{w_start},{w_end}] kritik_depo={', '.join(critical_depots) if critical_depots else '-'}")

        # Kritik depolari tek tek yaz: seviye, esik, tuketim, acik/fazla
        for d_id in critical_depots:
            trigger = critical_ctx.get(d_id, {}) or {}
            risk = risk_map.get(d_id, {}) or {}
            level = float(trigger.get("trigger_level", risk.get("level", 0.0)))
            threshold = float(trigger.get("threshold", risk.get("threshold", 0.0)))
            day_cons = int(round(trigger.get("day_consumption", get_depot_daily_consumption(depots[d_id], day) if d_id in depots else 0)))
            critical_gap_total = float(trigger.get("critical_gap_total", 0.0))
            relation_txt = f"kritik_urun_acik={int(round(max(0.0, critical_gap_total)))}"
            if not COMPACT_CONSOLE_OUTPUT:
                print(
                    f"    {d_id}: seviye={int(round(level))} | esik={int(round(threshold))} "
                    f"(={DEPOT_MIN_SAILING_DAYS}*{day_cons}+safety) | tuketim={day_cons} | {relation_txt}"
                )
            crit_products = str(trigger.get("critical_products", "")).strip()
            threshold_by_product = str(trigger.get("threshold_by_product", "")).strip()
            level_by_product = str(trigger.get("level_by_product", "")).strip()
            refill_by_product = str(trigger.get("refill_by_product", "")).strip()
            if (not COMPACT_CONSOLE_OUTPUT) and crit_products:
                print(f"      urun_kritik={crit_products}")
            if (not COMPACT_CONSOLE_OUTPUT) and level_by_product:
                print(f"      urun_seviye={level_by_product}")
            if (not COMPACT_CONSOLE_OUTPUT) and threshold_by_product:
                print(f"      urun_esik={threshold_by_product}")
            if (not COMPACT_CONSOLE_OUTPUT) and refill_by_product:
                print(f"      urun_refill_hedef={refill_by_product}")

            matched_moves: List[str] = []
            for a in rec.get("assignments", []):
                dep_delivs = a.get("depot_deliveries", []) or []
                if not dep_delivs:
                    continue
                for dep in dep_delivs:
                    dep_str = str(dep)
                    if not dep_str.startswith(f"{d_id}:"):
                        continue
                    delivered_qty = dep_str.split(":", 1)[1] if ":" in dep_str else "?"
                    matched_moves.append(
                        f"      -> gemi={a.get('ship_id')} rota={a.get('route_id')} "
                        f"start={a.get('start_day')} end={a.get('end_day')} teslim={delivered_qty}"
                    )
            if matched_moves:
                for mv in matched_moves:
                    if not COMPACT_CONSOLE_OUTPUT:
                        print(mv)
            else:
                if not COMPACT_CONSOLE_OUTPUT:
                    print("      -> atama yok (bu gun kritik ama teslimat planlanmadi)")
    if not critical_days_found:
        if not COMPACT_CONSOLE_OUTPUT:
            print("  kritik gun yok")

    print(f"\nFinal unassigned_queue size: {len(unassigned_queue)}")
    return  # Aktif mod burada biter: Greedy/Tabu devre disi, sadece rolling horizon akisi calisir.

    # Depo rotalarını üret ve ekle
    new_depot_routes = generate_depot_routes(ships, base_routes, depots, locs)
    base_routes.update(new_depot_routes)
    
    cfg = ConstraintConfig(
        enforce_duration_le_7=True,
        enforce_max_2_stops=True,
        enforce_ship_route_region_match=False,
        enforce_capacity=True,
        enforce_total_duration_le_35=True,
        enforce_unique_route_globally=True,
        enforce_cover_all_locations=True,
        enforce_time_window_pm2=True,
        include_inventory_penalty=False,
        enforce_depot_min_level=True 
    )


    # Farklı şans faktörleri (Seed'ler)
    seeds_to_try = [10, 20, 30, 40, 50] 
    
    global_best_sol = None
    global_best_obj = float('inf')
    global_best_dbg = {}

    print(f"\n🚀 Starting Multi-Run Optimization ({len(seeds_to_try)} runs)...")
    print("=" * 70)

    for i, seed in enumerate(seeds_to_try):
        print(f"\n▶ RUN {i+1}/{len(seeds_to_try)} (Seed: {seed})")
        print("-" * 30)
        
        # Her tur için rotaların temiz bir kopyasını al
        routes_copy = deepcopy(base_routes)

        # Tabu'suz baslangic cozumunu olustur ve raporla
        random.seed(seed)
        initial_sol = build_initial_solution_greedy_randomized(
            ships, routes_copy, depots, locs, cfg, rcl_size=5
        )
        initial_obj, initial_dbg = evaluate(initial_sol, ships, routes_copy, locs, depots, cfg)
        print(
            f"Tabu'suz: obj={initial_obj:,} viol={initial_dbg['violations_count']} "
            f"served={initial_dbg['served_count']}/{len(locs)}"
        )
        
        # Algoritmayı çalıştır
        best_sol, best_obj, dbg = tabu_search_multi_trip(
            ships, routes_copy, locs, depots, cfg,
            max_iters=1500,  
            neighborhood_size=200, 
            tabu_tenure=20, 
            seed=seed,
            max_trips_per_ship=9,
            initial_solution=initial_sol,
        )
        gain = initial_obj - best_obj
        print(
            f"Tabu sonrasi: obj={best_obj:,} viol={dbg['violations_count']} "
            f"served={dbg['served_count']}/{len(locs)} iyilesme={gain:,}"
        )
        
        print(f"\n🏁 Run {i+1} Result: {best_obj:,}")

        # Eğer bu sonuç, şimdiye kadarki en iyisinden daha iyiyse kaydet
        if best_obj < global_best_obj:
            global_best_obj = best_obj
            global_best_sol = best_sol
            global_best_dbg = dbg
            print("🏆 NEW GLOBAL BEST FOUND!")

    print("\n" + "=" * 70)
    print(f"🌟 GRAND FINAL BEST OBJECTIVE: {global_best_obj:,}")
    print("=" * 70)
    
    # En iyi sonucu detaylı yazdır
    pretty_print_solution(global_best_sol, base_routes, global_best_dbg, total_locations=len(locs))
    print("\nAUTO POST-OPT: no improvement found.")

if __name__ == "__main__":
    main()
