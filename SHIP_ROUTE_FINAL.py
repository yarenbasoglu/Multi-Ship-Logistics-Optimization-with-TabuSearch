from __future__ import annotations

import os
import random
import re
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set
from copy import deepcopy

# 1) CONSISTENT DATA

EXCEL_INPUT_PATH = "data_ship.xlsx"
XML_NS = {
    "a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}

BUSY_DAYS_BY_SHIP: Dict[str, Set[int]] = {}

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
    
@dataclass(frozen=True)
class Route:
    route_id: str
    ship_id: str
    start_port: str
    end_port: str
    stops: Tuple[str, ...]
    duration: int
    cost: int
    region: str
    demand_tons: int
    is_depot_delivery: bool = False 

Solution = Dict[str, Tuple[str, ...]]


# 3) PARSING

def normalize_code(raw: str) -> str:
    return str(raw or "").strip().upper()

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
        demand_day = non_zero[0][0] if non_zero else 18
        qty = max((q for _, q in non_zero), default=0)
        region = "North" if idx < customers_by_half else "South"
        locs[c] = Location(loc=c, region=region, product="mixed", qty=max(1, qty), demand_day=demand_day)

    raw_ships = sheets.get("Kapasite", [])
    if not raw_ships:
        raise ValueError("Kapasite sayfasi bos veya bulunamadi.")
    ships: Dict[str, Ship] = {}
    for row in raw_ships:
        sid = normalize_code(row.get("Gemi", ""))
        cap = safe_int(row.get("Kapasite", "0"), 0)
        if not sid or cap <= 0:
            continue
        ships[sid] = Ship(ship_id=sid, ownership="owned", capacity=cap, fixed_daily_cost=0)
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

    depots: Dict[str, Depot] = {}
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
        stops: List[str] = []
        for s in stop_candidates:
            if s.startswith("M") and s in locs and s not in stops:
                stops.append(s)
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
        else:
            duration = 1
        cost = max(1, safe_int(row.get("Maliyet", "1"), 1))

        stop_regions = {locs[s].region for s in stops}
        region = stop_regions.pop() if len(stop_regions) == 1 else "MIXED"
        demand_tons = sum(locs[s].qty for s in stops)
        rid = f"X{route_no}"
        route_no += 1

        routes[rid] = Route(
            route_id=rid,
            ship_id=sid,
            start_port=start_port,
            end_port=end_port,
            stops=tuple(stops),
            duration=duration,
            cost=cost,
            region=region,
            demand_tons=demand_tons,
            is_depot_delivery=False,
        )
    if not routes:
        raise ValueError("Gun_Maliyet sayfasindan gecerli rota okunamadi.")

    busy_days: Dict[str, Set[int]] = {}
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
            new_duration = r.duration + 1
            
            new_routes[new_id] = Route(
                route_id=new_id,
                ship_id=r.ship_id,
                start_port=r.start_port,
                end_port=r.end_port,
                stops=new_stops,
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
                duration=duration,
                cost=cost,
                region=d.region,
                demand_tons=0,
                is_depot_delivery=True
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
        current_inv = {d: depot.max_cap for d, depot in depots.items()}
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
            f"Excel veri dosyasi bulunamadi: {EXCEL_INPUT_PATH}. "
            "Bu uygulama sadece Excel verisi ile calisacak sekilde ayarlandi."
        )

    locs, ships, depots, base_routes, BUSY_DAYS_BY_SHIP = load_from_excel(EXCEL_INPUT_PATH)
    print(f"[DATA] Excel yüklendi: {EXCEL_INPUT_PATH}")
    print(f"[DATA] Musteri={len(locs)} Gemi={len(ships)} Rota={len(base_routes)}")

    # Depo rotalarını üret ve ekle
    new_depot_routes = generate_depot_routes(ships, base_routes, depots, locs)
    base_routes.update(new_depot_routes)

    cfg = ConstraintConfig(
        enforce_duration_le_7=True,
        enforce_max_2_stops=True,
        enforce_ship_route_region_match=False,
        # Excel'de rota bazli tasinan ton bilgisi olmadigi icin kapasiteyi burada zorlamiyoruz.
        enforce_capacity=False,
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
