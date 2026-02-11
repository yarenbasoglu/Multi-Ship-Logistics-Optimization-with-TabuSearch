from __future__ import annotations

import csv
import io
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set
from copy import deepcopy

# 1) CONSISTENT DATA

LOCATIONS_CSV = """Location,Region,Product,Qty,DemandDay
L1,North,gasoline,1100,5
L2,North,gasoline,1200,7
L3,North,diesel,900,12
L4,North,diesel,1000,10
L5,North,gasoline,1300,18
L6,North,gasoline,1000,20
L7,North,diesel,900,26
L8,North,diesel,850,27
L9,South,gasoline,1050,6
L10,South,diesel,900,14
L11,South,gasoline,1150,16
L12,South,diesel,1000,17
L13,South,diesel,850,22
L14,South,gasoline,950,24
L15,North,diesel,850,8
L16,North,gasoline,1000,11
L17,North,diesel,900,15
L18,North,gasoline,950,19
L19,North,diesel,800,23
L20,North,gasoline,1050,24
L21,South,gasoline,1000,9
L22,South,diesel,950,13
L23,South,gasoline,1100,18
L24,South,diesel,900,28
"""

SHIPS_CSV = """Ship,Ownership,ShipCapacity,FixedDailyCost
S1,owned,2700,0
S2,owned,2550,0
S3,owned,2400,0
S4,owned,2450,0
S5,owned,2850,0
S6,owned,2650,0
S7,owned,2350,0
S8,leased,2300,0
S9,leased,2500,0
S10,leased,2600,0
"""

# Depolar: MaxCap, MinCap, DailyDecay

DEPOTS_CSV = """Depot,Region,MaxCap,MinCap,DailyDecay
D_North,North,6000,1000,150
D_South1,South,5000,800,120
D_South2,South,5500,900,130
"""

ROUTES_CSV = """Route,Ship,StartPort,EndPort,Stops,DurationDays,CostUSD
R1,S1,P1,P1,L1-L2,7,52000
R2,S1,P1,P2,L1-L2,7,55500
R3,S1,P2,P1,L1-L2,6,54500
R4,S1,P2,P2,L1-L2,7,56500
R5,S1,P1,P1,L5-L6,7,61000
R6,S1,P1,P2,L5-L6,7,64500
R7,S1,P2,P1,L5-L6,6,63500
R8,S1,P2,P2,L5-L6,7,65500
R9,S1,P1,P1,L3-L4,6,50000
R10,S1,P2,P2,L7-L8,6,47000
R11,S1,P1,P2,L9,5,43000
R12,S1,P2,P1,L14,6,45500

R13,S2,P1,P1,L1-L2,7,59000
R14,S2,P1,P2,L1-L2,7,62500
R15,S2,P2,P1,L1-L2,6,61500
R16,S2,P2,P2,L1-L2,7,63500
R17,S2,P1,P1,L5-L6,7,70000
R18,S2,P1,P2,L5-L6,7,73500
R19,S2,P2,P1,L5-L6,6,72500
R20,S2,P2,P2,L5-L6,7,74500
R21,S2,P1,P1,L3-L4,7,61000
R22,S2,P2,P2,L3-L4,6,60000
R23,S2,P1,P2,L10,6,51000
R24,S2,P2,P1,L12,6,52000

R25,S3,P1,P1,L3-L4,7,66000
R26,S3,P1,P2,L3-L4,7,69500
R27,S3,P2,P1,L3-L4,6,68500
R28,S3,P2,P2,L3-L4,7,70500
R29,S3,P1,P1,L7-L8,7,54000
R30,S3,P1,P2,L7-L8,7,57500
R31,S3,P2,P1,L7-L8,6,56500
R32,S3,P2,P2,L7-L8,7,58500
R33,S3,P1,P1,L9,5,47000
R34,S3,P2,P2,L10,6,49500
R35,S3,P1,P2,L11,6,50500
R36,S3,P2,P1,L14,6,52000

R37,S4,P1,P1,L7-L8,7,61000
R38,S4,P1,P2,L7-L8,7,64500
R39,S4,P2,P1,L7-L8,6,63500
R40,S4,P2,P2,L7-L8,7,65500
R41,S4,P1,P1,L10,6,52000
R42,S4,P1,P2,L10,6,55500
R43,S4,P2,P1,L10,5,54500
R44,S4,P2,P2,L10,6,56500
R45,S4,P1,P1,L11,6,54000
R46,S4,P2,P2,L12,6,55000
R47,S4,P1,P2,L13,5,50000
R48,S4,P2,P1,L9,5,51000

R49,S5,P1,P1,L1-L2,7,72000
R50,S5,P1,P2,L1-L2,7,75500
R51,S5,P2,P1,L1-L2,6,74500
R52,S5,P2,P2,L1-L2,7,76500
R53,S5,P1,P1,L5-L6,7,82000
R54,S5,P1,P2,L5-L6,7,85500
R55,S5,P2,P1,L5-L6,6,84500
R56,S5,P2,P2,L5-L6,7,86500
R57,S5,P1,P1,L3-L4,7,76000
R58,S5,P2,P2,L3-L4,6,75000
R59,S5,P1,P2,L7-L8,6,70000
R60,S5,P2,P1,L14,6,68000

R61,S6,P1,P1,L1-L2,7,56000
R62,S6,P1,P2,L1-L2,7,59500
R63,S6,P2,P1,L1-L2,6,58500
R64,S6,P2,P2,L1-L2,7,60500
R65,S6,P1,P1,L5-L6,7,64000
R66,S6,P1,P2,L5-L6,7,67500
R67,S6,P2,P1,L5-L6,6,66500
R68,S6,P2,P2,L5-L6,7,68500
R69,S6,P1,P1,L3-L4,7,60000
R70,S6,P2,P2,L7-L8,6,57000
R71,S6,P1,P2,L11,6,52000
R72,S6,P2,P1,L12,6,53000

R73,S7,P1,P1,L1-L2,7,50000
R74,S7,P1,P2,L1-L2,7,53500
R75,S7,P2,P1,L1-L2,6,52500
R76,S7,P2,P2,L1-L2,7,54500
R77,S7,P1,P1,L5-L6,7,59000
R78,S7,P1,P2,L5-L6,7,62500
R79,S7,P2,P1,L5-L6,6,61500
R80,S7,P2,P2,L5-L6,7,63500
R81,S7,P1,P1,L7-L8,7,46000
R82,S7,P1,P2,L7-L8,7,49500
R83,S7,P2,P1,L7-L8,6,48500
R84,S7,P2,P2,L7-L8,7,50500

R85,S8,P1,P1,L9,5,41000
R86,S8,P1,P2,L9,5,44500
R87,S8,P2,P1,L9,4,43500
R88,S8,P2,P2,L9,5,45500
R89,S8,P1,P1,L10,6,42000
R90,S8,P1,P2,L11,6,44000
R91,S8,P2,P2,L12,6,45000
R92,S8,P2,P1,L13,5,40000
R93,S8,P1,P1,L14,6,43000

R97,S9,P1,P1,L1-L2,7,58000
R98,S9,P1,P2,L3-L4,7,61000
R99,S9,P2,P1,L3-L4,6,60000
R100,S9,P2,P2,L7-L8,7,56000
R101,S9,P1,P1,L5,6,52000
R102,S9,P1,P2,L6,6,50500
R103,S9,P2,P1,L9,5,50000
R104,S9,P2,P2,L10,6,51500
R105,S9,P1,P2,L11,6,53000
R106,S9,P2,P1,L12,6,52500
R107,S9,P1,P1,L13,5,48000
R108,S9,P2,P2,L14,6,54000

R117,S10,P1,P1,L3-L4,7,66000
R118,S10,P2,P2,L7-L8,6,62000
R119,S8,P1,P1,L15-L16,6,39000
R120,S8,P2,P2,L17-L18,6,40000
R121,S8,P1,P2,L19-L20,6,40500
R122,S8,P1,P1,L21-L22,6,39500
R123,S8,P2,P1,L23-L11,6,41000
R124,S8,P1,P1,L24,5,38500
R125,S9,P1,P1,L15-L16,6,42000
R126,S9,P2,P2,L17-L18,6,43000
R127,S9,P1,P2,L19-L20,6,43500
R128,S9,P1,P1,L21-L22,6,42500
R129,S9,P2,P1,L23-L11,6,44000
R130,S9,P1,P1,L24,5,41500

R131,S10,P1,P1,L15-L16,6,44000
R132,S10,P2,P2,L17-L18,6,45000
R133,S10,P1,P2,L19-L20,6,45500
R134,S10,P1,P1,L21-L22,6,44500
R135,S10,P2,P1,L23-L11,6,46000
R136,S10,P1,P1,L24,5,43500
"""

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

def parse_csv(text: str) -> List[Dict[str, str]]:
    f = io.StringIO(text.strip())
    reader = csv.DictReader(f)
    rows: List[Dict[str, str]] = []
    for row in reader:
        if not row:
            continue
        first_key = next(iter(row.keys()))
        if not row.get(first_key, "").strip():
            continue
        rows.append({k: (v.strip() if isinstance(v, str) else v) for k, v in row.items()})
    return rows

def load_locations() -> Dict[str, Location]:
    locs: Dict[str, Location] = {}
    for r in parse_csv(LOCATIONS_CSV):
        loc = Location(
            loc=r["Location"],
            region=r["Region"],
            product=r["Product"],
            qty=int(r["Qty"]),
            demand_day=int(r["DemandDay"]),
        )
        locs[loc.loc] = loc
    return locs

def load_ships() -> Dict[str, Ship]:
    ships: Dict[str, Ship] = {}
    for r in parse_csv(SHIPS_CSV):
        s = Ship(
            ship_id=r["Ship"],
            ownership=r["Ownership"],
            capacity=int(r["ShipCapacity"]),
            fixed_daily_cost=int(r["FixedDailyCost"]),
        )
        ships[s.ship_id] = s
    return ships

def load_depots() -> Dict[str, Depot]:
    depots: Dict[str, Depot] = {}
    for r in parse_csv(DEPOTS_CSV):
        d = Depot(
            depot=r["Depot"],
            region=r["Region"],
            max_cap=int(r["MaxCap"]),
            min_cap=int(r["MinCap"]),
            daily_decay=int(r["DailyDecay"]),
        )
        depots[d.depot] = d
    return depots

def load_routes(locs: Dict[str, Location]) -> Dict[str, Route]:
    routes: Dict[str, Route] = {}
    for r in parse_csv(ROUTES_CSV):
        rid = r["Route"]
        sid = r["Ship"]
        stops = tuple(s.strip() for s in r["Stops"].split("-") if s.strip())
        duration = int(r["DurationDays"])
        cost = int(r["CostUSD"])

        stop_regions = {locs[s].region for s in stops}
        region = stop_regions.pop() if len(stop_regions) == 1 else "MIXED"
        demand_tons = sum(locs[s].qty for s in stops)

        routes[rid] = Route(
            route_id=rid,
            ship_id=sid,
            start_port=r["StartPort"],
            end_port=r["EndPort"],
            stops=stops,
            duration=duration,
            cost=cost,
            region=region,
            demand_tons=demand_tons,
            is_depot_delivery=False
        )
    return routes


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
        
        target_depots = []
        if r.region == "North":
            target_depots.append(depots["D_North"])
        elif r.region == "South":
            target_depots.append(depots["D_South1"])
            target_depots.append(depots["D_South2"])
            
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

    def dfs(idx: int, current_time: int) -> bool: #yerleştirme
        if idx == len(items):
            return True
        rid, lo, hi, dur = items[idx]
        start_min = max(current_time, lo)
        for start in range(start_min, hi + 1):
            end_day = start + dur - 1
            if end_day > 35:
                continue
            chosen[rid] = start
            if dfs(idx + 1, end_day + 1):
                return True
            del chosen[rid]
        return False

    return chosen if dfs(0, 1) else None


# 7) EVALUATION

BIG_M = 10_000_000

def evaluate(
    sol: Solution,
    ships: Dict[str, Ship],
    routes: Dict[str, Route],
    locs: Dict[str, Location],
    depots: Dict[str, Depot],
    cfg: ConstraintConfig,
    rental_income_if_idle_ge_5: int = 1500, # #5 günden fazla boşta kalırsa günlük kira geliri
    idle_penalty_if_idle_lt_5: int = 1000,#5 günden az boşta kalırsa günlük ceza
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
            sched = schedule_trips_backtracking(trip_list, ship, routes, locs)
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
                    amount = ships[sid].capacity - r.demand_tons
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
#başlangıç çözümü oluşturma
def build_initial_solution_greedy_randomized(
    ships: Dict[str, Ship],
    routes: Dict[str, Route],
    locs: Dict[str, Location],
    cfg: ConstraintConfig,
    rcl_size: int = 5 #en ucuz 5 rotadan rastgele seçim yap
) -> Solution:
    customer_routes = {rid: r for rid, r in routes.items() if not r.is_depot_delivery}
    covers = route_covers(customer_routes)
    
    sol: Solution = {sid: tuple() for sid in ships.keys()}
    used_routes: Set[str] = set()
    uncovered: Set[str] = set(locs.keys())
    all_route_ids = sorted(customer_routes.keys(), key=lambda rid: customer_routes[rid].cost)
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
        
        if cfg.enforce_time_window_pm2:
            if schedule_trips_backtracking(new_list, ship, routes, locs) is None:
                return False
        return True

    for _ in range(400):
        if not uncovered: break
        target_loc = next(iter(uncovered))
        candidates = []
#bu lokasyonu kapsayan rotaları bul
        for rid in all_route_ids:
            if target_loc not in covers[rid]: continue
            if rid in used_routes: continue
            sid = routes[rid].ship_id
            if can_add_trip(sid, rid):
                candidates.append((rid, sid))
        
        if not candidates:
            uncovered.remove(target_loc)
            continue

        top_candidates = candidates[:rcl_size] #en ucuz rcl_size kadar rota alınır
        chosen_rid, chosen_sid = random.choice(top_candidates)

        sol[chosen_sid] = sol[chosen_sid] + (chosen_rid,)
        used_routes.add(chosen_rid)
        uncovered -= covers[chosen_rid]

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
):
    random.seed(seed)
    by_ship = routes_by_ship(routes)
    ship_ids = list(ships.keys())

    current = build_initial_solution_greedy_randomized(ships, routes, locs, cfg, rcl_size=5)
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
                f"viol={best_dbg['violations_count']} served={best_dbg['served_count']}/24"
            )

    return best, best_obj, best_dbg


# 9) PRINTING

def pretty_print_solution(sol: Solution, routes: Dict[str, Route], dbg: Dict):
    print("\nBEST OBJ:", f"{dbg['total_cost']:,}")
    print("violations:", dbg["violations_count"])
    print("served:", f"{dbg['served_count']}/24")
    
    print("\n=== SOLUTION (Ship -> trips) ===")
    sched = dbg.get("ship_schedules", {})
    for sid in sorted(sol.keys(), key=lambda x: int(x[1:])):
        trips = sol[sid]
        if not trips:
            print(f"{sid}: []")
            continue
        parts = []
        for rid in trips:
            r = routes[rid]
            day = sched.get(sid, {}).get(rid, None)
            extra = " [DEPOT]" if r.is_depot_delivery else ""
            parts.append(f"{rid}(stops={r.stops},cost={r.cost},day={day}{extra})")
        print(f"{sid}: " + "  ".join(parts))


# 10) RUN

def main():
    locs = load_locations()
    ships = load_ships()
    depots = load_depots()
    # Temel rotaları yükle
    base_routes = load_routes(locs)
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
        
        # Algoritmayı çalıştır
        best_sol, best_obj, dbg = tabu_search_multi_trip(
            ships, routes_copy, locs, depots, cfg,
            max_iters=1500,  
            neighborhood_size=200, 
            tabu_tenure=20, 
            seed=seed,
            max_trips_per_ship=9
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
    pretty_print_solution(global_best_sol, base_routes, global_best_dbg)
    print("\nAUTO POST-OPT: no improvement found.")

if __name__ == "__main__":
    main()