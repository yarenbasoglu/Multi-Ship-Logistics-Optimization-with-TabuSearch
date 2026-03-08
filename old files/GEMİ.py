from __future__ import annotations

import csv
import io
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set


# ============================================================
# 1) CONSISTENT DATA (same as we discussed)
# ============================================================

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
"""

SHIPS_CSV = """Ship,Ownership,ShipCapacity,FixedDailyCost,ShipRegion
S1,owned,2500,0,North
S2,owned,2300,0,North
S3,owned,2100,0,North
S4,owned,1900,0,South
S5,owned,2600,0,North
S6,owned,2400,0,North
S7,owned,2300,0,North
S8,leased,1300,0,South
S9,leased,2000,0,South
S10,leased,2200,0,North
"""

DEPOTS_CSV = """Depot,Region,IO_gas,IO_diesel,Target_gas,Target_diesel
D_N(L3),North,2000,2000,1500,1500
D_S1(L10),South,2500,1500,1800,1200
D_S2(L12),South,1500,2500,1200,1800
"""

# Ship-specific routes dataset (we keep it as is; multi-trip fixes coverage)
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
"""  # S10 south routes removed as discussed


# ============================================================
# 2) MODELS
# ============================================================

@dataclass(frozen=True)
class Location:
    loc: str
    region: str     # North / South
    product: str    # gasoline / diesel
    qty: int
    demand_day: int

@dataclass(frozen=True)
class Ship:
    ship_id: str
    ownership: str
    capacity: int
    fixed_daily_cost: int
    region: str     # North / South

@dataclass(frozen=True)
class Depot:
    depot: str
    region: str
    io_gas: int
    io_diesel: int
    target_gas: int
    target_diesel: int

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

# Solution: ship -> tuple(route_ids) (multiple trips)
Solution = Dict[str, Tuple[str, ...]]


# ============================================================
# 3) PARSERS
# ============================================================

def parse_csv(text: str) -> List[Dict[str, str]]:
    f = io.StringIO(text.strip())
    reader = csv.DictReader(f)
    rows: List[Dict[str, str]] = []
    for row in reader:
        if not row:
            continue
        # skip blank lines
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
            region=r["ShipRegion"],
        )
        ships[s.ship_id] = s
    return ships

def load_depots() -> List[Depot]:
    depots: List[Depot] = []
    for r in parse_csv(DEPOTS_CSV):
        depots.append(Depot(
            depot=r["Depot"],
            region=r["Region"],
            io_gas=int(r["IO_gas"]),
            io_diesel=int(r["IO_diesel"]),
            target_gas=int(r["Target_gas"]),
            target_diesel=int(r["Target_diesel"]),
        ))
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
        )
    return routes


# ============================================================
# 4) CONSTRAINT CONFIG
# ============================================================

@dataclass
class ConstraintConfig:
    enforce_duration_le_7: bool = True
    enforce_max_2_stops: bool = True
    enforce_ship_route_region_match: bool = True
    enforce_capacity: bool = True

    enforce_total_duration_le_35: bool = True          # NEW
    enforce_unique_route_globally: bool = True         # NEW

    enforce_cover_all_locations: bool = True
    enforce_time_window_pm2: bool = True

    include_inventory_penalty: bool = False            # keep off by default


# ============================================================
# 5) TIME WINDOWS + SCHEDULING (NEW)
# ============================================================

def feasible_service_days_for_route(route: Route, locs: Dict[str, Location]) -> List[int]:
    """
    Route stops demand day ±2 -> intersection window.
    Returns all feasible 'service start days' in [1..35].
    """
    lo = 1
    hi = 35
    for s in route.stops:
        d = locs[s].demand_day
        lo = max(lo, d - 2)
        hi = min(hi, d + 2)
    if lo > hi:
        return []
    return list(range(lo, hi + 1))

def schedule_trips_sequential(route_ids: Tuple[str, ...], ship: Ship, routes: Dict[str, Route], locs: Dict[str, Location]) -> Optional[Dict[str, int]]:
    """
    We schedule trips sequentially within 35 days without overlap.
    For each trip we pick the earliest feasible service day >= current_time.
    Trip occupies [start_day, start_day + duration - 1].
    Returns mapping route_id -> chosen start_day, or None if impossible.
    """
    current_time = 1
    chosen: Dict[str, int] = {}
    for rid in route_ids:
        r = routes[rid]
        days = feasible_service_days_for_route(r, locs)
        if not days:
            return None
        # earliest day >= current_time
        pick = None
        for d in days:
            if d >= current_time:
                pick = d
                break
        if pick is None:
            return None
        end_day = pick + r.duration - 1
        if end_day > 35:
            return None
        chosen[rid] = pick
        current_time = end_day + 1
    return chosen


# ============================================================
# 6) INVENTORY (optional, simple)
# ============================================================

def depot_assignment_for_south(loc: str) -> str:
    n = int(loc[1:])
    return "D_S1(L10)" if 9 <= n <= 11 else "D_S2(L12)"


# ============================================================
# 7) EVALUATION
# ============================================================

BIG_M = 10_000_000

def evaluate(
    sol: Solution,
    ships: Dict[str, Ship],
    routes: Dict[str, Route],
    locs: Dict[str, Location],
    depots: List[Depot],
    cfg: ConstraintConfig,
    rental_income_if_idle_ge_5: int = 15000,
    idle_penalty_if_idle_lt_5: int = 10000,
    inv_weight: int = 2,
) -> Tuple[int, Dict]:
    violations: List[str] = []
    total_cost = 0
    served: Set[str] = set()

    # global unique route usage
    used_routes: Set[str] = set()

    # inventory state (toy)
    inv_penalty = 0
    inv_debug = {}
    depot_map = {d.depot: d for d in depots}
    inv_after = {d.depot: {"gasoline": d.io_gas, "diesel": d.io_diesel} for d in depots}

    # store per ship schedule info
    ship_schedules: Dict[str, Dict[str, int]] = {}

    for ship_id, trip_list in sol.items():
        ship = ships[ship_id]

        # 0) validate each route belongs to ship, add costs, serve locations
        total_duration = 0
        for rid in trip_list:
            if rid not in routes:
                violations.append(f"{ship_id}: route {rid} not found")
                continue
            r = routes[rid]

            if r.ship_id != ship_id:
                violations.append(f"{ship_id}: picked route {rid} belongs to {r.ship_id}")

            if cfg.enforce_unique_route_globally:
                if rid in used_routes:
                    violations.append(f"route used multiple times: {rid}")
                used_routes.add(rid)

            if cfg.enforce_duration_le_7 and r.duration > 7:
                violations.append(f"{ship_id}/{rid}: duration {r.duration} > 7")

            if cfg.enforce_max_2_stops and len(r.stops) > 2:
                violations.append(f"{ship_id}/{rid}: stops {len(r.stops)} > 2")

            if cfg.enforce_ship_route_region_match:
                if r.region == "MIXED":
                    violations.append(f"{ship_id}/{rid}: mixed route region")
                elif r.region != ship.region:
                    violations.append(f"{ship_id}/{rid}: route region {r.region} != ship region {ship.region}")

            if cfg.enforce_capacity and r.demand_tons > ship.capacity:
                violations.append(f"{ship_id}/{rid}: demand {r.demand_tons} > cap {ship.capacity}")

            total_duration += r.duration

            # cost (fixed_daily_cost is 0 in your data but kept)
            total_cost += r.cost + ship.fixed_daily_cost * r.duration

            for s in r.stops:
                served.add(s)

                if cfg.include_inventory_penalty:
                    loc = locs[s]
                    if loc.region == "North":
                        depot_id = "D_N(L3)"
                    else:
                        depot_id = depot_assignment_for_south(loc.loc)
                    inv_after[depot_id][loc.product] -= loc.qty

        # 1) total duration <= 35
        if cfg.enforce_total_duration_le_35 and total_duration > 35:
            violations.append(f"{ship_id}: total_duration {total_duration} > 35")

        # 2) idle rule uses TOTAL duration
        idle = 35 - total_duration
        if idle >= 5:
            total_cost -= rental_income_if_idle_ge_5
        else:
            total_cost += idle_penalty_if_idle_lt_5

        # 3) time window scheduling (sequential, non-overlap)
        if cfg.enforce_time_window_pm2:
            sched = schedule_trips_sequential(trip_list, ship, routes, locs)
            if sched is None:
                violations.append(f"{ship_id}: cannot schedule trips within ±2 windows and 35 days")
            else:
                ship_schedules[ship_id] = sched

    # 4) cover all locations
    if cfg.enforce_cover_all_locations:
        all_locs = set(locs.keys())
        missing = sorted(all_locs - served)
        for m in missing:
            violations.append(f"missing location: {m}")

    # 5) inventory penalty (optional)
    if cfg.include_inventory_penalty:
        for d in depots:
            after_g = inv_after[d.depot]["gasoline"]
            after_d = inv_after[d.depot]["diesel"]
            inv_debug[d.depot] = {"after_gas": after_g, "after_diesel": after_d}
            inv_penalty += inv_weight * abs(after_g - d.target_gas)
            inv_penalty += inv_weight * abs(after_d - d.target_diesel)
        total_cost += inv_penalty

    objective = total_cost + BIG_M * len(violations)
    debug = {
        "total_cost": total_cost,
        "violations_count": len(violations),
        "violations": violations[:30],
        "served_count": len(served),
        "ship_schedules": ship_schedules,
        "inventory_penalty": inv_penalty,
        "inventory_after": inv_debug,
    }
    return objective, debug


# ============================================================
# 8) TABU SEARCH (MULTI-TRIP)
# ============================================================

def routes_by_ship(routes: Dict[str, Route]) -> Dict[str, List[str]]:
    d: Dict[str, List[str]] = {}
    for r in routes.values():
        d.setdefault(r.ship_id, []).append(r.route_id)
    return d

def route_covers(routes: Dict[str, Route]) -> Dict[str, Set[str]]:
    return {rid: set(routes[rid].stops) for rid in routes.keys()}

def build_initial_solution_greedy(
    ships: Dict[str, Ship],
    routes: Dict[str, Route],
    locs: Dict[str, Location],
    cfg: ConstraintConfig,
) -> Solution:
    """
    Greedy cover: try to cover all locations by adding trips to ships until coverage achieved.
    Keep it simple: iterate uncovered locations, assign a cheapest route (any ship) that covers it and is feasible locally.
    """
    by_ship = routes_by_ship(routes)
    covers = route_covers(routes)

    sol: Solution = {sid: tuple() for sid in ships.keys()}
    used_routes: Set[str] = set()
    uncovered: Set[str] = set(locs.keys())

    # Precompute candidate routes sorted by cost (global)
    all_route_ids = sorted(routes.keys(), key=lambda rid: routes[rid].cost)

    def can_add_trip(ship_id: str, rid: str) -> bool:
        r = routes[rid]
        ship = ships[ship_id]
        if r.ship_id != ship_id:
            return False
        if cfg.enforce_ship_route_region_match and (r.region == "MIXED" or r.region != ship.region):
            return False
        if cfg.enforce_capacity and r.demand_tons > ship.capacity:
            return False
        if cfg.enforce_duration_le_7 and r.duration > 7:
            return False
        if cfg.enforce_max_2_stops and len(r.stops) > 2:
            return False
        if cfg.enforce_unique_route_globally and rid in used_routes:
            return False
        # time window must exist
        if cfg.enforce_time_window_pm2 and not feasible_service_days_for_route(r, locs):
            return False
        # total duration check with new trip
        current = sol[ship_id]
        new_list = current + (rid,)
        total_dur = sum(routes[x].duration for x in new_list)
        if cfg.enforce_total_duration_le_35 and total_dur > 35:
            return False
        # schedule feasibility (sequential)
        if cfg.enforce_time_window_pm2 and schedule_trips_sequential(new_list, ship, routes, locs) is None:
            return False
        return True

    # Greedy loop: while uncovered remains, try to add a route covering something uncovered
    for _ in range(400):  # safety
        if not uncovered:
            break
        added = False
        # pick an uncovered location
        target_loc = next(iter(uncovered))

        # try cheapest route that covers it
        for rid in all_route_ids:
            if target_loc not in covers[rid]:
                continue
            sid = routes[rid].ship_id
            if can_add_trip(sid, rid):
                sol[sid] = sol[sid] + (rid,)
                used_routes.add(rid)
                uncovered -= covers[rid]
                added = True
                break

        if not added:
            # can't cover this loc with current constraints; break (tabu may still fix if you relax)
            break

    return sol

def tabu_search_multi_trip(
    ships: Dict[str, Ship],
    routes: Dict[str, Route],
    locs: Dict[str, Location],
    depots: List[Depot],
    cfg: ConstraintConfig,
    max_iters: int = 1200,
    neighborhood_size: int = 220,
    tabu_tenure: int = 22,
    seed: int = 11,
    max_trips_per_ship: int = 8,
):
    random.seed(seed)
    by_ship = routes_by_ship(routes)
    ship_ids = list(ships.keys())

    current = build_initial_solution_greedy(ships, routes, locs, cfg)
    best = dict(current)

    best_obj, best_dbg = evaluate(best, ships, routes, locs, depots, cfg)
    cur_obj, cur_dbg = best_obj, best_dbg

    # tabu: (ship_id, action, rid) -> remaining
    tabu: Dict[Tuple[str, str, str], int] = {}

    def decay_tabu():
        for k in list(tabu.keys()):
            tabu[k] -= 1
            if tabu[k] <= 0:
                del tabu[k]

    def add_tabu(ship_id: str, action: str, rid: str):
        tabu[(ship_id, action, rid)] = tabu_tenure

    def is_tabu(ship_id: str, action: str, rid: str) -> bool:
        return (ship_id, action, rid) in tabu

    all_route_ids = list(routes.keys())

    for it in range(1, max_iters + 1):
        decay_tabu()

        best_cand = None
        best_cand_obj = None
        best_cand_dbg = None
        best_move = None  # (sid, action, rid)

        for _ in range(neighborhood_size):
            sid = random.choice(ship_ids)
            trips = list(current[sid])

            action = random.choice(["add", "remove", "swap"])

            if action == "add":
                if len(trips) >= max_trips_per_ship:
                    continue
                rid = random.choice(by_ship.get(sid, []))
                if rid in trips:
                    continue
                if is_tabu(sid, "add", rid) and best_obj <= cur_obj:
                    # keep tabu unless aspiration later
                    pass

                cand = dict(current)
                cand[sid] = tuple(trips + [rid])

                cand_obj, cand_dbg = evaluate(cand, ships, routes, locs, depots, cfg)
                if is_tabu(sid, "add", rid) and cand_obj >= best_obj:
                    continue

                if best_cand is None or cand_obj < best_cand_obj:
                    best_cand, best_cand_obj, best_cand_dbg = cand, cand_obj, cand_dbg
                    best_move = (sid, "add", rid)

            elif action == "remove":
                if not trips:
                    continue
                rid = random.choice(trips)
                cand = dict(current)
                new_trips = trips[:]
                new_trips.remove(rid)
                cand[sid] = tuple(new_trips)

                cand_obj, cand_dbg = evaluate(cand, ships, routes, locs, depots, cfg)
                if is_tabu(sid, "remove", rid) and cand_obj >= best_obj:
                    continue

                if best_cand is None or cand_obj < best_cand_obj:
                    best_cand, best_cand_obj, best_cand_dbg = cand, cand_obj, cand_dbg
                    best_move = (sid, "remove", rid)
                    
            elif action == "move":
                # pick a route from one ship and move service to another ship with a replacement route
                from_sid = random.choice(ship_ids)
                from_trips = list(current[from_sid])
                if not from_trips:
                    continue
                rid_old = random.choice(from_trips)
                old_route = routes[rid_old]
                target_stops = old_route.stops  # e.g. ('L10',)

                # pick destination ship (same region)
                to_sid = random.choice([s for s in ship_ids if s != from_sid and ships[s].region == ships[from_sid].region])
                # find candidate routes on to_sid that cover the same stops (exact match)
                cand_rids = [rid for rid in routes_by_ship(routes).get(to_sid, [])
                            if routes[rid].stops == target_stops]

                if not cand_rids:
                    continue

                # choose a cheaper replacement if possible
                cand_rids.sort(key=lambda rid: routes[rid].cost)
                rid_new = cand_rids[0]

                # build candidate solution
                cand = dict(current)

                new_from = list(cand[from_sid])
                new_from.remove(rid_old)
                cand[from_sid] = tuple(new_from)

                new_to = list(cand[to_sid])
                if rid_new not in new_to:
                    new_to.append(rid_new)
                cand[to_sid] = tuple(new_to)

                cand_obj, cand_dbg = evaluate(cand, ships, routes, locs, depots, cfg)

                if best_cand is None or cand_obj < best_cand_obj:
                    best_cand, best_cand_obj, best_cand_dbg = cand, cand_obj, cand_dbg
                    best_move = (to_sid, "move", rid_new)


            else:  # swap
                if not trips:
                    continue
                old = random.choice(trips)
                new = random.choice(by_ship.get(sid, []))
                if new == old:
                    continue
                cand = dict(current)
                new_trips = trips[:]
                idx = new_trips.index(old)
                new_trips[idx] = new
                cand[sid] = tuple(new_trips)

                cand_obj, cand_dbg = evaluate(cand, ships, routes, locs, depots, cfg)
                if is_tabu(sid, "swap", new) and cand_obj >= best_obj:
                    continue

                if best_cand is None or cand_obj < best_cand_obj:
                    best_cand, best_cand_obj, best_cand_dbg = cand, cand_obj, cand_dbg
                    best_move = (sid, "swap", new)

        if best_cand is None:
            break

        current = best_cand
        cur_obj, cur_dbg = best_cand_obj, best_cand_dbg

        # register tabu move
        if best_move:
            add_tabu(*best_move)

        if cur_obj < best_obj:
            best = dict(current)
            best_obj, best_dbg = cur_obj, cur_dbg

        if it % 120 == 0:
            print(
                f"iter={it:4d} best_obj={best_obj:,} cur_obj={cur_obj:,} "
                f"viol={best_dbg['violations_count']} served={best_dbg['served_count']}/14"
            )

        # early stop: feasible and stable-ish
        if best_dbg["violations_count"] == 0 and it > 240:
            # you can keep it running for better cost; but this is a good stopping rule for homework
            pass

    return best, best_obj, best_dbg


# ============================================================
# 9) RUN
# ============================================================

def pretty_print_solution(sol: Solution, routes: Dict[str, Route], dbg: Dict):
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
            parts.append(f"{rid}(stops={r.stops},dur={r.duration},cost={r.cost},day={day})")
        print(f"{sid}: " + "  ".join(parts))

def main():
    locs = load_locations()
    ships = load_ships()
    depots = load_depots()
    routes = load_routes(locs)

    cfg = ConstraintConfig(
        enforce_duration_le_7=True,
        enforce_max_2_stops=True,
        enforce_ship_route_region_match=True,
        enforce_capacity=True,
        enforce_total_duration_le_35=True,
        enforce_unique_route_globally=True,
        enforce_cover_all_locations=True,
        enforce_time_window_pm2=True,
        include_inventory_penalty=False,
    )

    best, best_obj, dbg = tabu_search_multi_trip(
        ships, routes, locs, depots, cfg,
        max_iters=1500, neighborhood_size=240, tabu_tenure=24, seed=11,
        max_trips_per_ship=9
    )

    print("\nBEST OBJ:", f"{best_obj:,}")
    print("violations:", dbg["violations_count"])
    if dbg["violations_count"] > 0:
        print("sample violations:", dbg["violations"][:15])
    print("served:", f"{dbg['served_count']}/14")

    pretty_print_solution(best, routes, dbg)

if __name__ == "__main__":
    main()
