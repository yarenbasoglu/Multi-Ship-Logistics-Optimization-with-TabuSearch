from __future__ import annotations

import csv
import io
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set
from copy import deepcopy

# ============================================================
# 0) TEACHER DATASET FILE PATHS
# ============================================================
# Put the csv files in the same folder as this script, OR update the paths below.
SAYFA1_CSV_PATH      = "ENS001Dataset(Sayfa1).csv"
GUN_MALIYET_CSV_PATH = "ENS001Dataset(Gun_Maliyet).csv"
STOP1_CSV_PATH       = "ENS001Dataset(1).csv"
STOP2_CSV_PATH       = "ENS001Dataset(2).csv"
STOP3_CSV_PATH       = "ENS001Dataset(3).csv"

# ============================================================
# 1) LOCATIONS (your previous consistent location demand days)
#    If your teacher dataset has its own demand days/qty, replace this block with that.
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

# ============================================================
# 2) MODELS
# ============================================================

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
    ownership: str     # "owned" or "leased"
    capacity: int      # we keep, but set very large unless you have real capacities
    fixed_daily_cost: float
    region: str        # not used by teacher dataset; default "ALL"

@dataclass(frozen=True)
class Route:
    route_id: str
    ship_id: str
    code: str
    start_port: str
    end_port: str
    stops: Tuple[str, ...]              # e.g. ("L6","L10")
    duration: int                       # from Gun_Maliyet -> Gun3 (total days)
    daily_cost: float                   # from Gun_Maliyet -> Maliyet
    cost: float                         # duration * daily_cost
    stop_offsets: Tuple[int, ...]       # from 1.stop/2.stop/3.stop (arrival day within trip)

Solution = Dict[str, Tuple[str, ...]]   # ship_id -> tuple(route_ids)

# ============================================================
# 3) CSV HELPERS
# ============================================================

def read_csv_semicolon(path: str) -> List[Dict[str, str]]:
    """
    Teacher CSVs are ';' separated and use ',' as decimal separator.
    We read as raw strings first; we will convert explicitly.
    """
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        rows: List[Dict[str, str]] = []
        for r in reader:
            # skip fully empty
            if not r:
                continue
            # skip blank "first" field rows
            first_key = next(iter(r.keys()))
            if not (r.get(first_key) or "").strip():
                continue
            rows.append({k: (v.strip() if isinstance(v, str) else v) for k, v in r.items()})
        return rows

def to_int(x: Optional[str]) -> Optional[int]:
    if x is None:
        return None
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return None
    # teacher uses "Yok"
    if s.lower() == "yok":
        return None
    # sometimes "06" like strings -> int(6)
    try:
        return int(s)
    except ValueError:
        # try float then int
        try:
            return int(float(s.replace(",", ".")))
        except Exception:
            return None

def to_float(x: Optional[str]) -> Optional[float]:
    if x is None:
        return None
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return None
    if s.lower() == "yok":
        return None
    # teacher uses decimal comma
    s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None

# ============================================================
# 4) LOAD LOCATIONS
# ============================================================

def parse_csv_inline(text: str) -> List[Dict[str, str]]:
    f = io.StringIO(text.strip())
    reader = csv.DictReader(f)
    out: List[Dict[str, str]] = []
    for r in reader:
        if not r:
            continue
        out.append({k: (v.strip() if isinstance(v, str) else v) for k, v in r.items()})
    return out

def load_locations() -> Dict[str, Location]:
    locs: Dict[str, Location] = {}
    for r in parse_csv_inline(LOCATIONS_CSV):
        loc = Location(
            loc=r["Location"],
            region=r["Region"],
            product=r["Product"],
            qty=int(r["Qty"]),
            demand_day=int(r["DemandDay"]),
        )
        locs[loc.loc] = loc
    return locs

# ============================================================
# 5) LOAD TEACHER ROUTES (Gun_Maliyet + stop sheets)
# ============================================================

def load_stop_offsets() -> Dict[Tuple[str, str], Tuple[Optional[int], Optional[int], Optional[int]]]:
    """
    Returns:
      (route_code, ship_name) -> (offset1, offset2, offset3)
    where offsetk = that ship reaches k-th stop on that day (within the trip).
    """
    s1 = read_csv_semicolon(STOP1_CSV_PATH)
    s2 = read_csv_semicolon(STOP2_CSV_PATH)
    s3 = read_csv_semicolon(STOP3_CSV_PATH)

    # identify ship columns: everything that starts with "Gemi" or "Kiralik"
    ship_cols = [c for c in s1[0].keys() if c.startswith("Gemi") or c.startswith("Kiralik")]

    # route code column in these sheets is usually "Kod"
    def index_sheet(rows):
        idx: Dict[str, Dict[str, Optional[int]]] = {}
        for r in rows:
            code = (r.get("Kod") or "").strip()
            if not code:
                continue
            idx[code] = {sc: to_int(r.get(sc)) for sc in ship_cols}
        return idx

    i1 = index_sheet(s1)
    i2 = index_sheet(s2)
    i3 = index_sheet(s3)

    out: Dict[Tuple[str, str], Tuple[Optional[int], Optional[int], Optional[int]]] = {}
    for code in set(i1.keys()) | set(i2.keys()) | set(i3.keys()):
        for ship in ship_cols:
            out[(code, ship)] = (
                (i1.get(code, {}) or {}).get(ship),
                (i2.get(code, {}) or {}).get(ship),
                (i3.get(code, {}) or {}).get(ship),
            )
    return out

def build_ships_from_dataset(gun_rows: List[Dict[str, str]]) -> Dict[str, Ship]:
    """
    Teacher dataset ships are named: Gemi1..Gemi8, Kiralik1..Kiralik3.
    We create ships with huge capacity, region "ALL" (since dataset doesn't include region/capacity).
    """
    ship_names = sorted({(r.get("Ship") or "").strip() for r in gun_rows if (r.get("Ship") or "").strip()})
    ships: Dict[str, Ship] = {}
    for name in ship_names:
        ownership = "leased" if name.lower().startswith("kiralik") else "owned"
        ships[name] = Ship(
            ship_id=name,
            ownership=ownership,
            capacity=999999,         # unknown => set large so capacity won't block
            fixed_daily_cost=0.0,
            region="ALL",
        )
    return ships

def load_routes_teacher(locs: Dict[str, Location]) -> Tuple[Dict[str, Ship], Dict[str, Route]]:
    """
    Uses Gun_Maliyet as the main table:
      - duration = Gun3
      - daily_cost = Maliyet
      - cost = duration * daily_cost
      - stops come from columns j/k if numeric; 'Yok' => missing
      - stop_offsets come from stop sheets for that (code, ship)
    """
    gun_rows = read_csv_semicolon(GUN_MALIYET_CSV_PATH)
    offsets_map = load_stop_offsets()
    ships = build_ships_from_dataset(gun_rows)

    routes: Dict[str, Route] = {}

    for r in gun_rows:
        code = (r.get("Kod") or "").strip()
        ship_name = (r.get("Ship") or "").strip()
        if not code or not ship_name:
            continue

        start_port = str(to_int(r.get("i")) or "").zfill(2)
        stop1_n = to_int(r.get("j"))
        stop2_n = to_int(r.get("k"))
        end_port = str(to_int(r.get("o")) or "").zfill(2)

        stops: List[str] = []
        if stop1_n is not None:
            stops.append(f"L{stop1_n}")
        if stop2_n is not None:
            stops.append(f"L{stop2_n}")

        # guard: skip routes that mention locations we don't have
        if any(s not in locs for s in stops):
            continue

        duration = to_int(r.get("Gun3")) or 0
        daily_cost = to_float(r.get("Maliyet")) or 0.0
        cost = float(duration) * float(daily_cost)

        off1, off2, off3 = offsets_map.get((code, ship_name), (None, None, None))
        # build offsets aligned to stops; for 1-stop route use off1; for 2-stop use (off1, off2)
        stop_offsets: List[int] = []
        if len(stops) >= 1 and off1 is not None:
            stop_offsets.append(int(off1))
        if len(stops) >= 2 and off2 is not None:
            stop_offsets.append(int(off2))

        # if offset missing, keep it but it will make route infeasible later
        while len(stop_offsets) < len(stops):
            stop_offsets.append(9999)

        rid = f"{code}_{ship_name}"

        routes[rid] = Route(
            route_id=rid,
            ship_id=ship_name,
            code=code,
            start_port=start_port,
            end_port=end_port,
            stops=tuple(stops),
            duration=int(duration),
            daily_cost=float(daily_cost),
            cost=float(cost),
            stop_offsets=tuple(stop_offsets),
        )

    return ships, routes

# ============================================================
# 6) CONSTRAINT CONFIG
# ============================================================

@dataclass
class ConstraintConfig:
    enforce_total_duration_le_35: bool = True
    enforce_unique_route_globally: bool = True
    enforce_cover_all_locations: bool = True
    enforce_arrival_window_pm2: bool = True     # NEW: uses stop_offsets vs demand_day
    horizon_days: int = 35

# ============================================================
# 7) FEASIBILITY: arrival-window based on stop_offsets
# ============================================================

def feasible_start_days_for_route(route: Route, locs: Dict[str, Location], cfg: ConstraintConfig) -> List[int]:
    """
    If a route starts at day t (1-indexed), then it reaches its k-th stop at:
      arrival_day = t + offset_k - 1

    Constraint: for each stop location s:
      arrival_day must be within [demand_day(s)-2, demand_day(s)+2]

    We return all feasible t in [1 .. horizon - duration + 1].
    """
    if not cfg.enforce_arrival_window_pm2:
        return list(range(1, cfg.horizon_days - route.duration + 2))

    lo = 1
    hi = cfg.horizon_days - route.duration + 1
    if hi < 1:
        return []

    for s, off in zip(route.stops, route.stop_offsets):
        d = locs[s].demand_day
        # t + off - 1 in [d-2, d+2]  => t in [d-2 - (off-1), d+2 - (off-1)]
        lo = max(lo, (d - 2) - (off - 1))
        hi = min(hi, (d + 2) - (off - 1))

    if lo > hi:
        return []
    return list(range(int(lo), int(hi) + 1))

def schedule_trips_backtracking(
    route_ids: Tuple[str, ...],
    ship: Ship,
    routes: Dict[str, Route],
    locs: Dict[str, Location],
    cfg: ConstraintConfig,
) -> Optional[Dict[str, int]]:
    """
    Choose start day for each route sequentially (no overlap).
    Each route has feasible start days based on arrival offsets and demand windows.
    """
    items = []
    for rid in route_ids:
        r = routes[rid]
        days = feasible_start_days_for_route(r, locs, cfg)
        if not days:
            return None
        items.append((rid, days, r.duration))

    # most constrained first: fewest feasible start days
    items.sort(key=lambda x: (len(x[1]), min(x[1])))

    chosen: Dict[str, int] = {}

    def dfs(idx: int, current_time: int) -> bool:
        if idx == len(items):
            return True
        rid, days, dur = items[idx]

        for start in days:
            if start < current_time:
                continue
            end_day = start + dur - 1
            if end_day > cfg.horizon_days:
                continue

            chosen[rid] = start
            if dfs(idx + 1, end_day + 1):
                return True
            del chosen[rid]
        return False

    return chosen if dfs(0, 1) else None

# ============================================================
# 8) OBJECTIVE / EVALUATION
# ============================================================

BIG_M = 10_000_000

def evaluate(
    sol: Solution,
    ships: Dict[str, Ship],
    routes: Dict[str, Route],
    locs: Dict[str, Location],
    cfg: ConstraintConfig,
    rental_income_if_idle_ge_5: float = 15000.0,
    idle_penalty_if_idle_lt_5: float = 10000.0,
) -> Tuple[float, Dict]:

    violations: List[str] = []
    total_cost = 0.0
    served: Set[str] = set()
    used_routes: Set[str] = set()
    ship_schedules: Dict[str, Dict[str, int]] = {}

    for ship_id, trip_list in sol.items():
        ship = ships[ship_id]
        total_duration = 0

        for rid in trip_list:
            if rid not in routes:
                violations.append(f"{ship_id}: route {rid} not found")
                continue
            r = routes[rid]

            if r.ship_id != ship_id:
                violations.append(f"{ship_id}: picked {rid} belongs to {r.ship_id}")

            if cfg.enforce_unique_route_globally:
                if rid in used_routes:
                    violations.append(f"route used multiple times: {rid}")
                used_routes.add(rid)

            total_duration += r.duration
            total_cost += r.cost + ship.fixed_daily_cost * r.duration

            for s in r.stops:
                served.add(s)

        if cfg.enforce_total_duration_le_35 and total_duration > cfg.horizon_days:
            violations.append(f"{ship_id}: total_duration {total_duration} > {cfg.horizon_days}")

        # idle rule (same as your old model)
        idle = cfg.horizon_days - total_duration
        if idle >= 5:
            total_cost -= rental_income_if_idle_ge_5
        else:
            total_cost += idle_penalty_if_idle_lt_5

        # schedule feasibility using teacher arrival offsets
        sched = schedule_trips_backtracking(trip_list, ship, routes, locs, cfg)
        if sched is None:
            violations.append(f"{ship_id}: cannot schedule trips within arrival ±2 and {cfg.horizon_days} days")
        else:
            ship_schedules[ship_id] = sched

    if cfg.enforce_cover_all_locations:
        missing = sorted(set(locs.keys()) - served)
        for m in missing:
            violations.append(f"missing location: {m}")

    objective = total_cost + BIG_M * len(violations)
    debug = {
        "total_cost": total_cost,
        "violations_count": len(violations),
        "violations": violations[:50],
        "served_count": len(served),
        "ship_schedules": ship_schedules,
    }
    return objective, debug

# ============================================================
# 9) TABU SEARCH (multi-trip, clean)
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
    max_trips_per_ship: int,
) -> Solution:
    covers = route_covers(routes)
    by_ship = routes_by_ship(routes)

    sol: Solution = {sid: tuple() for sid in ships.keys()}
    used_routes: Set[str] = set()
    uncovered: Set[str] = set(locs.keys())

    all_route_ids = sorted(routes.keys(), key=lambda rid: routes[rid].cost)

    def can_add_trip(ship_id: str, rid: str) -> bool:
        if rid in used_routes and cfg.enforce_unique_route_globally:
            return False
        if routes[rid].ship_id != ship_id:
            return False
        if len(sol[ship_id]) >= max_trips_per_ship:
            return False

        new_list = sol[ship_id] + (rid,)
        total_dur = sum(routes[x].duration for x in new_list)
        if cfg.enforce_total_duration_le_35 and total_dur > cfg.horizon_days:
            return False

        # schedule must be feasible
        if schedule_trips_backtracking(new_list, ships[ship_id], routes, locs, cfg) is None:
            return False
        return True

    for _ in range(2000):
        if not uncovered:
            break
        target_loc = next(iter(uncovered))
        added = False

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
            # cannot cover some locations with current dataset/constraints
            break

    return sol

def tabu_search_multi_trip(
    ships: Dict[str, Ship],
    routes: Dict[str, Route],
    locs: Dict[str, Location],
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

    current = build_initial_solution_greedy(ships, routes, locs, cfg, max_trips_per_ship=max_trips_per_ship)
    best = dict(current)

    best_obj, best_dbg = evaluate(best, ships, routes, locs, cfg)
    cur_obj = best_obj

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

    for it in range(1, max_iters + 1):
        decay_tabu()

        best_cand = None
        best_cand_obj = None
        best_cand_dbg = None
        best_move = None  # (sid, action, rid)

        for _ in range(neighborhood_size):
            sid = random.choice(ship_ids)
            trips = list(current[sid])

            action = random.choice(["add", "remove", "swap_order"])

            if action == "add":
                if len(trips) >= max_trips_per_ship:
                    continue
                rid = random.choice(by_ship.get(sid, []))
                if rid in trips:
                    continue

                cand = dict(current)
                cand[sid] = tuple(trips + [rid])
                cand_obj, cand_dbg = evaluate(cand, ships, routes, locs, cfg)

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

                cand_obj, cand_dbg = evaluate(cand, ships, routes, locs, cfg)

                if is_tabu(sid, "remove", rid) and cand_obj >= best_obj:
                    continue

                if best_cand is None or cand_obj < best_cand_obj:
                    best_cand, best_cand_obj, best_cand_dbg = cand, cand_obj, cand_dbg
                    best_move = (sid, "remove", rid)

            else:  # swap_order
                if len(trips) < 2:
                    continue
                i, j = random.sample(range(len(trips)), 2)
                trips[i], trips[j] = trips[j], trips[i]

                cand = dict(current)
                cand[sid] = tuple(trips)
                cand_obj, cand_dbg = evaluate(cand, ships, routes, locs, cfg)

                if best_cand is None or cand_obj < best_cand_obj:
                    best_cand, best_cand_obj, best_cand_dbg = cand, cand_obj, cand_dbg
                    best_move = (sid, "swap_order", "ORDER")

        if best_cand is None:
            break

        current = best_cand
        cur_obj = best_cand_obj

        if best_move:
            add_tabu(*best_move)

        if cur_obj < best_obj:
            best = dict(current)
            best_obj, best_dbg = cur_obj, best_cand_dbg

        if it % 100 == 0:
            print(
                f"iter={it:4d} best_obj={best_obj:,.2f} cur_obj={cur_obj:,.2f} "
                f"viol={best_dbg['violations_count']} served={best_dbg['served_count']}/{len(locs)}"
            )

    return best, best_obj, best_dbg

# ============================================================
# 10) OUTPUT HELPERS
# ============================================================

def pretty_print_solution(sol: Solution, routes: Dict[str, Route], dbg: Dict):
    print("\n=== SOLUTION (Ship -> trips) ===")
    sched = dbg.get("ship_schedules", {})
    for sid in sorted(sol.keys()):
        trips = sol[sid]
        if not trips:
            continue
        parts = []
        for rid in trips:
            r = routes[rid]
            start_day = sched.get(sid, {}).get(rid, None)
            parts.append(
                f"{rid}(code={r.code}, stops={r.stops}, dur={r.duration}, daily={r.daily_cost}, cost={r.cost:.2f}, start={start_day})"
            )
        print(f"{sid}: " + "  ".join(parts))

# ============================================================
# 11) MAIN
# ============================================================

def main():
    locs = load_locations()
    ships, routes = load_routes_teacher(locs)

    cfg = ConstraintConfig(
        enforce_total_duration_le_35=True,
        enforce_unique_route_globally=True,
        enforce_cover_all_locations=True,
        enforce_arrival_window_pm2=True,
        horizon_days=35,
    )

    best, best_obj, dbg = tabu_search_multi_trip(
        ships, routes, locs, cfg,
        max_iters=1500,
        neighborhood_size=240,
        tabu_tenure=24,
        seed=11,
        max_trips_per_ship=9,
    )

    print("\nBEST OBJ:", f"{best_obj:,.2f}")
    print("violations:", dbg["violations_count"])
    print("served:", f"{dbg['served_count']}/{len(locs)}")
    pretty_print_solution(best, routes, dbg)


if __name__ == "__main__":
    main()
