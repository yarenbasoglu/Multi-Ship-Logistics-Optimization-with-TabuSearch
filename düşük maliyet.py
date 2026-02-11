from __future__ import annotations

import csv
import io
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict

# =========================================================
# INPUT VERİLERİ
# =========================================================

# Talep günleri (lokasyon -> ideal gün)
demand_day: Dict[str, int] = {
    "L1": 5,  "L2": 7,
    "L3": 12, "L4": 13,
    "L5": 20, "L6": 22,
    "L7": 28, "L8": 30,
    "L9": 9,  "L10": 16,
    "L11": 18, "L12": 25,
    "L13": 6,  "L14": 31,
}

# Lokasyon bazlı talep (ton)
total_demand: Dict[str, int] = {
    "L1": 1000, "L2": 1200, "L3": 900,  "L4": 1100,
    "L5": 1300, "L6": 1000, "L7": 800,  "L8": 950,
    "L9": 1050, "L10": 900, "L11": 1150, "L12": 1000,
    "L13": 850, "L14": 950,
}

# Gemi kapasiteleri
ships_data: Dict[str, int] = {
    "S1": 2500, "S2": 2200, "S3": 2000, "S4": 1800, "S5": 2600,
    "S6": 2300, "S7": 2400, "S8": 1700, "S9": 1900, "S10": 2100,
}

# --- SENİN 120 ROTALIK CSV ---
routes_csv = """Route,Ship,StartPort,EndPort,Stops,DemandTons,DurationDays,CostUSD
R1,S1,P1,P1,L1-L2,2300,7,52000
R2,S1,P1,P2,L1-L2,2300,7,55500
R3,S1,P2,P1,L1-L2,2300,6,54500
R4,S1,P2,P2,L1-L2,2300,7,56500
R5,S1,P1,P1,L5-L6,2400,7,61000
R6,S1,P1,P2,L5-L6,2400,7,64500
R7,S1,P2,P1,L5-L6,2400,6,63500
R8,S1,P2,P2,L5-L6,2400,7,65500
R9,S1,P1,P1,L3-L4,2000,6,50000
R10,S1,P2,P2,L7-L8,1750,6,47000
R11,S1,P1,P2,L9,1200,5,43000
R12,S1,P2,P1,L14,1200,6,45500

R13,S2,P1,P1,L1-L2,2200,7,59000
R14,S2,P1,P2,L1-L2,2200,7,62500
R15,S2,P2,P1,L1-L2,2200,6,61500
R16,S2,P2,P2,L1-L2,2200,7,63500
R17,S2,P1,P1,L5-L6,2200,7,70000
R18,S2,P1,P2,L5-L6,2200,7,73500
R19,S2,P2,P1,L5-L6,2200,6,72500
R20,S2,P2,P2,L5-L6,2200,7,74500
R21,S2,P1,P1,L3-L4,2000,7,61000
R22,S2,P2,P2,L3-L4,2000,6,60000
R23,S2,P1,P2,L10,1100,6,51000
R24,S2,P2,P1,L12,1200,6,52000

R25,S3,P1,P1,L3-L4,2000,7,66000
R26,S3,P1,P2,L3-L4,2000,7,69500
R27,S3,P2,P1,L3-L4,2000,6,68500
R28,S3,P2,P2,L3-L4,2000,7,70500
R29,S3,P1,P1,L7-L8,1750,7,54000
R30,S3,P1,P2,L7-L8,1750,7,57500
R31,S3,P2,P1,L7-L8,1750,6,56500
R32,S3,P2,P2,L7-L8,1750,7,58500
R33,S3,P1,P1,L9,1200,5,47000
R34,S3,P2,P2,L10,1100,6,49500
R35,S3,P1,P2,L11,1150,6,50500
R36,S3,P2,P1,L14,1200,6,52000

R37,S4,P1,P1,L7-L8,1750,7,61000
R38,S4,P1,P2,L7-L8,1750,7,64500
R39,S4,P2,P1,L7-L8,1750,6,63500
R40,S4,P2,P2,L7-L8,1750,7,65500
R41,S4,P1,P1,L10,1100,6,52000
R42,S4,P1,P2,L10,1100,6,55500
R43,S4,P2,P1,L10,1100,5,54500
R44,S4,P2,P2,L10,1100,6,56500
R45,S4,P1,P1,L11,1150,6,54000
R46,S4,P2,P2,L12,1200,6,55000
R47,S4,P1,P2,L13,1000,5,50000
R48,S4,P2,P1,L9,1200,5,51000

R49,S5,P1,P1,L1-L2,2500,7,72000
R50,S5,P1,P2,L1-L2,2500,7,75500
R51,S5,P2,P1,L1-L2,2500,6,74500
R52,S5,P2,P2,L1-L2,2500,7,76500
R53,S5,P1,P1,L5-L6,2500,7,82000
R54,S5,P1,P2,L5-L6,2500,7,85500
R55,S5,P2,P1,L5-L6,2500,6,84500
R56,S5,P2,P2,L5-L6,2500,7,86500
R57,S5,P1,P1,L3-L4,2100,7,76000
R58,S5,P2,P2,L3-L4,2100,6,75000
R59,S5,P1,P2,L7-L8,1750,6,70000
R60,S5,P2,P1,L14,1200,6,68000

R61,S6,P1,P1,L1-L2,2200,7,56000
R62,S6,P1,P2,L1-L2,2200,7,59500
R63,S6,P2,P1,L1-L2,2200,6,58500
R64,S6,P2,P2,L1-L2,2200,7,60500
R65,S6,P1,P1,L5-L6,2300,7,64000
R66,S6,P1,P2,L5-L6,2300,7,67500
R67,S6,P2,P1,L5-L6,2300,6,66500
R68,S6,P2,P2,L5-L6,2300,7,68500
R69,S6,P1,P1,L3-L4,2000,7,60000
R70,S6,P2,P2,L7-L8,1750,6,57000
R71,S6,P1,P2,L11,1150,6,52000
R72,S6,P2,P1,L12,1200,6,53000

R73,S7,P1,P1,L1-L2,2300,7,50000
R74,S7,P1,P2,L1-L2,2300,7,53500
R75,S7,P2,P1,L1-L2,2300,6,52500
R76,S7,P2,P2,L1-L2,2300,7,54500
R77,S7,P1,P1,L5-L6,2300,7,59000
R78,S7,P1,P2,L5-L6,2300,7,62500
R79,S7,P2,P1,L5-L6,2300,6,61500
R80,S7,P2,P2,L5-L6,2300,7,63500
R81,S7,P1,P1,L7-L8,1750,7,46000
R82,S7,P1,P2,L7-L8,1750,7,49500
R83,S7,P2,P1,L7-L8,1750,6,48500
R84,S7,P2,P2,L7-L8,1750,7,50500

R85,S8,P1,P1,L9,1050,5,41000
R86,S8,P1,P2,L9,1050,5,44500
R87,S8,P2,P1,L9,1050,4,43500
R88,S8,P2,P2,L9,1050,5,45500
R89,S8,P1,P1,L10,900,6,42000
R90,S8,P1,P2,L11,1150,6,44000
R91,S8,P2,P2,L12,1200,6,45000
R92,S8,P2,P1,L13,850,5,40000
R93,S8,P1,P1,L14,950,6,43000
R94,S8,P1,P2,L3,900,6,42500
R95,S8,P2,P2,L4,1100,6,43500
R96,S8,P2,P1,L8,950,6,41500

R97,S9,P1,P1,L1-L2,1900,7,58000
R98,S9,P1,P2,L3-L4,1900,7,61000
R99,S9,P2,P1,L3-L4,1900,6,60000
R100,S9,P2,P2,L7-L8,1750,7,56000
R101,S9,P1,P1,L5,1300,6,52000
R102,S9,P1,P2,L6,1000,6,50500
R103,S9,P2,P1,L9,1050,5,50000
R104,S9,P2,P2,L10,900,6,51500
R105,S9,P1,P2,L11,1150,6,53000
R106,S9,P2,P1,L12,1000,6,52500
R107,S9,P1,P1,L13,850,5,48000
R108,S9,P2,P2,L14,950,6,54000

R109,S10,P1,P1,L1-L2,2100,7,64000
R110,S10,P1,P2,L1-L2,2100,7,67500
R111,S10,P2,P1,L1-L2,2100,6,66500
R112,S10,P2,P2,L1-L2,2100,7,68500
R113,S10,P1,P1,L5-L6,2100,7,72000
R114,S10,P1,P2,L5-L6,2100,7,75500
R115,S10,P2,P1,L5-L6,2100,6,74500
R116,S10,P2,P2,L5-L6,2100,7,76500
R117,S10,P1,P1,L3-L4,2000,7,66000
R118,S10,P2,P2,L7-L8,1750,6,62000
R119,S10,P1,P2,L12,1000,6,59000
R120,S10,P2,P1,L14,950,6,60500
"""

# =========================================================
# PARAMETRELER
# =========================================================

CYCLE_DAYS = 35
TIME_TOLERANCE = 2  # talep günü ±2

# idle etkisi (isteğe bağlı; küçük tutalım)
IDLE_THRESHOLD_DAYS = 5
PENALTY_PER_MISSING_IDLE_DAY = 5_000
RENT_INCOME_PER_EXTRA_IDLE_DAY = 1_000

# cezalar
P_UNMET_DEMAND = 1_000_000          # her eksik ton
P_UNCOVERED_LOCATION = 2_000_000    # servis almayan lokasyon (served==0)
P_DUPLICATE_ROUTE = 2_000_000       # aynı route id birden fazla gemide
P_CAPACITY_VIOLATION = 2_000_000    # ton bazlı
P_TIME_WINDOW_VIOLATION = 800_000   # kesişmeyen 2-stop rota seçilirse

# greedy ağırlıkları
W_NEW_LOCATION = 600.0
W_DEMAND_COVER = 2.0

# tabu ayarları
TABU_TENURE = 12
NEIGHBORHOOD_SIZE = 200
MAX_ITERS = 600
SEED = 42
IDLE_PROB = 0.05  # tabu komşulukta idle seçme ihtimali

# =========================================================
# VERİ YAPILARI
# =========================================================

@dataclass(frozen=True)
class Route:
    rid: str
    ship: str
    start_port: str
    end_port: str
    stops: Tuple[str, ...]
    demand_tons: int
    duration_days: int
    cost_usd: int

    @property
    def num_stops(self) -> int:
        return len(self.stops)


@dataclass
class Ship:
    sid: str
    capacity_tons: int


# =========================================================
# CSV PARSE + FİLTRE
# =========================================================

def parse_routes_csv(csv_text: str) -> List[Route]:
    lines = []
    for line in csv_text.splitlines():
        if not line.strip():
            continue
        lines.append(line)

    reader = csv.DictReader(io.StringIO("\n".join(lines)))
    routes: List[Route] = []
    for row in reader:
        stops = tuple(row["Stops"].strip().split("-")) if row["Stops"] else tuple()
        routes.append(
            Route(
                rid=row["Route"].strip(),
                ship=row["Ship"].strip(),
                start_port=row["StartPort"].strip(),
                end_port=row["EndPort"].strip(),
                stops=stops,
                demand_tons=int(row["DemandTons"]),
                duration_days=int(row["DurationDays"]),
                cost_usd=int(row["CostUSD"]),
            )
        )
    return routes


def filter_candidate_routes(routes: List[Route]) -> List[Route]:
    filtered = []
    for r in routes:
        if r.duration_days > 7:
            continue
        if r.num_stops > 2:
            continue
        # güvenlik: lokasyon id bilinmiyorsa ele
        if any(loc not in demand_day for loc in r.stops):
            continue
        filtered.append(r)
    return filtered


# =========================================================
# ZAMAN PENCERESİ KESİŞİMİ
# =========================================================

def window_for_loc(loc: str) -> Tuple[int, int]:
    d = demand_day[loc]
    return (d - TIME_TOLERANCE, d + TIME_TOLERANCE)


def intersection_window(locs: Tuple[str, ...]) -> Optional[Tuple[int, int]]:
    lo = -10**9
    hi = 10**9
    for loc in locs:
        a, b = window_for_loc(loc)
        lo = max(lo, a)
        hi = min(hi, b)
    return (lo, hi) if lo <= hi else None


def choose_common_day(locs: Tuple[str, ...]) -> Optional[int]:
    inter = intersection_window(locs)
    if inter is None:
        return None
    lo, hi = inter
    return (lo + hi) // 2


# =========================================================
# MALİYET
# =========================================================

def idle_effect_cost(duration_days: int) -> int:
    idle = max(0, CYCLE_DAYS - duration_days)
    if idle < IDLE_THRESHOLD_DAYS:
        missing = IDLE_THRESHOLD_DAYS - idle
        return PENALTY_PER_MISSING_IDLE_DAY * missing
    extra = idle - IDLE_THRESHOLD_DAYS
    return -RENT_INCOME_PER_EXTRA_IDLE_DAY * extra


def route_effective_cost(r: Route) -> int:
    return r.cost_usd + idle_effect_cost(r.duration_days)


# =========================================================
# SERVİS HESABI
# =========================================================

def compute_deliveries(
    assignment: Dict[str, Optional[Route]],
    total_demand_map: Dict[str, int],
) -> Tuple[Dict[str, int], int, int]:
    remaining = dict(total_demand_map)
    served = {loc: 0 for loc in total_demand_map}
    time_viol = 0

    for r in assignment.values():
        if r is None:
            continue

        if r.num_stops == 2:
            common_day = choose_common_day(r.stops)
            if common_day is None:
                time_viol += 1
                continue
        # 1-stop her zaman feasible sayılıyor

        cap = r.demand_tons
        for loc in r.stops:
            take = min(remaining[loc], cap)
            served[loc] += take
            remaining[loc] -= take
            cap -= take
            if cap <= 0:
                break

    unmet_total = sum(remaining.values())
    return served, unmet_total, time_viol


def served_locations_set(served: Dict[str, int]) -> Set[str]:
    return {loc for loc, x in served.items() if x > 0}


# =========================================================
# OBJECTIVE
# =========================================================

def evaluate_solution(
    assignment: Dict[str, Optional[Route]],
    ships: Dict[str, Ship],
    all_locations: Set[str],
) -> Tuple[float, Dict[str, float]]:
    cost = 0
    penalties = 0

    # (A) maliyet
    for r in assignment.values():
        if r is None:
            continue
        cost += route_effective_cost(r)

    # (B) duplicate route id
    used = defaultdict(int)
    for r in assignment.values():
        if r is None:
            continue
        used[r.rid] += 1
    dup = sum(v - 1 for v in used.values() if v > 1)
    if dup > 0:
        penalties += P_DUPLICATE_ROUTE * dup

    # (C) kapasite ihlali
    cap_viol = 0
    for sid, r in assignment.items():
        if r is None:
            continue
        if r.demand_tons > ships[sid].capacity_tons:
            cap_viol += (r.demand_tons - ships[sid].capacity_tons)
    if cap_viol > 0:
        penalties += P_CAPACITY_VIOLATION * cap_viol

    # (D) demand + time
    served, unmet, time_viol = compute_deliveries(assignment, total_demand)
    if unmet > 0:
        penalties += P_UNMET_DEMAND * unmet
    if time_viol > 0:
        penalties += P_TIME_WINDOW_VIOLATION * time_viol

    # (E) coverage = gerçekten servis alan lokasyon
    served_set = served_locations_set(served)
    uncovered = all_locations - served_set
    if uncovered:
        penalties += P_UNCOVERED_LOCATION * len(uncovered)

    obj = cost + penalties
    breakdown = {
        "route_plus_idle_cost": float(cost),
        "penalties": float(penalties),
        "objective": float(obj),
        "unmet_demand_total_tons": float(unmet),
        "time_intersection_violations": float(time_viol),
        "duplicate_routes": float(dup),
        "capacity_violation_tons": float(cap_viol),
        "uncovered_locations": float(len(uncovered)),
        "served_locations": float(len(served_set)),
    }
    return float(obj), breakdown


# =========================================================
# GREEDY BAŞLANGIÇ + REPAIR
# =========================================================

def greedy_score(
    r: Route,
    remaining: Dict[str, int],
    already_served: Set[str],
    ship_cap: int,
    used_route_ids: Set[str],
) -> float:
    if r.rid in used_route_ids:
        return -1e18
    if r.demand_tons > ship_cap:
        return -1e18
    if r.num_stops == 2 and choose_common_day(r.stops) is None:
        return -1e18

    # ne kadar demand kapatır?
    cap = r.demand_tons
    demand_cover = 0
    for loc in r.stops:
        take = min(remaining[loc], cap)
        demand_cover += take
        cap -= take
        if cap <= 0:
            break

    new_locs = sum(1 for loc in r.stops if (loc not in already_served and remaining[loc] > 0))
    benefit = W_DEMAND_COVER * demand_cover + W_NEW_LOCATION * new_locs
    inc_cost = route_effective_cost(r)

    return benefit - 0.001 * inc_cost


def build_initial_solution(
    routes_by_ship: Dict[str, List[Route]],
    ships: Dict[str, Ship],
    all_locations: Set[str],
) -> Dict[str, Optional[Route]]:
    assignment = {sid: None for sid in ships.keys()}
    remaining = dict(total_demand)
    used_route_ids: Set[str] = set()
    served_so_far: Set[str] = set()

    ship_ids = list(ships.keys())
    random.shuffle(ship_ids)

    for sid in ship_ids:
        best_r = None
        best_s = -1e18

        for r in routes_by_ship[sid]:
            s = greedy_score(r, remaining, served_so_far, ships[sid].capacity_tons, used_route_ids)
            if s > best_s:
                best_s = s
                best_r = r

        if best_r is not None and best_s > 0:
            assignment[sid] = best_r
            used_route_ids.add(best_r.rid)

            # remaining düş
            cap = best_r.demand_tons
            for loc in best_r.stops:
                take = min(remaining[loc], cap)
                remaining[loc] -= take
                cap -= take
                if take > 0:
                    served_so_far.add(loc)
                if cap <= 0:
                    break

    # REPAIR: servis almayan lokasyonları kapatmaya zorla
    served, _, _ = compute_deliveries(assignment, total_demand)
    uncovered = all_locations - served_locations_set(served)

    for loc in list(uncovered):
        # bu loc'u servis ettirebilecek uygun bir rota bul, bir gemiye koy
        best_choice = None  # (delta_obj, sid, route)
        base_obj, _ = evaluate_solution(assignment, ships, all_locations)

        for sid in ships.keys():
            for r in routes_by_ship[sid]:
                if loc not in r.stops:
                    continue
                if r.demand_tons > ships[sid].capacity_tons:
                    continue
                if r.num_stops == 2 and choose_common_day(r.stops) is None:
                    continue
                if r.rid in {rr.rid for rr in assignment.values() if rr is not None and rr != assignment[sid]}:
                    continue

                trial = dict(assignment)
                trial[sid] = r
                obj, _ = evaluate_solution(trial, ships, all_locations)
                delta = obj - base_obj
                if best_choice is None or delta < best_choice[0]:
                    best_choice = (delta, sid, r)

        if best_choice is not None:
            _, sid, r = best_choice
            assignment[sid] = r

    return assignment


# =========================================================
# TABU SEARCH
# =========================================================

def tabu_search(
    routes_by_ship: Dict[str, List[Route]],
    ships: Dict[str, Ship],
    all_locations: Set[str],
    max_iters: int = MAX_ITERS,
    neighborhood_size: int = NEIGHBORHOOD_SIZE,
    tabu_tenure: int = TABU_TENURE,
    seed: int = SEED,
) -> Tuple[Dict[str, Optional[Route]], float, Dict[str, float]]:
    random.seed(seed)

    current = build_initial_solution(routes_by_ship, ships, all_locations)
    best = dict(current)
    best_obj, best_break = evaluate_solution(best, ships, all_locations)

    tabu: Dict[Tuple[str, str], int] = {}
    ship_ids = list(ships.keys())

    for it in range(1, max_iters + 1):
        # tabu süre azalt
        expired = [k for k, t in tabu.items() if t <= 1]
        for k in expired:
            tabu.pop(k, None)
        for k in list(tabu.keys()):
            tabu[k] -= 1

        candidates: List[Tuple[float, Dict[str, Optional[Route]], Tuple[str, str]]] = []

        for _ in range(neighborhood_size):
            sid = random.choice(ship_ids)

            # IDLE olasılığı düşük
            if random.random() < IDLE_PROB:
                proposed = None
                move_key = (sid, "NONE")
            else:
                proposed = random.choice(routes_by_ship[sid])
                move_key = (sid, proposed.rid)

            neigh = dict(current)
            neigh[sid] = proposed
            obj, _ = evaluate_solution(neigh, ships, all_locations)

            # tabu + aspirasyon
            if move_key in tabu and obj >= best_obj:
                continue

            candidates.append((obj, neigh, move_key))

        if not candidates:
            continue

        candidates.sort(key=lambda x: x[0])
        next_obj, next_sol, move_key = candidates[0]

        current = next_sol
        tabu[move_key] = tabu_tenure

        if next_obj < best_obj:
            best_obj = next_obj
            best = dict(current)
            best_break = evaluate_solution(best, ships, all_locations)[1]

        if it % 50 == 0:
            print(f"[iter {it:>3}] best objective = {best_obj:,.0f} | served_locs={int(best_break['served_locations'])}/14")

    return best, best_obj, best_break


# =========================================================
# RAPORLAMA
# =========================================================

def explain_solution(best: Dict[str, Optional[Route]], ships: Dict[str, Ship]) -> None:
    served, unmet, time_viol = compute_deliveries(best, total_demand)
    served_set = served_locations_set(served)

    print("\n=== ÖZET ===")
    print(f"Servis edilen lokasyon: {len(served_set)} / {len(total_demand)}")
    print(f"Karşılanmayan toplam talep (ton): {unmet}")
    print(f"Zaman-kesişim ihlali: {time_viol}")

    print("\n--- Lokasyon bazlı servis ---")
    for loc in sorted(total_demand.keys()):
        w = window_for_loc(loc)
        print(f"{loc}: served={served[loc]:>5} / demand={total_demand[loc]:>5}  day={demand_day[loc]} window={w}")

    print("\n--- Gemi -> Rota ---")
    for sid in sorted(best.keys()):
        r = best[sid]
        if r is None:
            print(f"{sid}: IDLE")
        else:
            print(f"{sid}: {r.rid} | {r.start_port}->{r.end_port} | Stops={'-'.join(r.stops)} | Tons={r.demand_tons} | Days={r.duration_days} | Cost={r.cost_usd}")


# =========================================================
# MAIN
# =========================================================

def main():
    random.seed(SEED)

    all_locations = set(total_demand.keys())
    ships: Dict[str, Ship] = {sid: Ship(sid=sid, capacity_tons=cap) for sid, cap in ships_data.items()}

    routes = filter_candidate_routes(parse_routes_csv(routes_csv))

    routes_by_ship: Dict[str, List[Route]] = defaultdict(list)
    for r in routes:
        routes_by_ship[r.ship].append(r)

    # güvenlik: geminin hiç rotası yoksa patlamasın
    for sid in ships.keys():
        if sid not in routes_by_ship or not routes_by_ship[sid]:
            raise RuntimeError(f"{sid} için rota yok! CSV'de {sid} satırları olmalı.")

    best, best_obj, breakdown = tabu_search(routes_by_ship, ships, all_locations)

    print("\n================ OBJECTIVE BREAKDOWN ================")
    for k, v in breakdown.items():
        print(f"{k}: {v:,.0f}")

    explain_solution(best, ships)
    print(f"\n✅ FINAL OBJECTIVE (MIN): {best_obj:,.0f}")


if __name__ == "__main__":
    main()
    