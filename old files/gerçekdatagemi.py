from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set

import pandas as pd


# ============================================================
# 1) LOAD DATA FROM EXCEL
# ============================================================

DATASET_XLSX = "ENS001Dataset.xlsx"   # aynı klasördeyse böyle; değilse full path ver

def norm_stop(v) -> Optional[int]:
    if pd.isna(v):
        return None
    if isinstance(v, str):
        t = v.strip().lower()
        if t == "" or "yok" in t:
            return None
    try:
        return int(v)
    except Exception:
        return None

@dataclass(frozen=True)
class Option:
    route_id: str              # e.g. 01061002_Gemi1
    route_code: str            # e.g. 01061002
    ship: str                  # e.g. Gemi1
    stops: Tuple[int, ...]     # e.g. (6,10)
    duration: int              # e.g. max(stop days)
    cost: float                # from Sayfa1


def load_options_from_excel(xlsx_path: str) -> Tuple[List[str], Dict[str, List[Option]], List[int]]:
    sayfa1 = pd.read_excel(xlsx_path, sheet_name="Sayfa1")
    stop1 = pd.read_excel(xlsx_path, sheet_name="1.Stop")
    stop2 = pd.read_excel(xlsx_path, sheet_name="2.Stop")
    stop3 = pd.read_excel(xlsx_path, sheet_name="3.Stop")

    ship_cols = [c for c in sayfa1.columns if str(c).startswith("Gemi") or str(c).startswith("Kiralik")]

    # route_code column in your file:
    ROUTE_COL = "Unnamed: 5"
    START_COL = "Unnamed: 1"
    S1_COL    = "Unnamed: 2"
    S2_COL    = "Unnamed: 3"
    END_COL   = "Unnamed: 4"

    options_by_ship: Dict[str, List[Option]] = {s: [] for s in ship_cols}
    universe: Set[int] = set()

    # map: route_code -> row index (works because sheets align row-by-row)
    code_to_idx = {}
    for i, row in sayfa1.iterrows():
        code = row.get(ROUTE_COL)
        if pd.isna(code):
            continue
        code_to_idx[str(code)] = i

    def get_day(df: pd.DataFrame, code: str, ship: str) -> Optional[int]:
        i = code_to_idx.get(code)
        if i is None:
            return None
        v = df.at[i, ship]
        if pd.isna(v):
            return None
        try:
            return int(v)
        except Exception:
            return None

    for i, row in sayfa1.iterrows():
        code = row.get(ROUTE_COL)
        if pd.isna(code):
            continue
        code = str(code)

        start = norm_stop(row.get(START_COL))
        s1    = norm_stop(row.get(S1_COL))
        s2    = norm_stop(row.get(S2_COL))
        end   = norm_stop(row.get(END_COL))

        stops = tuple([x for x in [s1, s2] if x is not None and x not in (start, end, 1, 2)])
        universe |= set(stops)

        for ship in ship_cols:
            cost = row.get(ship)
            if pd.isna(cost):
                continue  # this ship can't do this route

            d1 = get_day(stop1, code, ship)
            d2 = get_day(stop2, code, ship)
            d3 = get_day(stop3, code, ship)
            duration = max([d for d in [d1, d2, d3] if d is not None] + [0])
            if duration <= 0:
                continue

            rid = f"{code}_{ship}"
            options_by_ship[ship].append(
                Option(route_id=rid, route_code=code, ship=ship, stops=stops, duration=duration, cost=float(cost))
            )

    return ship_cols, options_by_ship, sorted(universe)


# ============================================================
# 2) EVALUATE (MIN COST + COVERAGE PENALTY)
# ============================================================

BIG_M = 1_000_000  # penalty per uncovered stop

def evaluate_solution(sol: Dict[str, str], options_by_ship: Dict[str, List[Option]], universe: List[int]) -> Tuple[float, Dict]:
    covered: Set[int] = set()
    total_cost = 0.0
    duration_ok = True

    # quick lookup
    opt_map: Dict[Tuple[str, str], Option] = {}
    for ship, opts in options_by_ship.items():
        for o in opts:
            opt_map[(ship, o.route_id)] = o

    for ship, route_id in sol.items():
        if route_id == "NONE":
            continue
        o = opt_map.get((ship, route_id))
        if o is None:
            # invalid choice
            return 1e18, {"error": f"{ship} picked unknown option {route_id}"}
        total_cost += o.cost
        covered |= set(o.stops)
        if o.duration > 35:
            duration_ok = False

    missing = set(universe) - covered
    penalty = BIG_M * len(missing)
    if not duration_ok:
        penalty += BIG_M

    obj = total_cost + penalty
    dbg = {
        "total_cost": total_cost,
        "missing": sorted(missing),
        "duration_ok": duration_ok,
        "covered_count": len(covered),
        "universe_count": len(universe),
    }
    return obj, dbg


# ============================================================
# 3) GREEDY START + TABU SEARCH
# ============================================================

def build_greedy_initial(ships: List[str], options_by_ship: Dict[str, List[Option]], universe: List[int]) -> Dict[str, str]:
    sol = {s: "NONE" for s in ships}
    uncovered = set(universe)

    # owned first, leased later
    ship_order = [s for s in ships if s.startswith("Gemi")] + [s for s in ships if s.startswith("Kiralik")]

    for ship in ship_order:
        if not uncovered:
            break
        best = None
        for o in options_by_ship[ship]:
            new = uncovered & set(o.stops)
            if not new:
                continue
            score = o.cost / len(new)  # cost efficiency
            if best is None or score < best[0]:
                best = (score, o)
        if best:
            sol[ship] = best[1].route_id
            uncovered -= set(best[1].stops)

    return sol

def tabu_search(
    ships: List[str],
    options_by_ship: Dict[str, List[Option]],
    universe: List[int],
    max_iters: int = 2000,
    neighborhood: int = 200,
    tabu_tenure: int = 25,
    seed: int = 11,
):
    random.seed(seed)

    # include NONE option
    options_with_none: Dict[str, List[str]] = {}
    for ship in ships:
        options_with_none[ship] = ["NONE"] + [o.route_id for o in options_by_ship[ship]]

    current = build_greedy_initial(ships, options_by_ship, universe)
    best = dict(current)

    best_obj, best_dbg = evaluate_solution(best, options_by_ship, universe)
    cur_obj = best_obj

    tabu: Dict[Tuple[str, str], int] = {}

    for it in range(1, max_iters + 1):
        # decay tabu
        for k in list(tabu.keys()):
            tabu[k] -= 1
            if tabu[k] <= 0:
                del tabu[k]

        best_cand = None
        best_cand_obj = None
        best_cand_dbg = None
        best_move = None

        for _ in range(neighborhood):
            ship = random.choice(ships)
            rid_new = random.choice(options_with_none[ship])
            if rid_new == current[ship]:
                continue

            move = (ship, rid_new)
            cand = dict(current)
            cand[ship] = rid_new

            obj, dbg = evaluate_solution(cand, options_by_ship, universe)

            # tabu unless aspiration (better than best)
            if move in tabu and obj >= best_obj:
                continue

            if best_cand is None or obj < best_cand_obj:
                best_cand = cand
                best_cand_obj = obj
                best_cand_dbg = dbg
                best_move = move

        if best_cand is None:
            break

        current = best_cand
        cur_obj = best_cand_obj
        tabu[best_move] = tabu_tenure

        if cur_obj < best_obj:
            best = dict(current)
            best_obj = cur_obj
            best_dbg = best_cand_dbg

        if it % 200 == 0:
            print(f"iter={it} best_obj={best_obj:.2f} cost={best_dbg['total_cost']:.2f} missing={len(best_dbg['missing'])}")

    return best, best_obj, best_dbg


def pretty_print(best_sol: Dict[str, str], options_by_ship: Dict[str, List[Option]]):
    # build lookup
    opt_map = {}
    for ship, opts in options_by_ship.items():
        for o in opts:
            opt_map[(ship, o.route_id)] = o

    print("\n=== BEST SOLUTION ===")
    for ship in sorted(best_sol.keys()):
        rid = best_sol[ship]
        if rid == "NONE":
            print(f"{ship}: NONE")
        else:
            o = opt_map[(ship, rid)]
            print(f"{ship}: {o.route_code}  stops={o.stops}  duration={o.duration}  cost={o.cost}")


def main():
    ships, options_by_ship, universe = load_options_from_excel(DATASET_XLSX)

    best_sol, best_obj, dbg = tabu_search(
        ships, options_by_ship, universe,
        max_iters=2000, neighborhood=220, tabu_tenure=25, seed=11
    )

    print("\nBEST OBJ:", best_obj)
    print("Total cost:", dbg["total_cost"])
    print("Missing stops:", dbg["missing"])
    print("Coverage:", dbg["covered_count"], "/", dbg["universe_count"])
    pretty_print(best_sol, options_by_ship)

if __name__ == "__main__":
    main()
