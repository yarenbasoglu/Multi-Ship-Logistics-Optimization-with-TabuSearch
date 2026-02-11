import csv
import io
import random

# =========================
# 1) ROUTES CSV (senin verdiğin)
# =========================
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

# =========================
# 2) DEMO DATA (Demand / Region / Ships / Depots)
# =========================

ships = ["S1","S2","S3","S4","S5","S6","S7","S8","S9","S10"]

ownership = {
    "S1":"owned","S2":"owned","S3":"owned","S4":"owned","S5":"owned","S6":"owned","S7":"owned",
    "S8":"leased","S9":"leased","S10":"leased"
}

ship_capacity = {
    "S1":2500,"S2":2300,"S3":2100,"S4":1900,"S5":2600,"S6":2400,"S7":2300,"S8":1300,"S9":2000,"S10":2200
}

north = set(["L1","L2","L3","L4","L5","L6","L7","L8"])
south = set(["L9","L10","L11","L12","L13","L14"])

# demand[L] = (product, qty, day)
demand = {
    "L1":("gasoline",1100,5),
    "L2":("gasoline",1200,7),
    "L3":("diesel",900,12),
    "L4":("diesel",1000,10),
    "L5":("gasoline",1300,18),
    "L6":("gasoline",1000,20),
    "L7":("diesel",900,26),
    "L8":("diesel",850,27),
    "L9":("gasoline",1050,6),
    "L10":("diesel",900,14),
    "L11":("gasoline",1150,16),
    "L12":("diesel",1000,17),
    "L13":("diesel",850,22),
    "L14":("gasoline",950,24),
}
locations = list(demand.keys())

depots = {
    "D_N":{"loc":"L3","region":"North"},
    "D_S1":{"loc":"L10","region":"South"},
    "D_S2":{"loc":"L12","region":"South"}
}

I0 = {
    "D_N":{"gasoline":2000,"diesel":2000},
    "D_S1":{"gasoline":2500,"diesel":1500},
    "D_S2":{"gasoline":1500,"diesel":2500},
}
Target = {
    "D_N":{"gasoline":1500,"diesel":1500},
    "D_S1":{"gasoline":1800,"diesel":1200},
    "D_S2":{"gasoline":1200,"diesel":1800},
}

south_to_depot = {
    "L9":"D_S1","L10":"D_S1","L11":"D_S1",
    "L12":"D_S2","L13":"D_S2","L14":"D_S2"
}

# =========================
# 3) Penalties / Weights
# =========================
BIG_UNMET = 1_000_000
BIG_COVER = 5_000_000
BIG_CAP   = 2_000_000
BIG_REGION= 2_000_000
TIME_MISMATCH = 50_000
W_INV = 10
PENALTY_IDLE = 10_000
BONUS_RENT = 15_000

# =========================
# 4) Parse routes
# =========================
def parse_routes(text):
    routes = []
    reader = csv.DictReader(io.StringIO(text))
    for row in reader:
        if not row["Route"]:
            continue
        r = {}
        r["Route"] = row["Route"].strip()
        r["Ship"] = row["Ship"].strip()
        r["StartPort"] = row["StartPort"].strip()
        r["EndPort"] = row["EndPort"].strip()
        r["Stops"] = row["Stops"].strip()
        r["StopsList"] = r["Stops"].split("-")
        r["Capacity"] = int(row["DemandTons"])
        r["Duration"] = int(row["DurationDays"])
        r["Cost"] = int(row["CostUSD"])
        routes.append(r)
    return routes

routes = parse_routes(routes_csv)

routes_by_ship = {s: [] for s in ships}
for i, r in enumerate(routes):
    routes_by_ship[r["Ship"]].append(i)

def region_of_stops(stops_list):
    if all(x in north for x in stops_list):
        return "North"
    if all(x in south for x in stops_list):
        return "South"
    return "Mixed"

for r in routes:
    r["Region"] = region_of_stops(r["StopsList"])

# =========================
# 5) Day alignment (GENEL: tüm çiftler için)
# =========================
def align_day_if_possible(L1, L2, demand_dict):
    """
    Eğer ±2 aralıkları kesişiyorsa (True, ortak_gün) döner.
    Kesişim yoksa (False, None).
    """
    if L1 not in demand_dict or L2 not in demand_dict:
        return False, None

    d1 = demand_dict[L1][2]
    d2 = demand_dict[L2][2]

    a1, b1 = d1 - 2, d1 + 2
    a2, b2 = d2 - 2, d2 + 2

    low = max(a1, a2)
    high = min(b1, b2)
    if low > high:
        return False, None

    # ortak gün seçimi: orta noktaya yakın olsun, kesişim içine kırp
    t_star = int(round((d1 + d2) / 2))
    if t_star < low:
        t_star = low
    if t_star > high:
        t_star = high
    return True, t_star

# =========================
# 6) Evaluate
# solution: ship -> route_index or None
# =========================
def evaluate(solution):
    route_cost = 0
    cap_pen = 0
    region_pen = 0
    time_pen = 0
    cover_pen = 0
    unmet_pen = 0
    inv_pen = 0
    idle_effect = 0

    served = set()
    delivered_qty = {L: 0 for L in locations}

    # depo stok kopya
    inv = {d: {"gasoline": I0[d]["gasoline"], "diesel": I0[d]["diesel"]} for d in depots}

    # seçilen 2-stop rotalarda hangi ortak gün seçildiğini kaydedelim (rapor için)
    aligned_days = []  # (ship, route, L1, L2, t_star) veya mismatch

    for s in ships:
        ridx = solution.get(s, None)
        if ridx is None:
            used = 0
        else:
            r = routes[ridx]
            route_cost += r["Cost"]

            # kapasite
            if r["Capacity"] > ship_capacity[s]:
                cap_pen += BIG_CAP

            # karışık bölge istemiyoruz
            if r["Region"] == "Mixed":
                region_pen += BIG_REGION

            # servis edilen lokasyonlar
            for L in r["StopsList"]:
                served.add(L)
                # basit demo: rota L'yi içeriyorsa talebi tam karşılar
                delivered_qty[L] = demand[L][1]

            # 2-stop ise ±2 aynı güne çekilebilir mi? (GENEL)
            if len(r["StopsList"]) == 2:
                A, B = r["StopsList"][0], r["StopsList"][1]
                ok, t_star = align_day_if_possible(A, B, demand)
                if not ok:
                    time_pen += TIME_MISMATCH
                    aligned_days.append((s, r["Route"], A, B, None))
                else:
                    aligned_days.append((s, r["Route"], A, B, t_star))

            used = r["Duration"]

        # idle kira/ceza
        idle = 35 - used
        if ownership[s] == "owned":
            if idle >= 5:
                idle_effect -= BONUS_RENT
            else:
                idle_effect += PENALTY_IDLE
        else:
            if idle < 5:
                idle_effect += PENALTY_IDLE

    # coverage cezası
    for L in locations:
        if L not in served:
            cover_pen += BIG_COVER

    # unmet + depo stok düşümü (basit)
    for L in locations:
        prod, qty, _ = demand[L]
        got = delivered_qty[L]

        if got < qty:
            unmet_pen += (qty - got) * BIG_UNMET

        # stoktan düş: talep karşılandıysa
        if got >= qty:
            if L in north:
                depot_id = "D_N"
            else:
                depot_id = south_to_depot[L]
            inv[depot_id][prod] -= qty

    # envanter hedef cezası (cycle sonu)
    for d in depots:
        for prod in ["gasoline","diesel"]:
            diff = inv[d][prod] - Target[d][prod]
            if diff < 0:
                diff = -diff
            inv_pen += diff * W_INV

    total = route_cost + cap_pen + region_pen + time_pen + cover_pen + unmet_pen + inv_pen + idle_effect

    breakdown = {
        "route_cost": route_cost,
        "idle_effect": idle_effect,
        "time_pen": time_pen,
        "cap_pen": cap_pen,
        "region_pen": region_pen,
        "cover_pen": cover_pen,
        "unmet_pen": unmet_pen,
        "inv_pen": inv_pen,
        "total": total,
        "aligned_days": aligned_days  # ekstra bilgi
    }
    return total, breakdown

# =========================
# 7) Initial solution: her ship için en ucuz rota
# =========================
def initial_solution():
    sol = {}
    for s in ships:
        best = None
        best_cost = None
        for ridx in routes_by_ship[s]:
            c = routes[ridx]["Cost"]
            if best_cost is None or c < best_cost:
                best_cost = c
                best = ridx
        sol[s] = best
    return sol

# =========================
# 8) Neighbor: 1 geminin rotasını değiştir / boş bırak
# =========================
def random_neighbor(sol):
    newsol = dict(sol)
    s = random.choice(ships)

    if random.random() < 0.15:
        newsol[s] = None
        return newsol, (s, None)

    candidates = routes_by_ship[s]
    if not candidates:
        newsol[s] = None
        return newsol, (s, None)

    new_r = random.choice(candidates)
    newsol[s] = new_r
    return newsol, (s, new_r)

# =========================
# 9) Tabu Search (basit)
# =========================
def tabu_search(max_iters=2500, tabu_tenure=12, seed=7):
    random.seed(seed)

    current = initial_solution()
    cur_cost, _ = evaluate(current)

    best = dict(current)
    best_cost, best_break = evaluate(best)

    tabu = {}  # move -> remaining

    for _it in range(max_iters):
        # tabu azalt
        expired = []
        for mv in tabu:
            tabu[mv] -= 1
            if tabu[mv] <= 0:
                expired.append(mv)
        for mv in expired:
            del tabu[mv]

        best_candidate = None
        best_candidate_cost = None
        best_candidate_move = None

        # birkaç komşu dene
        for _ in range(40):
            cand, move = random_neighbor(current)
            cost, _ = evaluate(cand)

            is_tabu = (move in tabu)
            # aspiration
            if is_tabu and cost >= best_cost:
                continue

            if best_candidate_cost is None or cost < best_candidate_cost:
                best_candidate = cand
                best_candidate_cost = cost
                best_candidate_move = move

        if best_candidate is None:
            continue

        current = best_candidate
        cur_cost = best_candidate_cost
        tabu[best_candidate_move] = tabu_tenure

        if cur_cost < best_cost:
            best = dict(current)
            best_cost, best_break = evaluate(best)

    return best, best_cost, best_break

# =========================
# 10) Run
# =========================
if __name__ == "__main__":
    best_sol, best_cost, br = tabu_search()

    print("=== BEST SOLUTION ===")
    print("Total Cost:", best_cost)

    print("\n--- Breakdown ---")
    for k in ["route_cost","idle_effect","time_pen","cap_pen","region_pen","cover_pen","unmet_pen","inv_pen","total"]:
        print(f"{k}: {br[k]}")

    print("\n--- Selected Routes ---")
    for s in ships:
        ridx = best_sol.get(s, None)
        if ridx is None:
            print(s, "-> IDLE")
        else:
            r = routes[ridx]
            print(s, "->", r["Route"], r["StartPort"], "->", r["EndPort"],
                  "| Stops:", r["Stops"], "| Dur:", r["Duration"], "| Cost:", r["Cost"])

    print("\n--- 2-Stop Day Alignment (ALL pairs) ---")
    for (s, route_id, A, B, t_star) in br["aligned_days"]:
        if t_star is None:
            print(f"{s} {route_id}: {A}-{B}  ->  ±2 KESİŞİM YOK  (penalty)")
        else:
            da = demand[A][2]
            db = demand[B][2]
            print(f"{s} {route_id}: {A}(d={da}) & {B}(d={db})  ->  ortak gün t*={t_star}")