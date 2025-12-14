
import csv
import io
import random
import math
from collections import defaultdict


routes_csv = """Route,Ship,StartPort,EndPort,Stops,DemandTons,DurationDays,CostUSD
R1,S1,P1,P1,L1,1100,4,28000
R2,S1,P1,P1,L1-L2,2000,7,49000
R3,S1,P1,P2,L3-L7,2000,7,49000
R4,S1,P1,P2,L5-L4,2000,9,63000
R5,S1,P1,P1,L5-L6,2000,9,63000
R6,S1,P2,P2,L1,1100,4,28000
R7,S1,P2,P2,L1-L2,2000,7,49000
R8,S1,P2,P1,L3-L7,2000,7,49000
R9,S1,P2,P1,L5-L4,2000,9,63000
R10,S1,P2,P2,L5-L6,2000,9,63000
R11,S2,P1,P1,L1,1100,4,34000
R12,S2,P1,P1,L1-L2,2000,7,59500
R13,S2,P1,P2,L3-L7,2300,7,59500
R14,S2,P1,P2,L5-L4,2500,9,76500
R15,S2,P1,P1,L5-L6,2500,9,76500
R16,S2,P2,P2,L1,1100,4,34000
R17,S2,P2,P2,L1-L2,2000,7,59500
R18,S2,P2,P1,L3-L7,2300,7,59500
R19,S2,P2,P1,L5-L4,2500,9,76500
R20,S2,P2,P2,L5-L6,2500,9,76500
R21,S3,P1,P1,L1,1100,4,40000
R22,S3,P1,P1,L1-L2,2000,7,70000
R23,S3,P1,P2,L3-L7,2300,7,70000
R24,S3,P1,P2,L5-L4,3000,9,90000
R25,S3,P1,P1,L5-L6,2900,9,90000
R26,S3,P2,P2,L1,1100,4,40000
R27,S3,P2,P2,L1-L2,2000,7,70000
R28,S3,P2,P1,L3-L7,2300,7,70000
R29,S3,P2,P1,L5-L4,3000,9,90000
R30,S3,P2,P2,L5-L6,2900,9,90000
R31,S4,P1,P1,L1,1100,4,48000
R32,S4,P1,P1,L1-L2,2000,7,84000
R33,S4,P1,P2,L3-L7,2300,7,84000
R34,S4,P1,P2,L5-L4,3000,9,108000
R35,S4,P1,P1,L5-L6,2900,9,108000
R36,S4,P2,P2,L1,1100,4,48000
R37,S4,P2,P2,L1-L2,2000,7,84000
R38,S4,P2,P1,L3-L7,2300,7,84000
R39,S4,P2,P1,L5-L4,3000,9,108000
R40,S4,P2,P2,L5-L6,2900,9,108000
R41,S5,P1,P1,L1,1100,4,54000
R42,S5,P1,P1,L1-L2,2000,7,94500
R43,S5,P1,P2,L3-L7,2300,7,94500
R44,S5,P1,P2,L5-L4,3000,9,121500
R45,S5,P1,P1,L5-L6,2900,9,121500
R46,S5,P2,P2,L1,1100,4,54000
R47,S5,P2,P2,L1-L2,2000,7,94500
R48,S5,P2,P1,L3-L7,2300,7,94500
R49,S5,P2,P1,L5-L4,3000,9,121500
R50,S5,P2,P2,L5-L6,2900,9,121500
R51,S6,P1,P1,L1,1100,4,30000
R52,S6,P1,P1,L1-L2,2000,7,52500
R53,S6,P1,P2,L3-L7,2300,7,52500
R54,S6,P1,P2,L5-L4,2400,9,67500
R55,S6,P1,P1,L5-L6,2400,9,67500
R56,S6,P2,P2,L1,1100,4,30000
R57,S6,P2,P2,L1-L2,2000,7,52500
R58,S6,P2,P1,L3-L7,2300,7,52500
R59,S6,P2,P1,L5-L4,2400,9,67500
R60,S6,P2,P2,L5-L6,2400,9,67500
R61,S7,P1,P1,L1,1100,4,22000
R62,S7,P1,P1,L1-L2,2000,7,38500
R63,S7,P1,P2,L3-L7,1400,7,38500
R64,S7,P1,P2,L5-L4,1400,9,49500
R65,S7,P1,P1,L5-L6,1400,9,49500
R66,S7,P2,P2,L1,1100,4,22000
R67,S7,P2,P2,L1-L2,2000,7,38500
R68,S7,P2,P1,L3-L7,1400,7,38500
R69,S7,P2,P1,L5-L4,1400,9,49500
R70,S7,P2,P2,L5-L6,1400,9,49500
R71,S8,P1,P1,L1,1100,4,24000
R72,S8,P1,P1,L1-L2,2000,7,42000
R73,S8,P1,P2,L3-L7,1800,7,42000
R74,S8,P1,P2,L5-L4,1800,9,54000
R75,S8,P1,P1,L5-L6,1800,9,54000
R76,S8,P2,P2,L1,1100,4,24000
R77,S8,P2,P2,L1-L2,2000,7,42000
R78,S8,P2,P1,L3-L7,1800,7,42000
R79,S8,P2,P1,L5-L4,1800,9,54000
R80,S8,P2,P2,L5-L6,1800,9,54000
R81,S9,P1,P1,L1,1100,4,30000
R82,S9,P1,P1,L1-L2,2000,7,52500
R83,S9,P1,P2,L3-L7,2200,7,52500
R84,S9,P1,P2,L5-L4,2200,9,67500
R85,S9,P1,P1,L5-L6,2200,9,67500
R86,S9,P2,P2,L1,1100,4,30000
R87,S9,P2,P2,L1-L2,2000,7,52500
R88,S9,P2,P1,L3-L7,2200,7,52500
R89,S9,P2,P1,L5-L4,2200,9,67500
R90,S9,P2,P2,L5-L6,2200,9,67500
R91,S10,P1,P1,L1,1100,4,36000
R92,S10,P1,P1,L1-L2,2000,7,63000
R93,S10,P1,P2,L3-L7,2300,7,63000
R94,S10,P1,P2,L5-L4,2900,9,81000
R95,S10,P1,P1,L5-L6,2900,9,81000
R96,S10,P2,P2,L1,1100,4,36000
R97,S10,P2,P2,L1-L2,2000,7,63000
R98,S10,P2,P1,L3-L7,2300,7,63000
R99,S10,P2,P1,L5-L4,2900,9,81000
R100,S10,P2,P2,L5-L6,2900,9,81000
"""

routes = []
reader = csv.DictReader(io.StringIO(routes_csv.strip()))
for row in reader:
    row["DemandTons"] = int(row["DemandTons"])
    row["DurationDays"] = int(row["DurationDays"])
    row["CostUSD"] = int(row["CostUSD"])
    routes.append(row)

total_demand = {
    "L1": 1100,
    "L2": 900,
    "L3": 1500,
    "L4": 1400,
    "L5": 1700,
    "L6": 1200,
    "L7": 800,
    "L8": 1000,
    "L9": 1100,
    "L10": 1000,
}

# %60 benzin, %40 dizel
demand_gas = {}
demand_diesel = {}
for loc, d in total_demand.items():
    gas = int(round(d * 0.6))
    diesel = d - gas
    demand_gas[loc] = gas
    demand_diesel[loc] = diesel

# Fiyatlar (örnek)
price_gas = {
    "L1": 950,
    "L2": 970,
    "L3": 1000,
    "L4": 1020,
    "L5": 1050,
    "L6": 980,
    "L7": 990,
    "L8": 1010,
    "L9": 1030,
    "L10": 1040,
}

price_diesel = {
    "L1": 850,
    "L2": 870,
    "L3": 880,
    "L4": 900,
    "L5": 920,
    "L6": 860,
    "L7": 870,
    "L8": 890,
    "L9": 905,
    "L10": 915,
}

north = {"L1", "L2", "L3", "L7", "L8", "L9"}
south = {"L4", "L5", "L6", "L10"}
depots = {"L1", "L3", "L5"}

# rota hangi bölgede?
for r in routes:
    stops = r["Stops"].split("-")  # "L3-L7" -> ["L3", "L7"]
    r["StopsList"] = stops
    if all(s in north for s in stops):
        r["Region"] = "North"
    elif all(s in south for s in stops):
        r["Region"] = "South"
    else:
        r["Region"] = "Mixed"

# gemilere göre rotalar
routes_by_ship = defaultdict(list)
for idx, r in enumerate(routes):
    routes_by_ship[r["Ship"]].append(idx)

ships = sorted(routes_by_ship.keys())
HORIZON = 35  # 35 gün toplam süre sınırı (her gemi için)

#her gemi toplam kaç gün çalışıyor
#Gemi S1 → Rota 5 (9 gün) + Rota 2 (7 gün) + Rota 1 (4 gün) = 20 gün
def ship_total_duration(assignments, ship):
    """
    assignments[ship] = [(route_idx, product), ...]
    """
    trips = assignments.get(ship, [])
    return sum(routes[ridx]["DurationDays"] for ridx, _ in trips)

#objective = (toplam maliyet + cezalar) – toplam gelir  
#objective ne kadar düşükse çözüm o kadar iyi
def evaluate_solution(
    assignments,   # assignments = {"S1": [(ridx1,"gas"), (ridx2,"diesel")], "S2": [], ...}
    shortage_penalty_per_ton=700,
    overstock_penalty_per_ton=70,
    depot_shortage_penalty_per_ton=400,
    depot_overstock_penalty_per_ton=40,
):
    # Maliyet parçaları için breakdown
    breakdown = {
        "route_cost": 0.0, # toplam rota maliyeti
        "duration_violation_penalty": 0.0, # gemi süresi HORIZON'u aştığında
        "idle_bonus_or_penalty": 0.0, # boşta kalma süresine göre
        "shortage_penalty": 0.0, # talep yetersizliği cezası
        "overstock_penalty": 0.0,   # fazla stok cezası
        "depot_shortage_penalty": 0.0, # depo yetersizliği cezası
        "depot_overstock_penalty": 0.0,     # depo fazla stok cezası
    }

    total_revenue = 0.0 #satılan ürünlerden gelen toplam gelir

    # hangi lokasyona ne kadar ürün gitti?
    delivered_gas = {loc: 0.0 for loc in total_demand.keys()}
    delivered_diesel = {loc: 0.0 for loc in total_demand.keys()}

    # her gemi için birden fazla rota/sefer olabilir
    for ship in ships:
        trips = assignments.get(ship, []) # gemiye atanmış seferler alınır
        total_duration = 0

        for (ridx, product) in trips:
            r = routes[ridx]
            breakdown["route_cost"] += r["CostUSD"] # rota maliyeti ekle
            total_duration += r["DurationDays"] # toplam süreyi güncelle

            stops = r["StopsList"] #rotanın durak listesi

            # Bu rota üzerindeki lokasyonların talep toplamı
            total_weight = 0.0
            for s in stops:
                total_weight += demand_gas[s] + demand_diesel[s]
            if total_weight <= 0:
                total_weight = len(stops) # talep yoksa eşit dağıtım için 2 durak varsa total weight 2 tonaj iki lokasyona eşit bölünür

            # Kapasiteyi duraklara dağıt
            for s in stops:
                share = (demand_gas[s] + demand_diesel[s]) / total_weight
                ton = r["DemandTons"] * share # bu durak için tonaj
                if product == "gas":
                    delivered_gas[s] += ton
                elif product == "diesel":
                    delivered_diesel[s] += ton

        # Geminin toplam süresi HORIZON'dan fazla ise ağır ceza ver
        if total_duration > HORIZON:
            breakdown["duration_violation_penalty"] += 1e9  # büyük ceza tabu search seçmek istemez

        # Idle / kiraya verme mantığı: kalan gün sayısına göre
        #breakdown maliyet parçalarını ayrı ayrı toplamanda kullanılır
        idle_days = max(0, HORIZON - total_duration)
        if idle_days >= 5:
            breakdown["idle_bonus_or_penalty"] -= 15000  # uzun süre boşta -> kira geliri
        else:
            breakdown["idle_bonus_or_penalty"] += 10000  # kısa süre boş -> ekstra maliyet

    # Lokasyon bazlı talep-ceza hesapları
    for loc in total_demand.keys():
        gas_sent = delivered_gas[loc] # lokasyona gönderilen benzin
        diesel_sent = delivered_diesel[loc] # lokasyona gönderilen dizel
        gas_req = demand_gas[loc] # lokasyonun benzin talebi
        diesel_req = demand_diesel[loc] # lokasyonun dizel talebi

        if loc in depots:
            # Depolar: hedef = talep, band: 0.8x – 1.2x
            min_g = 0.8 * gas_req #L1 talebi (gas): 700 ton 700*0.8 = 560 ton
            max_g = 1.2 * gas_req
            min_d = 0.8 * diesel_req
            max_d = 1.2 * diesel_req

            if gas_sent < min_g:
                pen = (min_g - gas_sent) * depot_shortage_penalty_per_ton
                breakdown["depot_shortage_penalty"] += pen # depo yetersizliği cezası
            elif gas_sent > max_g:
                pen = (gas_sent - max_g) * depot_overstock_penalty_per_ton
                breakdown["depot_overstock_penalty"] += pen # depo fazla stok cezası

            if diesel_sent < min_d:
                pen = (min_d - diesel_sent) * depot_shortage_penalty_per_ton
                breakdown["depot_shortage_penalty"] += pen # depo yetersizliği cezası
            elif diesel_sent > max_d:
                pen = (diesel_sent - max_d) * depot_overstock_penalty_per_ton
                breakdown["depot_overstock_penalty"] += pen # depo fazla stok cezası
        else:
            # Normal talep noktası
            shortage_g = max(0.0, gas_req - gas_sent) #eksik teslim
            shortage_d = max(0.0, diesel_req - diesel_sent)
            over_g = max(0.0, gas_sent - gas_req) #fazla teslim
            over_d = max(0.0, diesel_sent - diesel_req)

            breakdown["shortage_penalty"] += (shortage_g + shortage_d) * shortage_penalty_per_ton
            breakdown["overstock_penalty"] += (over_g + over_d) * overstock_penalty_per_ton

    # GELİR: teslim edilen ürün * fiyat
    for loc in total_demand.keys():
        total_revenue += delivered_gas[loc] * price_gas[loc]
        total_revenue += delivered_diesel[loc] * price_diesel[loc]

    total_cost_component = sum(breakdown.values())
    objective = total_cost_component - total_revenue  # minimize (maliyet+ceza - gelir)

    return objective, delivered_gas, delivered_diesel, total_revenue, breakdown

def tabu_search(
    max_iters=250, # maksimum iterasyon sayısı
    tabu_tenure=12, # tabu liste süresi
    neighborhood_size=60, # her iterasyonda bakılacak komşu sayısı
    seed=42, #raporu 3 kere çalıştırıp aynı sonucu alabilmek için
):
    random.seed(seed) #rastgelleliği sabitle

    # Başlangıç çözümü: her gemi için en ucuz rotalardan 35 gün dolana kadar ekle
    current = {}
    for ship in ships:
        trips = [] # bu gemi için seferler
        remaining = HORIZON
        candidates = sorted(routes_by_ship[ship], key=lambda idx: routes[idx]["CostUSD"]) #bu rotaları en ucuzdan en pahalıya sırala

        for ridx in candidates: #rotaları sırayla dene
            d = routes[ridx]["DurationDays"]
            if d <= remaining:
                product = random.choice(["gas", "diesel"])
                trips.append((ridx, product))
                remaining -= d #kalan süreden düş
        current[ship] = trips

    current_obj, _, _, _, _ = evaluate_solution(current) #mevcut çözümü değerlendir
    best = {s: list(trips) for s, trips in current.items()} #en iyi çözümü sakla
    best_obj = current_obj 

    tabu = {}  # move -> kalan iterasyon
    iterations = 0

    for _ in range(max_iters):
        iterations += 1 #kaç iterasyon geçtiğini saymak için


        #60 komşu deneniyor → içlerinden en iyisi seçiliyor → current çözüm buna güncelleniyor
        best_neighbor = None
        best_neighbor_obj = math.inf
        best_move = None
        
        #komşu çözüm üretme dögüsü
        for _ in range(neighborhood_size):
            ship = random.choice(ships)
            trips = current.get(ship, []) #seçilen geminin mevcut seferlrini al
            move_type = random.choice(["add", "remove", "flip"]) #ekleme, silme, ürün değiştirme

            new_trips = None
            move = None

            # ADD: Bu gemiye yeni bir rota ekle (süre sınırını aşmayacak şekilde)
            if move_type == "add":
                possible_routes = routes_by_ship[ship]
                if not possible_routes:
                    continue

                ridx = random.choice(possible_routes) #rasgele bir rota seç
                d = routes[ridx]["DurationDays"]
                cur_duration = sum(routes[r]["DurationDays"] for r, _ in trips) #mevcut seferlerin toplam süresi
                if cur_duration + d > HORIZON:
                    continue  # bu rotayı ekleyemiyoruz
                product = random.choice(["gas", "diesel"])
                new_trips = trips + [(ridx, product)] #yeni rotayı ekle
                move = ("add", ship, ridx, product)

            # REMOVE: Geminin seferlerinden birini sil
            elif move_type == "remove":
                if not trips:
                    continue
                idx = random.randrange(len(trips)) #geminin seferlerinden rastgele birini seç
                ridx, product = trips[idx]
                new_trips = trips[:idx] + trips[idx+1:] #o seferi çıkar
                move = ("remove", ship, ridx, product)

            # FLIP: Seferlerden birinin ürününü değiştir
            elif move_type == "flip":
                if not trips:
                    continue
                idx = random.randrange(len(trips))
                ridx, product = trips[idx]
                new_product = "diesel" if product == "gas" else "gas"
                new_trips = list(trips)
                new_trips[idx] = (ridx, new_product)
                move = ("flip", ship, ridx)

            if new_trips is None or move is None: #geçersiz komşuları atla add/remove/flip bir şekilde yapılamadıysa o komşu atlanır
                continue

            # Tabu kontrolü (aspiration: daha iyi global buluyorsa izin ver)
            if move in tabu and best_obj <= current_obj:
                continue

            neighbor = {s: list(trs) for s, trs in current.items()} #mevcut çözümü kopyala
            neighbor[ship] = new_trips

            obj, _, _, _, _ = evaluate_solution(neighbor) 
            if obj < best_neighbor_obj:
                best_neighbor_obj = obj
                best_neighbor = neighbor
                best_move = move 

        if best_neighbor is None: #Yani bu iterasyonda hiç düzgün komşu üretemedin → artık aramayı bırak, döngüden çık.
            break

        current = best_neighbor
        current_obj = best_neighbor_obj

        # Tabu listesi güncelle
        expired = [m for m, t in tabu.items() if t <= 1] #t → hamlenin tabu olarak kalacağı kalan iterasyon sayısı t <= 1 ise → bu hamlenin tabu süresi bu iterasyondan sonra bitecek, listeden çıkar.
        for m in expired:
            tabu.pop(m, None)
        for m in list(tabu):
            tabu[m] -= 1 #tabu süresini 1 azalt

        tabu[best_move] = tabu_tenure #yeni hamleyi tabu listesine ekle

        if current_obj < best_obj:
            best_obj = current_obj
            best = {s: list(trs) for s, trs in current.items()}  #list(trs)ile kopya alıyoruz  böylece ilerde current değişse bile best etkilenmez

    return best, best_obj, iterations #en iyi çözüm, en iyi obje değeri, toplam iterasyon sayısı

def find_best_known_solution(
    seeds=range(5),               # istersen burayı 20 yaparsın
    tabu_tenures=(5, 10, 20),
    neighborhoods=(40, 60, 80),
    max_iters=250
):
    best_global_obj = float("inf") #tüm koşularda bulunan en iyi obje değeri
    best_global_solution = None #tüm koşullarda bulunan en iyi çözüm
    best_params = None #en iyi çözümü veren parametreler

    print("\n==============================")
    print(" BEST-KNOWN COST SEARCH ")
    print("==============================\n")

    for seed in seeds:
        for tt in tabu_tenures:
            for nh in neighborhoods:

                solution, obj, iters = tabu_search(
                    max_iters=max_iters,
                    tabu_tenure=tt,
                    neighborhood_size=nh,
                    seed=seed
                )

                print(f"Seed={seed:2d}, TT={tt:2d}, NH={nh:2d} → Obj={obj:15,.2f} (iters={iters})")

                if obj < best_global_obj:
                    best_global_obj = obj
                    best_global_solution = solution
                    best_params = (seed, tt, nh)

    print("\n==============================")
    print(" BEST FOUND OVER ALL RUNS ")
    print("==============================")
    print(f"Best Objective: {best_global_obj:,.2f}")
    print(f"Best Params → seed={best_params[0]}, tabu_tenure={best_params[1]}, neighborhood={best_params[2]}")
    print("==============================")

    return best_global_solution, best_global_obj, best_params


if __name__ == "__main__":
    # Çoklu koşudan en iyi bilinen çözümü bul
    best_solution, best_obj, best_params = find_best_known_solution()

    # En iyi çözümü tekrar değerlendirip gelir / dağılımı al
    objective, delivered_gas, delivered_diesel, total_revenue, breakdown = evaluate_solution(best_solution) #if __name__ == "__main__":
    # Çoklu koşudan en iyi bilinen çözümü bul
    best_solution, best_obj, best_params = find_best_known_solution()

    # En iyi çözümü tekrar değerlendirip gelir / dağılımı al
    objective, delivered_gas, delivered_diesel, total_revenue, breakdown = evaluate_solution(best_solution)
    approx_total_cost = objective + total_revenue  # objective = cost - revenue

    print("\n=== BEST-KNOWN SOLUTION DETAILS ===")
    print(f"Objective (cost + penalties - revenue): {objective:,.2f}")
    print(f"Approx. total cost (routes + penalties): {approx_total_cost:,.2f} $") #toplam maliyet
    print(f"Total revenue: {total_revenue:,.2f} $") #toplam gelir
    print(f"Estimated profit (revenue - cost): {total_revenue - approx_total_cost:,.2f} $")
    print(f"Best params: seed={best_params[0]}, tabu_tenure={best_params[1]}, neighborhood={best_params[2]}") #en iyi parametreler (hangi seed, tabu süresi, komşu sayısı)
    print("===================================\n")

    print("Maliyet kırılımı (breakdown):")
    for k, v in breakdown.items():
        print(f"  {k}: {v:,.2f} $") #maliyet parçaları

#if __name__ == "__main__":
    # Çoklu koşudan en iyi bilinen çözümü bul
    best_solution, best_obj, best_params = find_best_known_solution()

    # En iyi çözümü tekrar değerlendirip gelir / dağılımı al
    objective, delivered_gas, delivered_diesel, total_revenue, breakdown = evaluate_solution(best_solution)
    approx_total_cost = objective + total_revenue  # objective = cost - revenue

    print("\n=== BEST-KNOWN SOLUTION DETAILS ===")
    print(f"Objective (cost + penalties - revenue): {objective:,.2f}") #toplam obje değeri
    print(f"Approx. total cost (routes + penalties): {approx_total_cost:,.2f} $") #toplam maliyet
    print(f"Total revenue: {total_revenue:,.2f} $") #toplam gelir
    print(f"Estimated profit (revenue - cost): {total_revenue - approx_total_cost:,.2f} $") #tahmini kar
    print(f"Best params: seed={best_params[0]}, tabu_tenure={best_params[1]}, neighborhood={best_params[2]}") #en iyi parametreler (hangi seed, tabu süresi, komşu sayısı)
    print("===================================\n")


    print("Maliyet kırılımı (breakdown):")
    for k, v in breakdown.items():
        print(f"  {k}: {v:,.2f} $") #maliyet parçaları

#gemilere göre seçilen rotaları yazdırma
#hangi gemi hangi rotaları kaçıncı seferde hangi ürünü taşıyor
    print("\nSeçilen rotalar (gemi → seferler):")
    for ship in ships:
        trips = best_solution.get(ship, [])
        if not trips:
            print(f"  {ship}: IDLE (rota yok)")
        else:
            print(f"  {ship}:")
            for k, (ridx, prod) in enumerate(trips, 1):
                r = routes[ridx]
                print(
                    f"    Trip {k}: {r['Route']} | Product={prod} | Stops={r['Stops']} | "
                    f"Duration={r['DurationDays']} gün | Cost={r['CostUSD']}"
                )

    print("\nLokasyon bazında gönderilen miktarlar (ton):")
    for loc in sorted(total_demand.keys()):
        print(
            f"  {loc}: "
            f"Gas={delivered_gas[loc]:.1f}/{demand_gas[loc]}  |  "
            f"Diesel={delivered_diesel[loc]:.1f}/{demand_diesel[loc]}"
        )

   
     

#Her gemi 35 günde birden fazla sefer yapabilsin
#Sadece maliyet değil kar maksimizasyonu
#Cezaları parçalayarak raporla 
#Parametre deneyleri : küçük bi deneme motoru 



#Genetik algoritma ile karşılaştır sonucu karşılaştır 