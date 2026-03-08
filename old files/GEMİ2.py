from __future__ import annotations

import csv
import io
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set
from copy import deepcopy

# ============================================================
# 1) CONSISTENT DATA
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
"""

# ============================================================
# 2) MODELS
# ============================================================

@dataclass(frozen=True)
class Location: 
    loc: str
    region: str
    product: str
    qty: int #lokasyona teslim edilen ton 
    demand_day: int #zaman penceresi +-2 gün kısıtını kuruyor

@dataclass(frozen=True) 
class Ship:
    ship_id: str
    ownership: str
    capacity: int
    fixed_daily_cost: int 
    region: str

@dataclass(frozen=True)
class Depot:
    depot: str
    region: str
    io_gas: int
    io_diesel: int # depo başlangıcındaki envanter
    target_gas: int # depo hedef envanteri
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

Solution = Dict[str, Tuple[str, ...]]

# ============================================================
# 3) VERİLER OKUNUYOR
# ============================================================

def parse_csv(text: str) -> List[Dict[str, str]]: #csv dosyasını okuyup satırları sözlük listesi olarak döndüren fonksiyon
    f = io.StringIO(text.strip())
    reader = csv.DictReader(f)
    rows: List[Dict[str, str]] = []
    for row in reader:
        if not row: # boş satırları atla
            continue
        first_key = next(iter(row.keys()))
        if not row.get(first_key, "").strip(): #ilk sütun boşsa atla
            continue
        rows.append({k: (v.strip() if isinstance(v, str) else v) for k, v in row.items()})
    return rows

def load_locations() -> Dict[str, Location]: #lokasyon verilerini yükleyen fonksiyon
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

def load_ships() -> Dict[str, Ship]: #gemi verilerini yükleyen fonksiyon
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

def load_depots() -> List[Depot]: #depo verilerini yükleyen fonksiyon
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

def load_routes(locs: Dict[str, Location]) -> Dict[str, Route]: #rota verilerini yükleyen fonksiyon
    routes: Dict[str, Route] = {}
    for r in parse_csv(ROUTES_CSV):
        rid = r["Route"]
        sid = r["Ship"]
        stops = tuple(s.strip() for s in r["Stops"].split("-") if s.strip()) #rota üzerindeki duraklar
        duration = int(r["DurationDays"])
        cost = int(r["CostUSD"])

        stop_regions = {locs[s].region for s in stops}
        region = stop_regions.pop() if len(stop_regions) == 1 else "MIXED" #rota bölgesi belirleme kuzeye giden gemi güneye gidemez ksıtı çalışır
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
    enforce_duration_le_7: bool = True #gemi rotasının süresi 7 günden uzun olamaz
    enforce_max_2_stops: bool = True #rota üzerindeki durak sayısı 2'den fazla olamaz
    enforce_ship_route_region_match: bool = True #gemi bölgesi ile rota bölgesi eşleşmeli
    enforce_capacity: bool = True #rota talebi gemi kapasitesini aşamaz
    enforce_total_duration_le_35: bool = True #her gemi için toplam rota süresi 35 günden uzun olamaz
    enforce_unique_route_globally: bool = True #her rota sadece bir gemi tarafından kullanılabilir
    enforce_cover_all_locations: bool = True #tüm lokasyonlar en az bir rota tarafından hizmet almalı
    enforce_time_window_pm2: bool = True #rota hizmet günleri, talep gününe ±2 gün içinde olmalı
    include_inventory_penalty: bool = False #depo envanteri hedef seviyeden sapma cezası dahil edilsin mi

# ============================================================
# 5) TIME WINDOWS + SCHEDULING
# ============================================================

def feasible_service_days_for_route(route: Route, locs: Dict[str, Location]) -> List[int]: #rota için uygun hizmet günlerini döndüren fonksiyon
    lo, hi = 1, 35  #lo= o rotanın başlayabileceği minimum gün hi= o rotanın başlayabileceği maksimum gün
   
    for s in route.stops: 
        d = locs[s].demand_day #talep günü
        lo = max(lo, d - 2) #talep gününden 2 gün öncesi
        hi = min(hi, d + 2) #talep gününden 2 gün sonrası
    if lo > hi: 
        return []
    return list(range(lo, hi + 1)) #uygun hizmet günleri
 
 #Bir gemiye verdiğin trip/rota listesini alıyor (route_ids) ve 35 gün içinde, rotaların zaman pencerelerine (±2) uyarak 
 # ve seferler üst üste binmeden ardışık bir takvim (schedule) bulmaya çalışıyor.
def schedule_trips_backtracking( #bir geminin seçtiği route listesine 35 gün içinde sıralı ve pencerelere uyan bi plan bulur
    route_ids: Tuple[str, ...], 
    ship: Ship,
    routes: Dict[str, Route],
    locs: Dict[str, Location],
) -> Optional[Dict[str, int]]:
    items = [] #(rota_id, en_erken_gün, en_geç_gün, süre)
    for rid in route_ids:
        r = routes[rid]
        days = feasible_service_days_for_route(r, locs) #rota için uygun hizmet günleri
        if not days:
            return None
        lo, hi = days[0], days[-1] #en erken ve en geç hizmet günleri
        items.append((rid, lo, hi, r.duration)) #rota_id, en erken gün, en geç gün, süre

    items.sort(key=lambda x: ((x[2] - x[1]), x[2])) #önce en dar pencere daha zor rota yani önce dene, sonra en erken bitiş günü

    chosen: Dict[str, int] = {}


#current_time: “şu anki gemi en erken hangi gün boş?”
#dfs fonksiyonu: rotaları sırayla deneyerek uygun bir plan bulmaya çalışır
    def dfs(idx: int, current_time: int) -> bool:  #geminin seferleri üst üste binemez o yüzden current time tututlur
        if idx == len(items):                       
            return True                              
        rid, lo, hi, dur = items[idx] #şu anki rotanın en erken, en geç başlama günleri ve süresi
        start_min = max(current_time, lo)
        for start in range(start_min, hi + 1): #rota için uygun başlama günleri
            end_day = start + dur - 1 #rota bitiş günü
            if end_day > 35:
                continue
            chosen[rid] = start
            if dfs(idx + 1, end_day + 1): #sonraki rotaya geç ve gemi en erken end_day + 1 gün boş
                return True
            del chosen[rid]
        return False

    return chosen if dfs(0, 1) else None

# ============================================================
# 6) INVENTORY (optional)
# ============================================================

#kuzeyde 1 depo olduğu için bi karar aşamasına gerek yok o yüzden güney için fonk yazdık
def depot_assignment_for_south(loc: str) -> str: #güney lokasyonları için depo ataması
    n = int(loc[1:])  
    return "D_S1(L10)" if 9 <= n <= 11 else "D_S2(L12)" #L9,L10,L11 için D_S1(L10) diğerleri için D_S2(L12)

# ============================================================
# 7) EVALUATION
# ============================================================

BIG_M = 10_000_000 #büyük ceza değeri amaç tabu search kuralları bozan çözümü asla iyi sanmasın 

def evaluate(
    sol: Solution, #her gemiye atanmış rota listesi
    ships: Dict[str, Ship],
    routes: Dict[str, Route],
    locs: Dict[str, Location],
    depots: List[Depot],
    cfg: ConstraintConfig,
    rental_income_if_idle_ge_5: int = 15000, #geminin 5 günden fazla boşta kalması durumunda elde edilen kira geliri
    idle_penalty_if_idle_lt_5: int = 10000,  #geminin 5 günden az boşta kalması durumunda ödenen ceza
    inv_weight: int = 2, #stok hedefinden sapmaya çarpan katsayı
) -> Tuple[int, Dict]:

    violations: List[str] = [] #kural bozulursa buraya yazılır
    total_cost = 0
    served: Set[str] = set()
    used_routes: Set[str] = set() #aynı rota iki kez kullanıldı mı kontrolü için
    inv_penalty = 0
    inv_debug = {}
    inv_after = {d.depot: {"gasoline": d.io_gas, "diesel": d.io_diesel} for d in depots} #depoların başlangıç envanteri
    ship_schedules: Dict[str, Dict[str, int]] = {} #gemi_id -> rota_id -> başlama günü

    for ship_id, trip_list in sol.items(): #gemi_id ve atanan rota listesi
        ship = ships[ship_id]
        total_duration = 0 #gemi için toplam rota süresi

        for rid in trip_list: #geminin her rotasını kontrol et
            if rid not in routes: #rota yoksa hata kaydı
                violations.append(f"{ship_id}: route {rid} not found")  
                continue

            r = routes[rid] #rotayı al

            if r.ship_id != ship_id: #rota gemiye atanmış mı kontrolü
                violations.append(f"{ship_id}: picked route {rid} belongs to {r.ship_id}") 
            
            #bu açıkken 1 rota 1 gemide 1 kez kullanılır
            if cfg.enforce_unique_route_globally: #aynı rota tekrar kullanılmasın
                if rid in used_routes:
                    violations.append(f"route used multiple times: {rid}")
                used_routes.add(rid)

            if cfg.enforce_duration_le_7 and r.duration > 7: #rota süresi 7 günden uzun olamaz
                violations.append(f"{ship_id}/{rid}: duration {r.duration} > 7")

            if cfg.enforce_max_2_stops and len(r.stops) > 2:    #rota üzerindeki durak sayısı 2'den fazla olamaz
                violations.append(f"{ship_id}/{rid}: stops {len(r.stops)} > 2")

            if cfg.enforce_ship_route_region_match: #gemi bölgesi ile rota bölgesi eşleşmeli
                if r.region == "MIXED":
                    violations.append(f"{ship_id}/{rid}: mixed route region")
                elif r.region != ship.region:
                    violations.append(f"{ship_id}/{rid}: route region {r.region} != ship region {ship.region}")

            if cfg.enforce_capacity and r.demand_tons > ship.capacity: #rota talebi gemi kapasitesini aşamaz
                violations.append(f"{ship_id}/{rid}: demand {r.demand_tons} > cap {ship.capacity}")

            total_duration += r.duration #gemi için toplam rota süresine ekle
            total_cost += r.cost + ship.fixed_daily_cost * r.duration   #rota maliyeti + gemi sabit günlük maliyeti * rota süresi
            
    
             
            # Stok kontrolü açık ise:
              #Lokasyonun region’ına göre depo seçiyor
              #O depodan ilgili ürünü (gasoline/diesel) qty kadar düşüyor
               #Stoklar girince cost’un uçmasının nedeni genelde buradaki “hedef stoktan sapma cezası” (aşağıda) ve inv_weight etkisi.
            for s in r.stops: #servis edilen lokasyonları işaretleme + stok düşme
                served.add(s)
                if cfg.include_inventory_penalty:
                    loc = locs[s]
                    depot_id = "D_N(L3)" if loc.region == "North" else depot_assignment_for_south(loc.loc)
                    inv_after[depot_id][loc.product] -= loc.qty

        if cfg.enforce_total_duration_le_35 and total_duration > 35: #her gemi için toplam rota süresi 35 günden uzun olamaz
            violations.append(f"{ship_id}: total_duration {total_duration} > 35")

        idle = 35 - total_duration #geminin boşta kaldığı gün sayısı
        if idle >= 5:
            total_cost -= rental_income_if_idle_ge_5
        else:
            total_cost += idle_penalty_if_idle_lt_5

        if cfg.enforce_time_window_pm2: #rota hizmet günleri, talep gününe ±2 gün içinde olmalı
            sched = schedule_trips_backtracking(trip_list, ship, routes, locs) #rota planlaması yap
            if sched is None:   #plan bulunamadıysa kural ihlali kaydı
                violations.append(f"{ship_id}: cannot schedule trips within ±2 windows and 35 days")
            else:   
                ship_schedules[ship_id] = sched

    if cfg.enforce_cover_all_locations: #tüm lokasyonlar en az bir rota tarafından hizmet almalı
        missing = sorted(set(locs.keys()) - served) 
        for m in missing: 
            violations.append(f"missing location: {m}")

    if cfg.include_inventory_penalty: #depo envanteri hedef seviyeden sapma cezası
        for d in depots:
            after_g = inv_after[d.depot]["gasoline"] #depo sonundaki benzin envanteri
            after_d = inv_after[d.depot]["diesel"] #depo sonundaki dizel envanteri
            inv_debug[d.depot] = {"after_gas": after_g, "after_diesel": after_d} #depo sonundaki envanter durumu
            inv_penalty += inv_weight * abs(after_g - d.target_gas) #hedef stoktan sapma cezası
            inv_penalty += inv_weight * abs(after_d - d.target_diesel) #hedef stoktan sapma cezası
        total_cost += inv_penalty

    objective = total_cost + BIG_M * len(violations) #kural ihlali varsa büyük ceza ekle
    debug = {
        "total_cost": total_cost,
        "violations_count": len(violations), 
        "violations": violations[:30],  #ilk 30 kural ihlali
        "served_count": len(served), 
        "ship_schedules": ship_schedules,
        "inventory_penalty": inv_penalty, #depo envanteri cezası
        "inventory_after": inv_debug,
    }
    return objective, debug #violation varsa obj uçuyor yoksa obj = total_cost

# ============================================================
# 8) TABU SEARCH (MULTI-TRIP)
# ============================================================
#	Greedy başlangıç çözümü (hızlıca “her lokasyon servis edilsin” diye kaba çözüm üretir)
#	Tabu Search (o kaba çözümü oynatarak maliyeti düşürür)

#tüm rotaları gruplara göre gruplandırır s5 e rota ekle dediğinde s5 in hangi rotaları seçebilecğini hızlı bulmak için 
def routes_by_ship(routes: Dict[str, Route]) -> Dict[str, List[str]]: 
    d: Dict[str, List[str]] = {}
    for r in routes.values():
        d.setdefault(r.ship_id, []).append(r.route_id)
    return d

#her rotanın hangi lokasyonları kapsadığını çıkarır 
#greedy başlangıç çözümünde hedef servis edilmeyen lokayonları kapatmak 
def route_covers(routes: Dict[str, Route]) -> Dict[str, Set[str]]:
    return {rid: set(routes[rid].stops) for rid in routes.keys()}


def build_initial_solution_greedy(
    ships: Dict[str, Ship],
    routes: Dict[str, Route],
    locs: Dict[str, Location],
    cfg: ConstraintConfig,
) -> Solution:
    by_ship = routes_by_ship(routes)
    covers = route_covers(routes)

    sol: Solution = {sid: tuple() for sid in ships.keys()} #başlangıçta her gemiye boş rota listesi ata
    used_routes: Set[str] = set() #kullanılan rotalar
    uncovered: Set[str] = set(locs.keys()) #henüz servis edilmeyen lokasyonlar
    all_route_ids = sorted(routes.keys(), key=lambda rid: routes[rid].cost) #rotaları maliyete göre sırala

    def can_add_trip(ship_id: str, rid: str) -> bool:
        r = routes[rid]
        ship = ships[ship_id]
        if r.ship_id != ship_id: #rota gemiye atanmış mı kontrolü
            return False
        if cfg.enforce_ship_route_region_match and (r.region == "MIXED" or r.region != ship.region): #gemi bölgesi ile rota bölgesi eşleşmeli
            return False
        if cfg.enforce_capacity and r.demand_tons > ship.capacity: #rota talebi gemi kapasitesini aşamaz
            return False
        if cfg.enforce_duration_le_7 and r.duration > 7: #rota süresi 7 günden uzun olamaz
            return False
        if cfg.enforce_max_2_stops and len(r.stops) > 2:    #rota üzerindeki durak sayısı 2'den fazla olamaz
            return False
        if cfg.enforce_unique_route_globally and rid in used_routes: #aynı rota tekrar kullanılmasın
            return False
        if cfg.enforce_time_window_pm2 and not feasible_service_days_for_route(r, locs): #rota için uygun hizmet günleri yoksa
            return False

        current = sol[ship_id]  #geminin mevcut rota listesi
        new_list = current + (rid,) #geminin mevcut rotalarına bu rotayı ekle
        total_dur = sum(routes[x].duration for x in new_list) #geminin toplam rota süresi
        if cfg.enforce_total_duration_le_35 and total_dur > 35: #her gemi için toplam rota süresi 35 günden uzun olamaz
            return False

        if cfg.enforce_time_window_pm2 and schedule_trips_backtracking(new_list, ship, routes, locs) is None: 
            return False

        return True

#rastgele bir uncovered lokasyon seçiliyor sonra ucuzdan pahalıya rotaları tarıyor
    for _ in range(400):  #greedy sonsuz döngüye girmesin diye
        if not uncovered:
            break
        target_loc = next(iter(uncovered))
        added = False
#hedef lokasyonun kapatan en ucuz rotayı bul
#o rotayı kendi gemisine ekle
#bu rotanın kapattığı lokasyonları uncovered setinden çıkar
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

        if not added: #hiçbir rota eklenemediyse döngüden çık
            break

    return sol

def tabu_search_multi_trip(
    ships: Dict[str, Ship],
    routes: Dict[str, Route],
    locs: Dict[str, Location],
    depots: List[Depot],
    cfg: ConstraintConfig,
    max_iters: int = 1500,
    neighborhood_size: int = 240,
    tabu_tenure: int = 24,
    seed: int = 11,
    max_trips_per_ship: int = 9,
):
    
#tabu search ile çözüm iyileştirme fonksiyonu
#greedy ile current başlatır
#best en iyi çzöümü tutar
#evaluate ile objective hesaplar
    random.seed(seed)
    by_ship = routes_by_ship(routes)
    ship_ids = list(ships.keys())

    current = build_initial_solution_greedy(ships, routes, locs, cfg) #başlangıç çözümü
    best = dict(current)

    best_obj, best_dbg = evaluate(best, ships, routes, locs, depots, cfg) #başlangıç çözümünü değerlendir
    cur_obj = best_obj #şimdiki çözümün objesi

    tabu: Dict[Tuple[str, str, str], int] = {}

    def decay_tabu(): #tabu listeyi güncelleme fonksiyonu
        for k in list(tabu.keys()): 
            tabu[k] -= 1  #tabu süresini azalt
            if tabu[k] <= 0: #süre dolduysa tabu listesinden çıkar
                del tabu[k]

    def add_tabu(ship_id: str, action: str, rid: str):
        tabu[(ship_id, action, rid)] = tabu_tenure

    def is_tabu(ship_id: str, action: str, rid: str) -> bool: 
        return (ship_id, action, rid) in tabu

    for it in range(1, max_iters + 1): #Her iterasyonda “komşu çözümler” üretip en iyisini seçiyor.
        decay_tabu()

        best_cand = None
        best_cand_obj = None
        best_cand_dbg = None
        best_move = None

#her denemede rastgele bir gemi seç rastgele bi hamle seç
        for _ in range(neighborhood_size):
            sid = random.choice(ship_ids)
            trips = list(current[sid])
            action = random.choice(["add", "remove", "swap_order"])


#o gemiye kendi rotalarıdan rastgele bir rota ekler sonra evaluate eder
            if action == "add":
                if len(trips) >= max_trips_per_ship: 
                    continue
                rid = random.choice(by_ship.get(sid, []))
                if rid in trips:
                    continue

                cand = dict(current) #mevcut çözümden kopya oluştur
                cand[sid] = tuple(trips + [rid]) #seçilen rotayı geminin rota listesine ekle
                cand_obj, cand_dbg = evaluate(cand, ships, routes, locs, depots, cfg) #yeni çözümü değerlendir

                if is_tabu(sid, "add", rid) and cand_obj >= best_obj: #tabu olsa bile eğer global best’i geçerse izin verebilir.Senin kodda şart: tabu ve best’ten iyi değilse reddet.
                    continue

                if best_cand is None or cand_obj < best_cand_obj: #şimdiki en iyi komşudan daha iyiyse güncelle
                    best_cand, best_cand_obj, best_cand_dbg = cand, cand_obj, cand_dbg
                    best_move = (sid, "add", rid)

#rastgele bir rota çıkar evaluate et 
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

# rota sırasını değiştir evaluate et ayı gemini trip sırasını değiştir 
#Neden? Çünkü scheduler ardışık çalıştığı için sıra değişimi feasibility/maliyet etkileyebilir (özellikle time window sıkışıklığında).
            else:  # swap_order
                if len(trips) < 2:
                    continue
                i, j = random.sample(range(len(trips)), 2)
                trips[i], trips[j] = trips[j], trips[i]

                cand = dict(current)
                cand[sid] = tuple(trips)
                cand_obj, cand_dbg = evaluate(cand, ships, routes, locs, depots, cfg)
               
               #her neighborhood denemesinde tabu kontrolü
               #yani o iterasyon boyunca bulduğum en iyi komşu çözüm tutuluyo
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

        if it % 120 == 0: #her 120 iterasyonda bir ilerleme raporu yazdır
            print(
                f"iter={it:4d} best_obj={best_obj:,} cur_obj={cur_obj:,} "
                f"viol={best_dbg['violations_count']} served={best_dbg['served_count']}/14"
            )

    return best, best_obj, best_dbg

# ============================================================
# 9) POST-OPT HELPERS
# ============================================================
#tabu search sonrası çözüm iyileştirme için yardımcı fonksiyonlar
#görselleştirme amaçlı fonk
def pretty_print_solution(sol: Solution, routes: Dict[str, Route], dbg: Dict):  #ekrana okunabilir şekilde rotalrı yazdırma
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

#verilen lokasyonu hangi gemi ve rotanın hizmet ettiğini bulur tüm gemileri dolaşır
# o geminin tüm rotalarını da dolaşır  
#bu lokasyonu şu an kim taşıyor
def find_serving_route(sol: Solution, routes: Dict[str, Route], loc: str) -> Optional[Tuple[str, str]]: 
    for sid, trips in sol.items():
        for rid in trips:
            if loc in routes[rid].stops:
                return sid, rid
    return None
#şu lokasyonları şu gemilre taşımayı dene
#bu fonk taşınacak hamle listesi
def build_moves_auto_from(sol: Solution, routes: Dict[str, Route], targets: List[Tuple[str, str]]):
    moves = []
    for loc, to_sid in targets:
        found = find_serving_route(sol, routes, loc)
        if found is None:
            print(f"[post-opt] {loc} not served, skip")
            continue
        from_sid, _ = found
        if from_sid == to_sid:
            print(f"[post-opt] {loc} already on {to_sid}, skip")
            continue
        moves.append((loc, from_sid, to_sid))
    return moves
#1. trip bitmeden 2. trip başlayamaz ayrıca her trip , durakların talep gününe ±2 gün içinde hizmet vermeli
#yeni rotayı en iyi pozisyona ekler “rota ekledim ama schedule patladı” sorununu engeller.
#Yeni bir rota (rid_new) eklerken, onu geminin trip listesine hangi sırada koyarsan schedule daha rahat olur?
def insert_route_best_position(
    trips: Tuple[str, ...],
    rid_new: str,
    ship: Ship,
    routes: Dict[str, Route],
    locs: Dict[str, Location],
) -> Optional[Tuple[str, ...]]:
    if rid_new in trips:
        return trips

    best_order = None
    best_end = None

    for pos in range(len(trips) + 1):
        cand = list(trips)
        cand.insert(pos, rid_new)
        cand_t = tuple(cand)

        sched = schedule_trips_backtracking(cand_t, ship, routes, locs)
        if sched is None:
            continue

        last_end = 0
        for rid in cand_t:
            start = sched[rid]
            last_end = max(last_end, start + routes[rid].duration - 1)

        if best_order is None or (best_end is not None and last_end < best_end):
            best_order = cand_t
            best_end = last_end

    return best_order
#Tabu Search çözümü bulduktan sonra, tek bir lokasyonu bir gemiden alıp başka gemiye taşırsak (maliyeti düşürerek) daha iyi çözüm çıkıyor mu diye denemek.
#Bu “post-opt” aslında lokal iyileştirme.

def try_move_location(
    current: Solution,
    ships: Dict[str, Ship],
    routes: Dict[str, Route],
    locs: Dict[str, Location],
    depots: List[Depot],
    cfg: ConstraintConfig,
    loc: str, #taşınacak lokasyon
    from_sid: str, #gönderici gemi
    to_sid: str, #hedef gemi
    prefer_single: bool = True,
    verbose: bool = False,
) -> Tuple[bool, Optional[Solution], int, Dict, int]:

    if ships[from_sid].region != ships[to_sid].region: #gemi bölgeleri uyuşmuyorsa taşıma yapılamaz
        if verbose:
            print(f"[move] region mismatch {from_sid}->{to_sid}")
        return False, None, 0, {}, 0

    found = None
    for rid in current.get(from_sid, tuple()): #gönderici geminin rotalarını dolaş
        if loc in routes[rid].stops: #o lokasyonu taşıyan rota var mı
            found = rid 
            break
    if found is None:
        if verbose:
            print(f"[move] {loc} not served by {from_sid}")
        return False, None, 0, {}, 0

    old_route = routes[found] 
    if len(old_route.stops) > 1:
        if verbose:
            print(f"[move] {loc} is in multi-stop route {found} {old_route.stops}; skip.")
        return False, None, 0, {}, 0
# en ucuz adayı seç
    candidates = [] #taşıma için uygun rotalar
    for rid, r in routes.items():
        if r.ship_id != to_sid:
            continue
        if loc not in r.stops:
            continue
        if prefer_single and tuple(r.stops) != (loc,):
            continue
        candidates.append(rid)

    if not candidates:
        if verbose:
            print(f"[move] no candidate route on {to_sid} for {loc}")
        return False, None, 0, {}, 0

    candidates.sort(key=lambda rid: routes[rid].cost)
    rid_new = candidates[0]

    if cfg.enforce_unique_route_globally: #aynı rota iki kez kullanılamaz kuralı
        for sid, trips in current.items():
            if rid_new in trips and sid != to_sid:
                if verbose:
                    print(f"[move] {rid_new} already used by {sid} (unique_route enabled)") #taşıma yapılamaz
                return False, None, 0, {}, 0

    cand = deepcopy(current)
    new_from = list(cand[from_sid])
    new_from.remove(found) #gönderici gemiden rotayı çıkar
    cand[from_sid] = tuple(new_from)

    to_ship = ships[to_sid]
    best_order = insert_route_best_position(cand[to_sid], rid_new, to_ship, routes, locs) #hedef gemiye rotayı en iyi pozisyona ekle
    if best_order is None:
        if verbose:
            print(f"[move] {loc}: inserting {rid_new} into {to_sid} made schedule impossible")
        return False, None, 0, {}, 0
    cand[to_sid] = best_order

    base_obj, _ = evaluate(current, ships, routes, locs, depots, cfg) #mevcut çözümü değerlendir
    cand_obj, cand_dbg = evaluate(cand, ships, routes, locs, depots, cfg) #aday çözümü değerlendir
    delta = cand_obj - base_obj #delta < 0 ise iyileşme var

    if verbose:
        print(f"[move] {loc}: {from_sid}:{found} -> {to_sid}:{rid_new}  delta={delta:,}  new_obj={cand_obj:,}")

    if cand_dbg["violations_count"] > 0:
        return False, None, cand_obj, cand_dbg, delta

    return True, cand, cand_obj, cand_dbg, delta

#birden fazla move denemesi yapar en iyi iyileştiren hamleyi uygular sonunda en iyi çözümü döndürür
#her hamlede taşımayı dener, feasible + iyileşme varsa, best sol günceller,sonra sıradaki hamleleri bu yeni çözüm üzerinden dener
def try_moves_batch(current, ships, routes, locs, depots, cfg, moves):
    base_obj, base_dbg = evaluate(current, ships, routes, locs, depots, cfg)
    print(f"\n=== BATCH TEST === base_obj={base_obj:,.0f} served={base_dbg['served_count']}/14 viol={base_dbg['violations_count']}\n")

    best_obj = base_obj
    best_sol = current

    for (loc, fr, to) in moves:
        ok, cand, obj, dbg, delta = try_move_location(
            best_sol, ships, routes, locs, depots, cfg,
            loc=loc, from_sid=fr, to_sid=to, prefer_single=True, verbose=True
        )
        if ok and obj < best_obj:
            best_obj = obj
            best_sol = cand

    return best_obj, (best_sol if best_sol != current else None) #çözüm değiştiyse döndür

# ============================================================
# 10) RUN
# ============================================================

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
    print("served:", f"{dbg['served_count']}/14")
    pretty_print_solution(best, routes, dbg)

    
    # ============================================================
    # ✅ HAMLE B: AUTO POST-OPT (South + North)
    # ============================================================

    region_locs = {
        "South": ["L9", "L10", "L11", "L12", "L13", "L14"],
        "North": ["L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8"],
    }

    region_ships = {
        "South": [sid for sid in ships if ships[sid].region == "South"],
        "North": [sid for sid in ships if ships[sid].region == "North"],
    }

    best_sol = best
    best_obj2, best_dbg2 = evaluate(best_sol, ships, routes, locs, depots, cfg)

    improved = True
    while improved:
        improved = False

        for region, loc_list in region_locs.items():
            ship_list = region_ships[region]

            for loc in loc_list:
                found = find_serving_route(best_sol, routes, loc)
                if found is None:
                    continue

                from_sid, _ = found

                # sadece aynı region gemileri arasında dene
                for to_sid in ship_list:
                    if to_sid == from_sid:
                        continue

                    ok, cand, obj, dbg, delta = try_move_location(
                        best_sol, ships, routes, locs, depots, cfg,
                        loc=loc, from_sid=from_sid, to_sid=to_sid,
                        prefer_single=True,  # tek stop route tercih ediyor
                        verbose=False #detaylı çıktı verme
                    )

                    if ok and obj < best_obj2:
                        best_sol = cand
                        best_obj2 = obj
                        best_dbg2 = dbg
                        improved = True

    if best_obj2 < best_obj:
        print("\n✅ AUTO POST-OPT improved:", f"{best_obj:,} -> {best_obj2:,}")
        pretty_print_solution(best_sol, routes, best_dbg2)
    else:
        print("\nAUTO POST-OPT: no improvement found.")


if __name__ == "__main__":
    main()
