import random
import copy
import matplotlib.pyplot as plt

class ShipRouteOptimizer:
    def __init__(self, routes_data):
        self.routes = routes_data
        self.available_ships = list(set(r['ship'] for r in self.routes))
        self.loc_region = {f'L{i}': 'North' if i <= 8 else 'South' for i in range(1, 15)}
        
        # GEMİ KAPASİTE TANIMLARI (Ton)
        self.ship_capacities = {
            'S1': 2500, 'S2': 2500, 'S3': 2200, 'S4': 2000, 'S5': 2800,
            'S6': 2500, 'S7': 2500, 'S8': 1300, 'S9': 2000, 'S10': 2300
        }
        
        self.cost_history = []
        self.best_solution = None
        self.best_cost = float('inf')

    def evaluate(self, solution):
        total_cost = 0
        visited_locs = set()
        
        for ship, assignments in solution.items():
            ship_days = 0
            ship_tons = 0
            ship_regs = set()
            
            for r in assignments:
                total_cost += r['cost']
                ship_days += r['days']
                ship_tons += r.get('demand', 0)
                
                stops = r['stops'].split('-')
                region = self.loc_region.get(stops[0].strip())
                if region: ship_regs.add(region)
                for s in stops: visited_locs.add(s.strip())
            
            # KISIT: Kapasite Aşımı
            capacity_limit = self.ship_capacities.get(ship, 2000)
            if ship_tons > capacity_limit:
                total_cost += 3000000 # Kapasite aşımı cezası
            
            # KISIT: 35 Gün Sınırı
            if ship_days > 35: total_cost += 5000000 
            
            # KISIT: Kuzey-Güney Çakışması
            if len(ship_regs) > 1: total_cost += 2000000 
            
            # Idle Time (Boşta Kalma)
            idle = 35 - ship_days
            if idle >= 5: total_cost -= 20000 
            else: total_cost += 15000 
            
        # KISIT: 14 Lokasyon Kapsama
        missing = 14 - len(visited_locs)
        if missing > 0: total_cost += (missing * 1500000)
        
        return total_cost, len(visited_locs)

    def run_tabu_search(self, iterations=30000):
        routes_by_ship = {s: [r for r in self.routes if r['ship'] == s] for s in self.available_ships}
        current_sol = {s: [random.choice(routes_by_ship[s])] for s in self.available_ships}
        
        self.best_solution = copy.deepcopy(current_sol)
        self.best_cost, _ = self.evaluate(self.best_solution)
        self.cost_history = [self.best_cost]
        
        tabu = []
        tabu_size = 500 

        print(f">>> Kapasite Odaklı Optimizasyon başladı ({iterations} adım)...")

        for i in range(iterations):
            neighbor = copy.deepcopy(current_sol)
            s_pick = random.choice(self.available_ships)
            
            # Gemiye atanacak rota sayısını dengeli tut (1-3)
            num_k = random.randint(1, min(3, len(routes_by_ship[s_pick])))
            neighbor[s_pick] = random.sample(routes_by_ship[s_pick], num_k)
            
            cost_n, v_count = self.evaluate(neighbor)
            
            sol_key = str({s: [r['id'] for r in rs] for s, rs in neighbor.items()})
            if sol_key not in tabu:
                current_sol = neighbor
                tabu.append(sol_key)
                if len(tabu) > tabu_size: tabu.pop(0)
                
                if cost_n < self.best_cost:
                    self.best_cost = cost_n
                    self.best_solution = copy.deepcopy(neighbor)
                    self.cost_history.append(self.best_cost)
                    
                    if v_count == 14 and self.best_cost < 2000000:
                        print(f"İterasyon {i:5}: Kapasite ve Zaman Uyumlu Çözüm Bulundu! Maliyet: {self.best_cost:.0f}")

    def visualize(self):
        plt.figure(figsize=(16, 6))
        
        # 1. Maliyet Grafiği
        plt.subplot(1, 2, 1)
        plt.plot(self.cost_history, color='#e74c3c', linewidth=2)
        plt.title('Maliyet Optimizasyon Eğrisi', fontsize=12)
        plt.xlabel('İyileşme Adımları')
        plt.ylabel('Maliyet (USD)')
        plt.grid(True, linestyle='--', alpha=0.6)

        # 2. Gemi Kapasite Kullanım Grafiği
        plt.subplot(1, 2, 2)
        usage_tons = [sum(r.get('demand', 0) for r in self.best_solution[s]) for s in self.available_ships]
        caps = [self.ship_capacities.get(s, 2000) for s in self.available_ships]
        
        # Doluluk oranına göre renk (Aşım varsa kırmızı)
        colors = ['#27ae60' if t <= c else '#c0392b' for t, c in zip(usage_tons, caps)]
        
        plt.bar(self.available_ships, usage_tons, color=colors, alpha=0.7, label='Yüklenen Miktar (Ton)')
        plt.scatter(self.available_ships, caps, color='black', marker='_', s=200, label='Maks. Kapasite')
        
        plt.title('Gemilerin Kapasite Kullanım Durumu', fontsize=12)
        plt.ylabel('Ton')
        plt.legend()
        
        plt.tight_layout()
        plt.show()

# --- ÇALIŞTIRMA BÖLÜMÜ ---
if __name__ == "__main__":
    raw_data = """Route,Ship,StartPort,EndPort,Stops,DemandTons,DurationDays,CostUSD
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
R120,S10,P2,P1,L14,950,6,60500""" 

    # Veriyi listeye dönüştürme
    data_list = []
    lines = raw_data.strip().split('\n')
    for line in lines:
        p = line.split(',')
        if "Route" in p or len(p) < 8:
            continue
        try:
            data_list.append({
                'id': p[0].strip(), 
                'ship': p[1].strip(), 
                'stops': p[4].strip(), 
                'days': float(p[6].strip()), 
                'cost': float(p[7].strip())
            })
        except ValueError:
            continue

    # BURASI EKSİKTİ: Sınıfı başlat ve fonksiyonları çağır
    if data_list:
        optimizer = ShipRouteOptimizer(data_list)
        optimizer.run_tabu_search(iterations=50000) # Test için 5000 yaptık
        print(f"\nOptimizasyon Tamamlandı. En İyi Maliyet: {optimizer.best_cost} USD")
        optimizer.visualize()
    else:
        print("Hata: Veri listesi boş, lütfen raw_data kısmını kontrol et.")