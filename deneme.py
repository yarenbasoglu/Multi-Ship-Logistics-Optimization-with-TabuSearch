import random
import copy
import matplotlib.pyplot as plt

class ShipRouteOptimizer:
    def __init__(self, routes_data):
        self.routes = routes_data
        # Verideki gemileri dinamik olarak al (boş gemi hatasını önler)
        self.available_ships = list(set(r['ship'] for r in self.routes))
        self.loc_region = {f'L{i}': 'North' if i <= 8 else 'South' for i in range(1, 15)}
        self.cost_history = []
        self.best_solution = None
        self.best_cost = float('inf')

    def evaluate(self, solution):
        total_cost = 0
        visited_locs = set()
        
        for ship, assignments in solution.items():
            ship_days = 0
            ship_regs = set()
            for r in assignments:
                total_cost += r['cost']
                ship_days += r['days']
                stops = r['stops'].split('-')
                # Bölge kontrolü
                region = self.loc_region.get(stops[0].strip())
                if region: ship_regs.add(region)
                for s in stops: visited_locs.add(s.strip())
            
            # KISITLAR
            if ship_days > 35: total_cost += 2000000
            if len(ship_regs) > 1: total_cost += 1000000
            
            # Idle Time
            idle = 35 - ship_days
            total_cost += -15000 if idle >= 5 else 10000
            
        # Lokasyon Kapsama
        missing = 14 - len(visited_locs)
        total_cost += (missing * 1000000)
        
        return total_cost, len(visited_locs)

    def run_tabu_search(self, iterations=5000):
        # Gemilere göre rotaları grupla
        routes_by_ship = {s: [r for r in self.routes if r['ship'] == s] for s in self.available_ships}
        
        # Başlangıç çözümü (Hata denetimli)
        current_sol = {s: [random.choice(routes_by_ship[s])] for s in self.available_ships}
        
        self.best_solution = copy.deepcopy(current_sol)
        self.best_cost, _ = self.evaluate(self.best_solution)
        self.cost_history.append(self.best_cost)
        
        tabu = []
        print(f"Optimizasyon başladı ({iterations} iterasyon)...")

        for i in range(iterations):
            neighbor = copy.deepcopy(current_sol)
            s_pick = random.choice(self.available_ships)
            
            # Maksimum 2 rota kısıtı
            num_k = random.randint(1, min(2, len(routes_by_ship[s_pick])))
            neighbor[s_pick] = random.sample(routes_by_ship[s_pick], num_k)
            
            cost_n, v_count = self.evaluate(neighbor)
            
            sol_key = str({s: [r['id'] for r in rs] for s, rs in neighbor.items()})
            if sol_key not in tabu:
                current_sol = neighbor
                tabu.append(sol_key)
                if len(tabu) > 50: tabu.pop(0)
                
                if cost_n < self.best_cost:
                    self.best_cost = cost_n
                    self.best_solution = copy.deepcopy(neighbor)
                    self.cost_history.append(self.best_cost)
                    if v_count == 14 and self.best_cost < 2000000:
                        print(f"İterasyon {i}: Maliyet iyileşti -> {self.best_cost}")

    def visualize(self):
        if not self.cost_history: return

        plt.figure(figsize=(14, 6))
        
        # 1. Maliyet Düşüşü
        plt.subplot(1, 2, 1)
        plt.plot(self.cost_history, color='red', marker='x', markersize=4)
        plt.title('Maliyetin Optimizasyon Süreci')
        plt.xlabel('İyileşme Adımları')
        plt.ylabel('Maliyet (USD)')
        plt.grid(alpha=0.3)

        # 2. Gemi Kullanımı
        plt.subplot(1, 2, 2)
        usage = [sum(r['days'] for r in self.best_solution[s]) for s in self.available_ships]
        colors = ['orange' if u > 35 else 'teal' for u in usage]
        plt.bar(self.available_ships, usage, color=colors)
        plt.axhline(y=35, color='black', linestyle='--', label='Limit (35)')
        plt.title('Gemi Kullanım Süreleri (Gün)')
        plt.legend()
        
        plt.tight_layout()
        plt.show()

# --- VERİ SETİ VE ÇALIŞTIRMA ---
if __name__ == "__main__":
    # Veri temizleme: Başındaki/sonundaki boşlukları temizleyerek listeye alıyoruz
    raw_data = """R1,S1,L1-L2,7,52000
R2,S1,L1-L2,7,55500
R11,S1,L9,5,43000
R13,S2,L1-L2,7,59000
R23,S2,L10,6,51000
R35,S3,L11,6,50500
R47,S4,L13,5,50000
R60,S5,L14,6,68000
R66,S6,L5-L6,7,67500
R70,S6,L7-L8,6,57000
R79,S7,L5-L6,6,61500
R89,S8,L10,6,42000
R106,S9,L12,6,52500
R120,S10,L14,6,60500""" # Buraya tüm 120 satırı virgüllü halde ekle

    data_list = []
    for line in raw_data.strip().split('\n'):
        p = line.split(',')
        data_list.append({
            'id': p[0].strip(), 'ship': p[1].strip(), 'stops': p[2].strip(),
            'days': float(p[3]), 'cost': float(p[4])
        })
    
    optimizer = ShipRouteOptimizer(data_list)
    optimizer.run_tabu_search(iterations=5000)
    
    print(f"\nSonuç: {optimizer.best_cost} USD")
    optimizer.visualize()