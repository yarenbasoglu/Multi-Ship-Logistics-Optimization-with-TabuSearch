import matplotlib.pyplot as plt
import pandas as pd

class ShipScheduleTracker:
    def __init__(self, final_assignments):
        """
        final_assignments: { 'S1': [{'id': 'R11', 'days': 5, 'stops': 'L9'}, ...], ... }
        """
        self.assignments = final_assignments
        self.max_cycle = 35  # Operasyon döngüsü 35 gün

    def generate_text_report(self):
        """Terminal üzerinde detaylı rota akış raporu basar."""
        print("\n" + "="*80)
        print(f"{'GEMİ':<6} | {'SIRA':<5} | {'ROTA':<6} | {'SÜRE':<6} | {'GÜN ARALIĞI':<15} | {'GÜZERGAH'}")
        print("-"*80)
        
        for ship, routes in self.assignments.items():
            current_day = 0
            for idx, r in enumerate(routes, 1):
                start = current_day
                end = current_day + r['days']
                print(f"{ship:<6} | {idx:<5} | {r['id']:<6} | {r['days']:<6} | {start:>2.1f} - {end:>2.1f}. Gün | {r['stops']}")
                current_day = end
            
            idle = self.max_cycle - current_day
            if idle > 0:
                print(f"{' ': <6} | {'-': <5} | {'BOŞ': <6} | {idle: <6.1f} | {current_day:>2.1f} - 35.0. Gün | LİMANDA BEKLEME")
            print("-"*80)

    def draw_gantt_chart(self):
        """Gemilerin rotalarını zaman tüneli (Gantt) üzerinde görselleştirir."""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        ship_names = list(self.assignments.keys())
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

        for i, ship in enumerate(ship_names):
            current_day = 0
            for r_idx, r in enumerate(self.assignments[ship]):
                # Her rota için bir bar çiz
                ax.broken_barh([(current_day, r['days'])], (i*10 + 2, 6), 
                                facecolors=colors[i % len(colors)], edgecolors='black')
                
                # Rota ismini barın üzerine yaz
                ax.text(current_day + r['days']/2, i*10 + 5, r['id'], 
                        ha='center', va='center', color='white', fontweight='bold')
                
                current_day += r['days']

        # Grafik Ayarları
        ax.set_ylim(0, len(ship_names) * 10)
        ax.set_xlim(0, self.max_cycle)
        ax.set_xlabel('Gün (35 Günlük Periyot)')
        ax.set_ylabel('Gemiler')
        ax.set_yticks([i*10 + 5 for i in range(len(ship_names))])
        ax.set_yticklabels(ship_names)
        ax.grid(True, axis='x', linestyle='--', alpha=0.7)
        ax.axvline(x=35, color='red', linestyle='-', linewidth=2, label='Döngü Sonu')
        
        plt.title('Gemi Operasyonel Rota ve Zaman Çizelgesi', fontsize=14)
        plt.tight_layout()
        plt.show()

# --- ÖRNEK KULLANIM ---
if __name__ == "__main__":
    # Optimizasyondan gelen örnek başarılı sonuç verisi
    best_plan = {
        'S1': [{'id': 'R7', 'days': 6, 'stops': 'L5-L6'}, {'id': 'R9', 'days': 6, 'stops': 'L3-L4'}],
        'S4': [{'id': 'R48', 'days': 5, 'stops': 'L9'}, {'id': 'R41', 'days': 6, 'stops': 'L10'}],
        'S10': [{'id': 'R111', 'days': 6, 'stops': 'L1-L2'}, {'id': 'R120', 'days': 6, 'stops': 'L14'}]
        # Diğer gemiler buraya eklenebilir...
    }

    tracker = ShipScheduleTracker(best_plan)
    tracker.generate_text_report()
    tracker.draw_gantt_chart()