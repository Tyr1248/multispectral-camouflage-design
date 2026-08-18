"""Genetic algorithm optimizer (single objective — optimize Band1 emissivity)

Supports two modes:
- "periodic": alternating Ge/ZnS
- "penalty":  8 materials + layer-count penalty

Fixes:
- Crossover and mutation are separated
- Offspring are automatically repaired to satisfy the adjacent-layer constraint
- Elitism + tournament selection
"""

import random
import numpy as np
import time
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
from optical_film_problem import OpticalFilmProblem
from config import (get_default_config, get_periodic_config, get_penalty_config,
                    get_fast_test_config, get_high_precision_config, get_large_population_config)


class GeneticAlgorithmOptimizer:
    def __init__(self, config=None):
        self.config = config if config else get_default_config()
        self.optical_problem = OpticalFilmProblem(self.config)
        self.mode = self.config.mode

        self.material_bits = self.config.material_bits
        self.max_layers = self.config.max_layers
        self.min_layers = self.config.min_layers

        self.results_dir = os.path.abspath(
            f"results_ga_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        os.makedirs(self.results_dir, exist_ok=True)
        self.save_config()

        self.evaluation_stats = {
            'total_evaluations': 0,
            'total_individuals': 0,
            'generation_stats': [],
            'best_fitness_history': []
        }

        # GA-specific parameters (set defaults if not present in config)
        if not hasattr(self.config, 'crossover_rate'):
            self.config.crossover_rate = 0.8
        if not hasattr(self.config, 'mutation_rate'):
            self.config.mutation_rate = 0.15
        if not hasattr(self.config, 'elite_ratio'):
            self.config.elite_ratio = 0.1

    def save_config(self):
        with open(os.path.join(self.results_dir, "ga_config.txt"), 'w') as f:
            f.write(str(self.config))

    # ========== Crossover (without mutation) ==========

    def activation_aware_crossover(self, a, b):
        """Activation-state-aware crossover — handles differences in layer count"""
        child = np.zeros_like(a)
        child[0:self.material_bits + 1] = a[0:self.material_bits + 1]

        for i in range(1, self.max_layers):
            start = i * (self.material_bits + 1)
            end = start + self.material_bits + 1
            a_active = self.optical_problem.is_layer_active(a, i)
            b_active = self.optical_problem.is_layer_active(b, i)

            if a_active and b_active:
                if random.random() < 0.5:
                    child[start:start + self.material_bits] = a[start:start + self.material_bits]
                else:
                    child[start:start + self.material_bits] = b[start:start + self.material_bits]
                child[start + self.material_bits] = \
                    (a[start + self.material_bits] + b[start + self.material_bits]) / 2
            elif a_active and not b_active:
                if random.random() < 0.7:
                    child[start:end] = a[start:end]
            elif not a_active and b_active:
                if random.random() < 0.7:
                    child[start:end] = b[start:end]

        child = self.optical_problem.repair_adjacent_layers_in_decision(child)
        return child

    def crossover(self, a, b):
        return self.activation_aware_crossover(a, b)

    # ========== Mutation ==========

    def mutation(self, solution):
        if random.random() < self.config.mutation_rate:
            solution = self.optical_problem.optical_mutation(solution)
        solution = self.optical_problem.repair_adjacent_layers_in_decision(solution)
        return solution

    # ========== Initialization ==========

    def initialize_optical_population(self):
        return [self.optical_problem.create_optical_film_decision()
                for _ in range(self.config.pop_size)]

    # ========== Fitness ==========

    def evaluate_fitness(self, population):
        obj1, obj2, obj3 = self.optical_problem.get_objective_values(population)
        self.evaluation_stats['total_evaluations'] += 1
        self.evaluation_stats['total_individuals'] += len(population)
        return obj1, obj2, obj3

    # ========== Selection ==========

    def tournament_selection(self, population, fitness_values, tournament_size=3):
        selected = []
        for _ in range(len(population)):
            candidates = random.sample(range(len(population)), tournament_size)
            best_idx = min(candidates, key=lambda i: fitness_values[i])
            selected.append(population[best_idx])
        return selected

    def elitism_selection(self, population, fitness_values, elite_size):
        sorted_idx = np.argsort(fitness_values)
        return [population[i] for i in sorted_idx[:elite_size]]

    # ========== Convergence detection ==========

    def check_convergence(self, best_history, generation):
        if generation < self.config.convergence_criteria['min_generations']:
            return False
        if len(best_history) > self.config.convergence_criteria['max_stagnation']:
            recent = best_history[-self.config.convergence_criteria['max_stagnation']:]
            if max(recent) - min(recent) < 1e-6:
                return True
        return False

    # ========== Main optimization loop ==========

    def optimize(self):
        mode_desc = "周期性结构 (Ge/ZnS交替)" if self.mode == "periodic" else "层数惩罚 (8种材料)"
        print(f"开始遗传算法优化（单目标 Band1）— {mode_desc}...")
        start_time = time.time()

        population = self.initialize_optical_population()
        obj1, obj2, obj3 = self.evaluate_fitness(population)

        gen_no = 0
        best_idx = np.argmin(obj1)
        best_fitness = obj1[best_idx]
        best_solution = population[best_idx].copy()
        fitness_history = [best_fitness]

        self.evaluation_stats['generation_stats'].append({
            'generation': 0, 'evaluations': 1,
            'individuals': len(population), 'best_fitness': best_fitness
        })

        while gen_no < self.config.max_gen:
            self.optical_problem.update_parameters(gen_no, self.config.max_gen)

            parents = self.tournament_selection(population, obj1)

            offspring = []
            for i in range(0, len(parents), 2):
                if i + 1 < len(parents):
                    if random.random() < self.config.crossover_rate:
                        c1 = self.crossover(parents[i], parents[i + 1])
                        c2 = self.crossover(parents[i + 1], parents[i])
                    else:
                        c1 = parents[i].copy()
                        c2 = parents[i + 1].copy()
                    c1 = self.mutation(c1)
                    c2 = self.mutation(c2)
                    offspring.extend([c1, c2])
            offspring = offspring[:self.config.pop_size]

            o1, o2, o3 = self.evaluate_fitness(offspring)

            combined_pop = population + offspring
            combined_fit = list(obj1) + list(o1)
            combined_o2 = list(obj2) + list(o2)
            combined_o3 = list(obj3) + list(o3)

            elite_size = max(1, int(self.config.elite_ratio * self.config.pop_size))
            elites = self.elitism_selection(combined_pop, combined_fit, elite_size)

            elite_idx = set(np.argsort(combined_fit)[:elite_size])
            non_elite = [(combined_pop[i], combined_fit[i])
                         for i in range(len(combined_pop)) if i not in elite_idx]
            if non_elite:
                non_elite_pop, non_elite_fit = zip(*non_elite)
                selected_rem = self.tournament_selection(
                    list(non_elite_pop), list(non_elite_fit),
                    tournament_size=2
                )[:self.config.pop_size - elite_size]
            else:
                selected_rem = []

            new_pop = elites + selected_rem
            population = new_pop
            obj1, obj2, obj3 = self.evaluate_fitness(population)

            curr_best_idx = np.argmin(obj1)
            if obj1[curr_best_idx] < best_fitness:
                best_fitness = obj1[curr_best_idx]
                best_solution = population[curr_best_idx].copy()
                if self.config.verbose:
                    print(f"第{gen_no}代: 新最优 = {best_fitness:.6f}")

            fitness_history.append(best_fitness)

            self.evaluation_stats['generation_stats'].append({
                'generation': gen_no + 1, 'evaluations': 2,
                'individuals': len(offspring) + len(new_pop),
                'best_fitness': best_fitness,
                'avg_fitness': float(np.mean(obj1))
            })

            if self.config.verbose and (gen_no % 10 == 0 or gen_no == self.config.max_gen - 1):
                avg_fit = np.mean(obj1)
                print(f"第{gen_no}代: 最优={best_fitness:.6f}, 平均={avg_fit:.6f}")

            gen_no += 1
            if self.check_convergence(fitness_history, gen_no):
                print(f"在第{gen_no}代提前收敛")
                break

        total_time = time.time() - start_time
        print(f"\n优化完成，用时{total_time:.1f}s，最优适应度={best_fitness:.6f}")

        mats, thicks = self.optical_problem.decode_solution(best_solution)
        print(f"最佳结构: {mats}")
        print(f"厚度: {[round(t, 1) for t in thicks]}")

        self.save_evaluation_stats()
        return population, obj1, obj2, obj3, best_solution, fitness_history

    def save_evaluation_stats(self):
        stats_file = os.path.join(self.results_dir, "evaluation_statistics.json")
        with open(stats_file, 'w') as f:
            json.dump(self.evaluation_stats, f, indent=2)

    # ========== Visualization and saving ==========

    def visualize_results(self, population, obj1, obj2, obj3, best_solution, fitness_history):
        if not self.config.plot_results:
            return
        print("生成可视化...")

        fig = plt.figure(figsize=(15, 10))

        # Convergence curve
        ax1 = plt.subplot(2, 3, 1)
        ax1.plot(fitness_history, 'b-', linewidth=2)
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Best Fitness (Band1)')
        ax1.set_title('Convergence Curve')
        ax1.grid(True, alpha=0.3)

        # Fitness distribution
        ax2 = plt.subplot(2, 3, 2)
        ax2.hist(obj1, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.axvline(min(obj1), color='red', linestyle='--', linewidth=2,
                    label=f'Best: {min(obj1):.6f}')
        ax2.set_xlabel('Fitness (Band1)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Final Fitness Distribution')
        ax2.legend()

        # 3D objective space
        ax3 = plt.subplot(2, 3, 3, projection='3d')
        ax3.scatter(obj1, obj2, obj3, c=obj1, cmap='viridis', s=20, alpha=0.5)
        best_idx = np.argmin(obj1)
        ax3.scatter([obj1[best_idx]], [obj2[best_idx]], [obj3[best_idx]],
                    c='red', s=100, marker='*')
        ax3.set_xlabel('Band1'); ax3.set_ylabel('Band2'); ax3.set_zlabel('Band3')
        ax3.set_title('Objective Space')

        # Band1 vs Band2
        ax4 = plt.subplot(2, 3, 4)
        sc = ax4.scatter(obj1, obj2, c=obj3, cmap='plasma', s=30, alpha=0.7)
        ax4.set_xlabel('Band1'); ax4.set_ylabel('Band2')
        ax4.set_title('Band1 vs Band2')
        plt.colorbar(sc, ax=ax4, label='Band3')

        # Layer count distribution
        ax5 = plt.subplot(2, 3, 5)
        layer_counts = [self.optical_problem.count_active_layers(ind) for ind in population]
        unique_layers = sorted(set(layer_counts))
        layer_dist = [layer_counts.count(l) for l in unique_layers]
        ax5.bar(unique_layers, layer_dist, color='lightcoral', edgecolor='black')
        ax5.set_xlabel('Active Layers')
        ax5.set_ylabel('Count')
        ax5.set_title('Layer Count Distribution')

        # Thicknesses of the best solution
        ax6 = plt.subplot(2, 3, 6)
        mats, thicks = self.optical_problem.decode_solution(best_solution)
        active_thicks = [t for t in thicks if t > 0]
        ax6.bar(range(1, len(active_thicks) + 1), active_thicks,
                color='lightgreen', edgecolor='black')
        ax6.set_xlabel('Layer Index')
        ax6.set_ylabel('Thickness (nm)')
        ax6.set_title(f'Best Solution ({len(active_thicks)} layers)')

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, "ga_results.png"), dpi=150)
        plt.close()

    def save_solutions(self, population, obj1, obj2, obj3, best_solution, top_n=10):
        sorted_idx = np.argsort(obj1)
        data = []
        for rank, idx in enumerate(sorted_idx[:top_n]):
            mats, thicks = self.optical_problem.decode_solution(population[idx])
            data.append({
                'rank': rank + 1,
                'materials': mats,
                'thicknesses': [round(t, 1) for t in thicks],
                'active_layers': self.optical_problem.count_active_layers(population[idx]),
                'objectives': {
                    'band1': round(obj1[idx], 6),
                    'band2': round(obj2[idx], 6),
                    'band3': round(obj3[idx], 6)
                }
            })
        with open(os.path.join(self.results_dir, "ga_solutions.json"), 'w') as f:
            json.dump(data, f, indent=2)
        print(f"解决方案已保存至: {self.results_dir}")


# ========== Main program entry point ==========
if __name__ == "__main__":
    import sys

    mode = "penalty"
    if len(sys.argv) > 1 and sys.argv[1] in ("periodic", "penalty"):
        mode = sys.argv[1]

    config = get_default_config(mode=mode)
    config.crossover_rate = 0.8
    config.mutation_rate = 0.15
    config.elite_ratio = 0.1

    opt = GeneticAlgorithmOptimizer(config)
    pop, o1, o2, o3, best, hist = opt.optimize()
    opt.visualize_results(pop, o1, o2, o3, best, hist)
    opt.save_solutions(pop, o1, o2, o3, best)
