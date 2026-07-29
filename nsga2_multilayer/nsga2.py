#!/usr/bin/env python3
"""NSGA-II 光学薄膜多目标优化器

支持两种优化模式:
- "periodic": 周期性结构 — 仅Ge/ZnS, 强制交替
- "penalty":  层数惩罚 — 8种材料, 层数惩罚项

修复内容:
- 拥挤距离按原始索引正确分配
- 交叉与变异完全分离
- 进度打印包含第一前沿示例解
- save_solutions 保存整个第一前沿
"""

import math
import random
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import json
from datetime import datetime

from optical_film_problem import OpticalFilmProblem
from config import (get_default_config, get_periodic_config, get_penalty_config,
                    get_fast_test_config, get_high_precision_config, get_large_population_config)


class NSGA2Optimizer:
    def __init__(self, config=None):
        self.config = config if config else get_default_config()
        self.optical_problem = OpticalFilmProblem(self.config)
        self.mode = self.config.mode

        self.material_bits = self.config.material_bits
        self.max_layers = self.config.max_layers
        self.min_layers = self.config.min_layers

        self.results_dir = os.path.abspath(
            f"results_nsga2_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        os.makedirs(self.results_dir, exist_ok=True)
        self.save_config()

        self.evaluation_stats = {
            'total_evaluations': 0,
            'total_individuals': 0,
            'generation_stats': []
        }

    def save_config(self):
        os.makedirs(self.results_dir, exist_ok=True)
        with open(os.path.join(self.results_dir, "optimization_config.txt"), 'w') as f:
            f.write(str(self.config))

    # ========== 非支配排序与拥挤距离 ==========

    def fast_non_dominated_sort(self, values1, values2, values3):
        size = len(values1)
        S = [[] for _ in range(size)]
        n = [0] * size
        rank = [0] * size
        front = [[]]

        for p in range(size):
            S[p] = []
            n[p] = 0
            for q in range(size):
                if (values1[p] <= values1[q] and values2[p] <= values2[q] and values3[p] <= values3[q]) and \
                   (values1[p] < values1[q] or values2[p] < values2[q] or values3[p] < values3[q]):
                    if q not in S[p]:
                        S[p].append(q)
                elif (values1[q] <= values1[p] and values2[q] <= values2[p] and values3[q] <= values3[p]) and \
                     (values1[q] < values1[p] or values2[q] < values2[p] or values3[q] < values3[p]):
                    n[p] += 1
            if n[p] == 0:
                rank[p] = 0
                if p not in front[0]:
                    front[0].append(p)

        i = 0
        while front[i]:
            Q = []
            for p in front[i]:
                for q in S[p]:
                    n[q] -= 1
                    if n[q] == 0:
                        rank[q] = i + 1
                        if q not in Q:
                            Q.append(q)
            i += 1
            front.append(Q)
        front.pop()
        return front

    def crowding_distance(self, values1, values2, values3, front):
        """计算拥挤距离，返回与front顺序一致的列表。已修复索引错位问题。"""
        if len(front) <= 2:
            return [float('inf')] * len(front)

        dist = {idx: 0.0 for idx in front}
        for values in [values1, values2, values3]:
            sorted_pairs = sorted([(idx, values[idx]) for idx in front], key=lambda x: x[1])
            min_val, max_val = sorted_pairs[0][1], sorted_pairs[-1][1]
            if max_val - min_val == 0:
                continue
            dist[sorted_pairs[0][0]] = float('inf')
            dist[sorted_pairs[-1][0]] = float('inf')
            for k in range(1, len(sorted_pairs) - 1):
                prev_idx, next_idx = sorted_pairs[k - 1][0], sorted_pairs[k + 1][0]
                dist[sorted_pairs[k][0]] += (values[next_idx] - values[prev_idx]) / (max_val - min_val)

        return [dist[idx] for idx in front]

    def get_front_rank(self, individual_idx, fronts):
        for rank, front in enumerate(fronts):
            if individual_idx in front:
                return rank
        return len(fronts)

    # ========== 选择算子 ==========

    def improved_tournament_selection(self, population_size, fronts, crowding_distances):
        tournament_size = 2
        candidates = random.sample(range(population_size), tournament_size)

        def sort_key(i):
            rank = self.get_front_rank(i, fronts)
            crowd = crowding_distances.get(i, 0)
            return (rank, -crowd)

        candidates.sort(key=sort_key)
        return candidates[0]

    # ========== 交叉（仅重组+修复，不含变异）==========

    def activation_aware_crossover(self, a, b):
        child = np.zeros_like(a)
        child[0:self.material_bits + 1] = a[0:self.material_bits + 1]

        for i in range(1, self.max_layers):
            start_idx = i * (self.material_bits + 1)
            end_idx = start_idx + self.material_bits + 1
            a_active = self.optical_problem.is_layer_active(a, i)
            b_active = self.optical_problem.is_layer_active(b, i)

            if a_active and b_active:
                if random.random() < 0.5:
                    child[start_idx:start_idx + self.material_bits] = \
                        a[start_idx:start_idx + self.material_bits]
                else:
                    child[start_idx:start_idx + self.material_bits] = \
                        b[start_idx:start_idx + self.material_bits]
                child[start_idx + self.material_bits] = \
                    (a[start_idx + self.material_bits] + b[start_idx + self.material_bits]) / 2
            elif a_active and not b_active:
                if random.random() < 0.7:
                    child[start_idx:end_idx] = a[start_idx:end_idx]
            elif not a_active and b_active:
                if random.random() < 0.7:
                    child[start_idx:end_idx] = b[start_idx:end_idx]

        child = self.optical_problem.repair_adjacent_layers_in_decision(child)
        return child

    def crossover(self, a, b):
        return self.activation_aware_crossover(a, b)

    # ========== 变异（外部调用，与交叉分离）==========

    def mutation(self, solution):
        if random.random() < 0.2:
            solution = self.optical_problem.optical_mutation(solution)
        solution = self.optical_problem.repair_adjacent_layers_in_decision(solution)
        return solution

    # ========== 种群初始化与评估 ==========

    def initialize_optical_population(self):
        return [self.optical_problem.create_optical_film_decision()
                for _ in range(self.config.pop_size)]

    def batch_evaluate_population(self, population):
        obj1, obj2, obj3 = self.optical_problem.get_objective_values(population)
        self.evaluation_stats['total_evaluations'] += 1
        self.evaluation_stats['total_individuals'] += len(population)
        return obj1, obj2, obj3

    def check_convergence(self, best_front_history, generation):
        if generation < self.config.convergence_criteria['min_generations']:
            return False
        if len(best_front_history) > self.config.convergence_criteria['max_stagnation']:
            recent_improvement = False
            for i in range(1, self.config.convergence_criteria['max_stagnation']):
                if best_front_history[-i] != best_front_history[-i - 1]:
                    recent_improvement = True
                    break
            if not recent_improvement:
                return True
        return False

    # ========== 主优化循环 ==========

    def optimize(self):
        mode_desc = "周期性结构 (Ge/ZnS交替)" if self.mode == "periodic" else "层数惩罚 (8种材料)"
        print(f"开始 NSGA-II 优化 — {mode_desc}...")
        print(self.config)

        start_time = time.time()
        population = self.initialize_optical_population()
        obj1_values, obj2_values, obj3_values = self.batch_evaluate_population(population)

        for i, indiv in enumerate(population):
            lc = self.optical_problem.count_active_layers(indiv)
            if lc < self.min_layers or lc > self.max_layers:
                print(f"警告: 初始个体 {i} 层数 {lc} 超范围")

        gen_no = 0
        best_front_history = []
        self.evaluation_stats['generation_stats'].append({
            'generation': gen_no, 'evaluations': 1,
            'individuals': self.config.pop_size, 'front_size': 0
        })

        while gen_no < self.config.max_gen:
            self.optical_problem.update_parameters(gen_no, self.config.max_gen)

            non_dominated_sorted_solution = self.fast_non_dominated_sort(
                obj1_values[:], obj2_values[:], obj3_values[:])

            individual_crowding = {}
            for front in non_dominated_sorted_solution:
                dist_list = self.crowding_distance(
                    obj1_values[:], obj2_values[:], obj3_values[:], front)
                for idx, d in zip(front, dist_list):
                    individual_crowding[idx] = d

            best_front_size = len(non_dominated_sorted_solution[0])
            best_front_history.append(best_front_size)

            if self.config.verbose and (gen_no % 10 == 0 or gen_no == self.config.max_gen - 1):
                print(f"第 {gen_no} 代:")
                print(f"  非支配前沿大小: {best_front_size}")
                print(f"  目标1范围: [{min(obj1_values):.4f}, {max(obj1_values):.4f}]")
                print(f"  目标2范围: [{min(obj2_values):.4f}, {max(obj2_values):.4f}]")
                print(f"  目标3范围: [{min(obj3_values):.4f}, {max(obj3_values):.4f}]")
                print(f"  累计评估次数: {self.evaluation_stats['total_evaluations']}")
                print(f"  累计评估个体: {self.evaluation_stats['total_individuals']}")

                first_front_indices = non_dominated_sorted_solution[0]
                print("  第一前沿示例解 (至多5个):")
                for i, idx in enumerate(first_front_indices[:5]):
                    layers = self.optical_problem.count_active_layers(population[idx])
                    print(f"    解{i + 1}: 层数={layers}, "
                          f"目标=({obj1_values[idx]:.4f}, {obj2_values[idx]:.4f}, {obj3_values[idx]:.4f})")
                print()

            offspring = []
            for _ in range(self.config.pop_size):
                p1 = self.improved_tournament_selection(
                    len(population), non_dominated_sorted_solution, individual_crowding)
                p2 = self.improved_tournament_selection(
                    len(population), non_dominated_sorted_solution, individual_crowding)
                child = self.crossover(population[p1], population[p2])
                child = self.mutation(child)
                offspring.append(child)

            invalid = sum(1 for c in offspring
                          if self.optical_problem.count_active_layers(c) < self.min_layers
                          or self.optical_problem.count_active_layers(c) > self.max_layers)
            if invalid:
                print(f"警告: {invalid} 个子代层数超范围")

            obj1_off, obj2_off, obj3_off = self.batch_evaluate_population(offspring)

            combined_pop = population + offspring
            combined_obj1 = obj1_values + obj1_off
            combined_obj2 = obj2_values + obj2_off
            combined_obj3 = obj3_values + obj3_off

            combined_fronts = self.fast_non_dominated_sort(
                combined_obj1, combined_obj2, combined_obj3)
            combined_crowding = []
            for front in combined_fronts:
                combined_crowding.append(
                    self.crowding_distance(combined_obj1, combined_obj2, combined_obj3, front))

            new_pop = []
            new_obj1, new_obj2, new_obj3 = [], [], []
            sel_count = 0
            front_idx = 0
            while sel_count < self.config.pop_size and front_idx < len(combined_fronts):
                cur_front = combined_fronts[front_idx]
                cur_crowd = combined_crowding[front_idx]
                if sel_count + len(cur_front) <= self.config.pop_size:
                    for idx in cur_front:
                        new_pop.append(combined_pop[idx])
                        new_obj1.append(combined_obj1[idx])
                        new_obj2.append(combined_obj2[idx])
                        new_obj3.append(combined_obj3[idx])
                    sel_count += len(cur_front)
                else:
                    sorted_idx = sorted(range(len(cur_front)),
                                        key=lambda i: cur_crowd[i], reverse=True)
                    remain = self.config.pop_size - sel_count
                    for i in range(remain):
                        idx = cur_front[sorted_idx[i]]
                        new_pop.append(combined_pop[idx])
                        new_obj1.append(combined_obj1[idx])
                        new_obj2.append(combined_obj2[idx])
                        new_obj3.append(combined_obj3[idx])
                    sel_count = self.config.pop_size
                front_idx += 1

            population = new_pop
            obj1_values, obj2_values, obj3_values = new_obj1, new_obj2, new_obj3

            self.evaluation_stats['generation_stats'].append({
                'generation': gen_no + 1,
                'evaluations': 1,
                'individuals': len(offspring),
                'front_size': best_front_size
            })

            gen_no += 1

            if self.check_convergence(best_front_history, gen_no):
                print(f"在第 {gen_no} 代提前收敛")
                break

        total_time = time.time() - start_time
        final_fronts = self.fast_non_dominated_sort(
            obj1_values[:], obj2_values[:], obj3_values[:])

        print(f"\n优化完成！")
        print(f"总用时: {total_time:.1f}秒")
        print(f"总代数: {gen_no}")
        print(f"总评估次数: {self.evaluation_stats['total_evaluations']}")
        print(f"总评估个体数: {self.evaluation_stats['total_individuals']}")
        print(f"最终帕累托前沿大小: {len(final_fronts[0])}")

        self.analyze_pareto_front(obj1_values, obj2_values, obj3_values, final_fronts)
        self.save_evaluation_stats()
        return population, obj1_values, obj2_values, obj3_values, final_fronts

    def analyze_pareto_front(self, obj1, obj2, obj3, fronts):
        print("\n帕累托前沿分析:")
        for i, front in enumerate(fronts):
            if i >= 5:
                break
            print(f"前沿 {i}: {len(front)} 个解")
            if front:
                print(f"  目标1: [{min(obj1[i] for i in front):.4f}, {max(obj1[i] for i in front):.4f}]")
                print(f"  目标2: [{min(obj2[i] for i in front):.4f}, {max(obj2[i] for i in front):.4f}]")
                print(f"  目标3: [{min(obj3[i] for i in front):.4f}, {max(obj3[i] for i in front):.4f}]")

        first = fronts[0]
        if len(first) > 1:
            o1f, o2f, o3f = [obj1[i] for i in first], [obj2[i] for i in first], [obj3[i] for i in first]
            corr12 = np.corrcoef(o1f, o2f)[0, 1]
            corr13 = np.corrcoef(o1f, o3f)[0, 1]
            corr23 = np.corrcoef(o2f, o3f)[0, 1]
            print(f"  相关性: Band1-2={corr12:.4f}, Band1-3={corr13:.4f}, Band2-3={corr23:.4f}")

    def save_evaluation_stats(self):
        os.makedirs(self.results_dir, exist_ok=True)
        stats_file = os.path.join(self.results_dir, "evaluation_statistics.json")
        with open(stats_file, 'w') as f:
            json.dump(self.evaluation_stats, f, indent=2)

    def visualize_results(self, population, obj1, obj2, obj3, fronts):
        if not self.config.plot_results:
            return
        os.makedirs(self.results_dir, exist_ok=True)
        print("生成可视化...")

        fig = plt.figure(figsize=(18, 6))

        # 3D: 第一前沿
        ax1 = fig.add_subplot(131, projection='3d')
        first = fronts[0]
        o1f = [obj1[i] for i in first]
        o2f = [obj2[i] for i in first]
        o3f = [obj3[i] for i in first]
        ax1.scatter(o1f, o2f, o3f, c=o1f, cmap='viridis')
        ax1.set_title('First Pareto Front')
        ax1.set_xlabel('Band1'); ax1.set_ylabel('Band2'); ax1.set_zlabel('Band3')

        # 3D: 前三个前沿
        ax2 = fig.add_subplot(132, projection='3d')
        colors = ['red', 'blue', 'green']
        for i in range(min(3, len(fronts))):
            o1f = [obj1[idx] for idx in fronts[i]]
            o2f = [obj2[idx] for idx in fronts[i]]
            o3f = [obj3[idx] for idx in fronts[i]]
            ax2.scatter(o1f, o2f, o3f, c=colors[i], s=30, alpha=0.6, label=f'Front {i + 1}')
        ax2.set_xlabel('Band1'); ax2.set_ylabel('Band2'); ax2.set_zlabel('Band3')
        ax2.set_title('First Three Pareto Fronts')
        ax2.legend()

        # 3D: 所有解按前沿着色
        ax3 = fig.add_subplot(133, projection='3d')
        front_colors = [self.get_front_rank(idx, fronts) for idx in range(len(population))]
        ax3.scatter(obj1, obj2, obj3, c=front_colors, cmap='plasma', s=20, alpha=0.6)
        ax3.set_xlabel('Band1'); ax3.set_ylabel('Band2'); ax3.set_zlabel('Band3')
        ax3.set_title('All Solutions (Colored by Front Rank)')

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, "pareto_front.png"), dpi=150)
        plt.close()

    def save_solutions(self, population, obj1_values, obj2_values, obj3_values, fronts):
        """保存整个第一帕累托前沿"""
        os.makedirs(self.results_dir, exist_ok=True)
        print("保存解决方案（完整第一前沿）...")
        solutions_data = []

        for idx, solution_idx in enumerate(fronts[0]):
            materials, thicknesses = self.optical_problem.decode_solution(population[solution_idx])
            active_layers = self.optical_problem.count_active_layers(population[solution_idx])

            solution_info = {
                'id': idx + 1,
                'front_rank': 1,
                'active_layers': active_layers,
                'materials': materials,
                'thicknesses': [round(t, 1) for t in thicknesses],
                'objectives': {
                    'band1': round(obj1_values[solution_idx], 6),
                    'band2': round(obj2_values[solution_idx], 6),
                    'band3': round(obj3_values[solution_idx], 6)
                }
            }
            solutions_data.append(solution_info)

            if idx < 5:
                print(f"解 {idx + 1}:")
                print(f"  激活层数: {active_layers}")
                print(f"  材料序列: {materials}")
                print(f"  厚度序列: {[round(t, 1) for t in thicknesses]}")
                print(f"  目标值: Band1={obj1_values[solution_idx]:.4f}, "
                      f"Band2={obj2_values[solution_idx]:.4f}, Band3={obj3_values[solution_idx]:.4f}")
                print()

        solutions_file = os.path.join(self.results_dir, "optimal_solutions.json")
        with open(solutions_file, 'w') as f:
            json.dump(solutions_data, f, indent=2)
        print(f"解决方案已保存至: {solutions_file} (共 {len(fronts[0])} 个非支配解)")


# ========== 主程序入口 ==========
if __name__ == "__main__":
    import sys

    # 从命令行或默认选择模式
    mode = "penalty"
    if len(sys.argv) > 1 and sys.argv[1] in ("periodic", "penalty"):
        mode = sys.argv[1]

    config = get_default_config(mode=mode)
    # 可调整 config.pop_size 等参数
    optimizer = NSGA2Optimizer(config)
    pop, o1, o2, o3, fronts = optimizer.optimize()
    optimizer.visualize_results(pop, o1, o2, o3, fronts)
    optimizer.save_solutions(pop, o1, o2, o3, fronts)
