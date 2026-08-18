import numpy as np
import random
import copy
from cal_emissivity import batch_calculate_weighted_score
from config import OptimizationConfig


class OpticalFilmProblem:
    """Optical thin-film optimization problem definition

    Supports two modes:
    - "periodic": alternating Ge/ZnS (material_bits=1)
    - "penalty":  8 materials, layer-count penalty (material_bits=3)
    """

    def __init__(self, config=None):
        self.config = config if config else OptimizationConfig()
        self.mode = self.config.mode

        self.max_layers = self.config.max_layers
        self.min_layers = self.config.min_layers
        self.thickness_range = self.config.thickness_range
        self.material_bits = self.config.material_bits
        self.total_vars = self.max_layers * (self.material_bits + 1)

        self.layer_penalty_weight = getattr(self.config, 'layer_penalty_weight', 0.01)
        self.ga_weights = getattr(self.config, 'ga_weights', [0.33, 0.33, 0.33])
        self.ga_use_laser_term = getattr(self.config, 'ga_use_laser_term', False)

        self.layer_mut_prob = self.config.layer_mut_prob
        self.material_mut_prob = self.config.material_mut_prob
        self.mut_prob = self.config.get_mutation_prob()
        self.eta_m = self.config.eta_m

        self.max_batch_size = getattr(self.config, 'max_batch_size', 200)

        if self.mode == "periodic":
            self.material_mapping = {0: "Ge", 1: "ZnS"}
        else:
            self.material_mapping = {
                0: "SiO2", 1: "ZnS", 2: "aSi", 3: "TiO2",
                4: "Ge", 5: "HfO2", 6: "ZnSe", 7: "Al2O3"
            }
        self.reverse_material_mapping = {v: k for k, v in self.material_mapping.items()}
        self.num_materials = len(self.material_mapping)

        self.config.validate()

    # ========== Constraint checking and repair ==========

    def check_adjacent_layer_constraint(self, materials):
        """Check whether adjacent layers use the same material; returns a list of conflicting positions"""
        conflicts = []
        for i in range(1, len(materials)):
            if materials[i] == materials[i - 1]:
                conflicts.append(i)
        return conflicts

    def repair_adjacent_layers(self, materials, thicknesses=None):
        """Repair the adjacent-layer material constraint

        periodic mode: force alternating Ge/ZnS
        penalty mode: randomly choose a material different from the previous layer
        """
        if len(materials) <= 1:
            return materials.copy(), thicknesses.copy() if thicknesses else None

        repaired_materials = list(materials)
        repaired_thicknesses = thicknesses.copy() if thicknesses else None

        for i in range(1, len(repaired_materials)):
            if repaired_materials[i] == repaired_materials[i - 1]:
                if self.mode == "periodic":
                    repaired_materials[i] = "Ge" if repaired_materials[i - 1] == "ZnS" else "ZnS"
                else:
                    available = [m for m in self.material_mapping.values()
                                 if m != repaired_materials[i - 1]]
                    repaired_materials[i] = random.choice(available)

        return repaired_materials, repaired_thicknesses

    def repair_adjacent_layers_in_decision(self, decision):
        """Decode -> repair the material constraint -> write back into the decision vector"""
        materials, thicknesses = self.decode_solution(decision)
        conflicts = self.check_adjacent_layer_constraint(materials)
        if not conflicts:
            return decision

        repaired_mats, _ = self.repair_adjacent_layers(materials, thicknesses)
        new_dec = decision.copy()
        for i, mat_name in enumerate(repaired_mats):
            if i >= self.max_layers:
                break
            start = i * (self.material_bits + 1)
            code = self.reverse_material_mapping[mat_name]
            if self.mode == "periodic":
                new_dec[start] = code
            else:
                for j in range(self.material_bits):
                    bit = (code >> j) & 1
                    new_dec[start + j] = bit
        return new_dec

    # ========== Layer-count constraint ==========

    def enforce_layer_constraints(self, decision):
        """Enforce the layer-count constraints"""
        new_decision = decision.copy()
        active = self.count_active_layers(new_decision)

        while active < self.min_layers:
            for i in range(1, self.max_layers):
                start = i * (self.material_bits + 1)
                if not any(new_decision[start:start + self.material_bits] > 0.5):
                    if self.mode == "periodic":
                        prev_code = self._get_material_at_periodic(new_decision, i - 1)
                        new_code = 1 - prev_code
                        new_decision[start] = new_code
                    else:
                        prev_code = self._get_material_at_penalty(new_decision, i - 1)
                        available = [c for c in range(self.num_materials) if c != prev_code]
                        new_code = random.choice(available)
                        for j in range(self.material_bits):
                            bit = (new_code >> j) & 1
                            new_decision[start + j] = bit
                    new_decision[start + self.material_bits] = random.random()
                    active += 1
                    break
            else:
                break

        while active > self.max_layers:
            for i in range(self.max_layers - 1, 0, -1):
                start = i * (self.material_bits + 1)
                if any(new_decision[start:start + self.material_bits] > 0.5):
                    new_decision[start:start + self.material_bits + 1] = 0
                    active -= 1
                    break

        if self.mode == "periodic":
            materials, thicknesses = self.decode_solution(new_decision)
            conflicts = self.check_adjacent_layer_constraint(materials)
            if conflicts:
                repaired_mats, _ = self.repair_adjacent_layers(materials, thicknesses)
                new_decision = self._encode_materials_to_decision(new_decision, repaired_mats)

        return new_decision

    def _get_material_at_periodic(self, decision, layer_idx):
        """periodic mode: get the material code of the given layer"""
        start = layer_idx * (self.material_bits + 1)
        return int(decision[start] > 0.5)

    def _get_material_at_penalty(self, decision, layer_idx):
        """penalty mode: get the material code of the given layer"""
        start = layer_idx * (self.material_bits + 1)
        code = 0
        for j in range(self.material_bits):
            if decision[start + j] > 0.5:
                code += (1 << j)
        return code

    def _encode_materials_to_decision(self, decision, materials):
        """Encode a list of material names back into the decision vector (periodic mode)"""
        new_dec = decision.copy()
        for i, mat_name in enumerate(materials):
            if i >= self.max_layers:
                break
            start = i * (self.material_bits + 1)
            code = self.reverse_material_mapping[mat_name]
            new_dec[start] = code
        return new_dec

    # ========== Initialization ==========

    def create_optical_film_decision(self):
        """Create a random individual that satisfies the constraints"""
        num_layers = random.randint(self.min_layers, self.max_layers)
        decision = np.zeros(self.total_vars)

        if self.mode == "periodic":
            decision[0] = 0  # Ge
            min_thick_norm = (self.config.first_layer_min_thickness - self.thickness_range[0]) / (
                    self.thickness_range[1] - self.thickness_range[0])
            decision[1] = min_thick_norm + (1 - min_thick_norm) * random.random()
            prev_code = 0
            for i in range(1, self.max_layers):
                start = i * (self.material_bits + 1)
                if i < num_layers:
                    new_code = 1 - prev_code
                    decision[start] = new_code
                    decision[start + self.material_bits] = random.random()
                    prev_code = new_code
        else:
            decision[0] = 1  # Ge=4: binary 100
            decision[1] = 0
            decision[2] = 0
            min_thick_norm = (self.config.first_layer_min_thickness - self.thickness_range[0]) / (
                    self.thickness_range[1] - self.thickness_range[0])
            decision[3] = min_thick_norm + (1 - min_thick_norm) * random.random()
            prev_code = 4
            for i in range(1, self.max_layers):
                start_idx = i * (self.material_bits + 1)
                if i < num_layers:
                    available = [c for c in range(self.num_materials) if c != prev_code]
                    if i == 1:
                        available = [c for c in available if c != 4]
                    material_code = random.choice(available)
                    prev_code = material_code
                    for j in range(self.material_bits):
                        bit = (material_code >> j) & 1
                        decision[start_idx + j] = bit
                    decision[start_idx + self.material_bits] = random.random()

        return decision

    # ========== Decoding ==========

    def decode_solution(self, decision):
        """Decode the decision vector into (material list, thickness list)"""
        materials = []
        thicknesses = []

        if self.mode == "periodic":
            materials.append("Ge")
            thick_norm = decision[1]
            thickness = self.thickness_range[0] + thick_norm * (self.thickness_range[1] - self.thickness_range[0])
            if thickness < self.config.first_layer_min_thickness:
                thickness = self.config.first_layer_min_thickness
            thicknesses.append(thickness)

            for i in range(1, self.max_layers):
                start = i * (self.material_bits + 1)
                mat_code = int(decision[start] > 0.5)
                thickness_norm = decision[start + self.material_bits]
                if thickness_norm > 0:
                    materials.append("ZnS" if mat_code == 1 else "Ge")
                    thickness = self.thickness_range[0] + thickness_norm * (self.thickness_range[1] - self.thickness_range[0])
                    thicknesses.append(thickness)
        else:
            material_code = 4  # Ge
            materials.append(self.material_mapping[material_code])
            thick_norm = decision[3]
            thickness = self.thickness_range[0] + thick_norm * (self.thickness_range[1] - self.thickness_range[0])
            if thickness < self.config.first_layer_min_thickness:
                thickness = self.config.first_layer_min_thickness + random.random() * (
                        self.thickness_range[1] - self.config.first_layer_min_thickness)
            thicknesses.append(thickness)

            for i in range(1, self.max_layers):
                start_idx = i * (self.material_bits + 1)
                if any(decision[start_idx:start_idx + self.material_bits] > 0.5):
                    code = 0
                    for j in range(self.material_bits):
                        if decision[start_idx + j] > 0.5:
                            code += (1 << j)
                    materials.append(self.material_mapping.get(code, f"Material_{code}"))
                    thick_norm = decision[start_idx + self.material_bits]
                    thickness = self.thickness_range[0] + thick_norm * (self.thickness_range[1] - self.thickness_range[0])
                    thicknesses.append(thickness)

        # Force-fill when the layer count is insufficient
        while len(materials) < self.min_layers:
            if self.mode == "periodic":
                prev_mat = materials[-1]
                next_mat = "ZnS" if prev_mat == "Ge" else "Ge"
            else:
                prev_mat = materials[-1]
                available = [m for m in self.material_mapping.values() if m != prev_mat]
                next_mat = random.choice(available)
            materials.append(next_mat)
            thicknesses.append(random.uniform(*self.thickness_range))

        # Repair the adjacent-layer constraint
        conflicts = self.check_adjacent_layer_constraint(materials)
        if conflicts:
            materials, thicknesses = self.repair_adjacent_layers(materials, thicknesses)

        return materials, thicknesses

    def decode_solution_with_repair(self, decision):
        """Decode and repair"""
        return self.decode_solution(decision)

    # ========== Activation state ==========

    def is_layer_active(self, decision, layer_idx):
        """Determine whether the given layer is active"""
        if layer_idx == 0:
            return True
        start = layer_idx * (self.material_bits + 1)
        if self.mode == "periodic":
            return any(decision[start:start + self.material_bits + 1] > 0.5)
        else:
            return any(decision[start:start + self.material_bits] > 0.5)

    def count_active_layers(self, decision):
        """Count the number of active layers"""
        count = 1
        for i in range(1, self.max_layers):
            if self.is_layer_active(decision, i):
                count += 1
        return count

    def can_add_layer(self, decision):
        return self.count_active_layers(decision) < self.max_layers

    def can_remove_layer(self, decision):
        return self.count_active_layers(decision) > self.min_layers

    # ========== Mutation operators ==========

    def layer_mutation(self, decision):
        """Layer-count mutation"""
        new_dec = decision.copy()

        if self.mode == "periodic":
            if random.random() < 0.5 and self.count_active_layers(new_dec) < self.max_layers:
                for i in range(1, self.max_layers):
                    if not self.is_layer_active(new_dec, i):
                        prev_code = self._get_material_at_periodic(new_dec, i - 1) if i > 0 else 0
                        new_code = 1 - prev_code
                        start = i * (self.material_bits + 1)
                        new_dec[start] = new_code
                        new_dec[start + self.material_bits] = random.random()
                        break
            elif self.count_active_layers(new_dec) > self.min_layers:
                for i in range(self.max_layers - 1, 0, -1):
                    if self.is_layer_active(new_dec, i):
                        start = i * (self.material_bits + 1)
                        new_dec[start:start + self.material_bits + 1] = 0
                        break
            return self.enforce_layer_constraints(new_dec)
        else:
            can_add = self.can_add_layer(new_dec)
            can_remove = self.can_remove_layer(new_dec)
            if not can_add and not can_remove:
                return self.enforce_layer_constraints(new_dec)

            if can_add and (not can_remove or random.random() < 0.5):
                for i in range(1, self.max_layers):
                    start_idx = i * (self.material_bits + 1)
                    if not any(new_dec[start_idx:start_idx + self.material_bits] > 0.5):
                        prev_code = self._get_material_at_penalty(new_dec, i - 1)
                        available = [c for c in range(self.num_materials) if c != prev_code]
                        material_code = random.choice(available)
                        for j in range(self.material_bits):
                            bit = (material_code >> j) & 1
                            new_dec[start_idx + j] = bit
                        new_dec[start_idx + self.material_bits] = random.random()
                        break
            elif can_remove:
                active_indices = [i for i in range(1, self.max_layers)
                                  if self.is_layer_active(new_dec, i)]
                if active_indices:
                    layer_to_remove = random.choice(active_indices)
                    start_idx = layer_to_remove * (self.material_bits + 1)
                    for j in range(self.material_bits + 1):
                        new_dec[start_idx + j] = 0
            return self.enforce_layer_constraints(new_dec)

    def material_mutation(self, decision):
        """Material mutation

        periodic mode: meaningless (alternation is enforced), return directly
        penalty mode: mutate materials of layers other than the first, keeping adjacent layers different
        """
        if self.mode == "periodic":
            return decision

        new_decision = copy.deepcopy(decision)
        active_indices = []
        for i in range(1, self.max_layers):
            start_idx = i * (self.material_bits + 1)
            if any(new_decision[start_idx:start_idx + self.material_bits] > 0.5):
                active_indices.append(i)

        if active_indices:
            layer_idx = random.choice(active_indices)
            start_idx = layer_idx * (self.material_bits + 1)

            current_code = self._get_material_at_penalty(new_decision, layer_idx)
            prev_code = -1
            next_code = -1

            if layer_idx > 0:
                prev_code = self._get_material_at_penalty(new_decision, layer_idx - 1)
            if layer_idx < self.max_layers - 1:
                next_start = (layer_idx + 1) * (self.material_bits + 1)
                if any(new_decision[next_start:next_start + self.material_bits] > 0.5):
                    next_code = self._get_material_at_penalty(new_decision, layer_idx + 1)

            available = [c for c in range(self.num_materials) if c != current_code]
            if prev_code >= 0 and prev_code in available:
                available.remove(prev_code)
            if next_code >= 0 and next_code in available:
                available.remove(next_code)

            if available:
                new_code = random.choice(available)
                for j in range(self.material_bits):
                    bit = (new_code >> j) & 1
                    new_decision[start_idx + j] = bit

        return new_decision

    def polynomial_mutate_single(self, x, eta, low, up):
        """Polynomial mutation of a single variable"""
        if random.random() < self.mut_prob:
            delta1 = (x - low) / (up - low)
            delta2 = (up - x) / (up - low)
            rand = random.random()
            mut_pow = 1.0 / (eta + 1.0)
            if rand < 0.5:
                xy = 1.0 - delta1
                val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (eta + 1.0))
                deltaq = val ** mut_pow - 1.0
            else:
                xy = 1.0 - delta2
                val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (eta + 1.0))
                deltaq = 1.0 - (val ** mut_pow)
            x = x + deltaq * (up - low)
            x = min(max(x, low), up)
        return x

    def thickness_mutation(self, decision):
        """Thickness mutation"""
        new_decision = copy.deepcopy(decision)
        for i in range(self.max_layers):
            start_idx = i * (self.material_bits + 1)
            if self.is_layer_active(new_decision, i) or i == 0:
                thickness_idx = start_idx + self.material_bits
                if i == 0:
                    min_norm = (self.config.first_layer_min_thickness - self.thickness_range[0]) / (
                            self.thickness_range[1] - self.thickness_range[0])
                    new_decision[thickness_idx] = self.polynomial_mutate_single(
                        new_decision[thickness_idx], self.eta_m, min_norm, 1)
                else:
                    new_decision[thickness_idx] = self.polynomial_mutate_single(
                        new_decision[thickness_idx], self.eta_m, 0, 1)
        return new_decision

    def optical_mutation(self, decision):
        """Mutation operator specific to optical thin films"""
        new_decision = copy.deepcopy(decision)

        if random.random() < self.layer_mut_prob:
            new_decision = self.layer_mutation(new_decision)

        if self.mode != "periodic" and random.random() < self.material_mut_prob:
            new_decision = self.material_mutation(new_decision)

        new_decision = self.thickness_mutation(new_decision)
        new_decision = self.enforce_layer_constraints(new_decision)

        if self.mode == "periodic":
            new_decision = self.repair_adjacent_layers_in_decision(new_decision)

        return new_decision

    # ========== Fitness evaluation ==========

    def batch_evaluate_optical_films(self, decisions):
        """Batch evaluation with batch splitting and layer-count penalty"""
        if len(decisions) <= self.max_batch_size:
            return self._evaluate_batch(decisions)

        all_obj = []
        for start in range(0, len(decisions), self.max_batch_size):
            batch = decisions[start:start + self.max_batch_size]
            all_obj.extend(self._evaluate_batch(batch))
        return all_obj

    def _evaluate_batch(self, decisions):
        """Evaluate a single batch"""
        input_data = []
        layer_counts = []
        for dec in decisions:
            mats, thicks = self.decode_solution_with_repair(dec)
            input_data.append((mats, thicks))
            layer_counts.append(self.count_active_layers(dec))

        try:
            batch_emissivity = batch_calculate_weighted_score(
                input_data,
                ga_weights=self.ga_weights,
                use_laser_term=self.ga_use_laser_term
            )
        except Exception as e:
            print(f"GPU计算失败: {e}")
            batch_emissivity = [[0.5, 0.5, 0.5] for _ in decisions]

        objectives = []
        for idx, em in enumerate(batch_emissivity):
            penalty = self.layer_penalty_weight * (layer_counts[idx] / self.max_layers)
            objectives.append([em[0] + penalty, em[1] + penalty, em[2] + penalty])
        return objectives

    def get_objective_values(self, decisions):
        """Get the three objective values"""
        objectives = self.batch_evaluate_optical_films(decisions)
        obj1 = [o[0] for o in objectives]
        obj2 = [o[1] for o in objectives]
        obj3 = [o[2] for o in objectives]
        return obj1, obj2, obj3

    def update_parameters(self, generation, max_generation):
        """Update parameters based on the evolution progress"""
        if self.config.adaptive_params['enabled']:
            adaptive = self.config.get_adaptive_parameters(generation, max_generation)
            self.layer_mut_prob = adaptive['layer_mut_prob']
            self.material_mut_prob = adaptive['material_mut_prob']
            self.mut_prob = adaptive['mut_prob']
