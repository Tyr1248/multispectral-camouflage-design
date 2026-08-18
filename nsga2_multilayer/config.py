# config.py
class OptimizationConfig:
    """NSGA-II optimization parameter configuration for optical thin films

    Supports two optimization modes:
    - "periodic": periodic structure — only Ge/ZnS, adjacent layers forced to alternate
    - "penalty":  layer-count penalty — 8 materials, adjacent layers must differ, penalty on layer count
    """

    def __init__(self, mode="penalty"):
        # === Optimization mode ===
        self.mode = mode  # "periodic" or "penalty"

        # === Population parameters ===
        self.pop_size = 1200
        self.max_gen = 500
        self.offspring_size_multiplier = 2

        # === Optical thin-film problem parameters ===
        self.max_layers = 15
        self.thickness_range = (10, 1200)
        self.first_layer_min_thickness = 300

        # === GPU batch processing parameters ===
        self.max_batch_size = 300

        # === Mutation parameters ===
        self.layer_mut_prob = 0.3
        self.material_mut_prob = 0.6
        self.base_mut_prob = 2
        self.eta_m = 25

        # === Crossover parameters ===
        self.crossover_prob = 0.7

        # === Adaptive parameters ===
        self.adaptive_params = {
            'enabled': True,
            'learning_rate': 0.1,
            'min_F': 0.3,
            'max_F': 0.9,
            'min_CR': 0.5,
            'max_CR': 1.0,
            'min_eta_c': 10 / 50,
            'max_eta_c': 40 / 50
        }

        # === Convergence criteria ===
        self.convergence_criteria = {
            'max_stagnation': 50,
            'tolerance': 5e-7,
            'min_generations': 50
        }

        # === Output settings ===
        self.verbose = True
        self.save_interval = 10
        self.plot_results = True

        # === Apply mode-specific parameter defaults ===
        self._apply_mode_defaults()

    def _apply_mode_defaults(self):
        """Set different default parameters based on the optimization mode"""
        if self.mode == "periodic":
            self.min_layers = 5
            self.material_bits = 1  # only 1 bit needed to encode two materials
            self.layer_penalty_weight = 0.05
            # GA weighting coefficients: MWIR, LWIR, (1-RC2), (1-laser)
            self.ga_weights = [0.25, 0.25, 0.4, 0.1]
            self.ga_use_laser_term = True
        else:  # "penalty"
            self.min_layers = 3
            self.material_bits = 3  # 3 bits to encode 8 materials
            self.layer_penalty_weight = 0.5
            # GA weighting coefficients: MWIR, (1-LWIR), (1-RC2)
            self.ga_weights = [0.33, 0.33, 0.33]
            self.ga_use_laser_term = False

    def update_for_problem(self, problem_type="optical_film"):
        """Update parameters for the given problem type"""
        if problem_type == "optical_film":
            self.pop_size = 100
            self.max_gen = 200
            self.max_layers = 20
            self.min_layers = 2
            self.max_batch_size = 200
        elif problem_type == "fast_test":
            self.pop_size = 30
            self.max_gen = 50
            self.max_layers = 10
            self.min_layers = 2
            self.max_batch_size = 100
            self.verbose = True
        elif problem_type == "large_population":
            self.pop_size = 300
            self.max_gen = 100
            self.max_batch_size = 150

    def get_mutation_prob(self):
        """Compute the dynamic mutation probability"""
        return self.base_mut_prob / (self.max_layers * (self.material_bits + 1))

    def get_adaptive_parameters(self, generation, max_generation):
        """Adjust parameters based on the evolution progress"""
        progress = generation / max_generation
        adaptive_layer_mut = self.layer_mut_prob * (1 - progress * 0.5)
        adaptive_material_mut = self.material_mut_prob * (1 - progress * 0.25)
        adaptive_mut_prob = (self.base_mut_prob + progress * 0.5) / (self.max_layers * (self.material_bits + 1))

        return {
            'layer_mut_prob': max(adaptive_layer_mut, 0.05),
            'material_mut_prob': max(adaptive_material_mut, 0.2),
            'mut_prob': adaptive_mut_prob
        }

    def validate(self):
        """Validate that the parameter values are valid"""
        assert self.pop_size > 0, "种群大小必须大于0"
        assert self.max_gen > 0, "最大代数必须大于0"
        assert self.max_layers >= self.min_layers, "最大层数必须大于等于最小层数"
        assert self.min_layers >= 2, "最小层数必须至少为2"
        assert self.thickness_range[1] > self.thickness_range[0], "厚度范围无效"
        assert 0 <= self.layer_mut_prob <= 1, "层数变异概率必须在[0,1]范围内"
        assert 0 <= self.material_mut_prob <= 1, "材料变异概率必须在[0,1]范围内"
        assert self.max_batch_size > 0, "最大批次大小必须大于0"
        assert self.mode in ("periodic", "penalty"), "模式必须是 'periodic' 或 'penalty'"
        return True

    def __str__(self):
        mode_desc = "周期性结构 (Ge/ZnS交替)" if self.mode == "periodic" else "层数惩罚 (8种材料)"
        return f"""
优化参数配置:
==============
优化模式: {mode_desc}
模式代码: {self.mode}

种群参数:
  种群大小: {self.pop_size}
  最大代数: {self.max_gen}
  子代倍数: {self.offspring_size_multiplier}

光学薄膜参数:
  最大层数: {self.max_layers}
  最小层数: {self.min_layers}
  厚度范围: {self.thickness_range} nm
  第一层最小厚度: {self.first_layer_min_thickness} nm
  材料编码位数: {self.material_bits}
  层数惩罚权重: {self.layer_penalty_weight}
  GA权重: {self.ga_weights}

GPU批次处理:
  最大批次大小: {self.max_batch_size}

变异参数:
  层数变异概率: {self.layer_mut_prob}
  材料变异概率: {self.material_mut_prob}
  基础变异系数: {self.base_mut_prob}
  多项式变异参数: {self.eta_m}

交叉参数:
  交叉概率: {self.crossover_prob}

自适应参数: {'启用' if self.adaptive_params['enabled'] else '禁用'}
收敛标准:
  最大停滞代数: {self.convergence_criteria['max_stagnation']}
  收敛容差: {self.convergence_criteria['tolerance']}
  最小运行代数: {self.convergence_criteria['min_generations']}
"""


# Predefined configuration templates
def get_default_config(mode="penalty"):
    """Get the default configuration"""
    return OptimizationConfig(mode=mode)


def get_periodic_config():
    """Get the periodic-structure configuration (alternating Ge/ZnS)"""
    return OptimizationConfig(mode="periodic")


def get_penalty_config():
    """Get the layer-count-penalty configuration (8 materials)"""
    return OptimizationConfig(mode="penalty")


def get_fast_test_config(mode="penalty"):
    """Get the fast-test configuration"""
    config = OptimizationConfig(mode=mode)
    config.update_for_problem("fast_test")
    return config


def get_high_precision_config(mode="penalty"):
    """Get the high-precision configuration"""
    config = OptimizationConfig(mode=mode)
    config.pop_size = 200
    config.max_gen = 500
    config.max_batch_size = 150
    config.adaptive_params['learning_rate'] = 0.05
    return config


def get_large_population_config(mode="penalty"):
    """Get the large-population configuration"""
    config = OptimizationConfig(mode=mode)
    config.pop_size = 300
    config.max_gen = 100
    config.max_batch_size = 100
    config.offspring_size_multiplier = 1.5
    return config


def get_gpu_memory_saver_config(mode="penalty"):
    """Get the GPU-memory-saver configuration"""
    config = OptimizationConfig(mode=mode)
    config.pop_size = 80
    config.max_batch_size = 50
    return config
