class DesignSpace:
	# DesignSpace object manages the generation, sampling, training, and analysis of a population of CGHModels.
	# The population is generated from a certain predicate, that refines our design space.
	# For instance:
	# 	flop_regime <= 4GF	-> The complexity of the model must be less than or equal to 4 Giga flop.
	# 	g_i = g 			-> Group widths are constant in all stages.
	# 	b_i = b 			-> Bottleneck ratios are constant in all stages.
	# 	d_i+1 >= d_i		-> The number of blocks in a stage *must* be greater than or equal to the previous stage.
	# 	w_i+1 >= w_i		-> The number of feature maps in a stage *must* be greater than or equal to the previous stage.

	def __init__(self, n, predicate):
		# Inputs:
		# n: 			Number of models in a population.
		# predicate		The predicate imposed on the design space.

		self.n = n
		self.predicate = predicate
		self.options = {}

	def set_limits(self, b, g, d, w):
		# Variable limits can be either lists or ranges
		assert len(b) > 0 and len(g) > 0 and len(d) > 0 and len(w) > 0, "Limits should be imposed on all variables."

		self.options["b"] = b
		self.options["g"] = g
		self.options["d"] = d
		self.options["w"] = w
