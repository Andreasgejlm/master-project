

class NetworkConfig:
	def __init__(self, n_in_subset=500, flop_regime=5, n_training=1024, training_types=("lines", "disks", "polys"), training_type_split=(0.5, 0.25, 0.25), predicate=""):
		self.n = n_in_subset
		self.nT = n_training
		assert len(training_types) == len(training_type_split), "Mismatch between training type and split."
		assert all(i in ("lines", "disks", "polys") for i in training_types), "Unknown training type."
		assert isinstance(flop_regime, int) or isinstance(flop_regime, float), "Flop regime must be either an int or float."
		self.train_types = training_types
		self.train_type_split = training_type_split
		self.predicate = predicate + f',f=({flop_regime} GF)'