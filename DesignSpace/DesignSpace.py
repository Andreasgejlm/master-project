from tqdm import tqdm
import numpy as np
import random
from DataProvider.CGHDataProvider import CGHDataProvider
from CGHModel.CGHModel import CGHNet
import re
import json
import datetime
import os
from SLM.SLM import SLM
from CGHSystem.CGHSystem import CGHSystem
from NetworkConfig.NetworkConfig import NetworkConfig



class DesignSpace:
	# DesignSpace object manages the generation, sampling, training, and analysis of a population of CGHModels.
	# The population is generated from a certain predicate, that refines our design space.
	# For instance:
	# 	f = ( 4GF )		    -> The complexity of a model must be around 4GF +- 10%.
	# 	g = ( CONST ) 		-> Group widths are constant in all stages.
	# 	b = ( CONST )		-> Bottleneck ratios are constant in all stages.
	# 	d = ( INCREASING )	-> The number of blocks in a stage *must* be greater than or equal to the previous stage.
	# 	w = ( INCREASING )	-> The number of feature maps in a stage *must* be greater than or equal to the previous stage.

	def __init__(self, slm, System, Config):
		# Inputs:
		self.date_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
		self.slm = SLM() if not slm else slm
		self.sys = CGHSystem() if not System else System
		self.conf = NetworkConfig() if not Config else Config
		self.stages = 6
		self.options = {}
		self.predicate = {}
		self.data_provider = CGHDataProvider(self.conf.nT, (self.slm.Mx, self.slm.My, self.slm.lp), self.sys.nz, self.sys.dz, self.sys.wl)
		self.parse_predicate(self.conf.predicate)

		dir_name = 'model_params/' + str(self.predicate["f"]*10**-6) + '/'
		self.param_file_dir = self.date_str + '-params.json'
		self.model_param_dir = os.path.join(os.getcwd(), dir_name)

	def parse_predicate(self, predicate_string):
		split_pred = predicate_string.replace(" ", "").split(",")
		for pred in split_pred:
			if len(pred) > 0:
				variable_to_limit = pred[0]
				operation = pred[pred.find("(") + 1:pred.find(")")]
				if variable_to_limit == "f":
					flop_regime = re.split('(\d+)', operation)
					prefix = float(flop_regime[1])
					suffix_str = flop_regime[2]
					if suffix_str == "MF":
						suffix_f = 10**6
					elif suffix_str == "GF":
						suffix_f = 10**9
					else:
						raise ValueError("Flop regime must be either MF or GF")
					self.predicate[variable_to_limit] = prefix * suffix_f
				else:
					assert operation in ["CONST", "INCREASING", "DECREASING"], "Operation in predicate not recognized."
					self.predicate[variable_to_limit] = operation

	def set_limits(self, b, g, d, w):
		# Variable limits can be either lists or ranges
		assert len(b) > 0 and len(g) > 0 and len(d) > 0 and len(w) > 0, "Limits should be imposed on all variables."

		self.options["b"] = list(b) if isinstance(b, range) else b
		self.options["g"] = list(g) if isinstance(g, range) else g
		self.options["d"] = list(d) if isinstance(d, range) else d
		self.options["w"] = list(w) if isinstance(w, range) else w
		#self.options["wb"] = wb

	def generate_models(self):
		assert len(self.options) > 0, "Limits must be put on the variables in the form of ranges or arrays of possible values."
		# TODO: Make it so that new data is not generated if already exists
		progress = tqdm(range(self.conf.n))
		for i in progress:
			progress.set_description(f'Generating and training model {i} of {self.conf.n} ...')
			valid = False
			while not valid:
				try_b = self.get_iteration("b")
				try_g = self.get_iteration("g")
				try_d = self.get_iteration("d")
				try_w = self.get_iteration("w")
				try_model = CGHNet(self.data_provider.get_params(try_b, try_d, try_g, try_w), phase_factors=self.data_provider.phase_factors)
				_, complexity = try_model.build_model()
				try_flops = complexity["flops"]
				print(try_flops)
				if self.predicate["f"]*0.5 <= try_flops <= self.predicate["f"]*1.5:
					valid = True

					try_model.train_network(1, self.conf.nT, self.data_provider.get_training_data())
					self.save_model_params(try_model.get_params())

	def save_model_params(self, params):
		json_data = {"predicate": self.predicate, "options": self.options,  "models": [params]}

		filepath = os.path.join(self.model_param_dir, self.param_file_dir)

		if not os.path.exists(self.model_param_dir):
			os.makedirs(self.model_param_dir)
		elif os.path.exists(filepath):
			with open(filepath) as fp:
				json_data = json.load(fp)
				prev_models = json_data["models"]
				prev_models.append(params)

		with open(filepath, 'w') as fp:
			json.dump(json_data, fp, sort_keys=True, indent=4)

	def _random_in_options(self, variable):
		return self._closest(random.randint(min(self.options[variable]), max(self.options[variable])), self.options[variable])

	def get_iteration(self, variable):
		if variable in self.predicate.keys():
			ops = self.predicate[variable]
			if ops == "CONST":
				const_val = self._random_in_options(variable)
				values = [const_val for _ in range(self.stages)]
			elif ops == "INCREASING":
				values = [self._random_in_options(variable) for _ in range(self.stages)]
				values.sort()
			elif ops == "DECREASING":
				values = [self._random_in_options(variable) for _ in range(self.stages)]
				values.sort(reverse=True)
			else:
				values = [self._random_in_options(variable) for _ in range(self.stages)]
			values.extend([self._random_in_options(variable), self._random_in_options(variable)])
			return values

		values = [self._random_in_options(variable) for _ in range(self.stages)]
		values.extend([self._random_in_options(variable), self._random_in_options(variable)])
		return values


	def _closest(self, val, arr):
		absolute_difference_function = lambda list_value: abs(list_value - val)
		return min(arr, key=absolute_difference_function)

	def loguniform(self, low=0, high=1, size=None):
		return np.exp(np.random.uniform(low, high, size))
