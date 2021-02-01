from tensorflow.keras import Model, Sequential, Input, layers, optimizers, utils, callbacks


class UNet(layers.Layer):
	def __init__(self, ds, bs, gs, ws, **kwargs):
		super(UNet, self).__init__(kwargs)
		self.ds = ds
		self.bs = bs
		self.gs = gs
		self.ws = ws

	def call(self, input, **kwargs):
		x = Stage(self.ds[0], self.bs[0], self.gs[0], self.ws[0])(input)  # r x r
		x = layers.MaxPooling2D((2, 2))(x)  # r/2 x r/2
		x = Stage(self.ds[1], self.bs[1], self.gs[1], self.ws[1])(x)  # r/2 x r/2
		x = layers.MaxPooling2D((2, 2))(x)  # r/4 x r/4
		x = Stage(self.ds[2], self.bs[2], self.gs[2], self.ws[2])(x)  # r/4 x r/4
		x = layers.MaxPooling2D((2, 2))(x) # r/8 x r/8
		x = Stage(self.ds[3], self.bs[3], self.gs[3], self.ws[3])(x)  # r/8 x r/8
		x = layers.MaxPooling2D((2, 2))(x)  # r/16 x r/16
		#x = layers.UpSampling2D()(x)  # r/2 x r/2
		#concat2 = layers.Concatenate()([x2, x])  # r/2 x r/2
		#x = self._stage(b[3], g[3], d[3], w[3], concat2)  # r/2 x r/2
		#x = layers.UpSampling2D()(x)  # r x r
		#concat1 = layers.Concatenate()([x1, x])  # r x r
		#x = self._stage(b[4], g[4], d[4], w[4], concat1)  # r x r
		return x

	def count_params(self):
		shape = self.input_shape
		cx = { "h": shape[1], "w": shape[2], "flops": 0, "params": 0, "acts": 0}
		return UNet.complexity(cx, shape[3], self.ds, self.bs, self.gs, self.ws)["params"]

	@staticmethod
	def complexity(cx, w_in, ds, bs, gs, ws):
		for d, b, g, w in zip(ds, bs, gs, ws):
			cx = Stage.complexity(cx, w_in, w, d, b, g)
			w_in = w
		return cx


class Stage(layers.Layer):
	def __init__(self, d_i, b_i, g_i, w_i, **kwargs):
		super(Stage, self).__init__(kwargs)
		self.d_i = d_i
		self.b_i = b_i
		self.g_i = g_i
		self.w_i = w_i

	def call(self, input, **kwargs):
		x = Block1s(self.b_i, self.g_i, self.w_i)(input)
		for block in range(1, self.d_i):
			x = Block1s(self.b_i, self.g_i, self.w_i)(x)
		return x

	@staticmethod
	def complexity(cx, w_in, w_out, d, b, g):
		for i in range(d):
			cx = Block1s.complexity(cx, w_in, w_out, b, g)
			w_in = w_out
			if i == 0:
				cx["h"] = cx["h"] // 2
				cx["w"] = cx["w"] // 2
		return cx


class Block1s(layers.Layer):
	def __init__(self, b, g, w, **kwargs):
		super(Block1s, self).__init__(kwargs)
		self.b = b
		self.g = g
		self.w = w

	def call(self, input, **kwargs):
		w_b = int(round(self.w / self.b))
		groups = w_b // self.g
		x = layers.Conv2D(w_b, (1, 1), activation='relu', padding='same')(input)
		x = layers.BatchNormalization()(x)
		x = layers.Conv2D(w_b, (3, 3), groups=groups, activation='relu', padding='same')(x)
		x = layers.BatchNormalization()(x)
		x = layers.Conv2D(self.w, (1, 1), activation='relu', padding='same')(x)
		x = layers.BatchNormalization()(x)

		# TODO: Fix residual
		res = layers.Conv2D(self.w, (1, 1), activation='relu', padding='same')(input)
		x = layers.add([res, x])
		return x

	@staticmethod
	def complexity(cx, w_in, w_out, b, g):
		w_b = int(round(w_out / b))
		groups = w_b // g
		cx = conv2d_cx(cx, w_in, w_b, k=1)
		cx = batchnorm2d_cx(cx, w_b)
		cx = conv2d_cx(cx, w_b, w_b, k=3, groups=groups)
		cx = batchnorm2d_cx(cx, w_b)
		cx = conv2d_cx(cx, w_b, w_out, k=1)
		cx = batchnorm2d_cx(cx, w_out)
		#cx = conv2d_cx(cx, w_in, w_out, k=1)
		return cx


# ---- COMPLEXITY OF PRIMITIVES ---- #
def conv2d_cx(cx, w_in, w_out, k, stride=1, groups=1):
	h, w, flops, params, acts = cx["h"], cx["w"], cx["flops"], cx["params"], cx["acts"]
	h, w = (h - 1) // stride + 1, (w - 1) // stride + 1
	f = k * k * w_in * w_out * h * w // groups
	flops += f  # + (w_out if bias else 0)
	params += k * k * w_in * w_out // groups  # + (w_out if bias else 0)
	acts += w_out * h * w
	return {"h": h, "w": w, "flops": flops, "params": params, "acts": acts}


def batchnorm2d_cx(cx, w_in):
	h, w, flops, params, acts = cx["h"], cx["w"], cx["flops"], cx["params"], cx["acts"]
	params += 2 * w_in
	return {"h": h, "w": w, "flops": flops, "params": params, "acts": acts}
