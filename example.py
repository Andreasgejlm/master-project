
from DesignSpace.DesignSpace import DesignSpace
from SLM.SLM import SLM
from CGHSystem.CGHSystem import CGHSystem
from NetworkConfig.NetworkConfig import NetworkConfig


slm = SLM(resolution=(256, 256), pixel_pitch=0.5*10**-5)
system = CGHSystem()
network_params = NetworkConfig(n_in_subset= 1, n_training=128, flop_regime=2)

design_space = DesignSpace(slm=slm, System=system, Config=network_params)
design_space.set_limits(b=[1, 2], g=[4, 8, 16], d=range(1, 5), w=range(64, 512))

design_space.generate_models()