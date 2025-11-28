"""
	This file is part of ComfyUI.
	Copyright (C) 2024 Comfy

	This program is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.

	This program is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 	See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with this program. 	If not, see <https://www.gnu.org/licenses/>.
"""

import psutil
import logging
from enum import Enum
from comfy.cli_args import args, PerformanceFeature
import torch
import sys
import importlib
import platform
import weakref
import gc

class VRAMState(Enum):
	DISABLED = 0 	#No vram present: no need to move models to vram
	NO_VRAM = 1 	#Very low vram: enable all the options to save vram
	LOW_VRAM = 2
	NORMAL_VRAM = 3
	HIGH_VRAM = 4
	SHARED = 5 	#No dedicated vram: memory shared between CPU and GPU but models still need to be moved between both.

class CPUState(Enum):
	GPU = 0
	CPU = 1
	MPS = 2

# Determine VRAM State
vram_state = VRAMState.NORMAL_VRAM
set_vram_to = VRAMState.NORMAL_VRAM
cpu_state = CPUState.GPU

total_vram = 0

def get_supported_float8_types():
	float8_types = []
	try:
		float8_types.append(torch.float8_e4m3fn)
	except:
		pass
	try:
		float8_types.append(torch.float8_e4m3fnuz)
	except:
		pass
	try:
		float8_types.append(torch.float8_e5m2)
	except:
		pass
	try:
		float8_types.append(torch.float8_e5m2fnuz)
	except:
		pass
	try:
		float8_types.append(torch.float8_e8m0fnu)
	except:
		pass
	return float8_types

FLOAT8_TYPES = get_supported_float8_types()

xpu_available = False
torch_version = ""
try:
	torch_version = torch.version.__version__
	temp = torch_version.split(".")
	torch_version_numeric = (int(temp[0]), int(temp[1]))
except:
	pass

lowvram_available = True
if args.deterministic:
	logging.info("Using deterministic algorithms for pytorch")
	torch.use_deterministic_algorithms(True, warn_only=True)

directml_enabled = False
if args.directml is not None:
	logging.warning("WARNING: torch-directml barely works, is very slow, has not been updated in over 1 year and might be removed soon, please don't use it, there are better options.")
	import torch_directml
	directml_enabled = True
	device_index = args.directml
	if device_index < 0:
		directml_device = torch_directml.device()
	else:
		directml_device = torch_directml.device(device_index)
	logging.info("Using directml with device: {}".format(torch_directml.device_name(device_index)))
	# torch_directml.disable_tiled_resources(True)
	lowvram_available = False #TODO: need to find a way to get free memory in directml before this can be enabled by default.

try:
	import intel_extension_for_pytorch as ipex 	# noqa: F401
except:
	pass

try:
	_ = torch.xpu.device_count()
	xpu_available = torch.xpu.is_available()
except:
	xpu_available = False

try:
	if torch.backends.mps.is_available():
		cpu_state = CPUState.MPS
		import torch.mps
except:
	pass

try:
	import torch_npu 	# noqa: F401
	_ = torch.npu.device_count()
	npu_available = torch.npu.is_available()
except:
	npu_available = False

try:
	import torch_mlu 	# noqa: F401
	_ = torch.mlu.device_count()
	mlu_available = torch.mlu.is_available()
except:
	mlu_available = False

try:
	ixuca_available = hasattr(torch, "corex")
except:
	ixuca_available = False

if args.cpu:
	cpu_state = CPUState.CPU

def is_intel_xpu():
	global cpu_state
	global xpu_available
	if cpu_state == CPUState.GPU:
		if xpu_available:
			return True
	return False

def is_ascend_npu():
	global npu_available
	if npu_available:
		return True
	return False

def is_mlu():
	global mlu_available
	if mlu_available:
		return True
	return False

def is_ixuca():
	global ixuca_available
	if ixuca_available:
		return True
	return False

def get_torch_device():
	global directml_enabled
	global cpu_state
	if directml_enabled:
		global directml_device
		return directml_device
	if cpu_state == CPUState.MPS:
		return torch.device("mps")
	if cpu_state == CPUState.CPU:
		return torch.device("cpu")
	else:
		if is_intel_xpu():
			return torch.device("xpu", torch.xpu.current_device())
		elif is_ascend_npu():
			return torch.device("npu", torch.npu.current_device())
		elif is_mlu():
			return torch.device("mlu", torch.mlu.current_device())
		else:
			# KORREKTUR: Explizite Prüfung, ob CUDA verfügbar ist, bevor .current_device() aufgerufen wird,
			# da dies der Punkt ist, an dem PyTorch ohne CUDA fehlschlägt.
			if torch.cuda.is_available():
				return torch.device(torch.cuda.current_device())
			# FALLBACK, falls keine unterstützte GPU/XPU gefunden wurde, aber GPU-Modus gewünscht ist
			return torch.device("cpu")

def get_total_memory(dev=None, torch_total_too=False):
	global directml_enabled
	if dev is None:
		dev = get_torch_device()

	if hasattr(dev, 'type') and (dev.type == 'cpu' or dev.type == 'mps'):
		mem_total = psutil.virtual_memory().total
		mem_total_torch = mem_total
	else:
		if directml_enabled:
			mem_total = 1024 * 1024 * 1024 #TODO
			mem_total_torch = mem_total
		elif is_intel_xpu():
			stats = torch.xpu.memory_stats(dev)
			mem_reserved = stats['reserved_bytes.all.current']
			mem_total_xpu = torch.xpu.get_device_properties(dev).total_memory
			mem_total_torch = mem_reserved
			mem_total = mem_total_xpu
		elif is_ascend_npu():
			stats = torch.npu.memory_stats(dev)
			mem_reserved = stats['reserved_bytes.all.current']
			_, mem_total_npu = torch.npu.mem_get_info(dev)
			mem_total_torch = mem_reserved
			mem_total = mem_total_npu
		elif is_mlu():
			stats = torch.mlu.memory_stats(dev)
			mem_reserved = stats['reserved_bytes.all.current']
			_, mem_total_mlu = torch.mlu.mem_get_info(dev)
			mem_total_torch = mem_reserved
			mem_total = mem_total_mlu
		else:
			# Fallback für CUDA-Geräte
			if torch.cuda.is_available():
				stats = torch.cuda.memory_stats(dev)
				mem_reserved = stats['reserved_bytes.all.current']
				_, mem_total_cuda = torch.cuda.mem_get_info(dev)
				mem_total_torch = mem_reserved
				mem_total = mem_total_cuda
			else:
				# Wenn keine dedizierte Hardware erkannt wird (einschließlich des Patches oben)
				mem_total = psutil.virtual_memory().total
				mem_total_torch = mem_total

	if torch_total_too:
		return (mem_total, mem_total_torch)
	else:
		return mem_total

def mac_version():
	try:
		return tuple(int(n) for n in platform.mac_ver()[0].split("."))
	except:
		return None

total_vram = get_total_memory(get_torch_device()) / (1024 * 1024)
total_ram = psutil.virtual_memory().total / (1024 * 1024)
logging.info("Total VRAM {:0.0f} MB, total RAM {:0.0f} MB".format(total_vram, total_ram))

try:
	logging.info("pytorch version: {}".format(torch_version))
	mac_ver = mac_version()
	if mac_ver is not None:
		logging.info("Mac Version {}".format(mac_ver))
except:
	pass

try:
	OOM_EXCEPTION = torch.cuda.OutOfMemoryError
except:
	OOM_EXCEPTION = Exception

XFORMERS_VERSION = ""
XFORMERS_ENABLED_VAE = True
if args.disable_xformers:
	XFORMERS_IS_AVAILABLE = False
else:
	try:
		import xformers
		import xformers.ops
		XFORMERS_IS_AVAILABLE = True
		try:
			XFORMERS_IS_AVAILABLE = xformers._has_cpp_library
		except:
			pass
		try:
			XFORMERS_VERSION = xformers.version.__version__
			logging.info("xformers version: {}".format(XFORMERS_VERSION))
			if XFORMERS_VERSION.startswith("0.0.18"):
				logging.warning("\nWARNING: This version of xformers has a major bug where you will get black images when generating high resolution images.")
				logging.warning("Please downgrade or upgrade xformers to a different version.\n")
				XFORMERS_ENABLED_VAE = False
		except:
			pass
	except:
		XFORMERS_IS_AVAILABLE = False

def is_nvidia():
	global cpu_state
	if cpu_state == CPUState.GPU:
		if torch.version.cuda:
			return True
	return False

def is_amd():
	global cpu_state
	if cpu_state == CPUState.GPU:
		if torch.version.hip:
			return True
	return False

def amd_min_version(device=None, min_rdna_version=0):
	if not is_amd():
		return False

	if is_device_cpu(device):
		return False

	arch = torch.cuda.get_device_properties(device).gcnArchName
	if arch.startswith('gfx') and len(arch) == 7:
		try:
			cmp_rdna_version = int(arch[4]) + 2
		except:
			cmp_rdna_version = 0
		if cmp_rdna_version >= min_rdna_version:
			return True

	return False

MIN_WEIGHT_MEMORY_RATIO = 0.4
if is_nvidia():
	MIN_WEIGHT_MEMORY_RATIO = 0.0

ENABLE_PYTORCH_ATTENTION = False
if args.use_pytorch_cross_attention:
	ENABLE_PYTORCH_ATTENTION = True
	XFORMERS_IS_AVAILABLE = False

try:
	if is_nvidia():
		if torch_version_numeric[0] >= 2:
			if ENABLE_PYTORCH_ATTENTION == False and args.use_split_cross_attention == False and args.use_quad_cross_attention == False:
				ENABLE_PYTORCH_ATTENTION = True
	if is_intel_xpu() or is_ascend_npu() or is_mlu() or is_ixuca():
		if args.use_split_cross_attention == False and args.use_quad_cross_attention == False:
			ENABLE_PYTORCH_ATTENTION = True
except:
	pass


SUPPORT_FP8_OPS = args.supports_fp8_compute

AMD_RDNA2_AND_OLDER_ARCH = ["gfx1030", "gfx1031", "gfx1010", "gfx1011", "gfx1012", "gfx906", "gfx900", "gfx803"]

try:
	if is_amd():
		arch = torch.cuda.get_device_properties(get_torch_device()).gcnArchName
		if not (any((a in arch) for a in AMD_RDNA2_AND_OLDER_ARCH)):
			torch.backends.cudnn.enabled = False 	# Seems to improve things a lot on AMD
			logging.info("Set: torch.backends.cudnn.enabled = False for better AMD performance.")

		try:
			rocm_version = tuple(map(int, str(torch.version.hip).split(".")[:2]))
		except:
			rocm_version = (6, -1)

		logging.info("AMD arch: {}".format(arch))
		logging.info("ROCm version: {}".format(rocm_version))
		if args.use_split_cross_attention == False and args.use_quad_cross_attention == False:
			if importlib.util.find_spec('triton') is not None: 	# AMD efficient attention implementation depends on triton. TODO: better way of detecting if it's compiled in or not.
				if torch_version_numeric >= (2, 7): 	# works on 2.6 but doesn't actually seem to improve much
					if any((a in arch) for a in ["gfx90a", "gfx942", "gfx1100", "gfx1101", "gfx1151"]): 	# TODO: more arches, TODO: gfx950
						ENABLE_PYTORCH_ATTENTION = True
				if rocm_version >= (7, 0):
					if any((a in arch) for a in ["gfx1201"]):
						ENABLE_PYTORCH_ATTENTION = True
		if torch_version_numeric >= (2, 7) and rocm_version >= (6, 4):
			if any((a in arch) for a in ["gfx1200", "gfx1201", "gfx950"]): 	# TODO: more arches, "gfx942" gives error on pytorch nightly 2.10 1013 rocm7.0
				SUPPORT_FP8_OPS = True

except:
	pass


if ENABLE_PYTORCH_ATTENTION:
	torch.backends.cuda.enable_math_sdp(True)
	torch.backends.cuda.enable_flash_sdp(True)
	torch.backends.cuda.enable_mem_efficient_sdp(True)


PRIORITIZE_FP16 = False 	# TODO: remove and replace with something that shows exactly which dtype is faster than the other
try:
	if (is_nvidia() or is_amd()) and PerformanceFeature.Fp16Accumulation in args.fast:
		torch.backends.cuda.matmul.allow_fp16_accumulation = True
		PRIORITIZE_FP16 = True 	# TODO: limit to cards where it actually boosts performance
		logging.info("Enabled fp16 accumulation.")
except:
	pass

if torch.cuda.is_available() and torch.backends.cudnn.is_available() and PerformanceFeature.AutoTune in args.fast:
	torch.backends.cudnn.benchmark = True

try:
	if torch_version_numeric >= (2, 5):
		torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(True)
except:
	logging.warning("Warning, could not set allow_fp16_bf16_reduction_math_sdp")

if args.lowvram:
	set_vram_to = VRAMState.LOW_VRAM
	lowvram_available = True
elif args.novram:
	set_vram_to = VRAMState.NO_VRAM
elif args.highvram or args.gpu_only:
	vram_state = VRAMState.HIGH_VRAM

FORCE_FP32 = False
if args.force_fp32:
	logging.info("Forcing FP32, if this improves things please report it.")
	FORCE_FP32 = True

if lowvram_available:
	if set_vram_to in (VRAMState.LOW_VRAM, VRAMState.NO_VRAM):
		vram_state = set_vram_to


if cpu_state != CPUState.GPU:
	vram_state = VRAMState.DISABLED

if cpu_state == CPUState.MPS:
	vram_state = VRAMState.SHARED

logging.info(f"Set vram state to: {vram_state.name}")

DISABLE_SMART_MEMORY = args.disable_smart_memory

if DISABLE_SMART_MEMORY:
	logging.info("Disabling smart memory management")

def get_torch_device_name(device):
	if hasattr(device, 'type'):
		if device.type == "cuda":
			try:
				allocator_backend = torch.cuda.get_allocator_backend()
			except:
				allocator_backend = ""
			return "{} {} : {}".format(device, torch.cuda.get_device_name(device), allocator_backend)
		elif device.type == "xpu":
			return "{} {}".format(device, torch.xpu.get_device_name(device))
		else:
			return "{}".format(device.type)
	elif is_intel_xpu():
		return "{} {}".format(device, torch.xpu.get_device_name(device))
	elif is_ascend_npu():
		return "{} {}".format(device, torch.npu.get_device_name(device))
	elif is_mlu():
		return "{} {}".format(device, torch.mlu.get_device_name(device))
	else:
		# Fallback für CUDA, muss aber prüfen, ob CUDA verfügbar ist
		if torch.cuda.is_available():
			return "CUDA {}: {}".format(device, torch.cuda.get_device_name(device))
		return "CPU (Fallback)"

try:
	logging.info("Device: {}".format(get_torch_device_name(get_torch_device())))
except:
	logging.warning("Could not pick default device.")


current_loaded_models = []

def module_size(module):
	module_mem = 0
	sd = module.state_dict()
	for k in sd:
		t = sd[k]
		module_mem += t.nelement() * t.element_size()
	return module_mem

class LoadedModel:
	def __init__(self, model):
		self._set_model(model)
		self.device = model.load_device
		self.real_model = None
		self.currently_used = True
		self.model_finalizer = None
		self._patcher_finalizer = None

	def _set_model(self, model):
		self._model = weakref.ref(model)
		if model.parent is not None:
			self._parent_model = weakref.ref(model.parent)
			self._patcher_finalizer = weakref.finalize(model, self._switch_parent)

	def _switch_parent(self):
		model = self._parent_model()
		if model is not None:
			self._set_model(model)

	@property
	def model(self):
		return self._model()

	def model_memory(self):
		return self.model.model_size()

	def model_loaded_memory(self):
		return self.model.loaded_size()

	def model_offloaded_memory(self):
		return self.model.model_size() - self.model.loaded_size()

	def model_memory_required(self, device):
		if device == self.model.current_loaded_device():
			return self.model_offloaded_memory()
		else:
			return self.model_memory()

	def model_load(self, lowvram_model_memory=0, force_patch_weights=False):
		self.model.model_patches_to(self.device)
		self.model.model_patches_to(self.model.model_dtype())

		# if self.model.loaded_size() > 0:
		use_more_vram = lowvram_model_memory
		if use_more_vram == 0:
			use_more_vram = 1e32
		self.model_use_more_vram(use_more_vram, force_patch_weights=force_patch_weights)

		real_model = self.model.model

		if is_intel_xpu() and not args.disable_ipex_optimize and 'ipex' in globals() and real_model is not None:
			with torch.no_grad():
				real_model = ipex.optimize(real_model.eval(), inplace=True, graph_mode=True, concat_linear=True)

		self.real_model = weakref.ref(real_model)
		self.model_finalizer = weakref.finalize(real_model, cleanup_models)
		return real_model

	def should_reload_model(self, force_patch_weights=False):
		if force_patch_weights and self.model.lowvram_patch_counter() > 0:
			return True
		return False

	def model_unload(self, memory_to_free=None, unpatch_weights=True):
		if memory_to_free is not None:
			if memory_to_free < self.model.loaded_size():
				freed = self.model.partially_unload(self.model.offload_device, memory_to_free)
				if freed >= memory_to_free:
					return False
		self.model.detach(unpatch_weights)
		self.model_finalizer.detach()
		self.model_finalizer = None
		self.real_model = None
		return True

	def model_use_more_vram(self, extra_memory, force_patch_weights=False):
		return self.model.partially_load(self.device, extra_memory, force_patch_weights=force_patch_weights)

	def __eq__(self, other):
		return self.model is other.model

	def __del__(self):
		if self._patcher_finalizer is not None:
			self._patcher_finalizer.detach()

	def is_dead(self):
		return self.real_model() is not None and self.model is None


def use_more_memory(extra_memory, loaded_models, device):
	for m in loaded_models:
		if m.device == device:
			extra_memory -= m.model_use_more_vram(extra_memory)
			if extra_memory <= 0:
				break

def offloaded_memory(loaded_models, device):
	offloaded_mem = 0
	for m in loaded_models:
		if m.device == device:
			offloaded_mem += m.model_offloaded_memory()
	return offloaded_mem

WINDOWS = any(platform.win32_ver())

EXTRA_RESERVED_VRAM = 400 * 1024 * 1024
if WINDOWS:
	EXTRA_RESERVED_VRAM = 600 * 1024 * 1024 #Windows is higher because of the shared vram issue
	if total_vram > (15 * 1024): 	# more extra reserved vram on 16GB+ cards
		EXTRA_RESERVED_VRAM += 100 * 1024 * 1024

if args.reserve_vram is not None:
	EXTRA_RESERVED_VRAM = args.reserve_vram * 1024 * 1024 * 1024
	logging.debug("Reserving {}MB vram for other applications.".format(EXTRA_RESERVED_VRAM / (1024 * 1024)))

def extra_reserved_memory():
	return EXTRA_RESERVED_VRAM

def minimum_inference_memory():
	return (1024 * 1024 * 1024) * 0.8 + extra_reserved_memory()

def free_memory(memory_required, device, keep_loaded=[]):
	cleanup_models_gc()
	unloaded_model = []
	can_unload = []
	unloaded_models = []

	for i in range(len(current_loaded_models) -1, -1, -1):
		shift_model = current_loaded_models[i]
		if shift_model.device == device:
			if shift_model not in keep_loaded and not shift_model.is_dead():
				can_unload.append((-shift_model.model_offloaded_memory(), sys.getrefcount(shift_model.model), shift_model.model_memory(), i))
				shift_model.currently_used = False

	for x in sorted(can_unload):
		i = x[-1]
		memory_to_free = None
		if not DISABLE_SMART_MEMORY:
			free_mem = get_free_memory(device)
			if free_mem > memory_required:
				break
			memory_to_free = memory_required - free_mem
		logging.debug(f"Unloading {current_loaded_models[i].model.model.__class__.__name__}")
		if current_loaded_models[i].model_unload(memory_to_free):
			unloaded_model.append(i)

	for i in sorted(unloaded_model, reverse=True):
		unloaded_models.append(current_loaded_models.pop(i))

	if len(unloaded_model) > 0:
		soft_empty_cache()
	else:
		if vram_state != VRAMState.HIGH_VRAM:
			mem_free_total, mem_free_torch = get_free_memory(device, torch_free_too=True)
			if mem_free_torch > mem_free_total * 0.25:
				soft_empty_cache()
	return unloaded_models

def load_models_gpu(models, memory_required=0, force_patch_weights=False, minimum_memory_required=None, force_full_load=False):
	cleanup_models_gc()
	global vram_state

	inference_memory = minimum_inference_memory()
	extra_mem = max(inference_memory, memory_required + extra_reserved_memory())
	if minimum_memory_required is None:
		minimum_memory_required = extra_mem
	else:
		minimum_memory_required = max(inference_memory, minimum_memory_required + extra_reserved_memory())

	models_temp = set()
	for m in models:
		models_temp.add(m)
		for mm in m.model_patches_models():
			models_temp.add(mm)

	models = models_temp

	models_to_load = []

	for x in models:
		loaded_model = LoadedModel(x)
		try:
			loaded_model_index = current_loaded_models.index(loaded_model)
		except:
			loaded_model_index = None

		if loaded_model_index is not None:
			loaded = current_loaded_models[loaded_model_index]
			loaded.currently_used = True
			models_to_load.append(loaded)
		else:
			if hasattr(x, "model"):
				logging.info(f"Requested to load {x.model.__class__.__name__}")
			models_to_load.append(loaded_model)

	for loaded_model in models_to_load:
		to_unload = []
		for i in range(len(current_loaded_models)):
			if loaded_model.model.is_clone(current_loaded_models[i].model):
				to_unload = [i] + to_unload
		for i in to_unload:
			model_to_unload = current_loaded_models.pop(i)
			model_to_unload.model.detach(unpatch_all=False)
			model_to_unload.model_finalizer.detach()

	total_memory_required = {}
	for loaded_model in models_to_load:
		total_memory_required[loaded_model.device] = total_memory_required.get(loaded_model.device, 0) + loaded_model.model_memory_required(loaded_model.device)

	for device in total_memory_required:
		if device != torch.device("cpu"):
			free_memory(total_memory_required[device] * 1.1 + extra_mem, device)

	for device in total_memory_required:
		if device != torch.device("cpu"):
			free_mem = get_free_memory(device)
			if free_mem < minimum_memory_required:
				models_l = free_memory(minimum_memory_required, device)
				logging.info("{} models unloaded.".format(len(models_l)))

	for loaded_model in models_to_load:
		model = loaded_model.model
		torch_dev = model.load_device
		if is_device_cpu(torch_dev):
			vram_set_state = VRAMState.DISABLED
		else:
			vram_set_state = vram_state
		lowvram_model_memory = 0
		if lowvram_available and (vram_set_state == VRAMState.LOW_VRAM or vram_set_state == VRAMState.NORMAL_VRAM) and not force_full_load:
			loaded_memory = loaded_model.model_loaded_memory()
			current_free_mem = get_free_memory(torch_dev) + loaded_memory

			lowvram_model_memory = max(0, (current_free_mem - minimum_memory_required), min(current_free_mem * MIN_WEIGHT_MEMORY_RATIO, current_free_mem - minimum_inference_memory()))
			lowvram_model_memory = lowvram_model_memory - loaded_memory

			if lowvram_model_memory == 0:
				lowvram_model_memory = 0.1

		if vram_set_state == VRAMState.NO_VRAM:
			lowvram_model_memory = 0.1

		loaded_model.model_load(lowvram_model_memory, force_patch_weights=force_patch_weights)
		current_loaded_models.insert(0, loaded_model)
	return

def load_model_gpu(model):
	return load_models_gpu([model])

def loaded_models(only_currently_used=False):
	output = []
	for m in current_loaded_models:
		if only_currently_used:
			if not m.currently_used:
				continue

		output.append(m.model)
	return output


def cleanup_models_gc():
	do_gc = False
	for i in range(len(current_loaded_models)):
		cur = current_loaded_models[i]
		if cur.is_dead():
			logging.info("Potential memory leak detected with model {}, doing a full garbage collect, for maximum performance avoid circular references in the model code.".format(cur.real_model().__class__.__name__))
			do_gc = True
			break

	if do_gc:
		gc.collect()
		soft_empty_cache()

		for i in range(len(current_loaded_models)):
			cur = current_loaded_models[i]
			if cur.is_dead():
				logging.warning("WARNING, memory leak with model {}. Please make sure it is not being referenced from somewhere.".format(cur.real_model().__class__.__name__))



def cleanup_models():
	to_delete = []
	for i in range(len(current_loaded_models)):
		if current_loaded_models[i].real_model() is None:
			to_delete = [i] + to_delete

	for i in to_delete:
		x = current_loaded_models.pop(i)
		del x

def dtype_size(dtype):
	dtype_size = 4
	if dtype == torch.float16 or dtype == torch.bfloat16:
		dtype_size = 2
	elif dtype == torch.float32:
		dtype_size = 4
	else:
		try:
			dtype_size = dtype.itemsize
		except: #Old pytorch doesn't have .itemsize
			pass
	return dtype_size

def unet_offload_device():
	if vram_state == VRAMState.HIGH_VRAM:
		return get_torch_device()
	else:
		return torch.device("cpu")

def unet_inital_load_device(parameters, dtype):
	torch_dev = get_torch_device()
	if vram_state == VRAMState.HIGH_VRAM or vram_state == VRAMState.SHARED:
		return torch_dev

	cpu_dev = torch.device("cpu")
	if DISABLE_SMART_MEMORY or vram_state == VRAMState.NO_VRAM:
		return cpu_dev

	model_size = dtype_size(dtype) * parameters

	mem_dev = get_free_memory(torch_dev)
	mem_cpu = get_free_memory(cpu_dev)
	if mem_dev > mem_cpu and model_size < mem_dev:
		return torch_dev
	else:
		return cpu_dev

def maximum_vram_for_weights(device=None):
	return (get_total_memory(device) * 0.88 - minimum_inference_memory())

def unet_dtype(device=None, model_params=0, supported_dtypes=[torch.float16, torch.bfloat16, torch.float32], weight_dtype=None):
	if model_params < 0:
		model_params = 1000000000000000000000
	if args.fp32_unet:
		return torch.float32
	if args.fp64_unet:
		return torch.float64
	if args.bf16_unet:
		return torch.bfloat16
	if args.fp16_unet:
		return torch.float16
	if args.fp8_e4m3fn_unet:
		return torch.float8_e4m3fn
	if args.fp8_e5m2_unet:
		return torch.float8_e5m2
	if args.fp8_e8m0fnu_unet:
		return torch.float8_e8m0fnu

	fp8_dtype = None
	if weight_dtype in FLOAT8_TYPES:
		fp8_dtype = weight_dtype

	if fp8_dtype is not None:
		if supports_fp8_compute(device): #if fp8 compute is supported the casting is most likely not expensive
			return fp8_dtype

		free_model_memory = maximum_vram_for_weights(device)
		if model_params * 2 > free_model_memory:
			return fp8_dtype

	if PRIORITIZE_FP16 or weight_dtype == torch.float16:
		if torch.float16 in supported_dtypes and should_use_fp16(device=device, model_params=model_params):
			return torch.float16

	for dt in supported_dtypes:
		if dt == torch.float16 and should_use_fp16(device=device, model_params=model_params):
			if torch.float16 in supported_dtypes:
				return torch.float16
		if dt == torch.bfloat16 and should_use_bf16(device, model_params=model_params):
			if torch.bfloat16 in supported_dtypes:
				return torch.bfloat16

	for dt in supported_dtypes:
		if dt == torch.float16 and should_use_fp16(device=device, model_params=model_params, manual_cast=True):
			if torch.float16 in supported_dtypes:
				return torch.float16
		if dt == torch.bfloat16 and should_use_bf16(device, model_params=model_params, manual_cast=True):
			if torch.bfloat16 in supported_dtypes:
				return torch.bfloat16

	return torch.float32

# None means no manual cast
def unet_manual_cast(weight_dtype, inference_device, supported_dtypes=[torch.float16, torch.bfloat16, torch.float32]):
	if weight_dtype == torch.float32 or weight_dtype == torch.float64:
		return None

	fp16_supported = should_use_fp16(inference_device, prioritize_performance=False)
	if fp16_supported and weight_dtype == torch.float16:
		return None

	bf16_supported = should_use_bf16(inference_device)
	if bf16_supported and weight_dtype == torch.bfloat16:
		return None

	fp16_supported = should_use_fp16(inference_device, prioritize_performance=True)
	if PRIORITIZE_FP16 and fp16_supported and torch.float16 in supported_dtypes:
		return torch.float16

	for dt in supported_dtypes:
		if dt == torch.float16 and fp16_supported:
			return torch.float16
		if dt == torch.bfloat16 and bf16_supported:
			return torch.bfloat16

	return torch.float32

def text_encoder_offload_device():
	if args.gpu_only:
		return get_torch_device()
	else:
		return torch.device("cpu")

def text_encoder_device():
	if args.gpu_only:
		return get_torch_device()
	elif vram_state == VRAMState.HIGH_VRAM or vram_state == VRAMState.NORMAL_VRAM:
		if should_use_fp16(prioritize_performance=False):
			return get_torch_device()
		else:
			return torch.device("cpu")
	else:
		return torch.device("cpu")

def text_encoder_initial_device(load_device, offload_device, model_size=0):
	if load_device == offload_device or model_size <= 1024 * 1024 * 1024:
		return offload_device

	if is_device_mps(load_device):
		return load_device

	mem_l = get_free_memory(load_device)
	mem_o = get_free_memory(offload_device)
	if mem_l > (mem_o * 0.5) and model_size * 1.2 < mem_l:
		return load_device
	else:
		return offload_device

def text_encoder_dtype(device=None):
	if args.fp8_e4m3fn_text_enc:
		return torch.float8_e4m3fn
	elif args.fp8_e5m2_text_enc:
		return torch.float8_e5m2
	elif args.fp16_text_enc:
		return torch.float16
	elif args.bf16_text_enc:
		return torch.bfloat16
	elif args.fp32_text_enc:
		return torch.float32

	if is_device_cpu(device):
		return torch.float16

	return torch.float16


def intermediate_device():
	if args.gpu_only:
		return get_torch_device()
	else:
		return torch.device("cpu")

def vae_device():
	if args.cpu_vae:
		return torch.device("cpu")
	return get_torch_device()

def vae_offload_device():
	if args.gpu_only:
		return get_torch_device()
	else:
		return torch.device("cpu")

def vae_dtype(device=None, allowed_dtypes=[]):
	if args.fp16_vae:
		return torch.float16
	elif args.bf16_vae:
		return torch.bfloat16
	elif args.fp32_vae:
		return torch.float32

	for d in allowed_dtypes:
		if d == torch.float16 and should_use_fp16(device):
			return d

		if d == torch.bfloat16 and should_use_bf16(device):
			return d

	return torch.float32

def get_autocast_device(dev):
	if hasattr(dev, 'type'):
		return dev.type
	return "cuda"

def supports_dtype(device, dtype): #TODO
	if dtype == torch.float32:
		return True
	if is_device_cpu(device):
		return False

# ---------------------------------------------------------------------------
# MyCandy / ARC compatibility shim for newer ComfyUI versions
# Ergänzt Hilfsfunktionen, die von neueren Modulen erwartet werden.
# ---------------------------------------------------------------------------

import torch as _torch

def supports_dtype(device, dtype):
    """Konservative Variante von supports_dtype.

    * float32 geht überall
    * auf CPU lassen wir half/bf16 vorsichtshalber nicht zu
    * auf GPU / DirectML erlauben wir float16 und bfloat16
    """
    if dtype == _torch.float32:
        return True

    dev_type = getattr(device, "type", str(device))
    if dev_type == "cpu":
        return False

    allowed = {_torch.float16, getattr(_torch, "bfloat16", None)}
    allowed.discard(None)
    return dtype in allowed

def supports_cast(device, dtype):
    """Nutzt dieselbe Logik wie supports_dtype.

    Wird u.a. von CLIP/TE genutzt, um zu entscheiden, ob der Cast
    auf dem jeweiligen Device überhaupt erlaubt ist.
    """
    return supports_dtype(device, dtype)

def xformers_enabled_vae():
    """In deinem Setup ist xformers nicht installiert -> immer False."""
    return False

def pytorch_attention_enabled_vae():
    """Wir nutzen standardmäßig die PyTorch-Attention im VAE."""
    return True

def force_channels_last():
    """Channels-last Layout ist hauptsächlich für GPU-Optimierung.

    Auf CPU eher überflüssig – daher hier konservativ False.
    """
    return False

# Anzahl der Streams, die das Offload-System benutzen darf.
# Für CPU / DirectML ist 1 sehr konservativ und sicher.
NUM_STREAMS = 1
