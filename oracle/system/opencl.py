import ctypes, sys
from oracle.util import OSX, error_wrap, LOG_ERROR

cl_device_map = {
    1: "CL_DEVICE_TYPE_DEFAULT",
    2: "CL_DEVICE_TYPE_CPU",
    4: "CL_DEVICE_TYPE_GPU",
    8: "CL_DEVICE_TYPE_ACCELERATOR",
    16: "CL_DEVICE_TYPE_CUSTOM",
}

cl_cache_map = {
    0: "CL_NONE",
    1: "CL_READ_ONLY_CACHE",
    2: "CL_READ_WRITE_CACHE",
}

cl_queue_map = {
    1: "CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE",
    2: "CL_QUEUE_PROFILING_ENABLE",
}

def __clGetPlatformInfo(_clGetPlatformInfo, platform_id, param_name):
    name_size = ctypes.c_uint()
    error_wrap("clGetPlatformInfo", _clGetPlatformInfo(platform_id, param_name, None, None, ctypes.byref(name_size)))
    name = (ctypes.c_char * name_size.value)()
    error_wrap("clGetPlatformInfo", _clGetPlatformInfo(platform_id, param_name, name_size.value, ctypes.byref(name), None))
    return name.value.decode("utf-8")


def __clGetDeviceInfo(_clGetDeviceInfo, device_id, param_name, size, ctype: ctypes._SimpleCData): 
    error_wrap("clGetDeviceInfo", _clGetDeviceInfo(device_id, param_name, size, ctypes.byref(ctype), None))
    return ctype

def __clGetDeviceInfoStr(_clGetDeviceInfo, device_id, param_name):
    name_size = ctypes.c_uint()
    error_wrap("clGetDeviceInfo", _clGetDeviceInfo(device_id, param_name, None, None, ctypes.byref(name_size)))
    return __clGetDeviceInfo(_clGetDeviceInfo, device_id, param_name, name_size.value, (ctypes.c_char * name_size.value)()).value.decode("utf-8").strip()

def __clGetDeviceInfoInt(_clGetDeviceInfo, device_id, param_name): return __clGetDeviceInfo(_clGetDeviceInfo, device_id, param_name, 8, ctypes.c_uint()).value
def __clGetDeviceInfoBool(_clGetDeviceInfo, device_id, param_name): return __clGetDeviceInfo(_clGetDeviceInfo, device_id, param_name, 4, ctypes.c_bool()).value
def __clGetDeviceInfoSize(_clGetDeviceInfo, device_id, param_name): return __clGetDeviceInfo(_clGetDeviceInfo, device_id, param_name, 16, ctypes.c_size_t()).value
def __clGetDeviceInfoLong(_clGetDeviceInfo, device_id, param_name): return __clGetDeviceInfo(_clGetDeviceInfo, device_id, param_name, 16, ctypes.c_long()).value
def __clGetDeviceInfoLongArr(_clGetDeviceInfo, device_id, param_name, size): return __clGetDeviceInfo(_clGetDeviceInfo, device_id, param_name, 16*size, (ctypes.c_long*size)())[:]


def opencl():
    try:
        opencl = ctypes.cdll.LoadLibrary("/System/Library/Frameworks/OpenCL.framework/OpenCL" if OSX else "libOpenCL.so")
        _clGetDeviceInfo = opencl["clGetDeviceInfo"]
        _clGetDeviceIDs = opencl["clGetDeviceIDs"]
        _clGetPlatformIDs = opencl["clGetPlatformIDs"]
        _clGetPlatformInfo = opencl["clGetPlatformInfo"]
        platform_count = ctypes.c_uint()
        error_wrap("clGetPlatformIDs", _clGetPlatformIDs(0, None, ctypes.byref(platform_count))) 
        platform_ids = (ctypes.c_uint * platform_count.value)()
        error_wrap("clGetPlatformIDs", _clGetPlatformIDs(platform_count.value, ctypes.byref(platform_ids), None))
        _platforms = dict()
        for platform_id in platform_ids:
            temp = {
                "profile": __clGetPlatformInfo(_clGetPlatformInfo, platform_id, 0x0900),
                "version": __clGetPlatformInfo(_clGetPlatformInfo, platform_id, 0x0901),
                "vendor": __clGetPlatformInfo(_clGetPlatformInfo, platform_id, 0x0903),
                "name": __clGetPlatformInfo(_clGetPlatformInfo, platform_id, 0x0902),
                "extensions": __clGetPlatformInfo(_clGetPlatformInfo, platform_id, 0x0904).split(" "),
            }
            device_count = ctypes.c_uint()
            error_wrap("clGetDeviceIDs", _clGetDeviceIDs(platform_id, 0xFFFFFFFF, 0, None, ctypes.byref(device_count)))
            devices = (ctypes.c_uint * device_count.value)()
            error_wrap("clGetDeviceIDs", _clGetDeviceIDs(platform_id, 0xFFFFFFFF, device_count.value, ctypes.byref(devices), None))
            _devices = dict()
            for i in devices:
                _devices[i] = {
                    "name": __clGetDeviceInfoStr(_clGetDeviceInfo, i, 0x102B),
                    "vendor": __clGetDeviceInfoStr(_clGetDeviceInfo, i, 0x102C),
                    "vendor_id": __clGetDeviceInfoInt(_clGetDeviceInfo, i, 0x1001),
                    "version": __clGetDeviceInfoStr(_clGetDeviceInfo, i, 0x102F),
                    "driver": __clGetDeviceInfoStr(_clGetDeviceInfo, i, 0x102D),
                    "profile": __clGetDeviceInfoStr(_clGetDeviceInfo, i, 0x102E),
                    "opencl_c_version": __clGetDeviceInfoStr(_clGetDeviceInfo, i, 0x103D),
                    "ecc_enabled": __clGetDeviceInfoBool(_clGetDeviceInfo, i, 0x1024),
                    "extensions": [i for i in __clGetDeviceInfoStr(_clGetDeviceInfo, i, 0x1030).split(" ") if i],
                    "type": cl_device_map[__clGetDeviceInfoInt(_clGetDeviceInfo, i, 0x1000)],
                    "max_compute_units": __clGetDeviceInfoInt(_clGetDeviceInfo, i, 0x1002),
                    "max_work_item_dimensions": __clGetDeviceInfoInt(_clGetDeviceInfo, i, 0x1003),
                    "max_work_group_size": __clGetDeviceInfoSize(_clGetDeviceInfo, i, 0x1004),
                    "max_work_item_size": __clGetDeviceInfoLongArr(_clGetDeviceInfo, i, 0x1005, 3),
                    "max_clock_frequency": __clGetDeviceInfoInt(_clGetDeviceInfo, i, 0x100C),
                    "address_bits": __clGetDeviceInfoInt(_clGetDeviceInfo, i, 0x100D),
                    "device_available": __clGetDeviceInfoBool(_clGetDeviceInfo, i, 0x1027),
                    "compiler_available": __clGetDeviceInfoBool(_clGetDeviceInfo, i, 0x1028),
                    "linker_available": __clGetDeviceInfoBool(_clGetDeviceInfo, i, 0x103A),
                    "endian_little": __clGetDeviceInfoBool(_clGetDeviceInfo, i, 0x1026),
                    "image_support": __clGetDeviceInfoBool(_clGetDeviceInfo, i, 0x1016),
                    "max_read_image_args": __clGetDeviceInfoInt(_clGetDeviceInfo, i, 0x100E),
                    "max_write_image_args": __clGetDeviceInfoInt(_clGetDeviceInfo, i, 0x100F),
                    "image2d_max_width": __clGetDeviceInfoSize(_clGetDeviceInfo, i, 0x1011),
                    "image2d_max_height": __clGetDeviceInfoSize(_clGetDeviceInfo, i, 0x1012),
                    "image3d_max_width": __clGetDeviceInfoSize(_clGetDeviceInfo, i, 0x1013),
                    "image3d_max_height": __clGetDeviceInfoSize(_clGetDeviceInfo, i, 0x1014),
                    "image3d_max_depth": __clGetDeviceInfoSize(_clGetDeviceInfo, i, 0x1015),
                    "image_max_buffer_size": __clGetDeviceInfoSize(_clGetDeviceInfo, i, 0x1040),
                    "image_max_array_size": __clGetDeviceInfoSize(_clGetDeviceInfo, i, 0x1041),
                    "image_pitch_alignment": __clGetDeviceInfoSize(_clGetDeviceInfo, i, 0x104A),
                    "image_base_address_alignment": __clGetDeviceInfoSize(_clGetDeviceInfo, i, 0x104B),
                    "host_unified_memory": __clGetDeviceInfoBool(_clGetDeviceInfo, i, 0x1035),
                    "preferred_interop_user_sync": __clGetDeviceInfoBool(_clGetDeviceInfo, i, 0x1048),
                    "mem_base_addr_align": __clGetDeviceInfoInt(_clGetDeviceInfo, i, 0x1019),
                    "min_data_type_align_size": __clGetDeviceInfoInt(_clGetDeviceInfo, i, 0x101A),
                    "max_parameter_size": __clGetDeviceInfoSize(_clGetDeviceInfo, i, 0x1017),
                    "max_samplers": __clGetDeviceInfoInt(_clGetDeviceInfo, i, 0x1018),
                    "max_constant_args": __clGetDeviceInfoInt(_clGetDeviceInfo, i, 0x1021),
                    "max_constant_buffer_size": __clGetDeviceInfoLong(_clGetDeviceInfo, i, 0x1020),
                    "max_mem_alloc_size": __clGetDeviceInfoLong(_clGetDeviceInfo, i, 0x1010),
                    "global_mem_size": __clGetDeviceInfoLong(_clGetDeviceInfo, i, 0x101F),
                    "local_mem_size": __clGetDeviceInfoLong(_clGetDeviceInfo, i, 0x1023),
                    "printf_buffer_size": __clGetDeviceInfoSize(_clGetDeviceInfo, i, 0x1049),
                    "partition_max_sub_devices": __clGetDeviceInfoInt(_clGetDeviceInfo, i, 0x1044),
                    "global_mem_cache_type": cl_cache_map[__clGetDeviceInfoInt(_clGetDeviceInfo, i, 0x101C)],
                    "local_mem_type": {1: "CL_LOCAL", 2: "CL_GLOBAL"}[__clGetDeviceInfoInt(_clGetDeviceInfo, i, 0x1022)],
                    "queue_properties": cl_queue_map[__clGetDeviceInfoInt(_clGetDeviceInfo, i, 0x102A)],
                    "execution_capabilities": {1: "CL_EXEC_KERNEL", 2: "CL_EXEC_NATIVE_KERNEL"}[__clGetDeviceInfoInt(_clGetDeviceInfo, i, 0x1029)],
                    "native_vector_sizes": {
                        "char": __clGetDeviceInfoInt(_clGetDeviceInfo, i, 0x1036),
                        "short": __clGetDeviceInfoInt(_clGetDeviceInfo, i, 0x1037),
                        "int": __clGetDeviceInfoInt(_clGetDeviceInfo, i, 0x1038),
                        "long": __clGetDeviceInfoInt(_clGetDeviceInfo, i, 0x1039),
                        "half": __clGetDeviceInfoInt(_clGetDeviceInfo, i, 0x103C),
                        "float": __clGetDeviceInfoInt(_clGetDeviceInfo, i, 0x103A),
                        "double": __clGetDeviceInfoInt(_clGetDeviceInfo, i, 0x103B),
                    },
                    "preferred_vector_sizes": {
                        "char": __clGetDeviceInfoInt(_clGetDeviceInfo, i, 0x1006),
                        "short": __clGetDeviceInfoInt(_clGetDeviceInfo, i, 0x1007),
                        "int": __clGetDeviceInfoInt(_clGetDeviceInfo, i, 0x1008),
                        "long": __clGetDeviceInfoInt(_clGetDeviceInfo, i, 0x1009),
                        "half": __clGetDeviceInfoInt(_clGetDeviceInfo, i, 0x1034),
                        "float": __clGetDeviceInfoInt(_clGetDeviceInfo, i, 0x100A),
                        "double": __clGetDeviceInfoInt(_clGetDeviceInfo, i, 0x100B),
                    }
                }
            temp["devices"] = _devices
            _platforms[platform_id] = temp 
        return _platforms
    except Exception as e:
        if LOG_ERROR: sys.stderr.write(f"{e}\n")
        return None       
