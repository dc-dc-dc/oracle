import subprocess, json

clinfo_raw = subprocess.check_output(['clinfo', "--raw"])
lines = [x.strip() for x in clinfo_raw.decode("utf-8").split("\n")]
numplatforms = 0
clinfo_res = dict()
for line in lines:
    if line.startswith("#"):
        numplatforms = int(line.split(" ")[-1])
    elif line.startswith("["):
        device = line[:5].split("/")[-1]
        if device == "*":
            continue
        line = line[6:].strip()
        name = line.split(" ")[0]
        value = line[len(name):].strip()
        if "devices" not in clinfo_res: clinfo_res["devices"] = dict()
        if device not in clinfo_res["devices"]: clinfo_res["devices"][device] = dict()
        if name != "" and value != "": clinfo_res["devices"][device][name] = value
    else:
        name = line.split(" ")[0]
        value = line[len(name):].strip()
        if name != "" and value != "": clinfo_res[name] = value

oracle_res = json.loads(subprocess.check_output(['./oracle.py', '--opencl']).decode("utf-8"))
def get_platform_oracle(platform: str): # -> dict
    for _,p in oracle_res.items():
        if p["name"] == platform: return p
    return None

def get_oracle_device(platform, name: str):
    for _,v in platform["devices"].items():
        if v['name'] == name: return v 
    return None

platform = None
for k,v in clinfo_res.items():
    if k == "CL_PLATFORM_NAME":
        platform = get_platform_oracle(v)
        break

if platform == None:
    print("ERROR platform not found in oracle")    

clinfo_map = { 
    "CL_PLATFORM_NAME": (lambda x: x["name"], lambda x: x),
    "CL_PLATFORM_VENDOR": (lambda x: x["vendor"], lambda x: x),
    "CL_PLATFORM_VERSION": (lambda x: x["version"], lambda x: x),
    "CL_PLATFORM_PROFILE": (lambda x: x["profile"], lambda x: x),
    "CL_PLATFORM_EXTENSIONS": (lambda x: x["extensions"], lambda x: x.split(" ")),
    "CL_DEVICE_NAME": (lambda x: x["name"], lambda x: x),
    "CL_DEVICE_VENDOR": (lambda x: x["vendor"], lambda x: x),
    "CL_DEVICE_VENDOR_ID": (lambda x: x["vendor_id"], lambda x: int(x,16)),
    "CL_DEVICE_VERSION": (lambda x: x["version"], lambda x: x.strip()),
    "CL_DRIVER_VERSION": (lambda x: x["driver"], lambda x: x),
    "CL_DEVICE_TYPE": (lambda x: x["type"], lambda x: x),
    "CL_DEVICE_PROFILE": (lambda x: x["profile"], lambda x: x),
    "CL_DEVICE_EXTENSIONS": (lambda x: x["extensions"], lambda x: x.split(" ")),
    "CL_DEVICE_AVAILABLE": (lambda x: x["device_available"], lambda x: x == "CL_TRUE"),
    "CL_DEVICE_COMPILER_AVAILABLE": (lambda x: x["compiler_available"], lambda x: x == "CL_TRUE"),
    "CL_DEVICE_LINKER_AVAILABLE": (lambda x: x["linker_available"], lambda x: x == "CL_TRUE"),
    "CL_DEVICE_IMAGE_SUPPORT": (lambda x: x["image_support"], lambda x: x == "CL_TRUE"),
    "CL_DEVICE_MAX_COMPUTE_UNITS": (lambda x: x["max_compute_units"], lambda x: int(x)),
    "CL_DEVICE_MAX_SAMPLERS": (lambda x: x["max_samplers"], lambda x: int(x)),
    "CL_DEVICE_MAX_CLOCK_FREQUENCY": (lambda x: x["max_clock_frequency"], lambda x: int(x)),
    "CL_DEVICE_ADDRESS_BITS": (lambda x: x["address_bits"], lambda x: int(x)),
    "CL_DEVICE_ENDIAN_LITTLE": (lambda x: x["endian_little"], lambda x: x == "CL_TRUE"),
    "CL_DEVICE_MAX_PARAMETER_SIZE": (lambda x: x["max_parameter_size"], lambda x: int(x)),
    "CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS": (lambda x: x["max_work_item_dimensions"], lambda x: int(x)),
    "CL_DEVICE_IMAGE2D_MAX_HEIGHT": (lambda x: x["image2d_max_height"], lambda x: int(x)),
    "CL_DEVICE_IMAGE2D_MAX_WIDTH": (lambda x: x["image2d_max_width"], lambda x: int(x)),
    "CL_DEVICE_IMAGE3D_MAX_HEIGHT": (lambda x: x["image3d_max_height"], lambda x: int(x)),
    "CL_DEVICE_IMAGE3D_MAX_WIDTH": (lambda x: x["image3d_max_width"], lambda x: int(x)),
    "CL_DEVICE_IMAGE3D_MAX_DEPTH": (lambda x: x["image3d_max_depth"], lambda x: int(x)),
    "CL_DEVICE_MAX_READ_IMAGE_ARGS": (lambda x: x["max_read_image_args"], lambda x: int(x)),
    "CL_DEVICE_MAX_WRITE_IMAGE_ARGS": (lambda x: x["max_write_image_args"], lambda x: int(x)),
    "CL_DEVICE_MAX_WORK_ITEM_SIZES": (lambda x: x["max_work_item_size"], lambda x: [int(y) for y in x.split(" ")]),
    "CL_DEVICE_MAX_WORK_GROUP_SIZE": (lambda x: x["max_work_group_size"], lambda x: int(x)),
    "CL_DEVICE_MAX_MEM_ALLOC_SIZE": (lambda x: x["max_mem_alloc_size"], lambda x: int(x)),
    "CL_DEVICE_GLOBAL_MEM_SIZE": (lambda x: x["global_mem_size"], lambda x: int(x)),
    "CL_DEVICE_MAX_CONSTANT_ARGS": (lambda x: x["max_constant_args"], lambda x: int(x)),
    "CL_DEVICE_LOCAL_MEM_SIZE": (lambda x: x["local_mem_size"], lambda x: int(x)),
    "CL_DEVICE_ERROR_CORRECTION_SUPPORT": (lambda x: x["ecc_enabled"], lambda x: x == "CL_TRUE"),
    "CL_DEVICE_IMAGE_PITCH_ALIGNMENT": (lambda x: x["image_pitch_alignment"], lambda x: int(x)),
    "CL_DEVICE_IMAGE_BASE_ADDRESS_ALIGNMENT": (lambda x: x["image_base_address_alignment"], lambda x: int(x)),
    "CL_DEVICE_HOST_UNIFIED_MEMORY": (lambda x: x["host_unified_memory"], lambda x: x == "CL_TRUE"),
    "CL_DEVICE_IMAGE_MAX_ARRAY_SIZE": (lambda x: x["image_max_array_size"], lambda x: int(x)),
    "CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE": (lambda x: x["min_data_type_align_size"], lambda x: int(x)),
    "CL_DEVICE_IMAGE_MAX_BUFFER_SIZE": (lambda x: x["image_max_buffer_size"], lambda x: int(x)),
    "CL_DEVICE_OPENCL_C_VERSION": (lambda x: x["opencl_c_version"], lambda x: x.strip()),
    "CL_DEVICE_MEM_BASE_ADDR_ALIGN": (lambda x: x["mem_base_addr_align"], lambda x: int(x)),
    "CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE": (lambda x: x["max_constant_buffer_size"], lambda x: int(x)),
    "CL_DEVICE_PREFERRED_INTEROP_USER_SYNC": (lambda x: x["preferred_interop_user_sync"], lambda x: x == "CL_TRUE"),
    "CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR": (lambda x: x["preferred_vector_sizes"]["char"], lambda x: int(x)),
    "CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT": (lambda x: x["preferred_vector_sizes"]["short"], lambda x: int(x)),
    "CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT": (lambda x: x["preferred_vector_sizes"]["int"], lambda x: int(x)),
    "CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG": (lambda x: x["preferred_vector_sizes"]["long"], lambda x: int(x)),
    "CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT": (lambda x: x["preferred_vector_sizes"]["float"], lambda x: int(x)),
    "CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE": (lambda x: x["preferred_vector_sizes"]["double"], lambda x: int(x)),
    "CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF": (lambda x: x["preferred_vector_sizes"]["half"], lambda x: int(x)),
    "CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR": (lambda x: x["native_vector_sizes"]["char"], lambda x: int(x)),
    "CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT": (lambda x: x["native_vector_sizes"]["short"], lambda x: int(x)),
    "CL_DEVICE_NATIVE_VECTOR_WIDTH_INT": (lambda x: x["native_vector_sizes"]["int"], lambda x: int(x)),
    "CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG": (lambda x: x["native_vector_sizes"]["long"], lambda x: int(x)),
    "CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT": (lambda x: x["native_vector_sizes"]["float"], lambda x: int(x)),
    "CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE": (lambda x: x["native_vector_sizes"]["double"], lambda x: int(x)),
    "CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF": (lambda x: x["native_vector_sizes"]["half"], lambda x: int(x)),
    "CL_DEVICE_PRINTF_BUFFER_SIZE": (lambda x: x["printf_buffer_size"], lambda x: int(x)),
    "CL_DEVICE_PARTITION_MAX_SUB_DEVICES": (lambda x: x["partition_max_sub_devices"], lambda x: int(x)),
    "CL_DEVICE_GLOBAL_MEM_CACHE_TYPE": (lambda x: x["global_mem_cache_type"], lambda x: x),
    "CL_DEVICE_LOCAL_MEM_TYPE": (lambda x: x["local_mem_type"], lambda x: x),
    "CL_DEVICE_EXECUTION_CAPABILITIES": (lambda x: x["execution_capabilities"], lambda x: x),
    "CL_DEVICE_QUEUE_PROPERTIES": (lambda x: x["queue_properties"], lambda x: x),
}

for k,v in clinfo_res.items():
    if k in clinfo_map: assert clinfo_map[k][0](platform) == clinfo_map[k][1](v), f"Value mismatch got {clinfo_map[k][0](platform)} expecting {clinfo_map[k][1](v)}"
    elif k == "devices":
        for di in v.keys():
            device = None
            for dk, dv in v[di].items():
                if dk == 'CL_DEVICE_NAME': 
                    device = get_oracle_device(platform, dv)
                    break
            if device == None:
                print(f"ERROR device {dv} not found in oracle")
            for dk, dv in v[di].items():
                if dk in clinfo_map: assert clinfo_map[dk][0](device) == clinfo_map[dk][1](dv), f"Value mismatch for {dk} got {clinfo_map[dk][0](device)} expecting {clinfo_map[dk][1](dv)}"
                else: print(f"ERROR: Key {dk} with val {dv} not in oracle device")
    else:
        print(f"ERROR: Key {k} not in oracle")
# print("clinfo:", clinfo_res)
# print("oracle:", oracle_res)