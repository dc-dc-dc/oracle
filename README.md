# Oracle

An all in one python script with zero dependencies to print system information.

Supports:
- OpenCL ✅
- METAL 🔜
- CUDA ✅
- ROCM 🔜
- CPU ✅
- RAM ✅

## Usage

```bash
./oracle.py
```

Want a clinfo like output?
```bash
./oracle.py --opencl
```

Want to only get the cuda ouput?
```bash
./oracle.py --cuda
```

## Testing

Compare the output of oracle to clinfo for opencl
```bash
python3 ./extra/clinfo.py
```