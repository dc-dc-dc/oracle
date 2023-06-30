import json, argparse, sys
from oracle.system.opencl import opencl
from oracle.system.cuda import cuda
from oracle.system.general import system 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get system information", prog="oracle")
    parser.add_argument("-j", "--json", action="store_true", help="Output as JSON")
    parser.add_argument("-e", "--error", action="store_true", help="Output errors to stderr")
    parser.add_argument("--opencl", action="store_true", help="Get OpenCL information")
    # parser.add_argument("--metal", action="store_true", help="Get Metal information")
    parser.add_argument("--cuda", action="store_true", help="Get CUDA information")
    args = parser.parse_args()
    
    LOG_ERROR = args.error
    if args.opencl: dets=opencl()
    # elif args.metal: dets=metal()
    elif args.cuda: dets=cuda()
    else: dets=system()
    
    sys.stdout.write(json.dumps(dets, indent=2)+"\n")