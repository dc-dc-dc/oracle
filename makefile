.PHONY: clean metal cuda_windows

metal:
	clang -lobjc -framework Metal -framework CoreGraphics -framework Foundation extra/metal.m -o ./out/metal-info && ./metal-info 

cuda_windows:
	clang ./extra/cuda.c -o ./out/cuda.exe -I"${CUDA_PATH}\include" -L"${CUDA_PATH}\lib\x64" -lcuda -lcudart
	./out/cuda.exe

clean:
	rm -rf ./out/*