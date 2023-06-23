metal:
	clang -lobjc -framework Metal -framework CoreGraphics -framework Foundation extra/metal.m -o metal-info && ./metal-info 
	rm metal-info