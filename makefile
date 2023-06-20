metal:
	clang -lobjc -framework Metal -framework CoreGraphics -framework Foundation examples/metal.m -o objc && ./objc 
	rm objc