metal:
	clang -lobjc -framework Metal -framework CoreGraphics -framework Foundation extra/metal.m -o objc && ./objc 
	rm objc