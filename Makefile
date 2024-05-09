SOURCES = BitNetMCU_MNIST_dll.c BitNetMCU_inference.c
HEADERS = BitNetMCU_model.h  BitNetMCU_inference.h
DLL = Bitnet_inf.dll

$(DLL): $(SOURCES) $(HEADERS)
	cc -fPIC -shared -o $@ -D _DLL $<

.PHONY: clean
clean:
	rm -f $(DLL)
