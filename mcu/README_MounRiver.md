# Note

## Makefile patch

* If you use `MounRiver Studio` RISC-V GCC toolchain, please modify the makefile at `ch32v003fun/ch32v003fun/ch32v003fun.mk`:

```diff
# Default prefix for Windows
ifeq ($(OS),Windows_NT)
-    PREFIX?=riscv64-unknown-elf
+    PREFIX?=riscv-none-elf
# Check if riscv64-linux-gnu-gcc exists
else ifneq ($(shell which riscv64-linux-gnu-gcc),)
    PREFIX?=riscv64-linux-gnu
# Check if riscv64-unknown-elf-gcc exists
else ifneq ($(shell which riscv64-unknown-elf-gcc),)
    PREFIX?=riscv64-unknown-elf
# Default prefix
else
    PREFIX?=riscv64-elf
endif
```

Or `riscv-none-embed` if you use `GCC` instead of `GCC 12`.
