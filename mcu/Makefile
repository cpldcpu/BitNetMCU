all : build

TARGET:=BitNetMCUdemo

# Make sure the 'CH32V003FUN' environment variable is set and points to the ch32v003fun directory.
# Alternatively, set the path to the ch32v003fun directory here
CH32V003FUN ?= /mnt/d/RISCV/ch32v003fun/ch32v003fun

include $(CH32V003FUN)/ch32v003fun.mk

flash : cv_flash
clean : cv_clean

