all : build

TARGET:=BitNetMCUdemo

# Use ch32v003fun as a submodule

CH32V003FUN ?= ch32v003fun/ch32v003fun


# Alternatively, set the path to the ch32v003fun directory here

# CH32V003FUN ?= /mnt/d/RISCV/ch32v003fun/ch32v003fun

include $(CH32V003FUN)/ch32v003fun.mk

flash : cv_flash
clean : cv_clean

