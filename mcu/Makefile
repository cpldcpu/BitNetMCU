all : build

TARGET:=BitNetMCUdemo
# TARGET_MCU?=CH32V003
TARGET_MCU?=CH32V002

CH32FUN ?= ch32v003fun/ch32fun

include $(CH32FUN)/ch32fun.mk

flash : cv_flash
clean : cv_clean

