bin_PROGRAMS = lz77_ref lz77_test
SUFFIXES = .lint

AM_CXXFLAGS = $(OPENCL_CFLAGS)

lz77_ref_SOURCES = lz77-reference.c

lz77_test_SOURCES = lz77_test.c
lz77_test_CFLAGS = -I../src
lz77_test_LDFLAGS = -L../src -lpz

check-local: lz77_test
	./lz77_test
