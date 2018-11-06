bin_PROGRAMS = lz77_ref lz77_test
SUFFIXES = .lint

AM_CXXFLAGS = $(OPENCL_CFLAGS)

lz77_ref_SOURCES = lz77-reference.c

lz77_test_SOURCES = lz77_test.c
lz77_test_CFLAGS = -I../src
lz77_test_LDFLAGS = -L../src -static
lz77_test_LDADD = ../src/libpz.la

lz77_fuzz: lz77_fuzz.c
	clang -g -O1 -fsanitize=fuzzer,address lz77_fuzz.c -o lz77_fuzz -I../src -L../src ../src/libpz.a

fuzz: lz77_fuzz
	./lz77_fuzz -runs=100000

check-local: lz77_test fuzz
	./lz77_test
