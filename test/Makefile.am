bin_PROGRAMS = lz77_ref #lz77_fuzz
check_PROGRAMS = lz77_test lz77_fuzz
SUFFIXES = .lint

AM_CXXFLAGS = $(OPENCL_CFLAGS)

lz77_ref_SOURCES = lz77-reference.c

lz77_test_SOURCES = lz77_test.c
lz77_test_CFLAGS = -I../src
lz77_test_LDFLAGS = -L../src -static
lz77_test_LDADD = ../src/libpz.la

#lz77_fuzz_CC = clang
lz77_fuzz_SOURCES = lz77_fuzz.c
lz77_fuzz_CFLAGS = -I../src -fsanitize=fuzzer,address -g -O0
lz77_fuzz_LDFLAGS = -L../src -static
lz77_fuzz_LDADD = ../src/libpz.la

#lz77_fuzz: lz77_fuzz.c
#	clang -g -O0 -fsanitize=fuzzer lz77_fuzz.c -o lz77_fuzz -I../src -L../src -lpz

fuzz: lz77_fuzz
	./lz77_fuzz -runs=100000

check-local: lz77_test fuzz
	./lz77_test

clean-local:
	rm -rf lz77_fuzz
