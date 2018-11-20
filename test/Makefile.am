#bin_PROGRAMS = lz77_ref #lz77_fuzz
CC=clang
check_PROGRAMS = lz77_test lz77_fuzz frequency_test huffman_fuzz huffman_test
SUFFIXES = .lint

AM_CPPFLAGS=-I$(top_builddir)/include

frequency_test_SOURCES = frequency_test.c
frequency_test_CFLAGS = $(AM_CPPFLAGS)
frequency_test_LDFLAGS = -L../src -static
frequency_test_LDADD = ../reference/libpz.la

lz77_test_SOURCES = lz77_test.c
lz77_test_CFLAGS = -I../src
lz77_test_LDFLAGS = -L../src -static
lz77_test_LDADD = ../reference/libpz.la

huffman_test_SOURCES = huffman_test.c
huffman_test_CFLAGS = $(AM_CPPFLAGS) # -fsanitize=address
huffman_test_LDFLAGS = -L../src -static
huffman_test_LDADD = ../reference/libpz.la

#lz77_fuzz_CC = clang
FUZZCFLAGS=-I../src -fsanitize=fuzzer,address -g -O0
FUZZLDFLAGS=-L../src -static
FUZZLDADD=../reference/libpz.la

lz77_fuzz_SOURCES = lz77_fuzz.c
lz77_fuzz_CFLAGS = $(FUZZCFLAGS)
lz77_fuzz_LDFLAGS = $(FUZZLDFLAGS)
lz77_fuzz_LDADD = $(FUZZLDADD)

huffman_fuzz_SOURCES = huffman_fuzz.c
huffman_fuzz_CFLAGS = $(FUZZCFLAGS)
huffman_fuzz_LDFLAGS = $(FUZZLDFLAGS)
huffman_fuzz_LDADD = $(FUZZLDADD)

#lz77_fuzz: lz77_fuzz.c
#	clang -g -O0 -fsanitize=fuzzer lz77_fuzz.c -o lz77_fuzz -I../src -L../src -lpz

fuzz: lz77_fuzz
	./lz77_fuzz -runs=100000

check-local: lz77_test fuzz
	./lz77_test
	./frequency_test

all-local: lz77_test lz77_fuzz frequency_test huffman_fuzz huffman_test

clean-local:
	rm -rf lz77_fuzz
