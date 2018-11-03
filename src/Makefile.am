SUFFIXES = .lint
bin_PROGRAMS = pz
#noinst_LIBRARIES = libpz.a#libpz.so
lib_LTLIBRARIES = libpz.la

libpz_la_SOURCES = lz77.c
include_HEADERS = lz77.h

AM_CXXFLAGS = $(OPENCL_CFLAGS)

pz_SOURCES = main.c
#pz_LDFLAGS = -L.
pz_LDADD = libpz.la $(OPENCL_LIBS)

LINT_FILES = $(pz_SOURCES) $(libpz_la_SOURCES)

# Make this OK to fail, tee our lint to the file as well.
LINTER_FLAGS ?= -- -I. $(AM_CPPFLAGS) $(DEFS)
FORMATTER_FLAGS ?= -style=Chromium -i

.c.lint:
	- $(LINTER) $? $(LINTER_FLAGS) | tee $@
# Select only C files
# We have to abuse the auto rule above so that -j works
lint: $(subst .c,.lint, $(filter %.c,$(LINT_FILES)))

format: $(LINT_FILES)
	$(FORMATTER) $(FORMATTER_FLAGS) $?

.PHONY: lint format

check-local: format lint

clean-local:
	rm -f *.lint
