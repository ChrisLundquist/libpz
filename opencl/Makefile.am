SUFFIXES = .lint
lib_LTLIBRARIES = libpz-opencl.la
CHECKdir = check

libpz_opencl_la_SOURCES = engine.c
#include_HEADERS = engine.h

OPENCL_LIBS=-lOpenCL

AM_CPPFLAGS=-I$(top_builddir)/include
CHECK_PROGRAMS = opencl_test
opencl_test_SOURCES = test.c
opencl_test_LDADD = libpz-opencl.la $(OPENCL_LIBS)

LINT_FILES = $(libpz_opencl_la_SOURCES)

# Make this OK to fail, tee our lint to the file as well.
LINTER_FLAGS ?= -- -I. $(AM_CPPFLAGS) $(DEFS)
FORMATTER_FLAGS ?= -style=Chromium -i

%.c.lint: %.c
	- $(LINTER) $? $(LINTER_FLAGS) | tee $@
# Select only C files
# We have to abuse the auto rule above so that -j works
lint: $(subst .c,.c.lint, $(filter %.c,$(LINT_FILES)))

format: $(LINT_FILES)
	$(FORMATTER) $(FORMATTER_FLAGS) $?

.PHONY: lint format

check-local: format lint

clean-local:
	rm -f *.lint