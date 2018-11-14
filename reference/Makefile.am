SUFFIXES = .lint
#noinst_LIBRARIES = libpz.a#libpz.so
lib_LTLIBRARIES = libpz.la

libpz_la_SOURCES = lz77.c

AM_CPPFLAGS=-I$(top_builddir)/include
#pz_LDFLAGS = -L.

LINT_FILES = $(pz_SOURCES) $(libpz_la_SOURCES)

# Make this OK to fail, tee our lint to the file as well.
LINTER_FLAGS ?= -- -I. -I.. $(AM_CPPFLAGS) $(DEFS)
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
