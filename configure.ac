AC_INIT([pz], [1.0], [bug-automake@gnu.org])
AM_INIT_AUTOMAKE([-Wall -Wno-portability foreign])
LT_INIT
AC_PROG_CC
AC_CONFIG_MACRO_DIRS([m4])
AC_CONFIG_HEADERS([config.h])
AC_CONFIG_FILES([
 Makefile
 opencl/Makefile
 samples/Makefile
])

#
# Linter, optional
#
AC_PATH_PROG([LINTER], [clang-tidy], [true])
AS_IF([test "x$LINTER" = xtrue],
    [AC_MSG_WARN([We recommend clang-tidy for developing this package.])])

#
# Formatter, optional
#
AC_PATH_PROG([FORMATTER], [clang-format], [true])
AS_IF([test "x$LINTER" = xtrue],
    [AC_MSG_WARN([We recommend clang-format for developing this package.])])

#
# Cflags -Wall -Wextra -std=c11
#
AC_LANG_CASE([C], [
    saved_cflags="$CFLAGS"
    AX_CHECK_COMPILE_FLAG([-std=gnu99], [
        AX_APPEND_FLAG(-std=c11, [])
    ])

    AX_CHECK_COMPILE_FLAG([-Wall], [
        AX_APPEND_FLAG(-Wall, [])
    ])

    AX_CHECK_COMPILE_FLAG([-Wextra], [
        AX_APPEND_FLAG(-Wextra, [])
    ])
])
# TODO AC_SEARCH_LIB OpenCL
#AX_COMPILER_FLAGS()
AC_OUTPUT
