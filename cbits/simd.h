#include <immintrin.h>
#include <memory.h>
#include <stddef.h>

#if __STDC_VERSION__ >= 199901L
  /* yay, C99 or newer!*/
#else
#warning "C99 or newer compliant compiler needed"
#endif



#ifdef __clang__
  /* yay, Clang!*/
#else
#warning "Clang is the only supported C compiler for now"
#endif