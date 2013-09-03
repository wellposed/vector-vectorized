#include <immintrin.h>
#include <memory.h>
#include <stddef.h>
#include <stdint.h>



#if __STDC_VERSION__ >= 199901L
  /* yay, C99 or newer!*/
#else
#warning "C99 or newer C standard compliant compiler is strongly recommended"
#endif


/*
       #if __GNUC__ > 3 || \
              (__GNUC__ == 3 && (__GNUC_MINOR__ > 2 || \
                                 (__GNUC_MINOR__ == 2 && \
                                  __GNUC_PATCHLEVEL__ > 0))

*/
#if defined(__clang__)
  /* yay, Clang! */
#elif defined(__GNUC__) && __GNUC__ > 4 || \
              (__GNUC__ == 4 && (__GNUC_MINOR__ > 4 || \
                                 (__GNUC_MINOR__ == 4 && \
                                  __GNUC_PATCHLEVEL__ >=  0))
/* yay recent gcc*/
#else 
#warning "Clang is the only supported C compiler for now, but recent GCC should be fine too"
#endif

