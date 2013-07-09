#include "simd.h"
#include <math.h>


/*
we have 3 versions: 
avx2 (untested)
sse4/avx1 (and i guess sse2 by accident there too, well
explicit dot product vs not, so some complation)
plus scalar fallback

*/

/*
+ - * , abs,

*/

/* CPP macro to generate declarations and body for strided scalar versions */

#define BinaryOpScalarArray(name,op) void name(int length, double  * restrict  left,int leftStride ,double  * restrict  right,int rightStride, double * restrict  result, int resultStride  ); \
 \
void name(int length, double  * restrict  left,int leftStride ,double  * restrict  right,int rightStride, double * restrict  result, int resultStride  ){ \
    int ix = 0 ;  \
    for (ix = 0; ix < length ; ix ++){ \
        result[ix* resultStride]= (left[ix*leftStride] ) + (right[ix*rightStride]  ) ; \
    } \
}




////////////
/// Scalar versions
///////////

/*
for now I shall assume that everything only works on < 4gb / 32 sized ranges


also, will try to write things so that if any of the read (input) arrays
*/

BinaryOpScalarArray(arrayGeneralPlusM,+)

BinaryOpScalarArray(arrayGeneralMinusM,-)

BinaryOpScalarArray(arrayGeneralTimesM,*)

BinaryOpScalarArray(arrayGeneralDivideM,/)

