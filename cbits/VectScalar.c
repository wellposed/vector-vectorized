#include "simd.h"
#include <tgmath.h>
#include <complex.h>

/*
we have 3 versions: 
avx2(untested)
sse4/avx1 (and i guess sse2 by accident there too, well
explicit dot product vs not, so some complation)
plus scalar fallback

*/

/*
+ - * , abs,

note, for now using 32bit int types for array operation sizes and strides 
because you really shouldn't do more than 4gb of work in one sequential ffi call!

*/

/* CPP macro to generate declarations and body for strided scalar versions */
#define negate(numexp)  (-(numexp))
#define reciprocal(numexp) (1.0/(numexp))


#define mkNumFracOpsScalar(type)  \
BinaryOpScalarArray(arrayGeneralPlus,+,type) \
BinaryOpScalarArray(arrayGeneralMinus,-,type) \
BinaryOpScalarArray(arrayGeneralTimes,*,type) \
BinaryOpScalarArray(arrayGeneralDivide,/,type) \
UnaryOpScalarArray(arrayGeneralNegate,negate,type) \
UnaryOpScalarArray(arrayGeneralAbs,fabs,type) \
UnaryOpScalarArray(arrayGeneralReciprocal,reciprocal,type) \


#define BinaryOpScalarArray(name,binaryop,type) void  name##_##type(unsigned int length, type  *   left, \
int leftStride ,type  *   right,int rightStride, type *   result, \
int resultStride  ); \
 \
void name##_##type(unsigned int length, type  *   left,int leftStride ,type  *   right,int rightStride, type *   result, int resultStride  ){ \
    int ix = 0 ;  \
    if( (leftStride == 1) && (rightStride == 1) && (resultStride ==1)){ \
        for (ix = 0; ix < length ; ix ++){ \
            result[ix* resultStride]= (left[ix*leftStride] ) binaryop (right[ix*rightStride]  ) ; \
        } ;  \
    }else \
        if(1 <= length){  \
        for (ix = 0; ix < length ; ix ++){ \
            result[ix* resultStride]= (left[ix*leftStride] ) binaryop (right[ix*rightStride]  ) ; \
        }  \
    } \
}

#define UnaryOpScalarArray(name,op,type) void name##_##type(unsigned int length, type *  in,int inStride,\
type *  out, int outStride); \
 \
void name##_##type(unsigned int length, type *  in,int inStride, type *  out, \
int outStride){ \
    int ix = 0 ; \
    for(ix = 0 ; ix < length ; ix ++){ \
        out[ix*outStride] = op(in[ix*inStride]); \
    } \
}

////////////
/// Scalar versions
///////////

/*
for now I shall assume that everything only works on < 4gb / 32 sized ranges


also, will try to write things so that if any of the read (input) arrays
*/

/*
do a typedef for the complex types so that writing the macro stuff 
mixes well
*/

typedef float complex complex_float ;
typedef double complex complex_double ;



mkNumFracOpsScalar(complex_double);
mkNumFracOpsScalar(complex_float);
mkNumFracOpsScalar(double) ; 
mkNumFracOpsScalar(float) ; 
