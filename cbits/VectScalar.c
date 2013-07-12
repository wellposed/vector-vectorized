#include "simd.h"
#include <tgmath.h>
#include <complex.h>
#include <stdint.h>
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


Assumption: 
stride is never zero

stride

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
UnaryOpScalarArray(arrayGeneralReciprocal,reciprocal,type)  \


#define AVX_Modulus (4 * sizeof(double))  
 // 4 * 64bits = 256
#define allEqualStride(val,a,b,c) (((val )== ((long int)a)) && ((val)== ((long int)b)) && ((val)== ((long int)c)) ) 

#define isAligned(x)  !((uintptr_t)x & 0xFF)

#define BinaryOpScalarArray(name,binaryop,type) void  name##_##type(int length, type  * restrict  left, \
int leftStride ,type  * restrict  right,int rightStride, type * restrict  result, \
int resultStride  ); \
 \
void name##_##type(int length, type  * restrict  left,int leftStride ,type  * restrict  right,int rightStride, type * restrict  result, int resultStride  ){ \
    int ix = 0 ;  \
    if(allEqual(1,leftStride,rightStride,resultStride) && isAligned(left) && isAligned(right) && isAligned(result)  ){ \
        int quotient =  length / 8 ; \
        int rem = length % 8 ; \
        int octMax =  8 * rem ; \
                \
        for(; ix < octMax ;){ \
            int counter = 0  ; \
            for(; counter < 8 ; counter ++ ){ \
                result[ix* resultStride]= (left[ix*leftStride] ) binaryop (right[ix*rightStride]  ) ; \
                ix++  ; \
            }} \
            for(; ix < length ; ix ++){ \
                result[ix* resultStride]= (left[ix*leftStride] ) binaryop (right[ix*rightStride]  ) ; \
            } \
        } \ 
    else{ \
        for(; ix < length ; ix ++){ \
            result[ix* resultStride]= (left[ix*leftStride] ) binaryop (right[ix*rightStride]  ) ; \
            } \
    } \
}

#define UnaryOpScalarArray(name,op,type) void name##_##type(int length, type * restrict in,int inStride,\
type * restrict out, int outStride); \
 \
void name##_##type(int length, type * restrict in,int inStride, type * restrict out, \
int outStride){ \
    int ix = 0 ; \
    for(; ix < length ; ix ++){ \
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
