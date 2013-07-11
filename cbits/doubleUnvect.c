#include "simd.h"
#include <tgmath.h>


/*
we have 3 versions: 
avx2 (untested)
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


#define BinaryOpScalarArray(name,binaryop,type) void  name##type(int length, type  * restrict  left, \
int leftStride ,type  * restrict  right,int rightStride, type * restrict  result, \
int resultStride  ); \
 \
void name##type(int length, double  * restrict  left,int leftStride ,double  * restrict  right,int rightStride, double * restrict  result, int resultStride  ){ \
    int ix = 0 ;  \
    for (ix = 0; ix < length ; ix ++){ \
        result[ix* resultStride]= (left[ix*leftStride] ) binaryop (right[ix*rightStride]  ) ; \
    } \
}

#define UnaryOpScalarArray(name,op,type) void name##type(int length, type * restrict in,int inStride,\
double * restrict out, int outStride); \
 \
void name##type(int length, double * restrict in,int inStride, double * restrict out, \
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

// inline static  double negateDouble(double in);
// inline  static double negateDouble(double in){
//     return -in ; 
// }

// inline static  float negateFloat(float in);
// inline  static float negateFloat(float in){
//     return -in ; 
// }
// inline static  double reciprocalDouble(double in);
// inline  static double reciprocalDouble(double in){
//     return 1.0/in ; 
// }

// inline static  float reciprocalDouble(float in);
// inline  static float reciprocalDouble(float in){
//     return 1.0/in ; 
// }



mkNumFracOpsScalar(double) ; 
