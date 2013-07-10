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

note, for now using 32bit int types for array operation sizes and strides 
because you really shouldn't do more than 4gb of work in one sequential ffi call!

*/

/* CPP macro to generate declarations and body for strided scalar versions */

#define BinaryOpScalarArray(name,type,op) void name(int length, type  * restrict  left, \
int leftStride ,type  * restrict  right,int rightStride, type * restrict  result, \
int resultStride  ); \
 \
void name(int length, double  * restrict  left,int leftStride ,double  * restrict  right,int rightStride, double * restrict  result, int resultStride  ){ \
    int ix = 0 ;  \
    for (ix = 0; ix < length ; ix ++){ \
        result[ix* resultStride]= (left[ix*leftStride] ) + (right[ix*rightStride]  ) ; \
    } \
}

#define UnaryOpScalarArray(name,op,type) void name(int length, type * restrict in,int inStride,\
double * restrict out, int outStride); \
 \
void name(int length, double * restrict in,int inStride, double * restrict out, \
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

inline static  double negateDouble(double in);
inline  static double negateDouble(double in){
    return -in ; 
}

inline static  float negateFloat(float in);
inline  static float negateFloat(float in){
    return -in ; 
}
inline static  double reciprocalDouble(double in);
inline  static double reciprocalDouble(double in){
    return 1.0/in ; 
}

inline static  float reciprocalDouble(float in);
inline  static float reciprocalDouble(float in){
    return 1.0/in ; 
}




BinaryOpScalarArray(arrayGeneralPlusDoubleM,+,double)

BinaryOpScalarArray(arrayGeneralMinusDoubleM,-,double)

BinaryOpScalarArray(arrayGeneralTimesDoubleM,*,double)

BinaryOpScalarArray(arrayGeneralDivideDoubleM,/,double)

UnaryOpScalarArray(arrayGeneralNegateDoubleM,negateDouble,double)

UnaryOpScalarArray(arrayGeneralAbsDoubleM,abs,double)

UnaryOpScalarArray(arrayGeneralReciprocalDoubleM,reciprocalDouble,double)


