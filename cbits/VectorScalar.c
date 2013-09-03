#include "simd.h"
#include <tgmath.h>
// #include <complex.h>



/*
note: using 32bit signed 
*/

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

// UnaryOpScalarArray(arrayGeneralLog)
// UnaryOpScalarArray(arrayGeneralAbs,fabs,type) \ // this is wrong for complex numbers

/*



double               carg(double complex);

double               cimag(double complex);




double               creal(double complex);

double complex       cacos(double complex);

double complex       cacosh(double complex);


double complex       casin(double complex);

double complex       casinh(double complex);


double complex       catan(double complex);

double complex       catanh(double complex);


double complex       ccos(double complex);

double complex       ccosh(double complex);

double complex       cexp(double complex);

double complex       clog(double complex);

double complex       conj(double complex);



double complex       cproj(double complex);

double complex       csin(double complex);

double complex       csinh(double complex);

double complex       csqrt(double complex);

double complex       ctan(double complex);

double complex       ctanh(double complex);


double complex       cpow(double complex, double complex);
*/

/*



atan2()
cbrt()
ceil()
copysign()
erf()
erfc()
exp2()
expm1()
fdim()
floor()
 


fma()
fmax()
fmin()
fmod()
frexp()
hypot()
ilogb()
ldexp()
lgamma()
llrint()
 


llround()
log10()
log1p()
log2()
logb()
lrint()
lround()
nearbyint()
nextafter()
nexttoward()
 


remainder()
remquo()
rint()
round()
scalbn()
scalbln()
tgamma()
trunc()
 */



#define BinaryOpScalarArray(name,binaryop,type) void  name##_##type(int32_t length, type  *   left, \
int32_t leftStride ,type  *   right,int32_t rightStride, type *   result, \
int resultStride  ); \
 \
void name##_##type(int32_t length, type  *   left,int32_t leftStride ,type  *   right,int32_t rightStride, type *   result, int32_t resultStride  ){ \
    int32_t ix = 0 ;  \
    for (ix = 0; ix < length ; ix ++){ \
        result[ix* resultStride]= (left[ix*leftStride] ) binaryop (right[ix*rightStride]  ) ; \
        }  \
}

#define UnaryOpScalarArray(name,op,type) void name##_##type(int32_t length, type *  in,int32_t inStride,\
type *  out, int32_t outStride); \
 \
void name##_##type(int32_t length, type *  in,int32_t inStride, type *  out, \
int32_t outStride){ \
    int32_t ix = 0 ; \
    for(ix = 0 ; ix < length ; ix ++){ \
        /*  WARNING / NOTE: in the complex float case, for reciprocal unary op, it does the divide using doubles,  then casts back to complex float, so may have  unexpectedly nice precision
        */ \
        out[ix*outStride] = (type) op(in[ix*inStride]); \
    } \
}

#define DotProductScalarArray(name,binaryop, type,init) type name##_##type(int32_t length, type  *   left, \
int32_t leftStride ,type  *   right,int32_t rightStride); \
 \
 type name##_##type(int32_t length, type  *   left, int32_t leftStride ,type  *   right,int32_t rightStride){ \
    type res = init ; \
    int32_t ix = 0; \
    for(ix=0 ; ix < length ; ix++ ){ \
        res +=  binaryop((left[ix*leftStride] ),(right[ix*rightStride]  )) ; \
        } \
    return res ; \
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


#define negate(numexp)  (-(numexp))
#define reciprocal(numexp) (1.0/(numexp))


#define mkNumFracOpsScalar(type)  \
BinaryOpScalarArray(arrayPlus,+,type) \
BinaryOpScalarArray(arrayMinus,-,type) \
BinaryOpScalarArray(arrayTimes,*,type) \
BinaryOpScalarArray(arrayDivide,/,type) \
UnaryOpScalarArray(arrayNegate,negate,type) \
UnaryOpScalarArray(arrayReciprocal,reciprocal,type) \
UnaryOpScalarArray(arraySqrt,sqrt,type) 


#define realtimes(x,y) (x * y ) 

#define complextimes(x,y)  (x * conj(y))

typedef float complex complex_float ;
typedef double complex complex_double ;

DotProductScalarArray(arrayDotProduct,realtimes,double,0.0) 
DotProductScalarArray(arrayDotProduct,realtimes,float,0.0) 

// name mangling the complex dot products because I need to 
//  wrap them to take a singlen array pointer where I write the result
// because haskell currently doesn't have an FFI story for structs and complex numbers
DotProductScalarArray(arrayDotProduct_internal,complextimes,complex_double,0.0 + I*0.0) 
DotProductScalarArray(arrayDotProduct_internal,complextimes,complex_float,0.0f + I*0.0f) 


/// NOTE: complex valued dot products have a different type than the real float dot products
/// this is important to remember!

void arrayDotProduct_complex_double(int32_t length, complex_double  *   left, int32_t leftStride ,complex_double  *   right,int32_t rightStride, complex_double * resultSingleton);

void arrayDotProduct_complex_double(int32_t length, complex_double  *   left, int32_t leftStride ,complex_double  *   right,int32_t rightStride, complex_double * resultSingleton){
    complex_double result = arrayDotProduct_internal_complex_double(length,left,leftStride,right,rightStride);
    resultSingleton[0] = result ; 
}


void arrayDotProduct_complex_float(int32_t length, complex_float  *   left, int32_t leftStride ,complex_float  *   right,int32_t rightStride, complex_float * resultSingleton);

void arrayDotProduct_complex_float(int32_t length, complex_float  *   left, int32_t leftStride ,complex_float  *   right,int32_t rightStride, complex_float * resultSingleton){
    complex_float result = arrayDotProduct_internal_complex_float(length,left,leftStride,right,rightStride);
    resultSingleton[0] = result ; 
}




UnaryOpScalarArray(arrayGeneralAbs,fabs,float)
UnaryOpScalarArray(arrayGeneralAbs,fabs,double)

mkNumFracOpsScalar(complex_double)
mkNumFracOpsScalar(complex_float)
mkNumFracOpsScalar(double) 
mkNumFracOpsScalar(float)  

