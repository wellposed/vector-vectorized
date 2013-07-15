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

UnaryOpScalarArray(arrayGeneralReciprocal,reciprocal,type) \
 \ 
UnaryOpScalarArray(arrayGeneralSqrt,sqrt,type) 
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

double complex       cpow(double complex, double complex);

double complex       cproj(double complex);

double complex       csin(double complex);

double complex       csinh(double complex);

double complex       csqrt(double complex);

double complex       ctan(double complex);

double complex       ctanh(double complex);

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



#define BinaryOpScalarArray(name,binaryop,type) void  name##_##type(uint32_t length, type  *   left, \
int32_t leftStride ,type  *   right,int32_t rightStride, type *   result, \
int resultStride  ); \
 \
void name##_##type(uint32_t length, type  *   left,int32_t leftStride ,type  *   right,int32_t rightStride, type *   result, int32_t resultStride  ){ \
    int ix = 0 ;  \
    for (ix = 0; ix < length ; ix ++){ \
        result[ix* resultStride]= (left[ix*leftStride] ) binaryop (right[ix*rightStride]  ) ; \
        }  \
}

#define UnaryOpScalarArray(name,op,type) void name##_##type(uint32_t length, type *  in,int32_t inStride,\
type *  out, int32_t outStride); \
 \
void name##_##type(uint32_t length, type *  in,int32_t inStride, type *  out, \
int32_t outStride){ \
    int ix = 0 ; \
    for(ix = 0 ; ix < length ; ix ++){ \
        out[ix*outStride] = op(in[ix*inStride]); \
    } \
}

#define DotProductScalarArray(name,binaryop, type,init) type name##_##type(uint32_t length, type  *   left, \
int32_t leftStride ,type  *   right,int32_t rightStride); \
 \
 type name##_##type(uint32_t length, type  *   left, int32_t leftStride ,type  *   right,int32_t rightStride){ \
    type res = init ; \
    uint32_t ix = 0; \
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

#define realtimes(x,y) (x * y ) 

#define complextimes(x,y)  (x * conj(y))

typedef float complex complex_float ;
typedef double complex complex_double ;

DotProductScalarArray(arrayGeneralDotProduct,realtimes,double,0.0) ;
DotProductScalarArray(arrayGeneralDotProduct,realtimes,float,0.0) ;
DotProductScalarArray(arrayGeneralDotProduct,complextimes,complex_double,0.0 + I*0.0) ;
DotProductScalarArray(arrayGeneralDotProduct,complextimes,complex_float,0.0 + I*0.0) ;

mkNumFracOpsScalar(complex_double);
mkNumFracOpsScalar(complex_float);
mkNumFracOpsScalar(double) ; 
mkNumFracOpsScalar(float) ; 
