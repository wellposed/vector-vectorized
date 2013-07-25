// #include "simd.h"
#include <tgmath.h>
#include <stdint.h>

#include <immintrin.h>
#include <memory.h>
#include <stddef.h>

/*
for now i'm only going to do these operations for floats
and doubles

will add complex operations at some later point, but not for now

this code only provides SIMD on SSE3 through AVX supported platforms
for doubles and floats

+,  - , * , / ,  sqrt, 
abs  (mask with 9223372036854775807 for doubles, )

plus dotproduct (NB, not worth using dot product instruction, i think)


lets for now assume we're only doing simd over 32byte aligned 32byte chunks

is 4 doubles, or 8 floats,

this will be 1 256bit vector or 2 128bit vectors

NOTE: for now I will probably only run simd on vectors that occupy at 
least 64 bytes, which guarantees at least one 32byte aligned 32byte region

note since pointers are in multiples of bytes, 
we want to ask about the byte alignment of the start of the pointer

eg for a binary op
we have basePtr's inL, inR, out, we assume / require 
for simd that   inL % 32 == inR % 32 == out %32

note we need that AND 
(inl %4 == 0) for floats 
(inl %8 == 0 ) for doubles 

lets assume those invariants  are enforced by client code
(considering i'm the only client code, yes)

*/


/*

dot product will be a loop
where i load, do pair wise, NB, worth using FMA  YMM ops when available

*/

/*
i'm writing it as 32byte aligned, need to know how many elements
are in a 16byte and 32byte 

__AVX__ , __SSE3__ and __FMA__ are the main ways for code, at least for now,
I will try to have the code perform favorably on both haswell and sandy-bridge 
micro architectures

basically every cycle I can try to have an independent 
vector mult/blend/fma, vector add/shuffle/fma, shuffle/blend,   load/storeaddress ,load/storeaddress 

(I *should* unroll my vectorized loops to exploit that, but for now I wont)

 elem32 =  2* elem16


__m256d means v4df (double precsion 64bit floats)
__m256  means v8sf (single precision 32 bit floats)

_mm256_load_pd
_mm256_store_pd for the avx versions

*/



/*
from the avx intrinsics header 
 typedef double __v4df __attribute__ ((__vector_size__ (32)));
 typedef float __v8sf __attribute__ ((__vector_size__ (32)));
 typedef long long __v4di __attribute__ ((__vector_size__ (32)));
 typedef int __v8si __attribute__ ((__vector_size__ (32)));
 typedef short __v16hi __attribute__ ((__vector_size__ (32)));
 typedef char __v32qi __attribute__ ((__vector_size__ (32)));
 
 typedef float __m256 __attribute__ ((__vector_size__ (32)));
 typedef double __m256d __attribute__((__vector_size__(32)));
 typedef long long __m256i __attribute__((__vector_size__(32)));

*/


#if   defined(__AVX__)      
#define BinaryOpSimdArray(name,binaryop,type,simdtypeAVX,simdloadAVX,simdstoreAVX,simdtypeSSE3,simdloadSSE3,simdstoreSSE3,sizeAVX, sizeSSE3) \ 
void  name##_##SIMD##_##type(uint32_t length, type  *   left,  type   *   right, type *   result); \
 \
void  name##_##SIMD##_##type(uint32_t length, type  *   left,type  *   right, type *   result){ \
    uint32_t ix = 0 ;  \
    for(ix = 0; ix < length ; ix+= sizeAVX){ \
        simdtypeAVX leftV = simdloadAVX(left + ix); \
        simdtypeAVX rightV =simdloadAVX(right+ix ) ; \
        simdtypeAVX resV =  leftV binaryop rightV ; \
        simdstoreAVX(result+ix , resV); \
        }  }
#elif defined(__SSE3__)   
#define BinaryOpSimdArray(name,binaryop,type,simdtypeAVX,simdloadAVX,simdstoreAVX,simdtypeSSE3,simdloadSSE3,simdstoreSSE3,sizeAVX, sizeSSE3) \ 
void  name##_##SIMD##_##type(uint32_t length, type  *   left, type   *   right,type *   result); \
 \
void  name##_##SIMD##_##type(uint32_t length, type  *   left,type  *   right, type *   result){ \        
  //  for pre sandybridge \
    uint32_t ix = 0 ;  \
    for(ix = 0; ix < length ; ix+=sizeAVX){ \
        simdtypeSSE3 leftV1 = simdloadSSE3(left + ix); \
        simdtypeSSE3 rightV1 =simdloadSSE3(right+ix ) ; \
        simdtypeSSE3 resV1 =  leftV1 binaryop rightV1 ; \
        simdstoreSSE3(result+ix , resV1); \
        simdtypeSSE3 leftV2 = simdloadSSE3(left + ix+sizeSSE3); \
        simdtypeSSE3 rightV2 =simdloadSSE3(right+ix+sizeSSE3 ) ; \
        simdtypeSSE3 resV2 =  leftV2 binaryop rightV2 ; \
        simdstoreSSE3(result+ix+sizeSSE3 , resV2); \
        } }
#else
#define BinaryOpSimdArray(name,binaryop,type,simdtypeAVX,simdloadAVX,simdstoreAVX,simdtypeSSE3,simdloadSSE3,simdstoreSSE3,sizeAVX, sizeSSE3) \ 
void  name##_##SIMD##_##type(uint32_t length, type  *   left,   type   *   right,type *   result); \
 \
void  name##_##SIMD##_##type(uint32_t length, type  *   left,type  *   right, type *   result){ \  
    //  scalar, sorry :) , for Old old intel x86, and for other architectures \
    uint32_t ix = 0 ; \
    for(ix = 0 ; ix < length ; ix+){ \
        result[ix] =(left[ix] ) binaryop (right[ix]  ) ;  \
    } }
#endif 



#ifdef  __AVX__    
#define UnaryOpSimdArray(name,unaryop,type,simdtypeAVX,simdloadAVX,simdstoreAVX,simdtypeSSE3,simdloadSSE3,simdstoreSSE3,sizeAVX, sizeSSE3) \ 
void  name##_##SIMD##_##type(uint32_t length, type   *   in,type *   result); \
 \
void  name##_##SIMD##_##type(uint32_t length, type  *   in, type *   result){ \
    uint32_t ix = 0 ;  \
    for(ix = 0; ix < length ; ix+= sizeAVX){ \
        simdtypeAVX inV = simdloadAVX(in + ix) ; \
        simdtypeAVX resV =   unaryop(inV, simdtypeAVX) ; \
        simdstoreAVX(result+ix , resV); \
        }  } ; 
#elif defined(__SSE3__)   
#define UnaryOpSimdArray(name,unaryop,type,simdtypeAVX,simdloadAVX,simdstoreAVX,simdtypeSSE3,simdloadSSE3,simdstoreSSE3,sizeAVX, sizeSSE3) \ 
void  name##_##SIMD##_##type(uint32_t length, type   *   in,type *   result); \
 \
void  name##_##SIMD##_##type(uint32_t length, type  *   in, type *   result){ \        
  //  for pre sandybridge \
    uint32_t ix = 0 ;  \
    for(ix = 0; ix < length ; ix+=sizeAVX){ \
        simdtypeSSE3 inV1 = simdloadSSE3(in + ix); \
        simdtypeSSE3 resV1 = unaryop(inV1, simdtypeSSE3) ; \
        simdstoreSSE3(result+ix , resV1); \
        simdtypeSSE3 inV2 = simdloadSSE3(in + ix+sizeSSE3); \
        simdtypeSSE3 resV2 =  unaryop(inV2, simdtypeSSE3) ; \
        simdstoreSSE3(result+ix+2 , resV2); \
        }  } ; 
#else  
#define UnaryOpSimdArray(name,unaryop,type,simdtypeAVX,simdloadAVX,simdstoreAVX,simdtypeSSE3,simdloadSSE3,simdstoreSSE3,sizeAVX, sizeSSE3) \ 
void  name##_##SIMD##_##type(uint32_t length, type   *   in,type *   result); \
 \
void  name##_##SIMD##_##type(uint32_t length, type  *   in, type *   result){ \
    //  scalar, sorry :) , for Old old intel x86, and for other architectures \
    uint32_t ix = 0 ; \
    for(ix = 0 ; ix < length ; ix ++){ \
        type inVal = (in[ix]) ; \
        result[ix] = unaryop( inVal, type ) ;  \
    } } ; 
#endif 


/*
note that 1 / v when v is a vector is casted to 1 being a replicated vector of 1s
(in both clang and gcc, wooo)
*/



#define negate(numexp,castType)  (-(numexp) )   

#define reciprocal(numexp,castType) ((castType)(1.0)/ (numexp)) 


#define mkNumFracOpsSIMD(type,simdtypeAVX,simdloadAVX,simdstoreAVX,simdtypeSSE3,simdloadSSE3,simdstoreSSE3,sizeAVX, sizeSSE3)  \
BinaryOpSimdArray(arrayPlus,+ ,type,simdtypeAVX,simdloadAVX,simdstoreAVX,simdtypeSSE3,simdloadSSE3,simdstoreSSE3,sizeAVX, sizeSSE3) ; \
BinaryOpSimdArray(arrayMinus,-,type,simdtypeAVX,simdloadAVX,simdstoreAVX,simdtypeSSE3,simdloadSSE3,simdstoreSSE3,sizeAVX, sizeSSE3) ; \
BinaryOpSimdArray(arrayTimes,*,type,simdtypeAVX,simdloadAVX,simdstoreAVX,simdtypeSSE3,simdloadSSE3,simdstoreSSE3,sizeAVX, sizeSSE3)  ; \
BinaryOpSimdArray(arrayDivide,/,type,simdtypeAVX,simdloadAVX,simdstoreAVX,simdtypeSSE3,simdloadSSE3,simdstoreSSE3,sizeAVX, sizeSSE3) ; \
UnaryOpSimdArray(arrayNegate,negate,type,simdtypeAVX,simdloadAVX,simdstoreAVX,simdtypeSSE3,simdloadSSE3,simdstoreSSE3,sizeAVX, sizeSSE3) ; \
UnaryOpSimdArray(arrayReciprocal ,reciprocal,type,simdtypeAVX,simdloadAVX,simdstoreAVX,simdtypeSSE3,simdloadSSE3,simdstoreSSE3,sizeAVX, sizeSSE3) ;
 

mkNumFracOpsSIMD(double,__m256d,_mm256_load_pd,_mm256_store_pd,__m128d,_mm128_load_pd,_mm128_store_pd,4,2);


// __m256d broadcastAVX(double val){
//     return (__builtin_shufflevector({val},{val},0,0,0,0));

// }

