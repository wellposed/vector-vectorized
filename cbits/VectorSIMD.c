#include "simd.h"
#include <tgmath.h>
#include <stdint.h>


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


*/

#define BinaryOpSimdArray(name,binaryop,type,elem16,elem32) void  name##_##SIMD##_##type(uint32_t length, type  *   left, \
    type  *   right,int32_t rightStride, type *   result); \
 \
void  name##_##SIMD##_##type(uint32_t length, type  *   left,type  *   right, type *   result){ \
    int ix = 0 ;  \
    for (ix = 0; ix < length ; ix+=elem32){ \
#ifdef         
        result[ix* resultStride]= (left[ix*leftStride] ) binaryop (right[ix*rightStride]  ) ; \
        }  \
}







