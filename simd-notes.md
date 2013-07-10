#simd notes


# XMM notes
most the the basic functionality I want is available as of SSE3, which came out
in 2005-2006. Assume any vectorization has at least that.

## SSSE3 
Has nothing handy for me

## SSE4.1
has Dot product and streaming loads, both handy, perhaps

## SSE4.2
nothing handy here

## AVX1 
in XMM land just lifts all the SSEn stuff to AVX nondestructive input land

# YMM Notes

## AVX1 
has pretty much everything i need, except for the 2 lane shuffle as a single op
vpermilpd  , vperm2f128  are needed to do the 2 lane shuffle, no single op

## AVX2
has Gather (strided reads) and  and 2 lane shuffle as a single op

# other questions:
for now lest ignore non temporal loads and stores.








