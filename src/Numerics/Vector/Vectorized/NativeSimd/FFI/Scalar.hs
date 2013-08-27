
module Numerics.Vector.Vectorized.NativeSimd.FFI.Scalar where

import Foreign hiding (unsafePerformIO)
import Foreign.C.Types
--import Unsafe.Coerce
import Prelude hiding (replicate)
--import Data.Storable
--import System.IO.Unsafe
--import Data.Vector.Storable.Mutable  
import GHC.Ptr (castPtr)


------
--- Double  operations
------


foreign import ccall unsafe "VectorScalar.c dotproduct_double" dotproduct_double :: 
        Int32 -> Ptr Double ->Int32 -> Ptr Double -> Int32 ->Double 


-----
-- binary operations, scalar versions
-----        

{-
for binary op opp, 
    "do O len inputl  strideL inputr strideR result strideResult"
acts like    result = inputL`op` inputR, lifted pointwise onto vectors
-}

foreign import ccall unsafe "VectorScalar.c arrayPlus_double" arrayPlus_double :: 
        Int32 -> Ptr Double -> Int32 ->Ptr Double-> Int32 -> Ptr Double ->Int32 -> IO ()  

foreign import ccall unsafe "VectorScalar.c arrayTimes_double" arrayTimes_double :: 
        Int32 -> Ptr Double -> Int32 ->Ptr Double-> Int32 -> Ptr Double ->Int32 -> IO ()  

foreign import ccall unsafe "VectorScalar.c arrayMinus_double" arrayMinus_double :: 
        Int32 -> Ptr Double -> Int32 ->Ptr Double-> Int32 -> Ptr Double ->Int32 -> IO ()  

foreign import ccall unsafe "VectorScalar.c arrayDivide_double" arrayDivide_double :: 
        Int32 -> Ptr Double -> Int32 ->Ptr Double-> Int32 -> Ptr Double ->Int32 -> IO ()  

----------
-- unary  operations    
------

{-
for unary op opp,
    "do opp len a a_stride b b_stride"
acts like
    b = opp a 

-}

foreign import ccall unsafe "VectorScalar.c arraySqrt_double" arraySquareRoot_double :: 
        Int32-> Ptr Double-> Int32-> Ptr Double -> Int32->  IO ()

foreign import ccall unsafe "VectorScalar.c arrayNegate_double" arrayNegate_double :: 
        Int32 -> Ptr Double-> Int32-> Ptr Double -> Int32-> IO ()        

foreign import ccall unsafe "VectorScalar.c arrayReciprocal_double" arrayReciprocal_double :: 
        Int32 -> Ptr Double-> Int32-> Ptr Double -> Int32->  IO ()  

---------
--- Float Simd operations 
---------


-- dot product acts like a pure function!


foreign import ccall unsafe "VectorScalar.c dotproduct_float" dotproduct_float :: 
        Int32 -> Ptr Float -> Int32-> Ptr Float -> Int32->  Float 


-----
-- binary operations, scalar versions
-----        

{-
for binary op opp, 
    "do O inputl inputr result"
acts like    result = inputL`op` inputR, lifted pointwise onto vectors
-}

foreign import ccall unsafe "VectorScalar.c arrayPlus_float" arrayPlus_float :: 
        Int32 -> Ptr Float -> Int32-> Ptr Float-> Int32-> Ptr Float -> Int32-> IO ()

foreign import ccall unsafe "VectorScalar.c arrayTimes_float" arrayTimes_float :: 
        Int32 -> Ptr Float -> Int32-> Ptr Float-> Int32-> Ptr Float -> Int32-> IO ()

foreign import ccall unsafe "VectorScalar.c arrayMinus_float" arrayMinus_float :: 
        Int32 -> Ptr Float -> Int32-> Ptr Float-> Int32-> Ptr Float -> Int32-> IO ()

foreign import ccall unsafe "VectorScalar.c arrayDivide_float" arrayDivide_float :: 
        Int32 -> Ptr Float -> Int32-> Ptr Float-> Int32-> Ptr Float -> Int32-> IO ()

----------
-- unary  operations    
------

{-
for unary op opp,
    "do opp a b"
acts like
    b = opp a 

-}

foreign import ccall unsafe "VectorScalar.c arraySqrt_float" arraySquareRoot_float :: 
        Int32 -> Ptr Float -> Int32-> Ptr Float-> Int32-> IO ()

foreign import ccall unsafe "VectorScalar.c arrayNegate_float" arrayNegate_float :: 
        Int32 -> Ptr Float -> Int32-> Ptr Float-> Int32-> IO ()

foreign import ccall unsafe "VectorScalar.c arrayReciprocal_float" arrayReciprocal_float :: 
        Int32 -> Ptr Float -> Int32-> Ptr Float-> Int32-> IO ()
