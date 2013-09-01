 
module Numerics.Vector.Vectorized.NativeSimd.FFI.Simd where

import Foreign hiding (unsafePerformIO)
import Foreign.C.Types
--import Unsafe.Coerce
import Prelude hiding (replicate)
--import Data.Storable
--import System.IO.Unsafe
--import Data.Vector.Storable.Mutable  
import GHC.Ptr (castPtr)


{-

all the SIMD codes have the following convention:
name##_##SIMD##_##type
where type = Double or Float

for binary operations: name is one of 
    arrayPlus,arrayTimes,arrayMinus,arrayPlus,arrayDivide 
for unary operations: name is one of
    arrayNegate, arrayReciprocal, arraySqrt (SquareRoot)



note: could use TH to generate the code, but theres
some nontrivial TH changes between 7.6 and 7.8 that I wish to ignore for now
and also CPP would be fragile
-}

--foreign import ccall unsafe "file.c funname" haskellname :: ffihaskelltype


------
--- Double Simd operations
------


foreign import ccall unsafe "VectorSIMD.c dotproduct_SIMD_double" dotproduct_SIMD_double :: 
        Int32 -> Ptr Double -> Ptr Double -> Double 


-----
-- binary operations, simd versions
-----        

{-
for binary op opp, 
    "do O inputl inputr result"
acts like    result = inputL`op` inputR, lifted pointwise onto vectors
-}

foreign import ccall unsafe "VectorSIMD.c arrayPlus_SIMD_double" arrayPlus_SIMD_double :: 
        Int32 -> Ptr Double -> Ptr Double-> Ptr Double -> IO ()

foreign import ccall unsafe "VectorSIMD.c arrayTimes_SIMD_double" arrayTimes_SIMD_double :: 
        Int32 -> Ptr Double -> Ptr Double-> Ptr Double -> IO ()

foreign import ccall unsafe "VectorSIMD.c arrayMinus_SIMD_double" arrayMinus_SIMD_double :: 
        Int32 -> Ptr Double -> Ptr Double-> Ptr Double -> IO ()        

foreign import ccall unsafe "VectorSIMD.c arrayDivide_SIMD_double" arrayDivide_SIMD_double :: 
        Int32 -> Ptr Double -> Ptr Double-> Ptr Double -> IO ()  

----------
-- unary  operations    
------

{-
for unary op opp,
    "do opp a b"
acts like
    b = opp a 

-}

foreign import ccall unsafe "VectorSIMD.c arraySqrt_SIMD_double" arraySquareRoot_SIMD_double :: 
        Int32-> Ptr Double-> Ptr Double -> IO ()

foreign import ccall unsafe "VectorSIMD.c arrayNegate_SIMD_double" arrayNegate_SIMD_double :: 
        Int32 -> Ptr Double-> Ptr Double -> IO ()        

foreign import ccall unsafe "VectorSIMD.c arrayReciprocal_SIMD_double" arrayReciprocal_SIMD_double :: 
        Int32 -> Ptr Double-> Ptr Double -> IO ()  

---------
--- Float Simd operations 
---------


-- dot product acts like a pure function!


foreign import ccall unsafe "VectorSIMD.c dotproduct_SIMD_float" dotproduct_SIMD_float :: 
        Int32 -> Ptr Float -> Ptr Float -> Float 


-----
-- binary operations, simd versions
-----        

{-
for binary op opp, 
    "do O inputl inputr result"
acts like    result = inputL`op` inputR, lifted pointwise onto vectors
-}

foreign import ccall unsafe "VectorSIMD.c arrayPlus_SIMD_float" arrayPlus_SIMD_float :: 
        Int32 -> Ptr Float -> Ptr Float-> Ptr Float -> IO ()

foreign import ccall unsafe "VectorSIMD.c arrayTimes_SIMD_float" arrayTimes_SIMD_float :: 
        Int32 -> Ptr Float -> Ptr Float-> Ptr Float -> IO ()

foreign import ccall unsafe "VectorSIMD.c arrayMinus_SIMD_float" arrayMinus_SIMD_float :: 
        Int32 -> Ptr Float -> Ptr Float-> Ptr Float -> IO ()

foreign import ccall unsafe "VectorSIMD.c arrayDivide_SIMD_float" arrayDivide_SIMD_float :: 
        Int32 -> Ptr Float -> Ptr Float-> Ptr Float -> IO ()

----------
-- unary  operations    
------

{-
for unary op opp,
    "do opp a b"
acts like
    b = opp a 

-}

foreign import ccall unsafe "VectorSIMD.c arraySqrt_SIMD_float" arraySquareRoot_SIMD_float :: 
        Int32 -> Ptr Float -> Ptr Float-> IO ()

foreign import ccall unsafe "VectorSIMD.c arrayNegate_SIMD_float" arrayNegate_SIMD_float :: 
        Int32 -> Ptr Float -> Ptr Float-> IO ()

foreign import ccall unsafe "VectorSIMD.c arrayReciprocal_SIMD_float" arrayReciprocal_SIMD_float :: 
        Int32 -> Ptr Float -> Ptr Float-> IO ()
