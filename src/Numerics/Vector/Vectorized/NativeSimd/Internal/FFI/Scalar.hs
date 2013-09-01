
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
--- Float operations 
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



{-
For any given type T, that in C would be a struct, the haskell FFI doesn't understand it
( ie currently types like Complex Double aren't understood, or their Storable analogues thereof)
This is a shame, because Complex Double and Complex Float have IEEE memory reps (citation?)

so they don't need to be part of the general struct story, though they 
wind up being modelable as structs

see GHC trac ticket http://ghc.haskell.org/trac/ghc/ticket/8061 #8061

This means I do need to have a singleton array for the result values 
for complex dot products! So the complex dot product operations take an 
additional pointer arge and return IO (), because of this (though 
operationally, they're just as pure as long as the singleton array is strictly localized)


-}


------
--- Complex Double  operations
------


foreign import ccall unsafe "VectorScalar.c dotproduct_complex_double" dotproduct_complex_double :: 
        Int32 -> Ptr (Complex Double) ->Int32 -> Ptr (Complex Double) -> Int32 ->Ptr (Complex Double) -> IO ()  


-----
-- binary operations, scalar versions
-----        

{-
for binary op opp, 
    "do O len inputl  strideL inputr strideR result strideResult"
acts like    result = inputL`op` inputR, lifted pointwise onto vectors
-}

foreign import ccall unsafe "VectorScalar.c arrayPlus_complex_double" arrayPlus_complex_double :: 
        Int32 -> Ptr (Complex Double) -> Int32 ->Ptr (Complex Double)-> Int32 -> Ptr (Complex Double) ->Int32 -> IO ()  

foreign import ccall unsafe "VectorScalar.c arrayTimes_complex_double" arrayTimes_complex_double :: 
        Int32 -> Ptr (Complex Double) -> Int32 ->Ptr (Complex Double)-> Int32 -> Ptr (Complex Double) ->Int32 -> IO ()  

foreign import ccall unsafe "VectorScalar.c arrayMinus_complex_double" arrayMinus_complex_double :: 
        Int32 -> Ptr (Complex Double) -> Int32 ->Ptr (Complex Double)-> Int32 -> Ptr (Complex Double) ->Int32 -> IO ()  

foreign import ccall unsafe "VectorScalar.c arrayDivide_complex_double" arrayDivide_complex_double :: 
        Int32 -> Ptr (Complex Double) -> Int32 ->Ptr (Complex Double)-> Int32 -> Ptr (Complex Double) ->Int32 -> IO ()  

----------
-- unary  operations    
------

{-
for unary op opp,
    "do opp len a a_stride b b_stride"
acts like
    b = opp a 

-}

foreign import ccall unsafe "VectorScalar.c arraySqrt_complex_double" arraySquareRoot_complex_double :: 
        Int32-> Ptr (Complex Double)-> Int32-> Ptr (Complex Double) -> Int32->  IO ()

foreign import ccall unsafe "VectorScalar.c arrayNegate_complex_double" arrayNegate_complex_double :: 
        Int32 -> Ptr (Complex Double)-> Int32-> Ptr (Complex Double) -> Int32-> IO ()        

foreign import ccall unsafe "VectorScalar.c arrayReciprocal_complex_double" arrayReciprocal_complex_double :: 
        Int32 -> Ptr (Complex Double)-> Int32-> Ptr (Complex Double) -> Int32->  IO ()  

---------
--- Complex Float operations 
---------


-- dot product acts like a pure function!


foreign import ccall unsafe "VectorScalar.c dotproduct_complex_float" dotproduct_complex_float :: 
        Int32 -> Ptr (Complex Float) -> Int32-> Ptr (Complex Float) -> Int32->  Ptr (Complex Float)-> IO ()


-----
-- binary operations, scalar versions
-----        

{-
for binary op opp, 
    "do O inputl inputr result"
acts like    result = inputL`op` inputR, lifted pointwise onto vectors
-}

foreign import ccall unsafe "VectorScalar.c arrayPlus_complex_float" arrayPlus_complex_float :: 
        Int32 -> Ptr (Complex Float) -> Int32-> Ptr (Complex Float)-> Int32-> Ptr (Complex Float) -> Int32-> IO ()

foreign import ccall unsafe "VectorScalar.c arrayTimes_complex_float" arrayTimes_complex_float :: 
        Int32 -> Ptr (Complex Float) -> Int32-> Ptr (Complex Float)-> Int32-> Ptr (Complex Float) -> Int32-> IO ()

foreign import ccall unsafe "VectorScalar.c arrayMinus_complex_float" arrayMinus_complex_float :: 
        Int32 -> Ptr (Complex Float) -> Int32-> Ptr (Complex Float)-> Int32-> Ptr (Complex Float) -> Int32-> IO ()

foreign import ccall unsafe "VectorScalar.c arrayDivide_complex_float" arrayDivide_complex_float :: 
        Int32 -> Ptr (Complex Float) -> Int32-> Ptr (Complex Float)-> Int32-> Ptr (Complex Float) -> Int32-> IO ()

----------
-- unary  operations    
------

{-
for unary op opp,
    "do opp a b"
acts like
    b = opp a 

-}

foreign import ccall unsafe "VectorScalar.c arraySqrt_complex_float" arraySquareRoot_complex_float :: 
        Int32 -> Ptr (Complex Float) -> Int32-> Ptr (Complex Float)-> Int32-> IO ()

foreign import ccall unsafe "VectorScalar.c arrayNegate_complex_float" arrayNegate_complex_float :: 
        Int32 -> Ptr (Complex Float) -> Int32-> Ptr (Complex Float)-> Int32-> IO ()

foreign import ccall unsafe "VectorScalar.c arrayReciprocal_complex_float" arrayReciprocal_complex_float :: 
        Int32 -> Ptr (Complex Float) -> Int32-> Ptr (Complex Float)-> Int32-> IO ()

