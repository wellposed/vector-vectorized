 
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
    arrayNegate, arrayReciprocal


-}

--foreign import ccall unsafe "file.c funname" haskellname :: ffihaskelltype

foreign import ccall unsafe "VectorSIMD.c dotproduct_SIMD_double" dotproduct_SIMD_double :: 
        Int32 -> Ptr Double -> Ptr Double -> Double 

-- dot product acts like a pure function!


foreign import ccall unsafe "VectorSIMD.c dotproduct_SIMD_float" dotproduct_SIMD_float :: 
        Int32 -> Ptr Float -> Ptr Float -> Float 