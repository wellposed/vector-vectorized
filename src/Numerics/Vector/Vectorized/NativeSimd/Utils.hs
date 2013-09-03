{-
This module provides the wrapper utils for the 
the pointer based FFI wrappers


1) the unary operations
2) the binary operations

3) dot products!


NB: for Float and Double, the memory layout for storable and unboxed are the same

for Complex Floats and Complex Doubles, we will only do the storable layout.

this does mean I incur the dependency of complex storable to make sure the type class
instances 


also need to work out the bounds checking story, but should do above these wrappers
-}

module Numerics.Vector.Vectorized.NativeSimd.Utils where

import Foreign.Storable.Complex   

import Data.Complex
import Foreign.Storable
import Foreign.Ptr
import Data.Int 
import Foreign.ForeignPtr.Safe 

import qualified Data.Vector.Storable.Mutable as SM 

{-# INLINE unsafeWith0 #-}
unsafeWith0 :: Storable a => SM.IOVector a -> (Int -> Ptr a -> IO b) -> IO b
unsafeWith0 (SM.MVector n fp)  =  \ f -> withForeignPtr fp  $! (f n)

{-# INLINE unsafeUnaryStorableSimdWrapper #-}
unsafeUnaryStorableSimdWrapper :: Storable a => 
        (Int32 -> Ptr a -> Ptr a -> IO ())-> Int32-> SM.IOVector a  -> SM.IOVector a -> IO ()
unsafeUnaryStorableSimdWrapper fun = \len  iovIN iovOUT ->
        unsafeWith0 iovIN $! \_ ptrIn -> 
            unsafeWith0 iovOUT $! \_ ptrOUT -> 
                do  fun len ptrIn ptrOUT
                    return () 

{-# INLINE unsafeBinaryStorableSimdWrapper #-}
unsafeBinaryStorableSimdWrapper :: Storable a => 
        (Int32 -> Ptr a -> Ptr a -> Ptr a -> IO ())-> Int32-> SM.IOVector a  -> SM.IOVector a  -> SM.IOVector a -> IO ()
unsafeBinaryStorableSimdWrapper fun = \len  iovInL iovInR iovOUT ->
        unsafeWith0 iovInL $! \_ ptrInL -> 
            unsafeWith0 iovInR $! \_ ptrInR -> 
                unsafeWith0 iovOUT $! \_ ptrOUT -> 
                    do  fun len ptrInL ptrInR ptrOUT
                        return () 


withSingletonStorable :: Storable a => IO (SM.IOVector a)
withSingletonStorable = SM.new 1                         