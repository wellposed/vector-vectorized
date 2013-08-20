{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE QuasiQuotes #-}

module Numerics.Vector.Vectorized.NativeSimd.FFI.TH where

import Language.Haskell.TH

{-
forImpD :: Callconv -> Safety -> String -> Name -> TypeQ -> DecQ


data Foreign = ImportF Callconv Safety String Name Type
             | ExportF Callconv        String Name Type
         deriving( Show, Eq, Data, Typeable )

data Callconv = CCall | StdCall
          deriving( Show, Eq, Data, Typeable )

data Safety = Unsafe | Safe | Interruptible
        deriving( Show, Eq, Data, Typeable )


mkName :: String -> Name        


[t| ... |], where the "..." is a type; the quotation has type Q Type.

Q Type == TypeQ


So for each type and simd vs 



-}    


