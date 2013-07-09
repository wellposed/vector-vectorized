
# notes on storable and unboxed reps that ghc uses

# for unboxed, see the primitives package
http://hackage.haskell.org/packages/archive/primitive/0.5.0.1/doc/html/src/Data-Primitive-Types.html#Prim  look like bool is a word8

#Storable
-- System-dependent, but rather obvious instances
--- bool is in a word
instance Storable Bool where
   sizeOf _          = sizeOf (undefined::HTYPE_INT)
   alignment _       = alignment (undefined::HTYPE_INT)
   peekElemOff p i   = liftM (/= (0::HTYPE_INT)) $ peekElemOff (castPtr p) i
   pokeElemOff p i x = pokeElemOff (castPtr p) i (if x then 1 else 0::HTYPE_INT)

#define STORABLE(T,size,align,read,write)       \
instance Storable (T) where {                   \
    sizeOf    _ = size;                         \
    alignment _ = align;                        \
    peekElemOff = read;                         \
    pokeElemOff = write }

#ifdef __GLASGOW_HASKELL__
STORABLE(Char,SIZEOF_INT32,ALIGNMENT_INT32,
         readWideCharOffPtr,writeWideCharOffPtr)
#elif defined(__HUGS__)
STORABLE(Char,SIZEOF_HSCHAR,ALIGNMENT_HSCHAR,
         readCharOffPtr,writeCharOffPtr)
#endif

STORABLE(Int,SIZEOF_HSINT,ALIGNMENT_HSINT,
         readIntOffPtr,writeIntOffPtr)

STORABLE(Word,SIZEOF_HSWORD,ALIGNMENT_HSWORD,
         readWordOffPtr,writeWordOffPtr)


STORABLE(Float,SIZEOF_HSFLOAT,ALIGNMENT_HSFLOAT,
         readFloatOffPtr,writeFloatOffPtr)

STORABLE(Double,SIZEOF_HSDOUBLE,ALIGNMENT_HSDOUBLE,
         readDoubleOffPtr,writeDoubleOffPtr)

STORABLE(Word8,SIZEOF_WORD8,ALIGNMENT_WORD8,
         readWord8OffPtr,writeWord8OffPtr)

STORABLE(Word16,SIZEOF_WORD16,ALIGNMENT_WORD16,
         readWord16OffPtr,writeWord16OffPtr)

STORABLE(Word32,SIZEOF_WORD32,ALIGNMENT_WORD32,
         readWord32OffPtr,writeWord32OffPtr)

STORABLE(Word64,SIZEOF_WORD64,ALIGNMENT_WORD64,
         readWord64OffPtr,writeWord64OffPtr)

STORABLE(Int8,SIZEOF_INT8,ALIGNMENT_INT8,
         readInt8OffPtr,writeInt8OffPtr)

STORABLE(Int16,SIZEOF_INT16,ALIGNMENT_INT16,
         readInt16OffPtr,writeInt16OffPtr)

STORABLE(Int32,SIZEOF_INT32,ALIGNMENT_INT32,
         readInt32OffPtr,writeInt32OffPtr)

STORABLE(Int64,SIZEOF_INT64,ALIGNMENT_INT64,
         readInt64OffPtr,writeInt64OffPtr)

#endif