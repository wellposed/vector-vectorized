{-# LANGUAGE NamedFieldPuns #-}
{-# LANGUAGE  BangPatterns #-}
{-# LANGUAGE CPP #-}
import Distribution.Simple
import Distribution.Simple.Program.Builtin
import Distribution.Simple.Program.Types
import Distribution.Simple.Program

import Distribution.PackageDescription
import Distribution.Simple.LocalBuildInfo
import Distribution.Simple.Setup
--main = defaultMain



main :: IO ()
#if x86_64_HOST_ARCH
main = 
    do 
        putStrLn "Please have Clang installed or the build will fail"
        defaultMainWithHooks myhook 
#else   
main = error "only x86_64 architectures are currently supported" 
#endif 


{-
this is to work around mac having a really old GCC version and AS (assembler)
darwin is OSX
-}
#if darwin_HOST_OS 
myhook =  set lensBuildHook2UserHooks myBuildHook simpleUserHooks 
    where 
        oldbuildhook = get lensBuildHook2UserHooks simpleUserHooks
        myBuildHook :: PackageDescription -> LocalBuildInfo -> UserHooks -> BuildFlags -> IO ()
        myBuildHook packDesc  locBuildInfo  = 
            oldbuildhook packDesc   (set lensProgramConfig2LocBuildInfo newProgramConfig locBuildInfo)
                where   !(Just !oldGhcConf) =lookupProgram ghcProgram oldProgramConfig

                        oldProgramConfig = get lensProgramConfig2LocBuildInfo locBuildInfo
                        oldOverrideArgs = get lensProgramOverrideArgs2ConfiguredProgram oldGhcConf
                        newOverrideArgs = oldOverrideArgs++["-pgma clang", "-pgmc clang"] 
                        newGhcConf = set lensProgramOverrideArgs2ConfiguredProgram newOverrideArgs oldGhcConf
                        newProgramConfig = updateProgram newGhcConf oldProgramConfig
#else
myhook = simpleUserHooks
#endif 
{-
    On mac we need to use Clang as the assembler and C compiler to enable
-}


    --simpleUserHooks { hookedPrograms = [ghcProgram { programPostConf = \ a b -> return ["-pgma clang", "-pgmc clang"] }] }
lensProgramOverrideArgs2ConfiguredProgram :: Lens [String] ConfiguredProgram
lensProgramOverrideArgs2ConfiguredProgram = 
        Lens (\ConfiguredProgram{programOverrideArgs}-> programOverrideArgs)
             (\newargs confprog -> confprog{programOverrideArgs = newargs})

lensBuildHook2UserHooks :: Lens (PackageDescription -> LocalBuildInfo -> UserHooks -> BuildFlags -> IO ()) UserHooks
lensBuildHook2UserHooks = Lens (\UserHooks{buildHook}->buildHook) (\ bh  uhook -> uhook{buildHook=bh}) 

lensProgramConfig2LocBuildInfo :: Lens (ProgramConfiguration) (LocalBuildInfo)
lensProgramConfig2LocBuildInfo = Lens (\LocalBuildInfo{withPrograms}-> withPrograms) 
                                    (\wprog lbi -> lbi{withPrograms= wprog})

-- my mini lens!
data Lens a b = Lens (b->a) (a ->b  -> b)
infixr 9 #

(#) :: Lens a b -> Lens b c -> Lens a c 
(#) lenA@(Lens getA setA) lenB@(Lens getB setB)=  Lens (getA . getB) (\a c -> setB (setA a $ getB c) c)

get :: Lens a b -> b -> a
get (Lens lensGet _) b = lensGet b

set :: Lens a b -> a -> b -> b 
set (Lens _ lensSet) a b = lensSet a b  


{-
minier lens thats lens compatible proposed by kmett on irc
http://hpaste.org/89830

-- {-# LANGUAGE Rank2Types, DeriveFunctor #-}
-- import Control.Applicative
-- type Lens s t a b = forall f. Functor f => (a -> f b) -> s -> f t
-- type Lens' s a = forall f. Functor f => (a -> f a) -> s -> f s
-- infixl 1 &
-- infixr 4 .~, %~
-- infixl 8 ^.
-- s ^. l = getConst (l Const s)
-- x & f = f x
-- (%~) l f s = l (\a () -> f a) s ()
-- (.~) l b s = l (\_ () -> b) s ()

-}
