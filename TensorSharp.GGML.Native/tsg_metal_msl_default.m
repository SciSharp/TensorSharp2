// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
//
// Restore the default Metal Shading Language version that a natively-linked host gets, so ggml's
// Metal 4 tensor-API detection can succeed inside a .NET process.
//
// Metal derives the default MTLCompileOptions.languageVersion from the SDK recorded in the *main
// executable*, not from the library making the call. Microsoft ships the .NET apphost -- and the
// `dotnet` muxer -- prebuilt against the macOS 15.5 SDK, so every .NET process inherits a stale
// default even on macOS 26:
//
//     main executable records SDK 26.5  ->  default languageVersion 0x40000 (MSL 4.0)
//     main executable records SDK 15.5  ->  default languageVersion 0x30002 (MSL 3.2)
//
// ggml creates MTLCompileOptions and sets only preprocessorMacros; it never sets languageVersion
// (ggml_metal_library_init and ggml_metal_library_init_from_source in ggml-metal-device.m). Under
// MSL 3.2 <metal_tensor> and MetalPerformancePrimitives declare nothing, so the probe kernel that
// decides ggml_metal_device_props::has_tensor fails to compile with "use of undeclared identifier
// 'mpp'" and ggml disables the tensor API for the whole process. On an M5 that measured as a 2.6x
// prefill throughput loss.
//
// The fix belongs upstream in ggml, but it cannot be carried as a local edit: ExternalProjects/ggml
// is git-ignored and eng/fetch-ggml.sh does `git reset --hard FETCH_HEAD` against ggml master on
// every build, so anything written there is erased by the next build. Raising the process-wide
// default here instead keeps the vendored checkout pristine and survives ggml updates.
//
// This only changes the value seen by code that asked for "whatever the OS defaults to". Callers
// that set languageVersion explicitly keep theirs -- MLX always does
// (mlx/backend/metal/device.cpp calls setLanguageVersion), so the MLX backend is unaffected.
//
// Deliberately narrow, so a wrong guess cannot make things worse than the stale default: it acts
// only when the GPU advertises the Metal 4 family, only when the inherited default is older than
// MSL 4.0, and it never lowers a version. TENSORSHARP_METAL_MSL_DEFAULT=off disables it;
// TENSORSHARP_METAL_MSL_DEFAULT=<major>.<minor> forces a specific version.
#if defined(__APPLE__)

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <objc/runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>   // strcasecmp

// MTLLanguageVersion encodes (major << 16) + minor; MTLGPUFamilyMetal4 is 5002. Both are spelled
// numerically so this file still builds against SDKs that predate the Metal 4 declarations, the
// same approach ggml takes for MTLGPUFamilyMetal4_GGML.
#define TSG_MSL_VERSION(major, minor) ((MTLLanguageVersion)(((major) << 16) + (minor)))

static const MTLLanguageVersion TSG_MSL_TARGET_DEFAULT = TSG_MSL_VERSION(4, 0);
static const NSInteger         TSG_MTLGPUFamilyMetal4  = 5002;

static IMP                g_orig_init   = NULL;
static MTLLanguageVersion g_target      = 0;      // 0 = leave every instance alone
static dispatch_once_t    g_decide_once = 0;

// Parse TENSORSHARP_METAL_MSL_DEFAULT. Returns 0 for "disabled", TSG_MSL_TARGET_DEFAULT when unset.
static MTLLanguageVersion tsg_msl_requested(void) {
    const char * env = getenv("TENSORSHARP_METAL_MSL_DEFAULT");
    if (env == NULL || env[0] == '\0') {
        return TSG_MSL_TARGET_DEFAULT;
    }

    if (strcasecmp(env, "off")   == 0 || strcasecmp(env, "0")  == 0 ||
        strcasecmp(env, "false") == 0 || strcasecmp(env, "no") == 0) {
        return 0;
    }

    int major = 0;
    int minor = 0;
    if (sscanf(env, "%d.%d", &major, &minor) == 2 && major > 0) {
        return TSG_MSL_VERSION(major, minor);
    }

    fprintf(stderr, "tensorsharp: ignoring unparseable TENSORSHARP_METAL_MSL_DEFAULT='%s'\n", env);
    return TSG_MSL_TARGET_DEFAULT;
}

// Decided once, on the first MTLCompileOptions created, so process startup pays nothing for it.
static void tsg_msl_decide(void * unused) {
    (void) unused;

    const MTLLanguageVersion requested = tsg_msl_requested();
    if (requested == 0) {
        return;
    }

    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (dev == nil) {
            return;
        }

        // Only Metal 4 hardware needs this, and only there is MSL 4.0 certain to be usable.
        const BOOL metal4 = [dev supportsFamily:(MTLGPUFamily) TSG_MTLGPUFamilyMetal4];

#if !__has_feature(objc_arc)
        [dev release];
#endif

        if (!metal4) {
            return;
        }
    }

    g_target = requested;
}

static id tsg_compile_options_init(id self, SEL _cmd) {
    self = ((id (*)(id, SEL)) g_orig_init)(self, _cmd);
    if (self == nil) {
        return nil;
    }

    dispatch_once_f(&g_decide_once, NULL, tsg_msl_decide);

    if (g_target != 0) {
        MTLCompileOptions * options = (MTLCompileOptions *) self;
        const MTLLanguageVersion inherited = options.languageVersion;

        if (inherited < g_target) {
            options.languageVersion = g_target;

            static BOOL logged = NO;
            if (!logged) {
                logged = YES;
                fprintf(stderr,
                        "tensorsharp: default Metal language version raised from %u.%u to %u.%u "
                        "(host executable records an old SDK; set TENSORSHARP_METAL_MSL_DEFAULT=off to disable)\n",
                        (unsigned) (inherited >> 16), (unsigned) (inherited & 0xffff),
                        (unsigned) (g_target   >> 16), (unsigned) (g_target   & 0xffff));
            }
        }
    }

    return self;
}

__attribute__((constructor))
static void tsg_metal_msl_default_install(void) {
    // objc_getClass rather than [MTLCompileOptions class] so a process without Metal just no-ops.
    Class cls = objc_getClass("MTLCompileOptions");
    if (cls == Nil) {
        return;
    }

    // MTLCompileOptions is a class cluster: [MTLCompileOptions new] hands back a private concrete
    // subclass (MTLCompileOptionsInternal today) and *that* is what applies the SDK-derived default
    // in its own -init. Hooking the abstract class would run via [super init], before the default is
    // written, and the subclass would then overwrite it. So discover the concrete class from a
    // throwaway instance -- it costs nothing, needs no MTLDevice, and avoids hardcoding a private
    // class name that Apple is free to rename.
    id probe = [[cls alloc] init];
    if (probe == nil) {
        return;
    }

    Class concrete = object_getClass(probe);

#if !__has_feature(objc_arc)
    [probe release];
#endif

    const SEL sel = @selector(init);

    Method method = class_getInstanceMethod(concrete, sel);
    if (method == NULL) {
        return;
    }

    // class_getInstanceMethod walks up the hierarchy, so -init may belong to a superclass. Adding
    // the override first keeps the swizzle off NSObject: if class_addMethod succeeds the class did
    // not implement -init itself and the inherited IMP is the one to chain to; if it fails the
    // class has its own -init and method_setImplementation hands back that. Either way our IMP runs
    // after the original, so it sees -- and can raise -- the fully initialized default.
    g_orig_init = method_getImplementation(method);

    if (!class_addMethod(concrete, sel, (IMP) tsg_compile_options_init, method_getTypeEncoding(method))) {
        g_orig_init = method_setImplementation(method, (IMP) tsg_compile_options_init);
    }
}

#endif // __APPLE__
