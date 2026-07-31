using System;
using System.Runtime.InteropServices;

namespace TensorSharp.Cuda.Interop
{
    internal static class CublasApi
    {
        private const string LibName = "cublas";

        [DllImport(LibName, EntryPoint = "cublasCreate_v2")]
        public static extern int cublasCreate(out IntPtr handle);

        [DllImport(LibName, EntryPoint = "cublasDestroy_v2")]
        public static extern int cublasDestroy(IntPtr handle);

        [DllImport(LibName, EntryPoint = "cublasSetStream_v2")]
        public static extern int cublasSetStream(IntPtr handle, IntPtr stream);

        [DllImport(LibName)]
        public static extern int cublasSetMathMode(IntPtr handle, int mode);

        [DllImport(LibName, EntryPoint = "cublasSgemm_v2")]
        public static extern int cublasSgemm(
            IntPtr handle,
            int transa,
            int transb,
            int m,
            int n,
            int k,
            ref float alpha,
            IntPtr a,
            int lda,
            IntPtr b,
            int ldb,
            ref float beta,
            IntPtr c,
            int ldc);

        [DllImport(LibName)]
        public static extern int cublasSgemmStridedBatched(
            IntPtr handle,
            int transa,
            int transb,
            int m,
            int n,
            int k,
            ref float alpha,
            IntPtr a,
            int lda,
            long strideA,
            IntPtr b,
            int ldb,
            long strideB,
            ref float beta,
            IntPtr c,
            int ldc,
            long strideC,
            int batchCount);

        [DllImport(LibName)]
        public static extern int cublasGemmEx(
            IntPtr handle,
            int transa,
            int transb,
            int m,
            int n,
            int k,
            ref float alpha,
            IntPtr a,
            int aType,
            int lda,
            IntPtr b,
            int bType,
            int ldb,
            ref float beta,
            IntPtr c,
            int cType,
            int ldc,
            int computeType,
            int algo);

        public const int CUBLAS_OP_N = 0;
        public const int CUBLAS_OP_T = 1;
        public const int CUBLAS_TENSOR_OP_MATH = 1;
        public const int CUDA_R_16F = 2;
        public const int CUDA_R_32F = 0;
        public const int CUDA_R_16BF = 14;
        public const int CUBLAS_COMPUTE_32F = 68;
        public const int CUBLAS_GEMM_DEFAULT = -1;
    }
}
