/*
    FP16 arithmetic -- 4 points
    +
    Multiple kernel implementations for different layer sizes -- 1 point
*/
#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#include <cuda_fp16.h>

#define TILE_WIDTH 16
#define MASK_MAX 4096

__constant__ __half mask_const[MASK_MAX];

__global__ void Float2HalfKernel(__half *output, const float *input, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < size)
    {
        output[tid] = __float2half(input[tid]);
    }
}

__global__ void conv_forward_kernel(float *output, __half *input, __half *mask, const int B, const int M, const int C, const int H, const int W, const int K,const int S)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.

    Function paramter definitions:
    output - output
    input - input
    mask - convolution kernel
    B - batch_size (number of images in x)
    M - number of output feature maps
    C - number of input feature maps
    H - input height dimension
    W - input width dimension
    K - kernel height and width (K x K)
    S - stride step length
    */

    const int H_out = (H - K)/S + 1;
    const int W_out = (W - K)/S + 1;
    // (void)H_out; // silence declared but never referenced warning. remove this line when you start working
    // (void)W_out; // silence declared but never referenced warning. remove this line when you start working

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)
    // out_4d(0,0,0,0) = a

    #define out_4d(i3, i2, i1, i0) output[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    #define mask_4d(i3, i2, i1, i0) mask[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
    #define const_mask(i3, i2, i1, i0) mask_const[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Insert your GPU convolution kernel code here
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = blockIdx.z;

    int RowWidth = ceil((float)W_out / TILE_WIDTH);
    // int ColWidth = ceil((float)H_out / TILE_WIDTH);

    int w_out = TILE_WIDTH * (bz % RowWidth) + tx;
    int h_out = TILE_WIDTH * (bz / RowWidth) + ty;
    int b_out = bx;
    int m_out = by;

    if (w_out < W_out && h_out < H_out) {
        __half sum = __float2half(0.0);
        for (int c = 0; c < C; c++) {
            for (int i = 0; i < K; i++) {
                for (int j = 0; j < K; j++) {
                    // sum += in_4d(b_out, c, h_out * S + i, w_out * S + j) * mask_4d(m_out, c, i, j);
                    sum = __hadd(sum, __hmul(in_4d(b_out, c, h_out * S + i, w_out * S + j), const_mask(m_out, c, i, j)));
                }
            }
        }
        out_4d(b_out, m_out, h_out, w_out) = __half2float(sum);
    }


    #undef out_4d
    #undef in_4d
    #undef mask_4d
}

	
__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    // Allocate memory and copy over the relevant data structures to the GPU

    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.

    // Useful snippet for error checking
    // cudaError_t error = cudaGetLastError();
    // if(error != cudaSuccess)
    // {
    //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
    //     exit(-1);
    // }
    const int H_out = (H - K)/S + 1;
    const int W_out = (W - K)/S + 1;

    // Set the kernel dimensions and call the kernel
    int input_size = B * C * H * W * sizeof(float);
    int output_size = B * M * H_out * W_out * sizeof(float);
    int mask_size = K * K * C * M * sizeof(float);

    cudaMalloc(device_input_ptr, input_size);
    cudaMalloc(device_output_ptr, output_size);
    cudaMalloc(device_mask_ptr, mask_size);

    // Copy the input and mask to the device
    cudaMemcpy(*device_input_ptr, host_input, input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(*device_mask_ptr, host_mask, mask_size, cudaMemcpyHostToDevice);
   
}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    // Set the kernel dimensions and call the kernel
    const int H_out = (H - K)/S + 1;
    const int W_out = (W - K)/S + 1;
    int input_size = B * C * H * W;
    int output_size = B * M * H_out * W_out;
    int mask_size = K * K * C * M;

    // Convert input and mask to half precision
    __half* input_half;
    __half* mask_half;

    // Allocate memory for half precision input and mask
    cudaMalloc((void **)&input_half, input_size * sizeof(__half));
    cudaMalloc((void **)&mask_half, mask_size * sizeof(__half));

    // Set device flags for mapping host memory
    cudaSetDeviceFlags(cudaDeviceMapHost);

    // Kernel 1 & 2: Convert input and mask to half precision
    Float2HalfKernel<<<ceil((float)input_size/1024), 1024>>>(input_half, device_input, input_size);
    cudaDeviceSynchronize();

    Float2HalfKernel<<<ceil((float)mask_size/1024), 1024>>>(mask_half, device_mask, mask_size);
    cudaDeviceSynchronize();

    // Copy mask to constant memory
    cudaMemcpyToSymbol(mask_const, mask_half, mask_size * sizeof(__half));

    // Kernel 3: Convolution
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 dimGrid(B, M, ceil((float)H_out/TILE_WIDTH)*ceil((float)W_out/TILE_WIDTH));
    conv_forward_kernel<<<dimGrid, dimBlock>>>(device_output, input_half, mask_half, B, M, C, H, W, K, S);
    cudaDeviceSynchronize();

    cudaFree(input_half);
    cudaFree(mask_half);

}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    // Copy the output back to host
    const int H_out = (H - K)/S + 1;
    const int W_out = (W - K)/S + 1;
    int output_size = B * M * H_out * W_out * sizeof(float);
    cudaMemcpy(host_output, device_output, output_size, cudaMemcpyDeviceToHost);

   
    // Free device memory
    cudaFree(device_input);
    cudaFree(device_output);
    cudaFree(device_mask);
    

}


__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}
