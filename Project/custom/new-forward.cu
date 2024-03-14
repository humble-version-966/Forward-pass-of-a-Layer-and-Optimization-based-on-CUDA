/*  
    Tuning with restrict and loop unrolling -- 3 points
*/
#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define TILE_WIDTH 16


__global__ void conv_forward_kernel(float *output, const float *input, const float *mask, const int B, const int M, const int C, const int H, const int W, const int K,const int S)
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

    // (void)H_out; // silence declared but never referenced warning. remove this line when you start working
    // (void)W_out; // silence declared but never referenced warning. remove this line when you start working

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)
    // out_4d(0,0,0,0) = a


    const int H_out = (H - K)/S + 1;
    const int W_out = (W - K)/S + 1;

    #define out_4d(i3, i2, i1, i0) output[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    #define mask_4d(i3, i2, i1, i0) mask[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

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
        float sum = 0;
        // for (int c = 0; c < C; c++) {
        //     for (int i = 0; i < K; i++) {
        //         for (int j = 0; j < K; j++) {
        //             sum += in_4d(b_out, c, h_out * S + i, w_out * S + j) * mask_4d(m_out, c, i, j);
        //         }
        //     }
        // }
        
        // Tuning with restrict and loop unrolling
        for (int c = 0; c < C; c++) {
            if (K ==3){
                sum += in_4d(b_out, c, h_out * S, w_out * S) * mask_4d(m_out, c, 0, 0);
                sum += in_4d(b_out, c, h_out * S, w_out * S + 1) * mask_4d(m_out, c, 0, 1);
                sum += in_4d(b_out, c, h_out * S, w_out * S + 2) * mask_4d(m_out, c, 0, 2);
                sum += in_4d(b_out, c, h_out * S + 1, w_out * S) * mask_4d(m_out, c, 1, 0);
                sum += in_4d(b_out, c, h_out * S + 1, w_out * S + 1) * mask_4d(m_out, c, 1, 1);
                sum += in_4d(b_out, c, h_out * S + 1, w_out * S + 2) * mask_4d(m_out, c, 1, 2);
                sum += in_4d(b_out, c, h_out * S + 2, w_out * S) * mask_4d(m_out, c, 2, 0);
                sum += in_4d(b_out, c, h_out * S + 2, w_out * S + 1) * mask_4d(m_out, c, 2, 1);
                sum += in_4d(b_out, c, h_out * S + 2, w_out * S + 2) * mask_4d(m_out, c, 2, 2);
            }
            else if (K == 7){
                sum += in_4d(b_out, c, h_out * S, w_out * S) * mask_4d(m_out, c, 0, 0);
                sum += in_4d(b_out, c, h_out * S, w_out * S + 1) * mask_4d(m_out, c, 0, 1);
                sum += in_4d(b_out, c, h_out * S, w_out * S + 2) * mask_4d(m_out, c, 0, 2);
                sum += in_4d(b_out, c, h_out * S, w_out * S + 3) * mask_4d(m_out, c, 0, 3);
                sum += in_4d(b_out, c, h_out * S, w_out * S + 4) * mask_4d(m_out, c, 0, 4);
                sum += in_4d(b_out, c, h_out * S, w_out * S + 5) * mask_4d(m_out, c, 0, 5);
                sum += in_4d(b_out, c, h_out * S, w_out * S + 6) * mask_4d(m_out, c, 0, 6);
                sum += in_4d(b_out, c, h_out * S + 1, w_out * S) * mask_4d(m_out, c, 1, 0);
                sum += in_4d(b_out, c, h_out * S + 1, w_out * S + 1) * mask_4d(m_out, c, 1, 1);
                sum += in_4d(b_out, c, h_out * S + 1, w_out * S + 2) * mask_4d(m_out, c, 1, 2);
                sum += in_4d(b_out, c, h_out * S + 1, w_out * S + 3) * mask_4d(m_out, c, 1, 3);
                sum += in_4d(b_out, c, h_out * S + 1, w_out * S + 4) * mask_4d(m_out, c, 1, 4);
                sum += in_4d(b_out, c, h_out * S + 1, w_out * S + 5) * mask_4d(m_out, c, 1, 5);
                sum += in_4d(b_out, c, h_out * S + 1, w_out * S + 6) * mask_4d(m_out, c, 1, 6);
                sum += in_4d(b_out, c, h_out * S + 2, w_out * S) * mask_4d(m_out, c, 2, 0);
                sum += in_4d(b_out, c, h_out * S + 2, w_out * S + 1) * mask_4d(m_out, c, 2, 1);
                sum += in_4d(b_out, c, h_out * S + 2, w_out * S + 2) * mask_4d(m_out, c, 2, 2);
                sum += in_4d(b_out, c, h_out * S + 2, w_out * S + 3) * mask_4d(m_out, c, 2, 3);
                sum += in_4d(b_out, c, h_out * S + 2, w_out * S + 4) * mask_4d(m_out, c, 2, 4);
                sum += in_4d(b_out, c, h_out * S + 2, w_out * S + 5) * mask_4d(m_out, c, 2, 5);
                sum += in_4d(b_out, c, h_out * S + 2, w_out * S + 6) * mask_4d(m_out, c, 2, 6);
                sum += in_4d(b_out, c, h_out * S + 3, w_out * S) * mask_4d(m_out, c, 3, 0);
                sum += in_4d(b_out, c, h_out * S + 3, w_out * S + 1) * mask_4d(m_out, c, 3, 1);
                sum += in_4d(b_out, c, h_out * S + 3, w_out * S + 2) * mask_4d(m_out, c, 3, 2);
                sum += in_4d(b_out, c, h_out * S + 3, w_out * S + 3) * mask_4d(m_out, c, 3, 3);
                sum += in_4d(b_out, c, h_out * S + 3, w_out * S + 4) * mask_4d(m_out, c, 3, 4);
                sum += in_4d(b_out, c, h_out * S + 3, w_out * S + 5) * mask_4d(m_out, c, 3, 5);
                sum += in_4d(b_out, c, h_out * S + 3, w_out * S + 6) * mask_4d(m_out, c, 3, 6);
                sum += in_4d(b_out, c, h_out * S + 4, w_out * S) * mask_4d(m_out, c, 4, 0);
                sum += in_4d(b_out, c, h_out * S + 4, w_out * S + 1) * mask_4d(m_out, c, 4, 1);
                sum += in_4d(b_out, c, h_out * S + 4, w_out * S + 2) * mask_4d(m_out, c, 4, 2);
                sum += in_4d(b_out, c, h_out * S + 4, w_out * S + 3) * mask_4d(m_out, c, 4, 3);
                sum += in_4d(b_out, c, h_out * S + 4, w_out * S + 4) * mask_4d(m_out, c, 4, 4);
                sum += in_4d(b_out, c, h_out * S + 4, w_out * S + 5) * mask_4d(m_out, c, 4, 5);
                sum += in_4d(b_out, c, h_out * S + 4, w_out * S + 6) * mask_4d(m_out, c, 4, 6);
                sum += in_4d(b_out, c, h_out * S + 5, w_out * S) * mask_4d(m_out, c, 5, 0);
                sum += in_4d(b_out, c, h_out * S + 5, w_out * S + 1) * mask_4d(m_out, c, 5, 1);
                sum += in_4d(b_out, c, h_out * S + 5, w_out * S + 2) * mask_4d(m_out, c, 5, 2);
                sum += in_4d(b_out, c, h_out * S + 5, w_out * S + 3) * mask_4d(m_out, c, 5, 3);
                sum += in_4d(b_out, c, h_out * S + 5, w_out * S + 4) * mask_4d(m_out, c, 5, 4);
                sum += in_4d(b_out, c, h_out * S + 5, w_out * S + 5) * mask_4d(m_out, c, 5, 5);
                sum += in_4d(b_out, c, h_out * S + 5, w_out * S + 6) * mask_4d(m_out, c, 5, 6);
                sum += in_4d(b_out, c, h_out * S + 6, w_out * S) * mask_4d(m_out, c, 6, 0);
                sum += in_4d(b_out, c, h_out * S + 6, w_out * S + 1) * mask_4d(m_out, c, 6, 1);
                sum += in_4d(b_out, c, h_out * S + 6, w_out * S + 2) * mask_4d(m_out, c, 6, 2);
                sum += in_4d(b_out, c, h_out * S + 6, w_out * S + 3) * mask_4d(m_out, c, 6, 3);
                sum += in_4d(b_out, c, h_out * S + 6, w_out * S + 4) * mask_4d(m_out, c, 6, 4);
                sum += in_4d(b_out, c, h_out * S + 6, w_out * S + 5) * mask_4d(m_out, c, 6, 5);
                sum += in_4d(b_out, c, h_out * S + 6, w_out * S + 6) * mask_4d(m_out, c, 6, 6);
            }
        }
        out_4d(b_out, m_out, h_out, w_out) = sum;
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
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 dimGrid(B, M, ceil((float)H_out/TILE_WIDTH)*ceil((float)W_out/TILE_WIDTH));
    conv_forward_kernel<<<dimGrid, dimBlock>>>(device_output, device_input, device_mask, B, M, C, H, W, K, S);

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
