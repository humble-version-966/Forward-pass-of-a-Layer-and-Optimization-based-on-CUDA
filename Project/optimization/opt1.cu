/*
    Shared memory matrix multiplication and input matrix unrolling -- 3 points
*/
#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define TILE_WIDTH 16
#define SHARE_MAX 12000

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

    // Insert your GPU convolution kernel code here
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = blockIdx.z;

    int W_grid = (W_out - 1) / TILE_WIDTH + 1;
    // int H_grid = (H_out - 1) / TILE_WIDTH + 1;

    int h_out = TILE_WIDTH * (by / W_grid) + ty;
    int w_out = TILE_WIDTH * (by % W_grid) + tx;
    int b_out = bz;
    int m_out = bx;

    // transfer h and w to input
    int h_in = TILE_WIDTH * (by / W_grid) * S;
    int w_in = TILE_WIDTH * (by % W_grid) * S;

    /*
        shared memory
    */
    __shared__ float input_shared[SHARE_MAX];
    int shared_size = (TILE_WIDTH - 1) * S + K;

    #define share_3d(i2, i1, i0) input_shared[(i2) * (shared_size * shared_size) + (i1) * (shared_size) + i0]

    // load input to shared mem
    for (int c = 0; c < C; c++) {
        for (int i = 0; i < ((shared_size - 1)/TILE_WIDTH + 1); i++) {
            for (int j = 0; j < ((shared_size - 1)/TILE_WIDTH + 1); j++) {
                int w_thread = TILE_WIDTH * i + tx;
                int h_thread = TILE_WIDTH * j + ty;
                if (w_thread < shared_size && h_thread < shared_size) {
                    share_3d(c, h_thread, w_thread) = in_4d(b_out, c, h_thread + h_in, w_thread + w_in);
                }
            }
        }
    }

    __syncthreads();

    // compute output
    float sum = 0.0f;

    if (h_out < H_out && w_out < W_out){
        for (int c = 0; c < C; c++) {
            for (int i = 0; i < K; i++) {
                for (int j = 0; j < K; j++) {
                    sum += share_3d(c, ty * S + i, tx * S + j) * mask_4d(m_out, c, i, j);
                }
            }
        }
        out_4d(b_out, m_out, h_out, w_out) = sum;
    }

    #undef out_4d
    #undef in_4d
    #undef mask_4d
    #undef share_3d
}

	
__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    // Allocate memory and copy over the relevant data structures to the GPU
    int input_size = H * W * B * C * sizeof(float);
    int mask_size = K * K * M * C * sizeof(float);
    int output_size = ((H - K)/S + 1) * ((W - K)/S + 1) * B * M * sizeof(float);

    // Allocate memory on the device
	cudaMalloc((void**)device_input_ptr, input_size);
	cudaMalloc((void**)device_mask_ptr, mask_size);
	cudaMalloc((void**)device_output_ptr, output_size);

    // Copy memory to the device
	cudaMemcpy(*device_input_ptr, host_input, input_size, cudaMemcpyHostToDevice);
	cudaMemcpy(*device_mask_ptr, host_mask, mask_size, cudaMemcpyHostToDevice);
}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    // Set the kernel dimensions and call the kernel
	dim3 BlockDim(TILE_WIDTH, TILE_WIDTH, 1);
	dim3 GridDim(M, ((H - K)/S/TILE_WIDTH+1)*((W - K)/S/TILE_WIDTH + 1), B);

	conv_forward_kernel<<<GridDim, BlockDim>>>(device_output, device_input, device_mask, B, M, C, H, W, K, S);	
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    // Copy the output back to host
	int output_size = ((H - K)/S + 1)*((W - K)/S + 1) * B * M;
	cudaMemcpy(host_output, device_output, output_size*sizeof(float), cudaMemcpyDeviceToHost);
   
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