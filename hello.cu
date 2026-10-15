#include <cstdio>

__global__ void hello()
{
    printf("Hello from GPU thread %d\n", threadIdx.x);
}

int main()
{
    hello<<<1, 4>>>();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(error));
        return 1;
    }
    error = cudaDeviceSynchronize();
    if (error != cudaSuccess) {
        std::fprintf(stderr, "Kernel execution failed: %s\n", cudaGetErrorString(error));
        return 1;
    }
    return 0;
}
