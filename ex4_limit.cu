#include <cstdio>

__global__ void probe()
{
}

int main()
{
    // The device limit is at most 1024 threads in one block.
    probe<<<1, 2048>>>();
    const cudaError_t error = cudaGetLastError();
    if (error == cudaSuccess) {
        std::printf("This GPU accepted 2048 threads in one block.\n");
        return 0;
    }
    std::printf("Kernel launch failed: %s\n", cudaGetErrorString(error));
    std::printf("The single-block thread limit is smaller than 2048.\n");
    return 0;
}
