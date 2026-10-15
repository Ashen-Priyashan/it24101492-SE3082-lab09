#include <cstdio>

__global__ void add(const int *a, const int *b, int *c)
{
    *c = *a + *b;
}

static bool check_cuda(cudaError_t error, const char *operation)
{
    if (error == cudaSuccess)
        return true;
    std::fprintf(stderr, "%s failed: %s\n", operation, cudaGetErrorString(error));
    return false;
}

int main()
{
    int a = 2, b = 7, c = 0;
    int *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    const size_t bytes = sizeof(int);

    if (!check_cuda(cudaMalloc(&d_a, bytes), "cudaMalloc(d_a)") ||
        !check_cuda(cudaMalloc(&d_b, bytes), "cudaMalloc(d_b)") ||
        !check_cuda(cudaMalloc(&d_c, bytes), "cudaMalloc(d_c)"))
        return 1;
    if (!check_cuda(cudaMemcpy(d_a, &a, bytes, cudaMemcpyHostToDevice), "copy a to device") ||
        !check_cuda(cudaMemcpy(d_b, &b, bytes, cudaMemcpyHostToDevice), "copy b to device"))
        return 1;

    add<<<1, 1>>>(d_a, d_b, d_c);
    if (!check_cuda(cudaGetLastError(), "add kernel launch") ||
        !check_cuda(cudaMemcpy(&c, d_c, bytes, cudaMemcpyDeviceToHost), "copy result to host"))
        return 1;

    std::printf("Result is %d\n", c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return c == 9 ? 0 : 1;
}
