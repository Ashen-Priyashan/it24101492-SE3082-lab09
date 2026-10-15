# SE3082 Lab 09 - Introduction to CUDA

Solutions for the CUDA lab sheet, organized by exercise. The programs need an
NVIDIA GPU and the CUDA compiler (`nvcc`), so run them in Google Colab with a
GPU runtime as described in the handout. This computer has no local `nvcc`.

## Start in Google Colab

1. Open `SE3082_Lab09_IT24101492.ipynb` in Google Colab.
2. Choose **Runtime → Change runtime type → T4 GPU** (or the available GPU).
3. Run the setup checks near the top of the notebook. The compile cells use
   `-arch=sm_75` for a T4; change it to match a different GPU as described in
   the notebook.
4. Run each exercise's write, compile, and run cells in order.

Colab's disk is temporary. Save the notebook to Drive; the `.cu` files are
recreated from notebook cells whenever the runtime is restarted.

## Exercise files

| Exercise | Source | What it demonstrates |
| --- | --- | --- |
| 1 | `hello.cu` | CUDA environment check and GPU thread output |
| 2 | `ex2_add.cu` | One GPU thread adds two integers |
| 3 | `ex3_blocks.cu` | 512 blocks, one thread per block |
| 4 | `ex4_threads.cu` | One block, 512 threads; includes the 2048-thread limit experiment |
| 5 | `ex5_vecmul.cu` | Element-wise multiplication of 10,000,000 values |
| 6 | `ex6_matrix.cu` | Element-wise multiplication of two 10,000 × 10,000 matrices |

Exercises 5 and 6 verify their GPU results against a CPU calculation. Exercise
6 allocates three 400 MB arrays on both the host and GPU, as the handout
specifies; use a Colab GPU runtime with enough available memory.
