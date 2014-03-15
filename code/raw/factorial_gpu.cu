#include <stdio.h>
#include <cuda.h>
//#include "cuPrintf.cu"

/*
 * CUDA parallel factorial computing program by bahramwhh
 * Note : This program is only tested for n <= 21
 * because of my hardware constraints
 * 
 * comments in the code are for debuggin purpose only ;)
 * */

__global__ void fact(int *n, int *a, int *b, size_t *result)
{
    extern __shared__ size_t output[];
    int arrays_size = (*n)/2;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    // first round
    output[tid] = a[tid] * b[tid];
    
    // sync
    __syncthreads();
    //cuPrintf("First Round  [tid=%d] : output[%d] = %d \n", tid, tid, output[tid]);
    
    // other rounds
    for(int i=0; i < 5; i++)
    {
	if(!(tid%2) && output[tid] != 0) 
	{
	    int next_tid = tid+1;
	    while(next_tid < arrays_size && output[next_tid] == 0)
			next_tid++;
	    if(next_tid < arrays_size)
	    {
		//cuPrintf("Currently I want to multiply output[%d] (%d) * output[%d] (%d) = %d\n", tid, output[tid], next_tid, output[next_tid], output[tid]*output[next_tid]);
		size_t temp = output[next_tid];
		output[next_tid] = 0;
		if(output[tid] != 0) 
		{
		    //cuPrintf(" ************ I [tid=%d] made output[%d] to zero - output[%d]=%d, output[%d]=%d, temp=%d ***************\n", tid, next_tid, tid, output[tid], next_tid, output[next_tid], temp);
		    //__syncthreads();
		    output[tid] *= temp;
		    //cuPrintf("I made the changes : output[%d] (%d) * output[%d] (%d) = %d, temp=%d\n", tid, output[tid], next_tid, output[next_tid], output[tid]*output[next_tid], temp);
		}
		else
		    output[next_tid] = temp;
	    }

	}
	__syncthreads();
	//cuPrintf("Round[%d]  [tid=%d] : output[%d] = %lld \n", i+2, tid, tid, output[tid]);
    }
    
    if(*n % 2)
	output[0] *= (*n);
    
    *result = output[0];
}

int main(void)
{
    int *n, *dev_n, *a, *b, *dev_a, *dev_b;
    size_t *result, *dev_result;
    
    n = (int *)malloc(sizeof(int));
    result = (size_t *)malloc(sizeof(size_t));
    
    printf("Please enter your value to calculate it's factorial: \n");
    scanf("%d", n);
    
    if(*n == 1 || *n == 0 || *n == 2)
	printf("Result = %d\n", *n);
    else if(*n < 0)
	printf("Factorial input can't be negative ;)\n");
    else
    {
	int arrays_size = (*n)/2;
	
	cudaMalloc((void **)&dev_n, sizeof(int));
	cudaMalloc((void **)&dev_a, arrays_size * sizeof(int));
	cudaMalloc((void **)&dev_b, arrays_size * sizeof(int));
	cudaMalloc((void **)&dev_result, sizeof(size_t));
	
	a = (int *)malloc(arrays_size * sizeof(int));
	b = (int *)malloc(arrays_size * sizeof(int));
	
	for(int i=1, a_c=0, b_c=0; i <= *n; i++)
	{
	    if(i % 2)
		a[a_c++] = i;
	    else
		b[b_c++] = i;
	}
	
	cudaMemcpy(dev_n, n, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_a, a, arrays_size * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, arrays_size * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_result, result, sizeof(size_t), cudaMemcpyHostToDevice);
	
	clock_t start, stop;
	double t = 0.0;
	
	//cudaPrintfInit(); // needed for debuggin ;)
	fact<<<1, arrays_size, arrays_size*sizeof(size_t)>>>(dev_n, dev_a, dev_b, dev_result);
	//cudaPrintfDisplay(stdout, true);
	//cudaPrintfEnd();
	
	cudaMemcpy(result, dev_result, sizeof(size_t), cudaMemcpyDeviceToHost);
	
	printf("Result is %lu\n", *result);
	
	free(n);
	free(a);
	free(b);
	free(result);
	
	cudaFree(dev_n);
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_result);
	
    }
}