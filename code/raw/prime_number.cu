#include <stdio.h>
#include <assert.h>
#include <time.h>
#include <cuda.h>

#define BLOCK_LOW(id,p,n) ((id)*(n)/(p))l;
#define BLOCK_HIGH(id,p,n) (BLOCK_LOW((id)+1,p,n)-1);
#define BLOCK_SIZE(id,p,n) (BLOCK_LOW((id)+1,p,n)-BLOCK_LOW(id,p,n));
#define BLOCK_OWNER(index,p,n) (((p)*((index)+1)-1)/(n));
int inputvalues[8] = {12010991,3059810,17756755,7362131,25485528,4134948,15051971,16947626};
//int inputvalues[8] = {100,3059810,17756755,7362131,25485528,4134948,15051971,16947626};
 
__global__ void seive(int *d_marked,int prime)
{
	
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int width=256;
	int idx=threadIdx.x;
	/*
	__shared__ int low_value =  blockIdx.x * blockDim.x + 0;
	__shared__ int high_value =  blockIdx.x * blockDim.x + blockDim.x;
	__shared__ int first;
	
    if (prime * prime > low_value)
         first = prime * prime - low_value;
      else {
         if (!(low_value % prime)) first = 0;
         else first = prime - (low_value % prime);
      }*/
	
	 __shared__ int  my_marked[256];
	
	 if( idx<width)	my_marked[idx]=d_marked[tid];

	
	
	  //if ((d_marked[tid]%prime)==0) d_marked[tid]=0 ;
	  if ((my_marked[idx]%prime)==0) d_marked[tid]=0 ;
	
	
}

int main (int argc, char** argv)
{
	int *h_marked;
	int *d_marked;
	int size;
	int i,ii,jj;
	int count=0;
	int prime=2;
	double t1, t2;
	FILE *f; 
	f =fopen("seive_CUDA_results.txt","w");
	fprintf(f,"index,counter,number,time,result\n");
	for (ii = 0;ii<8;ii++) 
	{
		size = inputvalues[ii];
		for(jj = 0;jj<10;jj++)
		{
	// Allocate host memory
	t1=clock();
	h_marked = (int *)malloc (size*sizeof(int));
	
	for (i = 0; i < size; i++) {h_marked[i] = i;}
	 
	
	
	// Allocate device memory
	cudaMalloc((void **) &d_marked, size*sizeof(int));
	
	// Copy host data to device
	cudaMemcpy(d_marked,h_marked, size*sizeof(int),cudaMemcpyHostToDevice);
	///------------------------------
	int myn;
   myn=sqrt(size)+1;
   
   	int *intarray = (int *) malloc(myn * sizeof(int));
	int *primearray = NULL;
	int  j = 0;
	int primecounter = 0;
	
	for(i = 0; i < myn; i++)
		intarray[i] = 0;
		
	/* at positions where there are multiples of i, mark as zero */
  for( i = 2; i * i < myn; i++ )
	if( intarray[i]==0 )
	  for( j = i + i; j < myn; j += i )
		intarray[j] = 1;

	  /* count all values that haven't been marked */
  for( j = 2; j < myn; j++ )
	if(intarray[j]==0)
	  primecounter++;
  
  primearray = (int *) malloc(primecounter  * sizeof(int));
  
  for(i = 0; i < primecounter; i++)
		primearray[i] = i;
		
   /* store all prime numbers from intarray into primearray */
  i=0;
  for( j = 2; j < myn; j++ )
	if(intarray[j]==0)	  {primearray[i] = j; i++;}
	///-----------------------------------
	// Launch Kernel

	/*dim3 dimGrid(1,1);
	dim3 dimBlock(128,1);
	
	seive <<< dimGrid,dimBlock >>> (d_marked,prime); */
   int block_size = 256;
   int n_blocks = size/block_size + (size%block_size == 0 ? 0:1);
   for(i=0;i<primecounter;i++){
   prime=primearray[i];
   seive <<< n_blocks, block_size >>> (d_marked, prime);
	}
	// Copy device data to host
	cudaMemcpy(h_marked,d_marked,size*sizeof(int),cudaMemcpyDeviceToHost);
	
	//for (i = 0; i < size; i++) printf("  %d ",h_marked[i]);
	count=0;
	for (i = 4; i < size; i++) if(h_marked[i]!=0) count++;
	t2 = clock();
	
	// Free memory locations
	cudaFree(d_marked); free(h_marked) ;
	
	fprintf(f,"%d,%d,%d,%.6f,%d\n",ii,jj,size,(t2-t1)/CLOCKS_PER_SEC,count+primecounter);
					 
	printf("%d,%d,%d %.6f seconds ",ii,jj,size,(t2-t1)/CLOCKS_PER_SEC);
  	printf(" # of primes is %d\n",count+primecounter);
					  
		}
	}
	fclose(f);
	
}



