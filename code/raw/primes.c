#include<stdio.h>
#include<stdlib.h>
#include<cuda.h>
#define SIZE 250000
#define BLOCK_NUM 96
#define THREAD_NUM 1024
int data[SIZE];
__global__ static void sieve(int *num,clock_t* time){
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    int tmp=bid*THREAD_NUM+tid;
    if(tid==0) time[bid] = clock();
    while(tmp<SIZE){
        int i=1;
        while(((2*tmp+3)*i+tmp+1)<SIZE){
            num[(2*tmp+3)*i+tmp+1] = 0;
            i++;
        }
        tmp+=BLOCK_NUM*THREAD_NUM;
    }
    if(tid==0) time[bid+BLOCK_NUM] = clock();
}
void GenerateNumbers(int *number,int size){
    for(int i=0;i<size;i++)
        number[i] = 2*i+1;
    number[0] = 2;
}
int main(){
    GenerateNumbers(data,SIZE);
    int *gpudata;
    clock_t* time;
    int cpudata[SIZE];
    cudaMalloc((void**)&gpudata,sizeof(int)*SIZE);
    cudaMalloc((void**)&time,sizeof(clock_t)*BLOCK_NUM*2);
    cudaMemcpy(gpudata,data,sizeof(int)*SIZE,cudaMemcpyHostToDevice);
    sieve<<<BLOCK_NUM,THREAD_NUM,0>>>(gpudata,time);
    clock_t time_used[BLOCK_NUM * 2];
    cudaMemcpy(&cpudata,gpudata,sizeof(int)*SIZE,cudaMemcpyDeviceToHost);
    cudaMemcpy(&time_used,time,sizeof(clock_t)*BLOCK_NUM*2,cudaMemcpyDeviceToHost);
    cudaFree(gpudata);
    for(int i=0;i<SIZE;i++)
        if(cpudata[i]!=0)
            printf("%d\t",cpudata[i]);
    clock_t min_start,max_end;
    min_start = time_used[0];
    max_end = time_used[BLOCK_NUM];
    for(int i=1;i<BLOCK_NUM;i++) {
        if(min_start>time_used[i])
            min_start=time_used[i];
        if(max_end<time_used[i+BLOCK_NUM])
            max_end=time_used[i+BLOCK_NUM];
    }
    printf("\nTime Cost: %d\n",max_end-min_start);
}