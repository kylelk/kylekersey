    #include <stdio.h>
    #include <cuda.h>
    #include <math.h>
    
    #define BLOCK_SIZE 16
    
    #define PI 3.141592653589793
    
    // GPU subroutines
    __global__ void GPU_find_potential(int Nr, int Nz, float dr2, float dz2, float *Phi_0, float *Phi_1, float *ChargeSource);
    
    //////////////////////////////////////////////
    int main(void)
    {
    
      // atmosphere properties
      int Nr, Nz;            // grid size [0,...,Nr][0,...,Nz]
      float Rmax, Zmax;      // grid boundaries (in km)
      float dr, dz;          // grid cell size
      float dr2, dz2;        // grid cell size squared
      float *Potential_0, *Potential_1; // electric potentital
      float *charge_source;  // external charge 
      float zcharge,a,Q;   // external charge parameters
    
      // GPU variables
      float *GPU_Potential_0, *GPU_Potential_1; // electric potential
      float *GPU_charge_source;      // external charge 
      dim3 dimBlock, dimGrid;
    
      int i,k,ik;   // counters over the grid
      int n, Nn; // iteration
      float r,z;    // coordinates
    
      // output variables
      FILE *DATAOUT;
      char filename[255];
      
       // Set GPU
       cudaSetDevice(0);
    
       // set atmosphere properties
       Rmax=300.0; // km
       Zmax=100.0; // km
       Nr = 300; Nz = 100; // this gives 1km x 1km grid
       dr = Rmax/Nr; dr2 = dr*dr;
       dz = Zmax/Nz; dz2 = dz*dz;
       zcharge = 10.0;
       a = 3.0;
       Q = 200.0; // total charge
       Q = Q/pow(a*sqrt(PI),3.0)/8.854E-12;
    
       Nn=20001; // iterations
    
       // set varaiables
       Potential_0 = (float *)malloc(sizeof(float)*(Nr+1)*(Nz+1));
       Potential_1 = (float *)malloc(sizeof(float)*(Nr+1)*(Nz+1));
       charge_source = (float *)malloc(sizeof(float)*(Nr+1)*(Nz+1));
    
       // allocate GPU variables
       cudaMalloc((void **) &GPU_Potential_0, sizeof(float)*(Nr+1)*(Nz+1));
       cudaMalloc((void **) &GPU_Potential_1, sizeof(float)*(Nr+1)*(Nz+1));
       cudaMalloc((void **) &GPU_charge_source, sizeof(float)*(Nr+1)*(Nz+1));
    
       // set the external charge 
       for (i=0;i<=Nr;i++) for (k=0;k<=Nz;k++) {
          ik = (Nr+1)*k+i;
          r = i*dr; z = k*dz;
          Potential_0[ik]=0.0;
          Potential_1[ik]=0.0;
          charge_source[ik]=Q*exp(-((z-zcharge)*(z-zcharge)+r*r)/(a*a));
       }
    
       // transfer data to GPU
       cudaMemcpy(GPU_Potential_0, Potential_0, sizeof(float)*(Nr+1)*(Nz+1), cudaMemcpyHostToDevice);
       cudaMemcpy(GPU_Potential_1, Potential_1, sizeof(float)*(Nr+1)*(Nz+1), cudaMemcpyHostToDevice);
       cudaMemcpy(GPU_charge_source, charge_source, sizeof(float)*(Nr+1)*(Nz+1), cudaMemcpyHostToDevice);
    
       // run the electric field calculation 
       dimBlock.x=BLOCK_SIZE;
       dimBlock.y=BLOCK_SIZE;
       dimGrid.x=(int)(Nr/BLOCK_SIZE)+1;
       dimGrid.y=(int)(Nz/BLOCK_SIZE)+1;
       for (n=1;n<=Nn;n++) {
         if (n%2==1)
           GPU_find_potential<<<dimGrid,dimBlock>>>(Nr,Nz,dr2,dz2,GPU_Potential_0,GPU_Potential_1,GPU_charge_source);
         else
           GPU_find_potential<<<dimGrid,dimBlock>>>(Nr,Nz,dr2,dz2,GPU_Potential_1,GPU_Potential_0,GPU_charge_source);
         cudaThreadSynchronize();
       }
    
       // retrieve data from GPU
       cudaMemcpy(Potential_1, GPU_Potential_1, sizeof(float)*(Nr+1)*(Nz+1), cudaMemcpyDeviceToHost);
    
       // data output
       sprintf (filename,"GPU_POTENTIAL.dat");
       DATAOUT = fopen(filename,"w");
       fprintf(DATAOUT,"%d %d\n",Nr,Nz);
       for (i=0;i<=Nr;i++) {
         for (k=0;k<=Nz;k++) {
           ik = (Nr+1)*k+i;
           fprintf(DATAOUT,"%E ",Potential_1[ik]);
         }
         fprintf(DATAOUT,"\n");
       }
       fclose(DATAOUT);
    
    
       // free the memory
       free(Potential_0);
       free(Potential_1);
       free(charge_source);
       cudaFree(GPU_Potential_0); 
       cudaFree(GPU_Potential_1); 
       cudaFree(GPU_charge_source); 
    }
    /////////////////////////////////////////////////////
    /////////////////////////////////////////////////////
    __global__ void GPU_find_potential(int Nr, int Nz, float dr2, float dz2, float *Phi_0, float *Phi_1, float *ChargeSource){
     
     // localy shared data for this block
     __shared__ float PhiBlock[BLOCK_SIZE+2][BLOCK_SIZE+2];
     // updated potential
     float Phi;
    
     // starting index for this block
     int i0 = BLOCK_SIZE * blockIdx.x;
     int k0 = BLOCK_SIZE * blockIdx.y;
    
     // index of this thread in the r-z grid
     int it = threadIdx.x;
     int kt = threadIdx.y;
     int i = i0 + it;
     int k = k0 + kt;
     int ik = (Nr+1)*k + i;
     
     // check if the point is within the r-z grid
     if (i<=Nr && k<=Nz) {
     
       // get the data from global into the shared memory
       PhiBlock[it+1][kt+1]=Phi_0[ik];
       if (i==0) PhiBlock[it][kt+1]=0.0;
       else if (it==0) PhiBlock[it][kt+1]=Phi_0[ik-1];
       if (i==Nr) PhiBlock[it+2][kt+1]=0.0;
       else if (it==BLOCK_SIZE-1) PhiBlock[it+2][kt+1]=Phi_0[ik+1];
       if (k==0) PhiBlock[it+1][kt]=0.0;
       else if (kt==0) PhiBlock[it+1][kt]=Phi_0[ik-(Nr+1)];
       if (k==Nz) PhiBlock[it+1][kt+2]=0.0;
       else if (kt==BLOCK_SIZE-1) PhiBlock[it+1][kt+2]=Phi_0[ik+(Nr+1)];
     }
    
     // synchornize threads to make sure that all data is loaded
     // (We exit "if" because this has to be done for all threads within the block)
     __syncthreads();
    
     if (i<=Nr && k<=Nz) {
    
       // calculate the updated potential
       if (k==0 || k==Nz) Phi = 0.0; // forced by boundary condition
       else if (i>0) {
         Phi=PhiBlock[it][kt+1]*dz2*(1.0-0.5/i);
         Phi += PhiBlock[it+1][kt]*dr2;
         Phi += PhiBlock[it+2][kt+1]*dz2*(1.0+0.5/i);
         Phi += PhiBlock[it+1][kt+2]*dr2;
         Phi += ChargeSource[ik]*dr2*dz2;
         Phi /= (2*(dr2+dz2));
       }
       else {
         Phi = PhiBlock[it+1][kt]*dr2;
         Phi += PhiBlock[it+2][kt+1]*dz2*4;
         Phi += PhiBlock[it+1][kt+2]*dr2;
         Phi += ChargeSource[ik]*dr2*dz2;
         Phi /= (2*(dr2+2*dz2));
       }
    
       // store the result
       Phi_1[ik]=Phi;
    
     } // end of: if (i<=Nr && k<=Nz) 
    
    }