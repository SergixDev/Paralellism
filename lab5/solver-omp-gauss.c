#include "omp.h"

#define lowerb(id, p, n)  ( id * (n/p) + (id < (n%p) ? id : n%p) )
#define numElem(id, p, n) ( (n/p) + (id < (n%p)) )
#define upperb(id, p, n)  ( lowerb(id, p, n) + numElem(id, p, n) - 1 )

#define min(a, b) ( (a < b) ? a : b )
#define max(a, b) ( (a > b) ? a : b )

extern int userparam;

// Function to copy one matrix into another
void copy_mat (double *u, double *v, unsigned sizex, unsigned sizey) {

    int nblocksi=omp_get_max_threads();
    int nblocksj=1;
	
	#pragma omp parallel
	{
		int blocki = omp_get_thread_num(); 
		int i_start = lowerb(blocki, nblocksi, sizex);
		int i_end = upperb(blocki, nblocksi, sizex);
		for (int blockj=0; blockj<nblocksj; ++blockj) {
			int j_start = lowerb(blockj, nblocksj, sizey);
			int j_end = upperb(blockj, nblocksj, sizey);
			for (int i=max(1, i_start); i<=min(sizex-2, i_end); i++)
				for (int j=max(1, j_start); j<=min(sizey-2, j_end); j++)
					v[i*sizey+j] = u[i*sizey+j];
		 }
	 }
	
	
}

// 2D-blocked solver: one iteration step
double solve (double *u, double *unew, unsigned sizex, unsigned sizey) {
	
    double tmp, diff, sum=0.0;

    int nblocksi=omp_get_max_threads();
    int nblocksj=22;
    int finished[nblocksi][nblocksj];
    for (int i = 0; i < nblocksi; ++i) {
		for (int j = 0; j < nblocksj; ++j) finished[i][j] = 0;
		}

    #pragma omp parallel reduction(+: sum) private(tmp, diff) shared(finished)
    {
      int blocki = omp_get_thread_num();
      int i_start = lowerb(blocki, nblocksi, sizex);
      int i_end = upperb(blocki, nblocksi, sizex);
      for (int blockj=0; blockj<nblocksj; ++blockj) {
		int prevFinished = 0;
        int j_start = lowerb(blockj, nblocksj, sizey);
        int j_end = upperb(blockj, nblocksj, sizey);
        
        if(u==unew && blocki!=0){ //Only enter if we are in Gauss and we are not in the first row
			do{
			//Check if the previous block has finished
			#pragma omp atomic read
			prevFinished= finished[blocki-1][blockj]; // COMPLETE
			}
			while(prevFinished != 1);//Wait if prev has not finished
		}
        
        for (int i=max(1, i_start); i<=min(sizex-2, i_end); i++) {
          for (int j=max(1, j_start); j<=min(sizey-2, j_end); j++) {
	        tmp = 0.25 * ( u[ i*sizey	   + (j-1) ] +  // left
                           u[ i*sizey	   + (j+1) ] +  // right
                           u[ (i-1)*sizey + j     ] +  // top
                           u[ (i+1)*sizey + j     ] ); // bottom
	        diff = tmp - u[i*sizey+ j];
	        sum += diff * diff;
	        unew[i*sizey+j] = tmp;
          }
        }
        if (u == unew) {
			#pragma omp atomic write
			finished[blocki][blockj] = 1;
		}
      }
    }
    return sum;
}
