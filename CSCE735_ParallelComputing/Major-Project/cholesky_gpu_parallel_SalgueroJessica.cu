// Compute triangular factors of an SPD matrix using GPU
// - given an SPD matrix A, compute upper triangular matrix R such that 
//   A = R'*R, where R' is the transpose of R
//
// SPD = Symmetric Positive Definite, doesn't require pivoting during factorization
//
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <new>

#define MAX_MATRIX_SIZE 4096
#define TOL 1.0e-8

#define ERR_MALLOC 1
#define ERR_MEMCPY 2
#define ERR_KERNEL 3

#define DEBUG 1

// Define Matrix

typedef struct {
        int  n;             // order of matrix (number of rows and columns)
        double **elements;      // Allows access to array values as a matrix
        double *array;			// Linear array, stores matrix row-by-row
} Matrix;

// Device routines ..............................................................

__global__ void device_create_matrix_on_device(Matrix); 
__global__ void device_cholesky_factorization(Matrix);

// Initializes A.elements of matrix A to point to the start of each row in A.array 
// - allows access to matrix entries in the standard way: 
//   A.elements[i][j] points to Aij element that is stored in A.array
// - both A.elements and A.array are allocated on the device, therefore not
//   accessible to host
//
__global__ void device_create_matrix_on_device(Matrix A) {

    for (int i = 0; i < A.n; i++) A.elements[i] = &(A.array[i*A.n]);
}

// Cholesky factorization on device
// - converts A to R where R is an upper triangular matrix
//   such that A = R'*R (R' = transpose of R)
// - A is overwritten with R
//
__global__ void device_cholesky_factorization(Matrix A) {
    double sqrt_pivot;
    for (int k = 0; k < A.n; k++) {
        sqrt_pivot = sqrt(A.elements[k][k]);
        for (int j = k; j < A.n; j++) {
            A.elements[k][j] = A.elements[k][j]/sqrt_pivot;
        }
        for (int i = k+1; i < A.n; i++) {
            for (int j = k+1; j < A.n; j++) {
                A.elements[i][j] -= A.elements[k][j] * A.elements[k][i];
            }
        }
        for (int j = k+1; j < A.n; j++) {
            A.elements[j][k] = 0.0;
        }
    }
}

// Parallel implementation - need to add another device routine
__global__ void parallel_cholesky(double* dA, int n) { // dA is a pointer to matrix, n is matrix size
    // Separate which subsections of the matrix each thread will be using
        // threadIdx.x = thread Id in x-axis
        // blockIdx.x = block Id in x-axis
        // blockDim.x = number of threads in a block in x-axis
    int thread = threadIdx.x + blockIdx.x * blockDim.x; // each thread will work on a different section

    __shared__ double diagonal; // shared memory variable = R[k][k], all threads will be able to access this variable
    
    // Loop over matrix by column
    for (int k = 0; k < n; ++k) {
        __syncthreads(); // want to make sure all threads have completed before moving to next loop iteration

        // calculate diagnonal, R[k][k]
        if (thread == 0) { // only needs to be calculated once, then update shared variable
            dA[k * n + k] = sqrt(dA[k * n + k]);
            diagonal = dA[k * n + k];
        }
        __syncthreads(); // make sure all threads have finished previous computation before moving on, so they can access the correct updated value

        // Divide each element by diagonal
        for (int j = k + 1 + thread; j < n; j += blockDim.x * gridDim.x) {
            dA[k * n + j] = dA[k * n + j] / diagonal;
        }
        __syncthreads(); // make sure all threads have finished previous loop

        // Update the matrix
        for (int i = k + 1 + thread; i < n; i += blockDim.x * gridDim.x) {
            for (int j = k + 1; j < n; ++j) {
                double curr_value = dA[i * n + j]; // current value
                double adjust = dA[k * n + i] * dA[k * n + j]; // will be subtracted from current value
                dA[i * n + j] = curr_value - adjust; 
            }
        }
        __syncthreads(); // wait for all threads to catch up

        // make sure everything below the diagonal is equal to zero in matrix
        for (int i = k + 1 + thread; i < n; i += blockDim.x * gridDim.x) {
            dA[i * n + k] = 0.0;
        }
        __syncthreads(); // wait for all threads to catch up
        
    }
}

// Host routines ..............................................................

Matrix cholesky_factorization(Matrix&);
Matrix product_with_transpose(Matrix& R);
int compare_matrix(Matrix&, Matrix&);
Matrix clone_matrix(Matrix& A); 
void initialize_spd_matrix(Matrix&, double);
Matrix create_matrix(int, int); 
void free_matrix_memory(Matrix&);
void print_matrix(Matrix&); 
void check_error(cudaError_t, int); 
void print_device_properties(); 

// Cholesky factorization on the host (provided for reference only)
// - return R where R is an upper triangular matrix
//   such that A = R'*R (R' = transpose of R)
// 
Matrix cholesky_factorization(Matrix& A) {
    double sqrt_pivot;
    Matrix R = clone_matrix(A);
    for (int k = 0; k < R.n; k++) {
        sqrt_pivot = sqrt(R.elements[k][k]);
        for (int j = k; j < R.n; j++) {
            R.elements[k][j] = R.elements[k][j]/sqrt_pivot;
        }
        for (int i = k+1; i < R.n; i++) {
            for (int j = k+1; j < R.n; j++) {
                R.elements[i][j] -= R.elements[k][j] * R.elements[k][i];
            }
        }
        for (int j = k+1; j < R.n; j++) {
            R.elements[j][k] = 0.0;
        }
    }
    return R;
}

// Product with transpose
// - return C = R' * R
Matrix product_with_transpose(Matrix& R) {
    Matrix C = create_matrix(R.n,R.n); 
    for (int i = 0; i < R.n; i++) {
        for (int j = 0; j < R.n; j++) {
            C.elements[i][j] = 0.0;
            for (int k = 0; k < R.n; k++)
                C.elements[i][j] += R.elements[k][i]*R.elements[k][j];
        }
    }
    return C;
}

// Compare if matrix is identical to another by checking if 
// their elements are identical within specified tolerance
int compare_matrix(Matrix& A, Matrix& B) {
    int error = 0;
    if (A.n != B.n) return error;
    if (A.n != B.n) return error;  
    for (int i = 0; i < A.n; i++) {
        for (int j = 0; j < A.n; j++) {
            if (fabs(A.elements[i][j] - B.elements[i][j]) > TOL) error = 1;
        }
    }
    return error;
}   

// Clone matrix 
Matrix clone_matrix(Matrix& A) {
    Matrix C = create_matrix(A.n, A.n);
    for (int i = 0; i < C.n; i++) {
        for (int j = 0; j < C.n; j++) {
            C.elements[i][j] = A.elements[i][j];
        }
    }
    return C;
}

// Initialize an SPD matrix (for testing factorization routine)
void initialize_spd_matrix(Matrix& A, double delta) {
    double value;
    for (int i = 0; i < A.n; i++) {
        A.elements[i][i] = delta;
    }
    for (int i = 0; i < A.n; i++) {
        for (int j = i+1; j < A.n; j++) {
            value =(double)(rand())/(double)(RAND_MAX);
            A.elements[i][j] = value;
            A.elements[j][i] = value;
            A.elements[i][i] += fabs(A.elements[i][j]);
            A.elements[j][j] += fabs(A.elements[i][j]);
        }
    }
}

// Create new matrix
// - matrix entries are uninitialized
Matrix create_matrix(int num_rows, int num_cols) {
    Matrix A;
    A.n = num_rows;
    A.n = num_cols;
    A.elements = new double *[A.n];
    A.array = new double[A.n*A.n];
    for (int i = 0; i < A.n; i++) A.elements[i] = &(A.array[i*A.n]);
    return A;
}

// Delete matrix arrays 
void free_matrix_memory(Matrix& A) {
    delete A.elements;
    delete A.array;
}

// Print matrix
void print_matrix(Matrix& A) {
    printf("\n... Printing matrix ... \n");
    for (int i = 0; i < A.n; i++) {
        for (int j = 0; j < A.n; j++) {
            printf(" %8.4f", A.elements[i][j]);
        }
        printf("\n");
    }
}

// Generic error
void check_error(cudaError_t err, int type) {
	if (err != cudaSuccess) {
	    switch(type) {
		    case ERR_MALLOC: 
				fprintf(stderr, "Failed cudaMalloc (error code %s)!\n", cudaGetErrorString(err));
				break;
		    case ERR_MEMCPY: 
				fprintf(stderr, "Failed cudaMemcpy (error code %s)!\n", cudaGetErrorString(err));
				break;
		    case ERR_KERNEL: 
				fprintf(stderr, "Failed kernel launch (error code %s)!\n", cudaGetErrorString(err));
				break;
		}
        exit(0);
	}
}

// Print device properties
void print_device_properties() {
    int i, deviceCount;
    cudaDeviceProp deviceProp;
    cudaGetDeviceCount(&deviceCount);
    printf("------------------------------------------------------------\n");
    printf("Number of GPU devices found = %d\n", deviceCount);
    for ( i = 0; i < deviceCount; ++i ) {
        cudaGetDeviceProperties(&deviceProp, i);
        printf("[Device: %1d] Compute Capability %d.%d.\n", i, deviceProp.major, deviceProp.minor);
        printf(" ... multiprocessor count  = %d\n", deviceProp.multiProcessorCount);
        printf(" ... max threads per multiprocessor = %d\n", deviceProp.maxThreadsPerMultiProcessor);
        printf(" ... max threads per block = %d\n", deviceProp.maxThreadsPerBlock);
        printf(" ... max block dimension   = %d, %d, %d (along x, y, z)\n",
                deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
        printf(" ... max grid size         = %d, %d, %d (along x, y, z)\n",
                deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
        printf(" ... warp size             = %d\n", deviceProp.warpSize);
        printf(" ... clock rate            = %d MHz\n", deviceProp.clockRate/1000);
    }
    printf("------------------------------------------------------------\n");
}

// Main Program ................................................................

int main(int argc, char *argv[]) {

	cudaError_t err = cudaSuccess;

    // Timing variables
    cudaEvent_t start, stop;            // GPU timing variables
    float time_array[5];

    // Print device properties
    print_device_properties();

    // Get device information and set device to use
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount > 0) {
        cudaSetDevice(0);
    } else {
        printf("Warning: No GPU device found ... results may be incorrect\n");
    }

    // Timing initializations
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Read input, validate
    if (argc != 2) {
        printf("Need one integer as input \n");
        printf("Use: <executable_name> <matrix_size>\n");
        exit(0);
    }
    int matrix_size = atoi(argv[argc-1]);
    if (matrix_size > MAX_MATRIX_SIZE) {
        printf("Maximum matrix size allowed: %d.\n", MAX_MATRIX_SIZE);
        exit(0);
    };

    // Initialize matrix A
    Matrix A = create_matrix(matrix_size, matrix_size); 
    initialize_spd_matrix(A, 1.0);

    // Create a copy of A on the device
    // - initialize number of rows and columns of the copy
    // - allocate dA.elements and dA.array on device
    // - copy A.array that has matrix entries to the device
    // - initialize dA.elements on the device 
    Matrix dA;
    dA.n = A.n;
    dA.n = A.n;
	
    // Allocate linear arrays on device
    size_t size_elements = dA.n*sizeof(double *);
    size_t size_array = dA.n*dA.n*sizeof(double);
    err = cudaMalloc(&dA.elements, size_elements); check_error(err, ERR_MALLOC); 
    err = cudaMalloc(&dA.array, size_array); check_error(err, ERR_MALLOC);

    // Copy matrix elements to device
    err = cudaMemcpy(dA.array, A.array, size_array, cudaMemcpyHostToDevice); check_error(err, ERR_MEMCPY);

    // Initialize row pointer array dA.elements on device
    device_create_matrix_on_device<<<1,1>>>(dA); 
    err = cudaGetLastError(); check_error(err, ERR_KERNEL);

    // Compute Cholesky factor on device
    cudaEventRecord( start, 0 );
    
    // comment out previous call
    // device_cholesky_factorization<<<1,1>>>(dA);
    
    // Replace single-threaded call
    int blockSize = 256; // number of threads per block
    int gridSize = (A.n + blockSize - 1) / blockSize; // number of blocks in grid for threads
    parallel_cholesky<<<gridSize, blockSize>>>(dA.array, A.n); 
    cudaDeviceSynchronize(); // allow for all CUDA tasks to catch up before moving on
    
    err = cudaGetLastError(); check_error(err, ERR_KERNEL);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&(time_array[0]), start, stop);

    // Copy result matrix from device to host
    Matrix R = create_matrix(dA.n, dA.n);
    size_array = R.n*R.n*sizeof(double);

    err = cudaMemcpy(R.array, dA.array, size_array, cudaMemcpyDeviceToHost); check_error(err, ERR_MEMCPY);

    // Compute C = R'*R
    Matrix RtR = product_with_transpose(R);
    
    //DEBUGGING - do the matrices look correct?
    // printf("Matrix A:");
    // print_matrix(A);
    // printf("Matrix R:");
    // print_matrix(R);
    // printf("Matrix RtR:");
    // print_matrix(RtR);

    // Compare A with C = R'*R
    int error = compare_matrix(A,RtR);

    if (error != 0) {
        printf("+++  Houston, we have a problem!\n");
    } else {
        printf("+++  Matrix successfully factored\n"); 
	printf("Matrix size: %d, GPU execution time: %8.4f ms\n", A.n, time_array[0]); 
    }

    // Free allocated arrays for A, R, RtR on host
    free_matrix_memory(A); 
    free_matrix_memory(R); 
    free_matrix_memory(RtR); 

    // Free allocated arrays for dA on device
    cudaFree(dA.elements);
    cudaFree(dA.array);
}
