// 
// Computes the product of two matrices: C = A * B
// using Strassen's algorithm
//
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <new>
#include <chrono>
#include <omp.h>

using namespace std;
using namespace std::chrono;

#define MAX_MATRIX_SIZE	65536
#define TOL 1.0e-12

#define DEBUG 1

// Global variables
int matrix_size; 
int leaf_matrix_size;

// Define Matrix

class Matrix {
    public:
        int  nrows;		// number of rows
        int  ncols;		// number of columns 
        double **elements;	// Matrix elements

	static Matrix strassens_product(Matrix, Matrix);
	static Matrix standard_product(Matrix&, Matrix&);
	static Matrix addition(Matrix&, Matrix&);
	static Matrix subtraction(Matrix&, Matrix&);
	static int compare_matrix(Matrix&, Matrix&); 

	Matrix extract_submatrix(int, int, int, int); 
	void update_submatrix(Matrix&, int, int, int, int); 
	void initialize_matrix(double);
	void print_matrix();

	static void matrix_error(int); 

	Matrix(int,int); 

    private:
	double *array;
};

// Strassen's matrix product 
// - return C = A * B
// - implemented only for square matrices where nrows=ncols
//
Matrix Matrix::strassens_product(Matrix A, Matrix B) {

    int n = A.nrows;
    if (A.nrows != A.ncols) Matrix::matrix_error(500); 
    if (B.nrows != B.ncols) Matrix::matrix_error(501); 
    if (A.ncols != B.nrows) Matrix::matrix_error(502); // Matrix not square

    if (n <= leaf_matrix_size) {

	Matrix C = Matrix::standard_product(A, B); 
	return C;

    } else {

	// Extract blocks of A: A11, A12, A21, A22
        Matrix A11 = A.extract_submatrix(0, n/2-1, 0, n/2-1);
        Matrix A12 = A.extract_submatrix(0, n/2-1, n/2, n-1);
        Matrix A21 = A.extract_submatrix(n/2, n-1, 0, n/2-1);
        Matrix A22 = A.extract_submatrix(n/2, n-1, n/2, n-1);

	// Extract blocks of B: B11, B12, B21, B22
        Matrix B11 = B.extract_submatrix(0, n/2-1, 0, n/2-1);
        Matrix B12 = B.extract_submatrix(0, n/2-1, n/2, n-1);
        Matrix B21 = B.extract_submatrix(n/2, n-1, 0, n/2-1);
        Matrix B22 = B.extract_submatrix(n/2, n-1, n/2, n-1);

        Matrix M1(n/2, n/2), M2(n/2, n/2), M3(n/2, n/2), M4(n/2, n/2), 
        M5(n/2, n/2), M6(n/2, n/2), M7(n/2, n/2); // Initialize matrices before executing parallel sections
    #pragma omp parallel
    {     
    // Chose parallel sections because this is a set of structured blocks with work that can be distributed among threads in a team
    // From reference card: parallel sections [2.10.2] [2.6.2] = Shortcut for specifying a parallel construct containing one sections construct and no other statements
    #pragma omp single nowait
    {
    // Compute products M1, M2, ..., M7
        #pragma omp task shared(M1)  // each structured block will be distributed amongst threads
        M1 = Matrix::strassens_product(Matrix::addition(A11,A22), Matrix::addition(B11,B22));
        #pragma omp task shared(M2) 
        M2 = Matrix::strassens_product(Matrix::addition(A21,A22), B11); 
        #pragma omp task shared(M3) 
        M3 = Matrix::strassens_product(A11, Matrix::subtraction(B12,B22));
        #pragma omp task shared(M4) 
        M4 = Matrix::strassens_product(A22, Matrix::subtraction(B21,B11));
        #pragma omp task shared(M5) 
        M5 = Matrix::strassens_product(Matrix::addition(A11,A12), B22); 
        #pragma omp task shared(M6) 
        M6 = Matrix::strassens_product(Matrix::subtraction(A21,A11), Matrix::addition(B11,B12));
        #pragma omp task shared(M7) 
        M7 = Matrix::strassens_product(Matrix::subtraction(A12,A22), Matrix::addition(B21,B22));
        
        #pragma omp taskwait
    }
    }

	// Compute blocks of C: C11, C12, C21, C22
	Matrix C11a = Matrix::addition(M1,M4);
	Matrix C11b = Matrix::subtraction(M7,M5);
        Matrix C11 = Matrix::addition(C11a, C11b);

        Matrix C12 = Matrix::addition(M3,M5);
        Matrix C21 = Matrix::addition(M2,M4);

	Matrix C22a = Matrix::subtraction(M1,M2);
	Matrix C22b = Matrix::addition(M3,M6);
        Matrix C22 = Matrix::addition(C22a, C22b);

	// Create the product matrix C
	Matrix C(n,n); 
        C.update_submatrix(C11, 0, n/2-1, 0, n/2-1);
        C.update_submatrix(C12, 0, n/2-1, n/2, n-1);
        C.update_submatrix(C21, n/2, n-1, 0, n/2-1);
        C.update_submatrix(C22, n/2, n-1, n/2, n-1);

        return C;
    }

}
 
// Standard matrix product
// - return C = A * B
Matrix Matrix::standard_product(Matrix& A, Matrix& B) {
    if (A.ncols != B.nrows) matrix_error(5); 
    Matrix C(A.nrows,B.ncols); 

    // Parallelize the outer loop
    // From reference sheet: parallel loop [2.10.1] [2.6.1] = Shortcut for specifying a parallel construct containing one or more associated loops and no other statements.
    #pragma omp parallel for 
    {  //don't use collapse clause because it could cause race condition while trying to parallelize, since inner-most loop depends on k
        for (int i = 0; i < C.nrows; i++) {
            for (int j = 0; j < C.ncols; j++) {
	            C.elements[i][j] = 0.0;
	            for (int k = 0; k < A.ncols; k++) 
	                C.elements[i][j] += A.elements[i][k]*B.elements[k][j];
	}
    } 
    }
    return C;
}

// Standard matrix addition
// - return C = A + B
Matrix Matrix::addition(Matrix& A, Matrix& B) {
    if (A.nrows != B.nrows) A.matrix_error(8); 
    if (A.ncols != B.ncols) A.matrix_error(9); 
    Matrix C(A.nrows,A.ncols);  
    // Parallelize loops
    // From reference sheet: loop [2.7.1] [2.5.1] = Specifies that the iterations of associated loops will be executed in parallel by threads in the team in the context of their implicit tasks.
    #pragma omp parallel for collapse(2) 
    {  // In this scenario, we can use collapse clause because the loop iterations are independent & do not depend on k
       // Multi-loop iterations can safely be split between threads
        for (int i = 0; i < C.nrows; i++) {
            for (int j = 0; j < C.ncols; j++) {
	            C.elements[i][j] = A.elements[i][j]+B.elements[i][j];
	}
    } 
    }
    
    return C;
}

// Standard matrix subtraction
// - return C = A - B
Matrix Matrix::subtraction(Matrix& A, Matrix& B) {
    if (A.nrows != B.nrows) A.matrix_error(88); 
    if (A.ncols != B.ncols) A.matrix_error(98); 
    Matrix C(A.nrows,A.ncols);
    // Parallelize loops  
    // From reference sheet: loop [2.7.1] [2.5.1] = Specifies that the iterations of associated loops will be executed in parallel by threads in the team in the context of their implicit tasks.
    #pragma omp parallel for collapse(2) 
    {  // In this scenario, we can use collapse clause because the loop iterations are independent & do not depend on k
       // Multi-loop iterations can safely be split between threads
    for (int i = 0; i < C.nrows; i++) {
        for (int j = 0; j < C.ncols; j++) {
	    C.elements[i][j] = A.elements[i][j]-B.elements[i][j];
    }
	}
    } 
    return C;
}

// Compare if matrix is identical to another by checking if 
// their elements are identical within specified tolerance
int Matrix::compare_matrix(Matrix& A, Matrix& B) {
    int error = 0;
    if (A.nrows != B.nrows) return error; 
    if (A.ncols != B.ncols) return error; 
    for (int i = 0; i < A.nrows; i++) {
        for (int j = 0; j < A.ncols; j++) {
            if (fabs(A.elements[i][j] - B.elements[i][j]) > TOL) error = 1;
	}
    }
    return error;
}

// Extract submatrix of A
// - return S = A[row_first:row_last][col_first:col_last]
Matrix Matrix::extract_submatrix(int row_first, int row_last, 
			  	 int col_first, int col_last) {
    Matrix S(row_last-row_first+1, col_last-col_first+1); 
    for (int i = 0; i < S.nrows; i++) {
        for (int j = 0; j < S.ncols; j++) {
	    S.elements[i][j] = elements[row_first+i][col_first+j];
	}
    } 
    return S;
}

// Update submatrix of A with matrix S
// - Copy S into A[row_first:row_last][col_first:col_last]
void Matrix::update_submatrix(Matrix& S, int row_first, int row_last, 
					int col_first, int col_last) {
    if (S.nrows != (row_last-row_first+1)) S.matrix_error(100);
    if (S.ncols != (col_last-col_first+1)) S.matrix_error(101);
    if (nrows < row_last) S.matrix_error(102); 
    if (ncols < col_last) S.matrix_error(103); 
    for (int i = 0; i < S.nrows; i++) {
        for (int j = 0; j < S.ncols; j++) {
	    elements[row_first+i][col_first+j] = S.elements[i][j];
	}
    } 
}

// Initialize matrix
// - for testing purpose only
void Matrix::initialize_matrix(double factor) {
    for (int i = 0; i < nrows; i++) {
        for (int j = 0; j < ncols; j++) {
            elements[i][j]= i + factor*j;
	}
    }
}

// Print matrix
void Matrix::print_matrix() {
    printf("\n... Printing matrix ... \n");
    for (int i = 0; i < nrows; i++) {
        for (int j = 0; j < ncols; j++) {
   	    printf(" %8.4f",elements[i][j]);
	}
	printf("\n");
    }
}

// Generic error
void Matrix::matrix_error(int error_number) {
    printf("Error encountered: %d ... aborting\n", error_number);  
    exit(0);
}

// Create new matrix
Matrix::Matrix(int num_rows, int num_cols) {
    nrows = num_rows;
    ncols = num_cols;
    elements = new double *[nrows];
    double * array = new double[nrows*ncols];
    for (int i = 0; i < nrows; i++) elements[i] = &(array[i*ncols]);
}

// =======================================================================
int main(int argc, char *argv[]) {
    time_t start, end;
    double standard_time, strassens_time;

    // Read input, validate
    if (argc != 3) {
        printf("Need two integers as input \n"); 
        printf("Use: <executable_name> <log_2(matrix_size)> <log_2(leaf_matrix_size)>\n"); 
        exit(0);
    }   
    int k = atoi(argv[argc-2]);
    matrix_size = (1 << k); 
    if (matrix_size > MAX_MATRIX_SIZE) {
        printf("Maximum matrix size allowed: %d.\n", MAX_MATRIX_SIZE);
        exit(0);
    };  
    int q = atoi(argv[argc-1]);
    leaf_matrix_size = (1 << q); 
    if (leaf_matrix_size > matrix_size) {
        printf("Leaf matrix size too large, setting to matrix size ...\n");
        leaf_matrix_size = matrix_size;
    };  

    // Initialize matrices A and B 
    Matrix A(matrix_size,matrix_size); A.initialize_matrix(1.0); 
    Matrix B(matrix_size,matrix_size); B.initialize_matrix(-1.0); 

    // ----------------------------------
    // Strassen's matrix multiplication
    start = omp_get_wtime();
    Matrix C = Matrix::strassens_product(A,B); 
    strassens_time = omp_get_wtime() - start;

    printf("Matrix size = %d, Leaf matrix size = %d, Strassen's (s) = %8.4f s,",
            matrix_size, leaf_matrix_size, strassens_time);

    // ----------------------------------
    // Standard matrix multiplication

    // Compute Cstd = A*B - standard matrix multiplication
    start = omp_get_wtime();
    // Matrix Cstd = Matrix::standard_product(A,B); 
    time(&end); 
    standard_time = omp_get_wtime() - start;

    printf(" Standard = %8.4f s,", standard_time);

    int error = Matrix::compare_matrix(C,Cstd);
    printf(" Error = %d\n", error);

    if (error != 0) {
        printf("Houston, we have a problem!\n");
    }
}

