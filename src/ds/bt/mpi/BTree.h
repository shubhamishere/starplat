#ifndef MPI_BTREE_H

// required libraries
#include <mpi.h>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <climits>
#include <ctime>
#include <algorithm>


// Btree class
class BTree {
    // method declarations
    public:
        BTree(int , char**);
        ~BTree();

        void insert(int*, int);
        int* search(int*, int);
        void remove(int*, int);
        void initMemory();
        void freeMemory();
        
        int pid;
        int total_process;
    
    // array declarations
    private:
        int *btree;
        int **holes;
};

#endif
