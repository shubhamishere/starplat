#ifndef OMP_BTREE_H

// required libraries
#include <omp.h>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <climits>
#include <ctime>
#include <algorithm>


// Btree class
class BTree {
    // user method declarations
    public:
        BTree();
        ~BTree();

        void insert(int*, int);
        int* search(int*, int);
        void remove(int*, int);
        void initMemory();
        void freeMemory();
    
    // arrays and parallel method declarations
    private:
        void parallelInsert(int*, int**, int*, int*, int*, int);
        int* parallelSearch(int*, int*, int , int*);
        void parallelRemove(int*, int**, int*, int*, int);

        int *btree;
        int** holes;
        int *holesCount;
        int *chunkElementCounter;
};

#endif
