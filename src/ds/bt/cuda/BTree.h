#ifndef CUDA_BTREE_H

// required libraries
#include <cuda.h>
#include <cstring>
#include <cstdlib>
#include <cmath>


// Btree class
class BTree {
    // method declarations
    public:
        BTree(int);
        ~BTree();

        void insert(int*, int);
        int* search(int*, int);
        void remove(int*, int);
        void initMemory(int);
        void freeMemory();
    
    // array declarations
    private:
        int *btree;
        int **holes;
        int *elementCounter;
        unsigned int *lastHoleCounter;
        
        int insertionChunkPos;
        
        int *insertionChunksHost;
};

#endif
