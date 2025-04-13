#ifndef CUDA_BTREE_H
#define CUDA_BTREE_H

// required libraries
#include <cuda.h>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <climits>


// a structure that defines each element node of the btree
struct Node {
    // integers representing the key and the value
    int key;
    int value;

    // constructor
    Node(int k, int v) : key(k), value(v) {};
};


// Btree class
class btree {
    // method declarations
    public:
        btree();
        ~btree();

        void insertNode(int, int);
        int* searchDevice(int);
        int* searchHost(int);
        void remove(int);

        void batchInsert(Node*, int);
        int* batchSearchHost(int*, int);
        int* batchSearchDevice(int*, int);
        void batchRemove(int*, int);

        
        Node *addBatch;
        int *searchBatch;
        int *deleteBatch;
        
        int addCount;
        int searchCount;
        int deleteCount;
        
    private:
        void initMemory();
        void freeMemory();

        Node *btreeArray;
        int **holes;
        int *elementCounter;
        unsigned int *lastHoleCounter;
        
        int insertionChunkPos;
        int *insertionChunksHost;
};

#endif