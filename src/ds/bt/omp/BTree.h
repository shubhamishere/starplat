#ifndef OMP_BTREE_H
#define OMP_BTREE_H

// required libraries
#include <omp.h>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <climits>
#include <ctime>
#include <algorithm>


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
    // user method declarations
    public:
        btree();
        ~btree();

        void insertNode(int, int);
        int* search(int);
        void remove(int);

        void batchInsert(Node*, int);
        int* batchSearch(int*, int);
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
        
        void parallelInsert(Node*, int**, int*, int*, Node*, int);
        int* parallelSearch(Node*, int*, int , int*);
        void parallelRemove(Node*, int**, int*, int*, int);

        Node *btreeArray;
        int** holes;
        int *holesCount;
        int *chunkElementCounter;
};

#endif
