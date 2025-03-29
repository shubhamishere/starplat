// including the header file
#include "BTree.h"

// The different parameters for the code
#define BLOCK_SIZE 1024
#define SEARCH_INNER_KERNEL 2
#define INSERTION_KERNEL 2
#define MAX_BTREE_SIZE 20000000
#define MAX_HOLES_SIZE 40000
#define CHUNK_SIZE 10000
#define MAX_VALUE_RANGE 10000000


// kernel to perform parallel insertion
__global__ void insertion(int *btree, int **holes, int *elementCounter, int *batchElements, int *insertionChunks, unsigned int *lastHoleCounter, int addCount){
    /*
        Parameters:
            btree : pointer to btree array
            holes : pointer to 2D holes array
            elementCounter : pointer to an array that stores the current total number of elements in each chunk
            batchElements : pointer to the batch to insert
            insertionChunks : pointer to an array that stores the chunk (randomly generated) in the cluster where each element should be inserted
            lastHoleCounter : pointer to an array that stores the number of elements present in the holes array for the current cluster
            addCount : an integer denoting the number of elements in the current batch to be inserted
    */

    // calculating the thread id
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    // running a loop for the number of elements that each thread should insert
    for (int i = 0; i < INSERTION_KERNEL; i++) {
        // adding out of bounds check
        if ((tid * INSERTION_KERNEL + i) >= addCount) break;

        // calculating the cluster where the element should be inserted
        int elem = batchElements[tid * INSERTION_KERNEL + i];
        int cluster = abs((elem + MAX_VALUE_RANGE) / (4 * CHUNK_SIZE));

        // a variable to be used to acquire lock on the holes array
        int hole = INT_MIN;

        // calculating the chunk bounds for the element
        int chunk = insertionChunks[tid * INSERTION_KERNEL + i];
        int chunkStart = (cluster * 4 * CHUNK_SIZE) + (chunk * CHUNK_SIZE);
        int chunkEnd = chunkStart + CHUNK_SIZE;

        // a boolean to check if any hole was successfully occupied or not
        bool elementInserted = false;

        // running a loop over the holes array for the selected cluster to find a hole (if any)
        for(int index = 0; index < lastHoleCounter[cluster]; index++){
            if(holes[cluster][index] >= chunkStart && holes[cluster][index] < chunkEnd){
                // gathering the hole index
                hole = holes[cluster][index];

                //acquiring lock and inserting the element
                int oldVal = atomicCAS(&holes[cluster][index], hole, INT_MIN);
                if(oldVal == hole){
                    btree[hole] = elem;
                    elementInserted = true;
                    break;
                }
            }
        }

        
        int chunkIndex = cluster * 4 + chunk;
        
        // if the hole was not found then add to a new index in the btree
        while(!elementInserted) {
            // check if the chunk is already full or not (if yes then move to the next chunk)
            if(elementCounter[chunkIndex] == CHUNK_SIZE) {
                if (chunk == 3) chunk = 0;
                else chunk++;
                chunkIndex = cluster * 4 + chunk;
                continue;
            }

            // insert the element to the btree by calculating the correct index
            int insertChunkIndex = atomicAdd(&elementCounter[chunkIndex], 1);
            int insertionIndex = cluster * 4 * CHUNK_SIZE + chunk * CHUNK_SIZE + insertChunkIndex;
            btree[insertionIndex] = elem;
            elementInserted = true;
        }
    }
}


// child kernel to perform parallel searching
__global__ void searching(int *btree, int *batchElements, int *searchResult, int searchResultIndex, int searchCount) {
    /*
        Parameters:
            btree : pointer to btree array
            batchElements : pointer to the batch to search
            searchResult : pointer to an array that stored the result of searching
            searchResultIndex : index of the batch for which the child kernel was launched
            searchCount : an integer denoting the number of elements in the current batch to be searched
    */

    // calculating the thread id
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    // running a loop over the number of elements each thread is supposed to search
    for (int i = 0; i < SEARCH_INNER_KERNEL; i++) {
        // adding out of bounds check
        if ((searchResultIndex * SEARCH_INNER_KERNEL + i) >= searchCount) break;

        int num = batchElements[searchResultIndex * SEARCH_INNER_KERNEL + i];

        int clusterIndex = abs((num + MAX_VALUE_RANGE) / (4 * CHUNK_SIZE));

        // if the element is found in the btree then add that index into result array
        if(btree[clusterIndex * 4 * CHUNK_SIZE + tid] == num) {
            searchResult[searchResultIndex * SEARCH_INNER_KERNEL + i] = clusterIndex * 4 * CHUNK_SIZE + tid;
        }
    }
}


// parent kernel to perform parallel searching
__global__ void batchSearch(int *btree, int *batchElements, int *searchResult, int searchCount) {
    /*
        Parameters:
            btree : pointer to btree array
            batchElements : pointer to the batch to search
            searchResult : pointer to an array that stored the result of searching
            searchCount : an integer denoting the number of elements in the current batch to be searched
    */

    // calculating the thread id
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    // running a loop over the number of elements each child thread is supposed to search for
    for (int i = 0; i < SEARCH_INNER_KERNEL; i++) {
        if ((tid * SEARCH_INNER_KERNEL + i) >= searchCount) break;
        searchResult[tid * SEARCH_INNER_KERNEL + i] = INT_MIN;
    }

    // launching the child kernel
    dim3 blocks = dim3(ceil(4 * CHUNK_SIZE / (double)BLOCK_SIZE), 1, 1);
    searching<<<blocks, BLOCK_SIZE>>> (btree, batchElements, searchResult, tid, searchCount);
}


// parent kernel to perform parallel deleting
__global__ void deletion(int *btree, int *batchElements , int *deletionIndex, int **holes, unsigned int *lastHoleCounter, int deleteCount){
     /*
        Parameters:
            btree : pointer to btree array
            batchElements : pointer to the batch to delete
            deletionIndex : pointer to array that stores the index in the btree where the element is located (so that deletion can be done there)
            holes : pointer to 2D holes array
            lastHoleCounter : pointer to an array that stores the number of elements present in the holes array for the current cluster
            deleteCount : an integer denoting the number of elements in the current batch to be deleted
    */

    // calculating the thread id
    int tid = blockDim.x * blockIdx.x + threadIdx.x; 

    int num = batchElements[tid];

    // setting the deletion index to INT_MIN (that will indicate that the element to be deleted does not exist in the btree)
    deletionIndex[tid] = INT_MIN;

    int clusterIndex = abs((num + MAX_VALUE_RANGE) / (4 * CHUNK_SIZE));

    // launching the search kernel to search for the element
    dim3 blocks=dim3(ceil(4 * CHUNK_SIZE / (double) BLOCK_SIZE), 1, 1);
    searching<<<blocks, BLOCK_SIZE>>> (btree, batchElements, deletionIndex, tid, deleteCount);

    // if the element actually exists in the btree then delete it
    if(deletionIndex[tid] >= 0) {
        // mark the index in the holes array
        int holeIndex = atomicInc(&lastHoleCounter[clusterIndex], MAX_HOLES_SIZE);
        holes[clusterIndex][holeIndex] = deletionIndex[tid];

        // make that index in the btree as INT_MIN that will indicate that the element no longer exists in the array (this is helpful for searching)
        btree[deletionIndex[tid]] = INT_MIN;
    }
}


// constructor
BTree::BTree(int MAX_SIZE) {
    // calling the function to allocate the memory for the different arrays
    initMemory(MAX_SIZE);
}


// destructor
BTree::~BTree() {
    // calling the function to deallocate the memory for the different arrays
    freeMemory();
}


// function to allocate memory for the different arrays
void BTree::initMemory(int MAX_SIZE) {
    // allocate btree on device and initialoze all indices to 0
    cudaMalloc(&btree, MAX_BTREE_SIZE * sizeof(int));
    cudaMemset(btree, 0, MAX_BTREE_SIZE * sizeof(int));

    // allocate holes array on device
    int numClusters = MAX_BTREE_SIZE / (4 * CHUNK_SIZE);
    cudaMalloc((void**)&holes, numClusters * 4 * CHUNK_SIZE * sizeof(int));

    // allocate the lastHoleCounter array on device
    cudaHostAlloc(&lastHoleCounter, numClusters * sizeof(unsigned int), 0);
    for (int i = 0; i < numClusters; i++) {
        lastHoleCounter[i] = 0;
    }

    // allocate the elementCounter array on device
    cudaMalloc(&elementCounter, ceil(MAX_BTREE_SIZE / (double)CHUNK_SIZE) * sizeof(int));
    cudaMemset(elementCounter, 0, ceil(MAX_BTREE_SIZE / (double)CHUNK_SIZE) * sizeof(int));

    // host side initialization
    insertionChunkPos = 0;
    insertionChunksHost = (int*)malloc(MAX_SIZE * sizeof(int));
}


// function to deallocate memory for the different arrays
void BTree::freeMemory() {
    // device deallocations
    cudaFree(btree);
    cudaFree(holes);
    cudaFree(elementCounter);

    // host deallocations
    free(insertionChunksHost);
}


// insert function for user
void BTree::insert(int *addBatch, int addCount) {
    /*
        Parameters:
            addBatch : pointer to the batch to add
            addCount : an integer denoting the number of elements in the current batch to be added
    */

    // randomly allocating one of the four chunks in the cluster to each element
    for (int index = 0; index < addCount; index++) {
        srand(time(0));
        insertionChunksHost[insertionChunkPos++] = rand() % 4;
    }

    // allocating memory on the device and copying the contents
    int *insertionChunks;
    cudaMalloc(&insertionChunks, addCount * sizeof(int));
    cudaMemcpy(insertionChunks, insertionChunksHost, addCount * sizeof(int), cudaMemcpyHostToDevice);

    int *batchElements;
    cudaMalloc(&batchElements, addCount * sizeof(int));
    cudaMemcpy(batchElements, addBatch, addCount * sizeof(int), cudaMemcpyHostToDevice);

    // launching the kernel
    dim3 threads = addCount <= 1024 ? dim3(addCount, 1, 1) : dim3(1024, 1, 1);
    dim3 blocks = dim3(ceil(addCount / (double) BLOCK_SIZE), 1, 1);

    insertion<<<blocks,threads>>> (btree, holes, elementCounter, batchElements, insertionChunks, lastHoleCounter, addCount);
    cudaDeviceSynchronize();

    // freeing the temporary device arrays
    cudaFree(batchElements);
    cudaFree(insertionChunks);

    insertionChunkPos = 0;
}


// search function for user
int* BTree::search(int *searchBatch, int searchCount) {
    /*
        Parameters:
            searchBatch : pointer to the batch to search
            searchCount : an integer denoting the number of elements in the current batch to be searched
    */

    // allocating memory on the device and copying the contents
    int *batchElements;
    cudaMalloc(&batchElements, searchCount * sizeof(int));
    cudaMemcpy(batchElements, searchBatch, searchCount * sizeof(int), cudaMemcpyHostToDevice);

    int *searchResult;
    cudaMalloc(&searchResult, searchCount * sizeof(int));

    // launching the kernel
    int threads = ceil((double)searchCount / SEARCH_INNER_KERNEL);
    
    batchSearch<<<1, threads>>> (btree, batchElements, searchResult, searchCount);                        
    
    // copying back the results from the device to host
    int *searchResultHost = (int*)malloc(searchCount * sizeof(int));
    
    cudaMemcpy(searchResultHost, searchResult, searchCount * sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    
    // freeing the temporary device arrays
    cudaFree(batchElements);
    cudaFree(searchResult);
    free(searchResultHost);

    return searchResultHost;
}


// delete function for user
void BTree::remove(int *deleteBatch, int deleteCount) {
    /*
        Parameters:
            deleteBatch : pointer to the batch to delete
            deleteCount : an integer denoting the number of elements in the current batch to be deleted
    */

    // allocating memory on the device and copying the contents
    int *deletionIndex;
    cudaMalloc(&deletionIndex, deleteCount * sizeof(int));

    int *batchElements;
    cudaMalloc(&batchElements, deleteCount * sizeof(int));
    cudaMemcpy(batchElements, deleteBatch, deleteCount * sizeof(int), cudaMemcpyHostToDevice);
    
    // launching the kernel
    dim3 blocks = dim3(ceil(4 * CHUNK_SIZE / (double)BLOCK_SIZE), 1, 1);

    deletion<<<1, deleteCount>>> (btree, batchElements, deletionIndex, holes, lastHoleCounter, deleteCount);
    cudaDeviceSynchronize();

    // freeing the temporary device arrays
    cudaFree(batchElements);
    cudaFree(deletionIndex);
}