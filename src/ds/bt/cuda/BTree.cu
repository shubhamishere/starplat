// including the header file
#include "BTree.h"

// The different parameters for the code
#define MAX_SIZE 32
#define SEARCH_BLOCK_SIZE 64
#define BLOCK_SIZE 1024
#define SEARCH_INNER_KERNEL 2
#define INSERTION_KERNEL 2
#define MAX_BTREE_SIZE 20000000
#define MAX_HOLES_SIZE 40000
#define CHUNK_SIZE 10000
#define MAX_VALUE_RANGE 10000000


// kernel to perform parallel insertion
__global__ void insertion(Node *btreeArray, int **holes, int *elementCounter, Node *batchElements, int *insertionChunks, unsigned int *lastHoleCounter, int addCount){
    /*
        Parameters:
            btreeArray : pointer to btree array
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
        int elem = batchElements[tid * INSERTION_KERNEL + i].key;
        int elemValue = batchElements[tid * INSERTION_KERNEL + i].value;

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
                    // inserting the key and the value into the btree node
                    btreeArray[hole].key = elem;
                    btreeArray[hole].value = elemValue;
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

            // icalculating the correct index where the element should be inserted
            int insertChunkIndex = atomicAdd(&elementCounter[chunkIndex], 1);
            int insertionIndex = cluster * 4 * CHUNK_SIZE + chunk * CHUNK_SIZE + insertChunkIndex;
            // inserting the key and the value into the btree node
            btreeArray[insertionIndex].key = elem;
            btreeArray[insertionIndex].value = elemValue;
            elementInserted = true;
        }
    }
}


// child kernel to perform parallel searching
__global__ void parallelSearching(Node *btreeArray, int *batchElements, int *searchResult, int searchResultIndex, int searchCount) {
    /*
        Parameters:
            btreeArray : pointer to btree array
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

        // if the key of the element is found in the btree then add the value corresponding to the key into result array
        if(btreeArray[clusterIndex * 4 * CHUNK_SIZE + tid].key == num) {
            searchResult[searchResultIndex * SEARCH_INNER_KERNEL + i] = btreeArray[clusterIndex * 4 * CHUNK_SIZE + tid].value;
        }
    }
}


// parent kernel to perform parallel searching
__global__ void searching(Node *btreeArray, int *batchElements, int *searchResult, int searchCount) {
    /*
        Parameters:
            btreeArray : pointer to btree array
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
    parallelSearching<<<blocks, BLOCK_SIZE>>> (btreeArray, batchElements, searchResult, tid, searchCount);
}


// child kernel to perform searching and populating the during deletion
__global__ void deletionSearching(Node *btreeArray, int *batchElements, int searchResultIndex, int searchCount) {
    /*
        Parameters:
            btreeArray : pointer to btree array
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

        // if the element is found in the btree then mark that index as deleted (replace by INT_MIN)
        if(btreeArray[clusterIndex * 4 * CHUNK_SIZE + tid].key == num) {
            btreeArray[clusterIndex * 4 * CHUNK_SIZE + tid].key = INT_MIN;
            btreeArray[clusterIndex * 4 * CHUNK_SIZE + tid].value = INT_MIN;
        }
    }
}


// parent kernel to perform parallel deleting
__global__ void deletion(Node *btreeArray, int *batchElements, int **holes, unsigned int *lastHoleCounter, int deleteCount){
     /*
        Parameters:
            btreeArray : pointer to btree array
            batchElements : pointer to the batch to delete
            holes : pointer to 2D holes array
            lastHoleCounter : pointer to an array that stores the number of elements present in the holes array for the current cluster
            deleteCount : an integer denoting the number of elements in the current batch to be deleted
    */

    // calculating the thread id
    int tid = blockDim.x * blockIdx.x + threadIdx.x; 

    // adding out of bounds check
    if (tid >= deleteCount) return;

    // launching the search kernel to search for the element
    dim3 blocks = dim3(ceil(4 * CHUNK_SIZE / (double) BLOCK_SIZE), 1, 1);
    deletionSearching<<<blocks, BLOCK_SIZE>>> (btreeArray, batchElements, tid, deleteCount);
}


// constructor
btree::btree() {
    // calling the function to allocate the memory for the different arrays
    initMemory();
}


// destructor
btree::~btree() {
    // calling the function to deallocate the memory for the different arrays
    freeMemory();
}


// function to allocate memory for the different arrays
void btree::initMemory() {
    // allocate btree on device and initialize all indices to 0
    cudaMalloc(&btreeArray, MAX_BTREE_SIZE * sizeof(Node));
    cudaMemset(btreeArray, 0, MAX_BTREE_SIZE * sizeof(Node));

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

    // allocating memory for the batch arrays
    addBatch = (Node*)malloc(MAX_SIZE * sizeof(Node));
    searchBatch = (int*)malloc(SEARCH_BLOCK_SIZE * sizeof(int));
    deleteBatch = (int*)malloc(MAX_SIZE * sizeof(int));

    // initializing the count of elements in the batches
    addCount = 0;
    searchCount = 0;
    deleteCount = 0;
}


// function to deallocate memory for the different arrays
void btree::freeMemory() {
    // device deallocations
    cudaFree(btreeArray);
    cudaFree(holes);
    cudaFree(elementCounter);

    // host deallocations
    free(insertionChunksHost);
    free(addBatch);
    free(searchBatch);
    free(deleteBatch);
}


// insert function for user to insert a batch
void btree::batchInsert(Node *addBatch, int addCount) {
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

    Node *batchElements;
    cudaMalloc(&batchElements, addCount * sizeof(Node));
    cudaMemcpy(batchElements, addBatch, addCount * sizeof(Node), cudaMemcpyHostToDevice);

    // launching the kernel
    dim3 threads = addCount <= 1024 ? dim3(addCount, 1, 1) : dim3(1024, 1, 1);
    dim3 blocks = dim3(ceil(addCount / (double) BLOCK_SIZE), 1, 1);

    insertion<<<blocks,threads>>> (btreeArray, holes, elementCounter, batchElements, insertionChunks, lastHoleCounter, addCount);
    cudaDeviceSynchronize();

    // freeing the temporary device arrays
    cudaFree(batchElements);
    cudaFree(insertionChunks);

    // resetting the random chunk array count
    insertionChunkPos = 0;
}


// search function for user to search a batch on the host end
int* btree::batchSearchHost(int *searchBatch, int searchCount) {
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
    
    searching<<<1, threads>>> (btreeArray, batchElements, searchResult, searchCount);                        
    
    // copying back the results from the device to host
    int *searchResultHost = (int*)malloc(searchCount * sizeof(int));
    
    cudaMemcpy(searchResultHost, searchResult, searchCount * sizeof(int), cudaMemcpyDeviceToHost);
    
    // freeing the temporary device arrays
    cudaFree(batchElements);
    cudaFree(searchResult);

    // returning the result of the search back to the caller function
    return searchResultHost;
}


// search function for user to search a batch on the device end
int* btree::batchSearchDevice(int *batchElements, int searchCount) {
    /*
        Parameters:
            batchElements : pointer to the batch to search
            searchCount : an integer denoting the number of elements in the current batch to be searched
    */

    // allocating memory on the device and copying the contents
    int *searchResult;
    cudaMalloc(&searchResult, searchCount * sizeof(int));

    // launching the kernel
    int threads = ceil((double)searchCount / SEARCH_INNER_KERNEL);
    
    searching<<<1, threads>>> (btreeArray, batchElements, searchResult, searchCount);                        
    cudaDeviceSynchronize();
    
    // freeing the temporary device arrays
    cudaFree(batchElements);

    // returning the result of the search back to the caller function
    return searchResult;
}


// delete function for user to delete a batch
void btree::batchRemove(int *deleteBatch, int deleteCount) {
    /*
        Parameters:
            deleteBatch : pointer to the batch to delete
            deleteCount : an integer denoting the number of elements in the current batch to be deleted
    */

    // allocating memory on the device and copying the contents
    int *batchElements;
    cudaMalloc(&batchElements, deleteCount * sizeof(int));
    cudaMemcpy(batchElements, deleteBatch, deleteCount * sizeof(int), cudaMemcpyHostToDevice);
    
    // launching the kernel
    int threads = ceil((double)deleteCount / BLOCK_SIZE);

    deletion<<<1, threads>>> (btreeArray, batchElements, holes, lastHoleCounter, deleteCount);
    cudaDeviceSynchronize();

    // freeing the temporary device arrays
    cudaFree(batchElements);
}


// declaring a host device function to insert one key-value pair
__host__ __device__ void btree::insertNode(int key, int value) {
    /*
        Parameters:
            key : key of the element which is to be inserted
            value : value corresponding to the key of the element which is to be inserted
    */

    // creating the node object
    Node newNode(key, value);

    // adding the element to the batch
    addBatch[addCount++] = newNode;

    // calling the batch insert function to insert the element (the size of the batch passed is of one element)
    batchInsert(addBatch, addCount);

    // resetting the batch size count
    addCount = 0;
}


// declaring a host function to search one key-value pair
__host__ int* btree::searchHost(int key) {
    /*
        Parameters:
            key : key of the element which is to be searched
    */

    // adding the key to the batch
    searchBatch[searchCount++] = key;

    // calling the batch search function to search for the key (the size of the batch passed is of one element)
    int* searchResult = batchSearchHost(searchBatch, searchCount);

    // resetting the batch size count
    searchCount = 0;

    // returing back the pointer to the array that stores the result of the search
    return searchResult;
}


// declaring a host function to search one key-value pair
__device__ int* btree::searchDevice(int key) {
    /*
        Parameters:
            key : key of the element which is to be searched
    */

    // adding the key to the batch
    int *batchElements;
    cudaMalloc(&batchElements, sizeof(int));
    batchElements[0] = key;

    // calling the batch search function to search for the key (the size of the batch passed is of one element)
    int* searchResult = batchSearchDevice(batchElements, 1);

    // returing back the pointer to the array that stores the result of the 
    return searchResult;
}


// declaring a host device function to delete one key-value pair
__host__ __device__ void btree::remove(int key) {
    /*
        Parameters:
            key : key of the element which is to be deleted
    */

    // adding the key to the batch
    deleteBatch[deleteCount++] = key;

    // calling the batch search function to search for the key (the size of the batch passed is of one element)
    batchRemove(deleteBatch, deleteCount);

    // resetting the batch size count
    deleteCount = 0;
}