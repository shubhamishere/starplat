// including the header file
#include "BTree.h"

// The different parameters for the code
#define MAX_BTREE_SIZE 20000000
#define MAX_VALUE_RANGE 10000000
#define MAX_HOLES_SIZE 10000000
#define CHUNK_SIZE 100
#define CLUSTER_HOLES_SIZE 20000


// function to perform parallel insertion
void BTree::parallelInsert(int *btree, int **holes, int *holesCount, int *chunkElementCounter, int *addBatch, int addCount) {
    /*
        Parameters:
            btree : pointer to btree array
            holes : pointer to 2D holes array
            holesCount : pointer to an array that stores the number of elements present in the holes array for the current cluster
            chunkElementCounter : pointer to an array that stores the current total number of elements in each chunk
            addBatch : pointer to the batch to insert
            addCount : an integer denoting the number of elements in the current batch to be inserted
    */

    // variables to store the different ids of threads
    int index, tid, numt;

    // parallel execution starts
    // tid and index are private to each thread
    // numt is shared across all threads
    #pragma omp parallel private(tid, index) shared(numt)
    {
        // num represents total number of threads
        numt = omp_get_num_threads();
        // tid represents id of each thread
        tid  = omp_get_thread_num();

        // finding the range of cluster values for each process
        int startClusterValue = (MAX_BTREE_SIZE / numt) * tid;
        int endClusterValue = (MAX_BTREE_SIZE / numt) * (tid + 1) - 1;
        if (tid == numt - 1)
            endClusterValue = MAX_BTREE_SIZE - 1;

        // finding the range of elements from the insert batch on which each thread will work on (done by binary search)
        int* start_itr = std::lower_bound(addBatch, addBatch + addCount, startClusterValue - MAX_VALUE_RANGE);
        int* end_itr = std::upper_bound(addBatch, addBatch + addCount, endClusterValue - MAX_VALUE_RANGE);

        // if the thread has some elements to insert
        if (start_itr != end_itr)
        {
            // the start and end indices in the array
            int startIdx = start_itr - addBatch;
            int endIdx = end_itr - addBatch;

            // the thread is responsible for inserting these elements
            for (index = startIdx; index < endIdx; index++) {
                int elem = addBatch[index];

                // random chunk calculation
                int cluster = abs((elem + MAX_VALUE_RANGE) / (4 * CHUNK_SIZE));             //cluster Size = 4 * chunk size
                srand(time(0));
                int chunk = rand() % 4;
                int chunkStart = (cluster * 4 * CHUNK_SIZE) + (chunk * CHUNK_SIZE);
                int chunkEnd = chunkStart + CHUNK_SIZE - 1;

                int insertIndex = INT_MIN;

                // checking if a hole is present in the cluster
                for (int holeIndex = 0; holeIndex < holesCount[cluster]; holeIndex++) {
                    if(holes[cluster][holeIndex] != INT_MIN && holes[cluster][holeIndex] >= chunkStart && holes[cluster][holeIndex] <= chunkEnd) {
                        // acquiring the hole
                        //no need of synchronization as no other thread will work on this cluster simultaneously
                        insertIndex = holes[cluster][holeIndex];
                        holes[cluster][holeIndex] = INT_MIN;
                    }
                }

                // if no hole is found
                if (insertIndex == INT_MIN) {
                    // iterate over the four chunks in the cluster
                    while (chunkElementCounter[cluster * 4 + chunk] == CHUNK_SIZE) {
                        if (chunk == 3) chunk = 0;
                        else chunk++;
                    }

                    // calculate the index where the element should be inserted
                    insertIndex = cluster * 4 * CHUNK_SIZE + chunk * CHUNK_SIZE + chunkElementCounter[cluster * 4 + chunk];
                    chunkElementCounter[cluster * 4 + chunk]++;
                }

                // insert the element
                btree[insertIndex] = elem;
            }
        }
    }
}


// function to perform parallel searching
int* BTree::parallelSearch (int *btree, int *searchBatch, int searchCount, int *searchResult) {
    /*
        Parameters:
            btree : pointer to btree array
            searchBatch : pointer to the batch to search
            searchCount : an integer denoting the number of elements in the current batch to be searched
            searchResult : pointer to an array that stores the result of the searching
    */

    // variables to store the different ids of threads
    int index, tid, numt;
    
    // parallel execution starts
    // tid and index are private to each thread
    // numt is shared across all threads
    #pragma omp parallel private(tid, index) shared(numt)
    {
        // num represents total number of threads
        numt = omp_get_num_threads();
        // tid represents id of each thread
        tid  = omp_get_thread_num();

        // finding the range of cluster values for each process
        int startClusterValue = (MAX_BTREE_SIZE / numt) * tid;
        int endClusterValue = (MAX_BTREE_SIZE / numt) * (tid + 1) - 1;
        if (tid == numt - 1)
            endClusterValue = MAX_BTREE_SIZE - 1;

        // finding the range of elements from the search batch on which each thread will work on (done by binary search)
        int* start_itr = std::lower_bound(searchBatch, searchBatch + searchCount, startClusterValue - MAX_VALUE_RANGE);
        int* end_itr = std::upper_bound(searchBatch, searchBatch + searchCount, endClusterValue - MAX_VALUE_RANGE);

        // if the thread has some elements to search
        if (start_itr != end_itr)
        {
            // the start and end indices in the array
            int startIdx = start_itr - searchBatch;
            int endIdx = end_itr - searchBatch;
            
            // the thread is responsible for searching these elements
            for (index = startIdx; index < endIdx; index++) {
                int elem = searchBatch[index];
                
                // calculating the cluster where the element should be present
                int cluster = abs((elem + MAX_VALUE_RANGE) / (4 * CHUNK_SIZE));
                int clusterStart = cluster * 4 * CHUNK_SIZE;
                int clusterEnd = clusterStart + 4 * CHUNK_SIZE;

                bool searchFlag = false;

                //searching over the cluster for the element
                for (int searchIndex = clusterStart; searchIndex < clusterEnd; searchIndex++) {
                    if (btree[searchIndex] == elem) {
                        // store the result of successful searching into the searchResult array
                        searchResult[index] = searchIndex;
                        searchFlag = true;
                        break;
                    }
                }

                // store the result of unsuccessful searching into the searchResult array
                if (!searchFlag) {
                    searchResult[index] = INT_MIN;
                }
            }
        }
    }

    return searchBatch;
}


// function to perform parallel deletion
void BTree::parallelRemove (int *btree, int **holes, int *holesCount, int *deleteBatch, int deleteCount) {
    /*
        Parameters:
            btree : pointer to btree array
            holes : pointer to 2D holes array
            holesCount : pointer to an array that stores the number of elements present in the holes array for the current cluster
            deleteBatch : pointer to the batch to delete
            deleteCount : an integer denoting the number of elements in the current batch to be deleted
    */

    // variables to store the different ids of threads
    int index, tid, numt;

    // parallel execution starts
    // tid and index are private to each thread
    // numt is shared across all threads
    #pragma omp parallel private(tid, index) shared(numt)
    {
        // num represents total number of threads
        numt = omp_get_num_threads();
        // tid represents id of each thread
        tid  = omp_get_thread_num();
        
        // finding the range of cluster values for each process
        int startClusterValue = (MAX_BTREE_SIZE / numt) * tid;
        int endClusterValue = (MAX_BTREE_SIZE / numt) * (tid + 1) - 1;
        if (tid == numt - 1)
            endClusterValue = MAX_BTREE_SIZE - 1;

        // finding the range of elements from the delete batch on which each thread will work on (done by binary search)
        int* start_itr = std::lower_bound(deleteBatch, deleteBatch + deleteCount, startClusterValue - MAX_VALUE_RANGE);
        int* end_itr = std::upper_bound(deleteBatch, deleteBatch + deleteCount, endClusterValue - MAX_VALUE_RANGE);

        // if the thread has some elements to delete
        if (start_itr != end_itr)
        {
            // the start and end indices in the array
            int startIdx = start_itr - deleteBatch;
            int endIdx = end_itr - deleteBatch;

            // the thread is responsible for deleting these elements
            for (index = startIdx; index < endIdx; index++) {
                int elem = deleteBatch[index];

                // calculating the cluster where the element should be present
                int cluster = abs((elem + MAX_VALUE_RANGE) / (4 * CHUNK_SIZE));
                int clusterStart = cluster * 4 * CHUNK_SIZE;
                int clusterEnd = clusterStart + 4 * CHUNK_SIZE;

                //searching over the cluster for the element
                for (int searchIndex = clusterStart; searchIndex < clusterEnd; searchIndex++) {
                    if (btree[searchIndex] == elem) {
                        //no need of synchronization as no other thread will work on this cluster simultaneously
                        // mark the index in the holes array
                        holes[cluster][holesCount[cluster]] = searchIndex;
                        // mark the index as deleted in the btree array
                        btree[searchIndex] = INT_MIN;
                        // increment the holes count
                        if (holesCount[cluster] < CLUSTER_HOLES_SIZE)
                            holesCount[cluster]++;
                        break;
                    }
                }
            }
        }
    }
}


// constructor
BTree::BTree() {
    // calling the function to allocate the memory for the different arrays
    initMemory();
}


// destructor
BTree::~BTree() {
    // calling the function to deallocate the memory for the different arrays
    freeMemory();
}


// function to allocate memory for the different arrays
void BTree::initMemory() {
    // allocate memory for btree
    btree = (int*) malloc (MAX_BTREE_SIZE * sizeof(int));
    
    // allocate memory for holes
    int numClusters = ceil((double)MAX_BTREE_SIZE / (4 * CHUNK_SIZE));
    holes = (int**)malloc(numClusters * sizeof(int*));

    // allocate memory for holesCount and initilize them to 0
    holesCount = (int*)malloc(numClusters * sizeof(int));
    for (int i = 0; i < numClusters; i++) {
        holes[i] = (int*)malloc(CLUSTER_HOLES_SIZE * sizeof(int));
        holesCount[i] = 0;
    }
    
    // allocate memory for chunkElementCounter
    chunkElementCounter = (int*) malloc ((ceil(MAX_BTREE_SIZE/(double)CHUNK_SIZE)*sizeof(int)) * sizeof(int));
}


// function to deallocate memory for the different arrays
void BTree::freeMemory() {
    free (btree);
    free (holes);
    free (holesCount);
    free (chunkElementCounter);
}


// insert function for user
void BTree::insert(int* addBatch, int addCount) { 
    /*
        Parameters:
            addBatch : pointer to the batch to add
            addCount : an integer denoting the number of elements in the current batch to be added
    */

    // sort the batch (to allow binary searching)
    std::sort(addBatch, addBatch + addCount);

    // parallel insert function call
    parallelInsert(btree, holes, holesCount, chunkElementCounter, addBatch, addCount);
}


// search function for user
int* BTree::search(int* searchBatch, int searchCount) {
    /*
        Parameters:
            searchBatch : pointer to the batch to search
            searchCount : an integer denoting the number of elements in the current batch to be searched
    */

    // sort the batch (to allow binary searching)
    std::sort(searchBatch, searchBatch + searchCount);

    // allocate memory for the result array
    int *searchResult = (int*)malloc(searchCount * sizeof(int));

    // parallel search function call
    searchResult = parallelSearch (btree, searchBatch, searchCount, searchResult);

    return searchResult;
}


// delete function for user
void BTree::remove(int* deleteBatch, int deleteCount) {
    /*
        Parameters:
            deleteBatch : pointer to the batch to delete
            deleteCount : an integer denoting the number of elements in the current batch to be deleted
    */

    // sort the batch (to allow binary searching)
    std::sort(deleteBatch, deleteBatch + deleteCount);

    // parallel delete function call
    parallelRemove (btree, holes, holesCount, deleteBatch, deleteCount);
}