/*
    This file is made as a sample structure on how can the mpi Btree implementation be used.
    Any user can mimic this code structure of the object creation, batch formation and function calls
*/

// including the header file
#include "BTree.h"

// other imports for file reading and printing
#include <string.h>
#include <iostream>
#include <sstream>
#include <fstream>

// defining the batch size
#define BATCH_SIZE 30000

using namespace std;


// main function
int main(int argc, char** argv) {
    // creating a btree object
    BTree BT (argc, argv);

    // declaring the batches and allocating memory for them
    int *addBatch;
    int *deleteBatch;
    int *searchBatch;
    int addCount;
    int deleteCount;
    int searchCount;

    addCount = 0;
    deleteCount = 0;
    searchCount = 0;

    addBatch = (int*)malloc(BATCH_SIZE * sizeof(int));
    searchBatch = (int*)malloc(BATCH_SIZE * sizeof(int));
    deleteBatch = (int*)malloc(BATCH_SIZE * sizeof(int));
    
    // file reading starts
    ifstream inputFile(argv[1]);

    char command;
    int num;

    while (inputFile >> command >> num) {
        // insertion
        if (command == 'a') {
            // batch formation
            addBatch[addCount++] = num;
            
            if (addCount == BATCH_SIZE) {
                // function call
                BT.insert(addBatch, addCount);
                addCount = 0;
            }
        }
        // searching
        else if(command == 's') {
            // batch formation
            searchBatch[searchCount++] = num;

            if (searchCount == BATCH_SIZE) {
                // function call
                int *searchResult = BT.search(searchBatch, searchCount);
                
                // the following code can be uncommented to print the result of searching
                // if (BT.pid == 0) {
                //     for (int i = 0; i < searchCount; i++) {
                //         if(searchResult[i] == INT_MAX)  std::cout << "Element " << searchBatch[i] << " is not found" << std::endl;
                //         else std::cout << "Element " << searchBatch[i] << " found at index " << searchResult[i] << std::endl;
                //     }
                // }
                
                searchCount = 0;
            }
        }
        // deletion
        else if (command == 'd') {
            // batch formation
            deleteBatch[deleteCount++] = num;

            if (deleteCount == BATCH_SIZE) {
                // function call
                BT.remove(deleteBatch, deleteCount);
                deleteCount = 0;
            }
        }
    }

    // file reading ends
    inputFile.close();

    // if some elements are left to be operated on (since the batch size was not full) then operate on them

    // insertion
    if (addCount > 0)
        BT.insert(addBatch, addCount);

    // searching
    if (searchCount > 0) {
        int *searchResult = BT.search(searchBatch, searchCount);
        
        // the following code can be uncommented to print the result of searching
        // if (BT.pid == 0) {
        //     for (int i = 0; i < searchCount; i++) {
        //         if(searchResult[i] < 0)  std::cout << "Element " << searchBatch[i] << " is not found" << std::endl;
        //         else std::cout << "Element " << searchBatch[i] << " found at index " << searchResult[i] << std::endl;
        //     }
        // }
    }

    // deletion
    if (deleteCount > 0)
        BT.remove(deleteBatch, deleteCount);

    return 0;
}