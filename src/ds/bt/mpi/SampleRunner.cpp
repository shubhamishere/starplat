/*
    This file is made as a sample structure on how can the mpi Btree implementation be used.
    Any user can mimic this code structure of the object creation and function calls
*/

// including the header file
#include "BTree.h"

// other imports for file reading and printing
#include <string.h>
#include <iostream>
#include <sstream>
#include <fstream>

using namespace std;

// main function
int main(int argc, char** argv) {
    // creating a btree object
    btree bt;
    
    // file reading starts
    ifstream inputFile(argv[1]);

    char command;
    int num;

    while (inputFile >> command >> num) {
        // insertion
        if (command == 'a') {
            int value;
            inputFile >> value;
            bt.insertNode(num, value);
        }
        // searching
        else if(command == 's') {
            int *searchResult = bt.search(num);

            if (bt.pid == 0) {
                if(searchResult[0] == INT_MIN)  cout << "Key " << num << " is not found" << endl;
                else cout << "Value " << searchResult[0] << " is found for key " << num << endl;
            }
        }
        // deletion
        else if (command == 'd') {
            bt.remove(num);
        }
    }

    // file reading ends
    inputFile.close();

    return 0;
}