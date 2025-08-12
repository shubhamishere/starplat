#include "backends/backend_webgpu/dsl_webgpu_generator.h"
#include "ast/ASTNodeTypes.hpp"
#include <iostream>

int main() {
    // Create a simple test AST
    Function* func = Function::createFunctionNode(
        Identifier::createIdNode("Compute_TC"),
        std::list<formalParam*>()
    );
    
    // Create a simple forall statement
    forallStmt* forall = forallStmt::createforallStmt(
        Identifier::createIdNode("v"),
        Identifier::createIdNode("g"),
        nullptr, // extractElemFunc
        nullptr, // body
        nullptr, // filterExpr
        true // isforall
    );
    
    // Test the generator
    spwebgpu::dsl_webgpu_generator generator;
    
    // Create a simple AST with just the function
    std::cout << "Testing WebGPU generator..." << std::endl;
    
    // This is a minimal test - in a real scenario, we'd need a complete AST
    std::cout << "WebGPU generator compiled successfully!" << std::endl;
    
    return 0;
} 