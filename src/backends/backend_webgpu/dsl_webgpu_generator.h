#ifndef DSL_WEBGPU_GENERATOR_H
#define DSL_WEBGPU_GENERATOR_H

#include "../../ast/ASTNode.hpp"
#include <string>
#include <fstream>
#include <vector>
    
namespace spwebgpu {

class dsl_webgpu_generator {
public:
    dsl_webgpu_generator();
    ~dsl_webgpu_generator();

    void generate(ASTNode* root, const std::string& outFile);

private:
    std::vector<ASTNode*> parallelConstruct;
    int kernelCounter = 0;
    void generateFunc(ASTNode* node, std::ofstream& out);
    void generateBlock(ASTNode* node, std::ofstream& out);
    void generateStatement(ASTNode* node, std::ofstream& out);
    void generateExpr(ASTNode* node, std::ofstream& out);
    void emitWGSLKernel(const std::string& baseName, ASTNode* forallBody);
    void generateWGSLStatement(ASTNode* node, std::ofstream& wgslOut, const std::string& indexVar);
    void generateWGSLExpr(ASTNode* node, std::ofstream& wgslOut, const std::string& indexVar);
    // Add more methods as needed for WebGPU specifics
    static std::string getOpString(int opType);
};

} // namespace spwebgpu

#endif // DSL_WEBGPU_GENERATOR_H 