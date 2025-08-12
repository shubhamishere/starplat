#ifndef DSL_WEBGPU_GENERATOR_H
#define DSL_WEBGPU_GENERATOR_H

#include "../../ast/ASTNode.hpp"
#include "../../ast/ASTNodeTypes.hpp"
#include <string>
#include <fstream>
#include <vector>

namespace spwebgpu {

class dsl_webgpu_generator {
public:
  dsl_webgpu_generator();
  ~dsl_webgpu_generator();

  // Entry point: generate JavaScript host + WGSL kernels for a function AST
  void generate(ASTNode* root, const std::string& outFile);

private:
  int kernelCounter = 0;

  void generateFunc(ASTNode* node, std::ofstream& out);
  void generateBlock(ASTNode* node, std::ofstream& out);
  void generateStatement(ASTNode* node, std::ofstream& out);
  void generateExpr(ASTNode* node, std::ofstream& out);
  void generateFixedPoint(fixedPointStmt* fp, std::ofstream& out);
  void generateDoWhile(dowhileStmt* dw, std::ofstream& out);
  // Generate host-side sequencing for statements; replaces forall with kernel launches
  void generateHostBody(ASTNode* node, std::ofstream& out, int& launchIndex);

  void emitWGSLKernel(const std::string& baseName, ASTNode* forallBody);
          void generateWGSLStatement(ASTNode* node, std::ofstream& wgslOut, const std::string& indexVar, int indentLevel = 1);
  void generateWGSLExpr(ASTNode* node, std::ofstream& wgslOut, const std::string& indexVar);
  void generateProcCall(proc_callStmt* stmt, std::ofstream& out);
  void generatePropertyAccess(PropAccess* prop, std::ofstream& wgslOut, const std::string& indexVar);
  void generateReductionStmt(reductionCallStmt* stmt, std::ofstream& out);

  static std::string getOpString(int opType);
};

} // namespace spwebgpu

#endif // DSL_WEBGPU_GENERATOR_H

