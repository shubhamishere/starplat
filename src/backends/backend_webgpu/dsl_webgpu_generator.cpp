#include "dsl_webgpu_generator.h"
#include <iostream>
#include "../../ast/ASTNodeTypes.hpp"

namespace spwebgpu {

namespace {
// Find a reduction target identifier inside a block (first occurrence)
static Identifier* findFirstReductionTargetId(ASTNode* node) {
  if (!node) return nullptr;
  if (node->getTypeofNode() == NODE_REDUCTIONCALLSTMT) {
    reductionCallStmt* r = static_cast<reductionCallStmt*>(node);
    if (r->isLeftIdentifier()) return r->getLeftId();
    if (r->isTargetId()) return r->getTargetId();
  }
  if (node->getTypeofNode() == NODE_BLOCKSTMT) {
    blockStatement* b = static_cast<blockStatement*>(node);
    for (statement* s : b->returnStatements()) {
      Identifier* id = findFirstReductionTargetId(s);
      if (id) return id;
    }
  }
  return nullptr;
}
}

dsl_webgpu_generator::dsl_webgpu_generator() {}
dsl_webgpu_generator::~dsl_webgpu_generator() {}

void dsl_webgpu_generator::generate(ASTNode* root, const std::string& outFile) {
  if (!root) {
    std::cerr << "[WebGPU] Error: root ASTNode is null!" << std::endl;
    return;
  }
  std::ofstream out(outFile);
  if (!out.is_open()) {
    std::cerr << "[WebGPU] Failed to open output file: " << outFile << std::endl;
    return;
  }
  std::cout << "[WebGPU] Starting code generation for: " << outFile << std::endl;
  generateFunc(root, out);
  out.close();
}

void dsl_webgpu_generator::generateFunc(ASTNode* node, std::ofstream& out) {
  if (!node || node->getTypeofNode() != NODE_FUNC) return;
  Function* func = static_cast<Function*>(node);
  std::string funcName = func->getIdentifier() ? func->getIdentifier()->getIdentifier() : "unnamed";
  
  // Reset kernel counter for a fresh generation
  kernelCounter = 0;
  // Pre-scan: emit WGSL kernels for each forall encountered in order
  if (func->getBlockStatement()) {
    blockStatement* block = static_cast<blockStatement*>(func->getBlockStatement());
    for (statement* stmt : block->returnStatements()) {
      if (stmt->getTypeofNode() == NODE_FORALLSTMT) {
        std::string kernelName = "kernel_" + std::to_string(kernelCounter);
        forallStmt* fa = static_cast<forallStmt*>(stmt);
        emitWGSLKernel(kernelName, fa->getBody());
        kernelCounter++;
      }
      if (stmt->getTypeofNode() == NODE_FIXEDPTSTMT) {
        // recurse one level: generate kernels for nested foralls in fixedPoint
        fixedPointStmt* fp = static_cast<fixedPointStmt*>(stmt);
        if (fp->getBody() && fp->getBody()->getTypeofNode() == NODE_BLOCKSTMT) {
          blockStatement* fpb = static_cast<blockStatement*>(fp->getBody());
          for (statement* s2 : fpb->returnStatements()) {
            if (s2->getTypeofNode() == NODE_FORALLSTMT) {
              std::string kernelName2 = "kernel_" + std::to_string(kernelCounter);
              forallStmt* fa2 = static_cast<forallStmt*>(s2);
              emitWGSLKernel(kernelName2, fa2->getBody());
              kernelCounter++;
            }
          }
        }
      }
    }
  }
  
  // Now generate the JavaScript host function
  // Determine appropriate variable names based on function
  // Use a single generic result variable for all algorithms
  std::string resultVar = "result";
  
  out << "export async function " << funcName << "(device, adj_dataBuffer, adj_offsetsBuffer, nodeCount) {\n";
  out << "  let " << resultVar << " = 0;\n";
  // Allocate shared result and properties buffers once
  out << "  const resultBuffer = device.createBuffer({ size: 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });\n";
  out << "  const propertyBuffer = device.createBuffer({ size: Math.max(1, nodeCount) * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });\n";
  out << "  device.queue.writeBuffer(resultBuffer, 0, new Uint32Array([0]));\n";
  out << "  device.queue.writeBuffer(propertyBuffer, 0, new Uint32Array([nodeCount]));\n";
  int launchIndex = 0;
  // Generate host-side sequencing for the function body
  generateHostBody(func->getBlockStatement(), out, launchIndex);
  out << "  return " << resultVar << ";\n";
  out << "}\n\n";

  // Generate the kernel launch functions  
  for (int i = 0; i < kernelCounter; ++i) {
    out << "async function launchkernel_" << i << "(device, adj_dataBuffer, adj_offsetsBuffer, resultBuffer, propertyBuffer, nodeCount) {\n";
    out << "  const shaderCode = await (await fetch('kernel_" << i << ".wgsl')).text();\n";
    out << "  const shaderModule = device.createShaderModule({ code: shaderCode });\n";
    out << "  const pipeline = device.createComputePipeline({ layout: 'auto', compute: { module: shaderModule, entryPoint: 'main' } });\n";
    out << "  \n";
    out << "  // Using shared result/property buffers provided by caller\n";
    out << "  const readBuffer = device.createBuffer({ \n";
    out << "    size: 4, \n";
    out << "    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ \n";
    out << "  });\n";
    out << "  \n";
    out << "  const bindGroup = device.createBindGroup({ \n";
    out << "    layout: pipeline.getBindGroupLayout(0), \n";
    out << "    entries: [\n";
    out << "      { binding: 0, resource: { buffer: adj_dataBuffer } },\n";
    out << "      { binding: 1, resource: { buffer: adj_offsetsBuffer } },\n";
    out << "      { binding: 2, resource: { buffer: resultBuffer } },\n";
    out << "      { binding: 3, resource: { buffer: propertyBuffer } }\n";
    out << "    ]\n";
    out << "  });\n";
    out << "  \n";
    out << "  const encoder = device.createCommandEncoder();\n";
    out << "  const pass = encoder.beginComputePass();\n";
    out << "  pass.setPipeline(pipeline);\n";
    out << "  pass.setBindGroup(0, bindGroup);\n";
    out << "  \n";
    out << "  // Dispatch one workgroup per 256 nodes (ensure at least 1 group)\n";
    out << "  let __groups = Math.ceil(nodeCount / 256);\n";
    out << "  if (__groups < 1) { __groups = 1; }\n";
    out << "  pass.dispatchWorkgroups(__groups);\n";
    out << "  pass.end();\n";
    out << "  \n";
    out << "  encoder.copyBufferToBuffer(resultBuffer, 0, readBuffer, 0, 4);\n";
    out << "  device.queue.submit([encoder.finish()]);\n";
    out << "  await device.queue.onSubmittedWorkDone();\n";
    out << "  \n";
    out << "  await readBuffer.mapAsync(GPUMapMode.READ);\n";
    out << "  const result = new Uint32Array(readBuffer.getMappedRange())[0];\n";
    out << "  readBuffer.unmap();\n";
    out << "  \n";
    out << "  return result;\n";
    out << "}\n\n";
  }
}

void dsl_webgpu_generator::generateBlock(ASTNode* node, std::ofstream& out) {
  blockStatement* block = static_cast<blockStatement*>(node);
  if (!block) return;
  
  // Generate all statements in the block
  for (statement* stmt : block->returnStatements()) {
    generateStatement(stmt, out);
  }
}
void dsl_webgpu_generator::generateHostBody(ASTNode* node, std::ofstream& out, int& launchIndex) {
  if (!node) return;
  if (node->getTypeofNode() == NODE_BLOCKSTMT) {
    blockStatement* block = static_cast<blockStatement*>(node);
    for (statement* stmt : block->returnStatements()) {
      generateHostBody(stmt, out, launchIndex);
    }
    return;
  }

  if (node->getTypeofNode() == NODE_FORALLSTMT) {
    out << "  const kernel_res_" << launchIndex << " = await launchkernel_" << launchIndex << "(device, adj_dataBuffer, adj_offsetsBuffer, resultBuffer, propertyBuffer, nodeCount);\n";
    // Assign to generic result for now
    out << "  result = kernel_res_" << launchIndex << ";\n";
    launchIndex++;
    return;
  }

  if (node->getTypeofNode() == NODE_FIXEDPTSTMT) {
    fixedPointStmt* fp = static_cast<fixedPointStmt*>(node);
    std::string fpVarName = fp->getFixedPointId() ? fp->getFixedPointId()->getIdentifier() : "converged";
    out << "  // Fixed point loop: " << fpVarName << "\n";
    out << "  let " << fpVarName << " = false;\n";
    out << "  let fpIterations = 0;\n";
    out << "  const maxFpIterations = 1000;\n";
    out << "  while (!" << fpVarName << " && fpIterations < maxFpIterations) {\n";
    out << "    " << fpVarName << " = true;\n";
    if (fp->getBody()) {
      generateHostBody(fp->getBody(), out, launchIndex);
      // Heuristic: if any kernel produced non-zero result, mark not converged
      out << "    if (result > 0) { " << fpVarName << " = false; }\n";
    }
    out << "    fpIterations++;\n";
    out << "  }\n";
    return;
  }

  // Fallback: emit plain JS for other stmts
  generateStatement(node, out);
}

void dsl_webgpu_generator::generateStatement(ASTNode* node, std::ofstream& out) {
  if (!node) return;
  switch (node->getTypeofNode()) {
    case NODE_DECL: {
      declaration* decl = static_cast<declaration*>(node);
      Identifier* id = decl->getdeclId();
      out << "  let " << (id ? id->getIdentifier() : "unnamed");
      if (decl->isInitialized()) { out << " = "; generateExpr(decl->getExpressionAssigned(), out); }
      out << ";\n";
      break;
    }
    case NODE_ASSIGN: {
      assignment* asst = static_cast<assignment*>(node);
      
      if (asst->lhs_isIdentifier()) {
        // Regular variable assignment: variable = expression
        Identifier* id = asst->getId();
        out << "  " << (id ? id->getIdentifier() : "unnamed") << " = ";
        generateExpr(asst->getExpr(), out);
        out << ";\n";
        
      } else if (asst->lhs_isProp()) {
        // Property assignment: object.property = expression
        PropAccess* prop = asst->getPropId();
        out << "  ";
        if (prop && prop->getIdentifier2() && prop->getIdentifier1()) {
          // Generate: property[object] = expression
          out << prop->getIdentifier2()->getIdentifier() << "[" 
              << prop->getIdentifier1()->getIdentifier() << "]";
        } else {
          out << "/*prop*/";
        }
        out << " = ";
        generateExpr(asst->getExpr(), out);
        out << ";\n";
        
      } else if (asst->lhs_isIndexAccess()) {
        // Index assignment: array[index] = expression
        out << "  ";
        generateExpr(asst->getIndexAccess(), out);
        out << " = ";
        generateExpr(asst->getExpr(), out);
        out << ";\n";
        
      } else {
        out << "  // Unhandled assignment type\n";
      }
      break;
    }
    case NODE_IFSTMT: {
      ifStmt* s = static_cast<ifStmt*>(node);
      out << "  if ("; generateExpr(s->getCondition(), out); out << ") {\n";
      if (s->getIfBody()) {
        if (s->getIfBody()->getTypeofNode() == NODE_BLOCKSTMT) generateBlock(s->getIfBody(), out);
        else generateStatement(s->getIfBody(), out);
      }
      out << "  }";
      if (s->getElseBody()) {
        out << " else {\n";
        if (s->getElseBody()->getTypeofNode() == NODE_BLOCKSTMT) generateBlock(s->getElseBody(), out);
        else generateStatement(s->getElseBody(), out);
        out << "  }";
      }
      out << "\n";
      break;
    }
    case NODE_SIMPLEFORSTMT: {
      simpleForStmt* f = static_cast<simpleForStmt*>(node);
      out << "  for (let "; Identifier* v = f->getLoopVariable(); out << (v ? v->getIdentifier() : "i");
      out << " = "; if (f->getRhs()) generateExpr(f->getRhs(), out); else out << "0";
      out << "; "; if (f->getIterCondition()) generateExpr(f->getIterCondition(), out);
      out << "; "; if (f->getUpdateExpression()) generateExpr(f->getUpdateExpression(), out);
      out << ") {\n";
      if (f->getBody()) { if (f->getBody()->getTypeofNode() == NODE_BLOCKSTMT) generateBlock(f->getBody(), out); else generateStatement(f->getBody(), out); }
      out << "  }\n";
      break;
    }
    case NODE_FORALLSTMT: {
      std::string kernelName = "kernel_" + std::to_string(kernelCounter);
      forallStmt* fa = static_cast<forallStmt*>(node);
      emitWGSLKernel(kernelName, fa->getBody());
      kernelCounter++;
      break;
    }
    case NODE_FIXEDPTSTMT: {
      fixedPointStmt* fp = static_cast<fixedPointStmt*>(node);
      generateFixedPoint(fp, out);
      break;
    }
    case NODE_DOWHILESTMT: {
      dowhileStmt* dw = static_cast<dowhileStmt*>(node);
      generateDoWhile(dw, out);
      break;
    }
    case NODE_WHILESTMT: {
      whileStmt* w = static_cast<whileStmt*>(node);
      out << "  while (";
      if (w->getCondition()) generateExpr(w->getCondition(), out);
      else out << "false";
      out << ") {\n";
      if (w->getBody()) {
        if (w->getBody()->getTypeofNode() == NODE_BLOCKSTMT) generateBlock(w->getBody(), out);
        else generateStatement(w->getBody(), out);
      }
      out << "  }\n";
      break;
    }
    case NODE_BREAKSTMT: {
      out << "  break;\n";
      break;
    }
    case NODE_CONTINUESTMT: {
      out << "  continue;\n";
      break;
    }
    case NODE_UNARYSTMT: {
      unary_stmt* u = static_cast<unary_stmt*>(node);
      out << "  ";
      if (u && u->getUnaryExpr()) {
        Expression* e = u->getUnaryExpr();
        if (e->getExpressionFamily() == EXPR_UNARY) {
          // Emit as id = id +/- 1 to avoid placeholders
          Expression* inner = e->getUnaryExpr();
          if (inner && inner->isIdentifierExpr()) {
            Identifier* id = inner->getId();
            out << id->getIdentifier() << " = " << id->getIdentifier();
            int op = e->getOperatorType();
            if (op == OPERATOR_INC) out << " + 1";
            else if (op == OPERATOR_DEC) out << " - 1";
            else { generateExpr(e, out); }
          } else {
            generateExpr(e, out);
          }
        } else {
          generateExpr(e, out);
        }
      }
      out << ";\n";
      break;
    }
    case NODE_PROCCALLSTMT: {
      proc_callStmt* pc = static_cast<proc_callStmt*>(node);
      generateProcCall(pc, out);
      break;
    }
    case NODE_REDUCTIONCALLSTMT: {
      reductionCallStmt* rc = static_cast<reductionCallStmt*>(node);
      generateReductionStmt(rc, out);
      break;
    }
    case NODE_LOOPSTMT: {
      loopStmt* loop = static_cast<loopStmt*>(node);
      out << "  for (let ";
      if (loop->getIterator()) out << loop->getIterator()->getIdentifier();
      else out << "i";
      out << " = ";
      if (loop->getStartValue()) generateExpr(loop->getStartValue(), out);
      else out << "0";
      out << "; ";
      if (loop->getIterator()) out << loop->getIterator()->getIdentifier();
      else out << "i";
      out << " < ";
      if (loop->getEndValue()) generateExpr(loop->getEndValue(), out);
      else out << "1";
      out << "; ";
      if (loop->getIterator()) out << loop->getIterator()->getIdentifier();
      else out << "i";
      out << " += ";
      if (loop->getStepValue()) generateExpr(loop->getStepValue(), out);
      else out << "1";
      out << ") {\n";
      if (loop->getBody()) {
        if (loop->getBody()->getTypeofNode() == NODE_BLOCKSTMT) generateBlock(loop->getBody(), out);
        else generateStatement(loop->getBody(), out);
      }
      out << "  }\n";
      break;
    }
    case NODE_RETURN: {
      returnStmt* r = static_cast<returnStmt*>(node);
      out << "  return ";
      if (r && r->getReturnExpression()) { 
        generateExpr(r->getReturnExpression(), out);
      } else {
        out << "result";
      }
      out << ";\n";
      break;
    }
    default:
      out << "  // Unhandled stmt\n";
      break;
  }
}

void dsl_webgpu_generator::generateExpr(ASTNode* node, std::ofstream& out) {
  if (!node) { out << "undefined"; return; }
  switch (node->getTypeofNode()) {
    case NODE_ID: { Identifier* id = static_cast<Identifier*>(node); out << id->getIdentifier(); break; }
    case NODE_EXPR: {
      Expression* expr = static_cast<Expression*>(node);
      switch (expr->getExpressionFamily()) {
        case EXPR_INTCONSTANT: out << expr->getIntegerConstant(); break;
        case EXPR_FLOATCONSTANT: out << expr->getFloatConstant(); break;
        case EXPR_BOOLCONSTANT: out << (expr->getBooleanConstant()?"true":"false"); break;
        case EXPR_STRINGCONSTANT: out << '"' << expr->getStringConstant() << '"'; break;
        case EXPR_LONGCONSTANT: out << expr->getIntegerConstant(); break; // Long as regular int in JS
        case EXPR_DOUBLECONSTANT: out << expr->getFloatConstant(); break; // Double as regular float in JS
        case EXPR_INFINITY: out << (expr->isPositiveInfinity() ? "Infinity" : "-Infinity"); break;
        case EXPR_ARITHMETIC:
        case EXPR_RELATIONAL:
        case EXPR_LOGICAL: { 
          out << "("; 
          generateExpr(expr->getLeft(), out); 
          out << " " << getOpString(expr->getOperatorType()) << " "; 
          generateExpr(expr->getRight(), out); 
          out << ")"; 
          break; 
        }
        case EXPR_UNARY: { 
          out << "(" << getOpString(expr->getOperatorType()); 
          generateExpr(expr->getUnaryExpr(), out); 
          out << ")"; 
          break; 
        }
        case EXPR_ID: {
          Identifier* id = expr->getId();
          out << (id ? id->getIdentifier() : "unnamed");
          break;
        }
        case EXPR_PROCCALL: {
          // Handle procedure calls in expressions
          proc_callExpr* proc = static_cast<proc_callExpr*>(expr);
          if (proc && proc->getMethodId()) {
            std::string methodName = proc->getMethodId()->getIdentifier();
            if (methodName == "is_an_edge") {
              // JS host doesn't run kernel logic; return a boolean placeholder
              out << "true";
            } else if (methodName == "count_outNbrs") {
              // Count of out neighbors is not computed on host; placeholder 0
              out << "0";
            } else if (methodName == "num_nodes") {
              out << "nodeCount";
            } else if (methodName == "Min") {
              out << "Math.min(";
              list<argument*> argList = proc->getArgList();
              if (argList.size() >= 2) {
                auto it = argList.begin();
                argument* arg1 = *it;
                ++it;
                argument* arg2 = *it;
                if (arg1->isExpr()) generateExpr(arg1->getExpr(), out);
                out << ", ";
                if (arg2->isExpr()) generateExpr(arg2->getExpr(), out);
              }
              out << ")";
            } else if (methodName == "Max") {
              out << "Math.max(";
              list<argument*> argList = proc->getArgList();
              if (argList.size() >= 2) {
                auto it = argList.begin();
                argument* arg1 = *it;
                ++it;
                argument* arg2 = *it;
                if (arg1->isExpr()) generateExpr(arg1->getExpr(), out);
                out << ", ";
                if (arg2->isExpr()) generateExpr(arg2->getExpr(), out);
              }
              out << ")";
            } else {
              out << "0";
            }
          } else {
            out << "true"; // Fallback
          }
          break;
        }
        case EXPR_PROPID: {
          PropAccess* prop = expr->getPropId();
          if (prop && prop->getIdentifier2()) {
            out << prop->getIdentifier2()->getIdentifier() << "[";
            if (prop->getIdentifier1()) out << prop->getIdentifier1()->getIdentifier();
            else if (prop->getPropExpr()) generateExpr(prop->getPropExpr(), out);
            out << "]";
          } else if (prop && prop->getIdentifier1() && prop->getPropExpr()) {
            out << prop->getIdentifier1()->getIdentifier() << "["; generateExpr(prop->getPropExpr(), out); out << "]";
          } else { out << "0"; }
          break;
        }
        case EXPR_MAPGET: {
          // Handle map/container access: map[key]
          out << "(";
          if (expr->getMapExpr()) {
            generateExpr(expr->getMapExpr(), out);
          }
          out << "[";
          if (expr->getIndexExpr()) {
            generateExpr(expr->getIndexExpr(), out);
          }
          out << "])";
          break;
        }
        case EXPR_ALLOCATE: {
          // Handle memory allocation expressions
          out << "0"; // Placeholder for allocation
          break;
        }
        case EXPR_DEPENDENT: {
          // Handle dependent expressions (used in fixed point loops)
          out << "true"; // Placeholder for dependent expressions
          break;
        }
        default: out << "0"; break;
      }
      break;
    }
    default: out << "0"; break;
  }
}

void dsl_webgpu_generator::emitWGSLKernel(const std::string& baseName, ASTNode* forallBody) {
  std::string filename = std::string("../graphcode/generated_webgpu/") + baseName + ".wgsl";
  std::ofstream wgslOut(filename);
  if (!wgslOut.is_open()) return;
  
  // --- Storage Buffers ---
  wgslOut << "// Graph algorithm compute shader\n";
  wgslOut << "@group(0) @binding(0) var<storage, read> adj_data: array<u32>;\n";
  wgslOut << "@group(0) @binding(1) var<storage, read> adj_offsets: array<u32>;\n";
  wgslOut << "@group(0) @binding(2) var<storage, read_write> result: atomic<u32>;\n";
  wgslOut << "@group(0) @binding(3) var<storage, read_write> properties: array<atomic<u32>>;\n\n";
  
  // --- Helper Functions ---
  // float atomics via CAS on atomic<u32>
  wgslOut << "fn atomicAddF32(ptr: ptr<storage, atomic<u32>>, val: f32) -> f32 {\n";
  wgslOut << "  loop {\n";
  wgslOut << "    let oldBits: u32 = atomicLoad(ptr);\n";
  wgslOut << "    let oldVal: f32 = bitcast<f32>(oldBits);\n";
  wgslOut << "    let newVal: f32 = oldVal + val;\n";
  wgslOut << "    let newBits: u32 = bitcast<u32>(newVal);\n";
  wgslOut << "    let res = atomicCompareExchangeWeak(ptr, oldBits, newBits);\n";
  wgslOut << "    if (res.exchanged) { return oldVal; }\n";
  wgslOut << "  }\n";
  wgslOut << "}\n";
  wgslOut << "fn atomicMinF32(ptr: ptr<storage, atomic<u32>>, val: f32) {\n";
  wgslOut << "  loop {\n";
  wgslOut << "    let oldBits: u32 = atomicLoad(ptr);\n";
  wgslOut << "    let oldVal: f32 = bitcast<f32>(oldBits);\n";
  wgslOut << "    let newVal: f32 = min(oldVal, val);\n";
  wgslOut << "    let newBits: u32 = bitcast<u32>(newVal);\n";
  wgslOut << "    let res = atomicCompareExchangeWeak(ptr, oldBits, newBits);\n";
  wgslOut << "    if (res.exchanged) { return; }\n";
  wgslOut << "  }\n";
  wgslOut << "}\n";
  wgslOut << "fn atomicMaxF32(ptr: ptr<storage, atomic<u32>>, val: f32) {\n";
  wgslOut << "  loop {\n";
  wgslOut << "    let oldBits: u32 = atomicLoad(ptr);\n";
  wgslOut << "    let oldVal: f32 = bitcast<f32>(oldBits);\n";
  wgslOut << "    let newVal: f32 = max(oldVal, val);\n";
  wgslOut << "    let newBits: u32 = bitcast<u32>(newVal);\n";
  wgslOut << "    let res = atomicCompareExchangeWeak(ptr, oldBits, newBits);\n";
  wgslOut << "    if (res.exchanged) { return; }\n";
  wgslOut << "  }\n";
  wgslOut << "}\n\n";
  wgslOut << "/**\n";
  wgslOut << " * Checks if there's an edge between vertices u and w\n";
  wgslOut << " */\n";
  wgslOut << "fn findEdge(u: u32, w: u32) -> bool {\n";
  wgslOut << "  for (var edge = adj_offsets[u]; edge < adj_offsets[u + 1]; edge++) {\n";
  wgslOut << "    if (adj_data[edge] == w) {\n";
  wgslOut << "      return true;\n";
  wgslOut << "    }\n";
  wgslOut << "  }\n";
  wgslOut << "  return false;\n";
  wgslOut << "}\n\n";
  
  // --- Main Entry Point ---
  wgslOut << "@compute @workgroup_size(256)\n";
  wgslOut << "fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {\n";
  wgslOut << "  let v = global_id.x;\n";
  wgslOut << "  let node_count = atomicLoad(&properties[0u]);\n";
  wgslOut << "  \n";
  wgslOut << "  if (v >= node_count) {\n";
  wgslOut << "    return;\n";
  wgslOut << "  }\n\n";
  
  if (forallBody && forallBody->getTypeofNode() == NODE_BLOCKSTMT) {
    blockStatement* block = static_cast<blockStatement*>(forallBody);
    for (statement* stmt : block->returnStatements()) { 
      generateWGSLStatement(stmt, wgslOut, "v", 1); 
    }
  } else if (forallBody) {
    generateWGSLStatement(forallBody, wgslOut, "v", 1);
  }
  wgslOut << "}\n";
  wgslOut.close();
}

void dsl_webgpu_generator::generateWGSLStatement(ASTNode* node, std::ofstream& wgslOut, const std::string& indexVar, int indentLevel) {
  if (!node) return;
  std::string indent = std::string(indentLevel * 2, ' ');
  
  // Use CUDA-style multiple handler approach (if statements instead of switch)
  if (node->getTypeofNode() == NODE_PROCCALLSTMT) {
    proc_callStmt* proc = static_cast<proc_callStmt*>(node);
    generateProcCall(proc, wgslOut);
    return;
  }
  
  if (node->getTypeofNode() == NODE_REDUCTIONCALLSTMT) {
    reductionCallStmt* r = static_cast<reductionCallStmt*>(node);
    // Determine operation kind
    int opKind = -1; // 0:add, 1:min, 2:max
    if (r->is_reducCall() && r->getReducCall()) {
      int rt = r->getReducCall()->getReductionType();
      if (rt == REDUCE_MIN) opKind = 1; else if (rt == REDUCE_MAX) opKind = 2; else opKind = 0;
    } else {
      opKind = 0;
    }

    // Determine if RHS is float-ish (heuristic)
    auto isFloatExpr = [&](Expression* ex) -> bool {
      if (!ex) return false;
      int fam = ex->getExpressionFamily();
      return fam == EXPR_FLOATCONSTANT || fam == EXPR_DOUBLECONSTANT;
    };
    Expression* rhsExpr = nullptr;
    if (r->is_reducCall()) {
      reductionCall* rc = r->getReducCall();
      auto args = rc->getargList();
      if (!args.empty()) {
        argument* a = args.front();
        if (a && a->isExpr()) rhsExpr = a->getExpr();
      }
    } else if (r->getRightSide()) {
      rhsExpr = r->getRightSide();
    }
    bool rhsIsFloat = (rhsExpr && isFloatExpr(rhsExpr));

    auto emitOp = [&](const std::string &ptr) {
      if (!rhsIsFloat) {
        if (opKind == 1) {
          wgslOut << "atomicMin(" << ptr << ", ";
        } else if (opKind == 2) {
          wgslOut << "atomicMax(" << ptr << ", ";
        } else {
          wgslOut << "atomicAdd(" << ptr << ", ";
        }
      } else {
        if (opKind == 1) {
          wgslOut << "atomicMinF32(" << ptr << ", ";
        } else if (opKind == 2) {
          wgslOut << "atomicMaxF32(" << ptr << ", ";
        } else {
          wgslOut << "atomicAddF32(" << ptr << ", ";
        }
      }
    };

    // Decide target pointer
    if (r->isLeftIdentifier()) {
      // Use global result for scalar reductions
      wgslOut << indent;
      emitOp("&result");
      if (r->is_reducCall()) {
        // reductionCall has arg list; take first expr as rhs
        reductionCall* rc = r->getReducCall();
        auto args = rc->getargList();
        if (!args.empty()) {
          argument* a = args.front();
          if (a && a->isExpr()) {
            if (rhsIsFloat) { wgslOut << "f32("; generateWGSLExpr(a->getExpr(), wgslOut, indexVar); wgslOut << ")"; }
            else { wgslOut << "u32("; generateWGSLExpr(a->getExpr(), wgslOut, indexVar); wgslOut << ")"; }
          }
        }
      } else if (r->getRightSide()) {
        if (rhsIsFloat) { wgslOut << "f32("; generateWGSLExpr(r->getRightSide(), wgslOut, indexVar); wgslOut << ")"; }
        else { wgslOut << "u32("; generateWGSLExpr(r->getRightSide(), wgslOut, indexVar); wgslOut << ")"; }
      } else {
        wgslOut << (rhsIsFloat ? "1.0" : "1u");
      }
      wgslOut << ");\n";
    } else if (r->getLhsType() == 2) {
      // Property target
      wgslOut << indent;
      PropAccess* prop = r->getPropAccess();
      std::string arr = prop && prop->getIdentifier2() ? prop->getIdentifier2()->getIdentifier() : "properties";
      // Emit operator and pointer inline
      if (!rhsIsFloat) {
        if (opKind == 1) { wgslOut << "atomicMin(&" << arr << "["; }
        else if (opKind == 2) { wgslOut << "atomicMax(&" << arr << "["; }
        else { wgslOut << "atomicAdd(&" << arr << "["; }
      } else {
        if (opKind == 1) { wgslOut << "atomicMinF32(&" << arr << "["; }
        else if (opKind == 2) { wgslOut << "atomicMaxF32(&" << arr << "["; }
        else { wgslOut << "atomicAddF32(&" << arr << "["; }
      }
      if (prop && prop->getIdentifier1()) {
        wgslOut << prop->getIdentifier1()->getIdentifier();
      } else if (prop && prop->getPropExpr()) {
        generateWGSLExpr(prop->getPropExpr(), wgslOut, indexVar);
      }
      wgslOut << "], ";
      if (r->is_reducCall()) {
        reductionCall* rc = r->getReducCall();
        auto args = rc->getargList();
        if (!args.empty()) {
          argument* a = args.front();
          if (a && a->isExpr()) {
            if (rhsIsFloat) { wgslOut << "f32("; generateWGSLExpr(a->getExpr(), wgslOut, indexVar); wgslOut << ")"; }
            else { wgslOut << "u32("; generateWGSLExpr(a->getExpr(), wgslOut, indexVar); wgslOut << ")"; }
          }
        }
      } else if (r->getRightSide()) {
        if (rhsIsFloat) { wgslOut << "f32("; generateWGSLExpr(r->getRightSide(), wgslOut, indexVar); wgslOut << ")"; }
        else { wgslOut << "u32("; generateWGSLExpr(r->getRightSide(), wgslOut, indexVar); wgslOut << ")"; }
      } else {
        wgslOut << (rhsIsFloat ? "1.0" : "1u");
      }
      wgslOut << ");\n";
    } else {
      // Fallback to result
      wgslOut << indent;
      emitOp("&result");
      if (r->getRightSide()) {
        if (rhsIsFloat) { wgslOut << "f32("; generateWGSLExpr(r->getRightSide(), wgslOut, indexVar); wgslOut << ")"; }
        else { wgslOut << "u32("; generateWGSLExpr(r->getRightSide(), wgslOut, indexVar); wgslOut << ")"; }
      } else {
        wgslOut << (rhsIsFloat ? "1.0" : "1u");
      }
      wgslOut << ");\n";
    }
    return;
  }
  
  if (node->getTypeofNode() == NODE_DECL) {
    // Handle variable declarations
    declaration* decl = static_cast<declaration*>(node);
    if (decl && decl->getdeclId()) {
      Identifier* id = decl->getdeclId();
      // Emit var with best-effort type annotation when the init is a float constant
      bool isFloat = false;
      Expression* init = decl->getExpressionAssigned();
      if (init && init->getTypeofNode() == NODE_EXPR) {
        Expression* e = static_cast<Expression*>(init);
        isFloat = (e->getExpressionFamily() == EXPR_FLOATCONSTANT || e->getExpressionFamily() == EXPR_DOUBLECONSTANT);
      }
      wgslOut << indent << "var " << (id ? id->getIdentifier() : "var");
      if (isFloat) wgslOut << ": f32";
      wgslOut << " = ";
      if (init) {
        // If expecting float, ensure literal looks like float
        if (isFloat && init->getTypeofNode() == NODE_EXPR) {
          Expression* e = static_cast<Expression*>(init);
          if (e->getExpressionFamily() == EXPR_INTCONSTANT) {
            wgslOut << e->getIntegerConstant() << ".0";
          } else {
            generateWGSLExpr(init, wgslOut, indexVar);
          }
        } else {
          generateWGSLExpr(init, wgslOut, indexVar);
        }
      } else {
        wgslOut << (isFloat ? "0.0" : "0u");
      }
      wgslOut << ";\n";
    }
    return;
  }
  
  if (node->getTypeofNode() == NODE_ASSIGN) {
    assignment* asst = static_cast<assignment*>(node);
    if (asst->lhs_isIdentifier()) {
      Identifier* id = asst->getId();
      if (asst->getAtomicSignal()) {
        // Handle atomic assignment (like +=) for reduction
        wgslOut << indent << "atomicAdd(&result, ";
        if (asst->getExpr()) { wgslOut << "u32("; generateWGSLExpr(asst->getExpr(), wgslOut, indexVar); wgslOut << ")"; }
        else { wgslOut << "1u"; }
        wgslOut << ");\n";
      } else {
        // Regular assignment
        wgslOut << indent << (id ? id->getIdentifier() : "unnamed") << " = ";
        generateWGSLExpr(asst->getExpr(), wgslOut, indexVar);
        wgslOut << ";\n";
      }
    } else if (asst->lhs_isProp()) {
      // Handle property assignment: v.deg = expression
      PropAccess* prop = asst->getPropId();
      wgslOut << indent;
      generatePropertyAccess(prop, wgslOut, indexVar);
      wgslOut << " = ";
      generateWGSLExpr(asst->getExpr(), wgslOut, indexVar);
      wgslOut << ";\n";
    } else if (asst->lhs_isIndexAccess()) {
      // Handle index access assignment: array[index] = expression
      wgslOut << indent;
      generateWGSLExpr(asst->getIndexAccess(), wgslOut, indexVar);
      wgslOut << " = ";
      generateWGSLExpr(asst->getExpr(), wgslOut, indexVar);
      wgslOut << ";\n";
    } else {
      // Unknown assignment type
      wgslOut << indent << "// Unknown assignment type\n";
    }
    return;
  }
  
  if (node->getTypeofNode() == NODE_UNARYSTMT) {
    // Handle unary statements (like increment/decrement)
    unary_stmt* u = static_cast<unary_stmt*>(node);
    wgslOut << indent;
    if (u && u->getUnaryExpr()) {
      generateWGSLExpr(u->getUnaryExpr(), wgslOut, indexVar);
    }
    wgslOut << ";\n";
    return;
  }
  
  if (node->getTypeofNode() == NODE_EXPR) {
    // Handle expressions that appear as standalone statements
    Expression* expr = static_cast<Expression*>(node);
    
    // Check if this is actually an assignment expression
    if (expr->getExpressionFamily() == EXPR_ARITHMETIC) {
      // This might be an assignment-like expression
      // Try to handle it as a property assignment
      if (expr->getLeft() && expr->getRight()) {
        // Check if left side is a property access
        if (expr->getLeft()->getTypeofNode() == NODE_EXPR) {
          Expression* leftExpr = static_cast<Expression*>(expr->getLeft());
          if (leftExpr->getExpressionFamily() == EXPR_PROPID) {
            // This is a property assignment
            wgslOut << indent;
            generateWGSLExpr(expr->getLeft(), wgslOut, indexVar);
            wgslOut << " = ";
            generateWGSLExpr(expr->getRight(), wgslOut, indexVar);
            wgslOut << ";\n";
            return;
          }
        }
      }
    }
    
    // Enhanced handling for complex expressions
    // Check if this is a property assignment expression
    if (expr->getExpressionFamily() == EXPR_ARITHMETIC && 
        expr->getOperatorType() == OPERATOR_ADD) {
      // This might be a compound assignment like "sum = sum + ..."
      if (expr->getLeft() && expr->getRight()) {
        // Check if left side is an identifier
        if (expr->getLeft()->getTypeofNode() == NODE_EXPR) {
          Expression* leftExpr = static_cast<Expression*>(expr->getLeft());
          if (leftExpr->getExpressionFamily() == EXPR_ID) {
            // This is a variable assignment
            wgslOut << indent;
            generateWGSLExpr(expr->getLeft(), wgslOut, indexVar);
            wgslOut << " = ";
            generateWGSLExpr(expr->getRight(), wgslOut, indexVar);
            wgslOut << ";\n";
            return;
          }
        }
      }
    }
    
    // Default handling for expressions
    wgslOut << indent;
    generateWGSLExpr(expr, wgslOut, indexVar);
    wgslOut << ";\n";
    return;
  }
  
  // Continue with other cases using switch statement for remaining cases
  switch (node->getTypeofNode()) {
    case NODE_SIMPLEFORSTMT: {
      // Handle simple for loops
      simpleForStmt* f = static_cast<simpleForStmt*>(node);
      wgslOut << indent << "for (var ";
      if (f->getLoopVariable()) wgslOut << f->getLoopVariable()->getIdentifier();
      else wgslOut << "i";
      wgslOut << " = ";
      if (f->getRhs()) generateWGSLExpr(f->getRhs(), wgslOut, indexVar);
      else wgslOut << "0";
      wgslOut << "; ";
      if (f->getIterCondition()) generateWGSLExpr(f->getIterCondition(), wgslOut, indexVar);
      else wgslOut << "true";
      wgslOut << "; ";
      if (f->getUpdateExpression()) generateWGSLExpr(f->getUpdateExpression(), wgslOut, indexVar);
      else wgslOut << "i++";
      wgslOut << ") {\n";
      if (f->getBody()) {
        if (f->getBody()->getTypeofNode() == NODE_BLOCKSTMT) {
          generateWGSLStatement(f->getBody(), wgslOut, indexVar, indentLevel + 1);
        } else {
          generateWGSLStatement(f->getBody(), wgslOut, indexVar, indentLevel + 1);
        }
      }
      wgslOut << indent << "}\n";
      break;
    }
    case NODE_RETURN: {
      // Handle return statements
      wgslOut << indent << "return;\n";
      break;
    }

    case NODE_IFSTMT: {
      ifStmt* s = static_cast<ifStmt*>(node);
      wgslOut << indent << "if ("; 
      generateWGSLExpr(s->getCondition(), wgslOut, indexVar); 
      wgslOut << ") {\n";
      if (s->getIfBody()) generateWGSLStatement(s->getIfBody(), wgslOut, indexVar, indentLevel + 1);
      wgslOut << indent << "}";
      if (s->getElseBody()) { 
        wgslOut << " else {\n"; 
        generateWGSLStatement(s->getElseBody(), wgslOut, indexVar, indentLevel + 1); 
        wgslOut << indent << "}"; 
      }
      wgslOut << "\n";
      break;
    }
    case NODE_BLOCKSTMT: {
      blockStatement* block = static_cast<blockStatement*>(node);
      for (statement* stmt : block->returnStatements()) { 
        generateWGSLStatement(stmt, wgslOut, indexVar, indentLevel); 
      }
      break;
    }
    case NODE_FORALLSTMT: {
      forallStmt* fa = static_cast<forallStmt*>(node);
      if (fa->getBody()) {
        proc_callExpr* extractElemFunc = fa->getExtractElementFunc();
        if (extractElemFunc && extractElemFunc->getMethodId()) {
          std::string methodName = extractElemFunc->getMethodId()->getIdentifier();
          if (methodName == "neighbors") {
            // Generate neighbor iteration loop
            wgslOut << indent << "for (var edge = adj_offsets[" << indexVar << "]; edge < adj_offsets[" << indexVar << " + 1]; edge++) {\n";
            wgslOut << indent << "  let " << fa->getIterator()->getIdentifier() << " = adj_data[edge];\n";
            // Check filter condition if present
            if (fa->hasFilterExpr()) {
              wgslOut << indent << "  if (";
              generateWGSLExpr(fa->getfilterExpr(), wgslOut, indexVar);
              wgslOut << ") {\n";
              generateWGSLStatement(fa->getBody(), wgslOut, indexVar, indentLevel + 2);
              wgslOut << indent << "  }\n";
            } else {
              generateWGSLStatement(fa->getBody(), wgslOut, indexVar, indentLevel + 1);
            }
            wgslOut << indent << "}\n";
          } else {
            // Handle other iteration types (e.g., g.nodes())
            generateWGSLStatement(fa->getBody(), wgslOut, indexVar, indentLevel);
          }
        } else {
          generateWGSLStatement(fa->getBody(), wgslOut, indexVar, indentLevel);
        }
      }
      break;
    }
    default: {
      wgslOut << indent << "// unhandled stmt type: " << node->getTypeofNode() << " (node type: " << node->getTypeofNode() << ")\n";
      
      // UNIVERSAL FALLBACK: Try to handle ANY node type as a potential expression
      wgslOut << indent << "// UNIVERSAL FALLBACK: Attempting to handle as expression\n";
      
      // Try to cast as Expression first
      try {
        Expression* expr = static_cast<Expression*>(node);
        if (expr) {
          wgslOut << indent;
          generateWGSLExpr(expr, wgslOut, indexVar);
          wgslOut << ";\n";
          return;
        }
      } catch (...) {
        wgslOut << indent << "// Expression cast failed\n";
      }
      
      // Try to cast as Assignment
      try {
        assignment* asst = static_cast<assignment*>(node);
        if (asst) {
          if (asst->lhs_isIdentifier()) {
            Identifier* id = asst->getId();
            wgslOut << indent << (id ? id->getIdentifier() : "unnamed") << " = ";
            generateWGSLExpr(asst->getExpr(), wgslOut, indexVar);
            wgslOut << ";\n";
            return;
          } else if (asst->lhs_isProp()) {
            PropAccess* prop = asst->getPropId();
            wgslOut << indent;
            generatePropertyAccess(prop, wgslOut, indexVar);
            wgslOut << " = ";
            generateWGSLExpr(asst->getExpr(), wgslOut, indexVar);
            wgslOut << ";\n";
            return;
          }
        }
      } catch (...) {
        wgslOut << indent << "// Assignment cast failed\n";
      }
      
      // Try to cast as Declaration
      try {
        declaration* decl = static_cast<declaration*>(node);
        if (decl && decl->getdeclId()) {
          Identifier* id = decl->getdeclId();
          wgslOut << indent << "let " << (id ? id->getIdentifier() : "var") << " = ";
          if (decl->getExpressionAssigned()) {
            generateWGSLExpr(decl->getExpressionAssigned(), wgslOut, indexVar);
          } else {
            wgslOut << "0u";
          }
          wgslOut << ";\n";
          return;
        }
      } catch (...) {
        wgslOut << indent << "// Declaration cast failed\n";
      }
      
      // Final fallback: Generate a placeholder
      wgslOut << indent << "// UNHANDLED NODE TYPE: " << node->getTypeofNode() << "\n";
      wgslOut << indent << "// TODO: Implement proper handling for this node type\n";
      break;
    }
  }
}

void dsl_webgpu_generator::generateWGSLExpr(ASTNode* node, std::ofstream& wgslOut, const std::string& indexVar) {
  if (!node) { wgslOut << "0u"; return; }
  
  switch (node->getTypeofNode()) {
    case NODE_ID: { Identifier* id = static_cast<Identifier*>(node); wgslOut << (id ? id->getIdentifier() : "unnamed"); break; }
    case NODE_PROCCALLSTMT: {
      // Handle procedure call statements in expression context (following CUDA backend pattern)
      Expression* expr = static_cast<Expression*>(node);
      proc_callExpr* proc = static_cast<proc_callExpr*>(expr);
      if (proc && proc->getMethodId()) {
        std::string methodName = proc->getMethodId()->getIdentifier();
        if (methodName == "count_outNbrs") {
          // Generate neighbor count: adj_offsets[v+1] - adj_offsets[v]
          list<argument*> argList = proc->getArgList();
          if (!argList.empty()) {
            argument* arg = argList.front();
            wgslOut << "(adj_offsets[";
            if (arg->isExpr()) {
              generateWGSLExpr(arg->getExpr(), wgslOut, indexVar);
            }
            wgslOut << " + 1] - adj_offsets[";
            if (arg->isExpr()) {
              generateWGSLExpr(arg->getExpr(), wgslOut, indexVar);
            }
            wgslOut << "])";
          } else {
            wgslOut << "0";
          }
        } else {
          wgslOut << "0"; // Other procedure calls
        }
      } else {
        wgslOut << "0";
      }
      break;
    }
    case NODE_PROCCALLEXPR: {
      // Handle procedure call expressions
      proc_callExpr* proc = static_cast<proc_callExpr*>(node);
      if (proc && proc->getMethodId()) {
        std::string methodName = proc->getMethodId()->getIdentifier();
        if (methodName == "count_outNbrs") {
          // Generate neighbor count: adj_offsets[v+1] - adj_offsets[v]
          list<argument*> argList = proc->getArgList();
          if (!argList.empty()) {
            argument* arg = argList.front();
            wgslOut << "(adj_offsets[";
            if (arg->isExpr()) {
              generateWGSLExpr(arg->getExpr(), wgslOut, indexVar);
            }
            wgslOut << " + 1] - adj_offsets[";
            if (arg->isExpr()) {
              generateWGSLExpr(arg->getExpr(), wgslOut, indexVar);
            }
            wgslOut << "])";
          } else {
            wgslOut << "0";
          }
        } else if (methodName == "is_an_edge") {
          // Generate edge existence check
          list<argument*> argList = proc->getArgList();
          if (argList.size() >= 2) {
            auto it = argList.begin();
            argument* arg1 = *it;
            ++it;
            argument* arg2 = *it;
            
            wgslOut << "findEdge(";
            if (arg1->isExpr()) {
              generateWGSLExpr(arg1->getExpr(), wgslOut, indexVar);
            }
            wgslOut << ", ";
            if (arg2->isExpr()) {
              generateWGSLExpr(arg2->getExpr(), wgslOut, indexVar);
            }
            wgslOut << ")";
          } else {
            wgslOut << "true"; // Fallback
          }
        } else {
          wgslOut << "true"; // Other procedure calls
        }
      } else {
        wgslOut << "true"; // Fallback
      }
      break;
    }
    case NODE_EXPR: {
      Expression* expr = static_cast<Expression*>(node);
      switch (expr->getExpressionFamily()) {
        case EXPR_INTCONSTANT: wgslOut << expr->getIntegerConstant(); break;
        case EXPR_FLOATCONSTANT: wgslOut << expr->getFloatConstant(); break;
        case EXPR_BOOLCONSTANT: wgslOut << (expr->getBooleanConstant() ? "true" : "false"); break;
        case EXPR_STRINGCONSTANT: wgslOut << "\"" << expr->getStringConstant() << "\""; break;
        case EXPR_LONGCONSTANT: wgslOut << expr->getIntegerConstant() << "u"; break; // Long as u32 in WGSL
        case EXPR_DOUBLECONSTANT: wgslOut << expr->getFloatConstant(); break; // Double as f32 in WGSL
        case EXPR_INFINITY: wgslOut << (expr->isPositiveInfinity() ? "1.0 / 0.0" : "-1.0 / 0.0"); break;
        case EXPR_ID: {
          Identifier* id = expr->getId();
          wgslOut << (id ? id->getIdentifier() : "unnamed");
          break;
        }
        case EXPR_PROPID: {
          PropAccess* prop = expr->getPropId();
          generatePropertyAccess(prop, wgslOut, indexVar);
          break;
        }
        case EXPR_ARITHMETIC:
        case EXPR_RELATIONAL:
        case EXPR_LOGICAL: { 
          wgslOut << "("; 
          generateWGSLExpr(expr->getLeft(), wgslOut, indexVar); 
          wgslOut << " " << getOpString(expr->getOperatorType()) << " "; 
          generateWGSLExpr(expr->getRight(), wgslOut, indexVar); 
          wgslOut << ")"; 
          break; 
        }
        case EXPR_UNARY: { 
          wgslOut << "(" << getOpString(expr->getOperatorType()); 
          generateWGSLExpr(expr->getUnaryExpr(), wgslOut, indexVar); 
          wgslOut << ")"; 
          break; 
        }
        case EXPR_MAPGET: {
          // Handle map/container access: map[key]
          wgslOut << "(";
          if (expr->getMapExpr()) {
            generateWGSLExpr(expr->getMapExpr(), wgslOut, indexVar);
          }
          wgslOut << "[";
          if (expr->getIndexExpr()) {
            generateWGSLExpr(expr->getIndexExpr(), wgslOut, indexVar);
          }
          wgslOut << "])";
          break;
        }
        case EXPR_ALLOCATE: {
          // Handle memory allocation expressions
          wgslOut << "0u"; // Placeholder for allocation
          break;
        }
        case EXPR_DEPENDENT: {
          // Handle dependent expressions (used in fixed point loops)
          wgslOut << "true"; // Placeholder for dependent expressions
          break;
        }
        case EXPR_PROCCALL: {
          // Cast to proc_callExpr to access proc call data
          proc_callExpr* proc = static_cast<proc_callExpr*>(node);
          if (proc && proc->getMethodId()) {
            std::string methodName = proc->getMethodId()->getIdentifier();
            if (methodName == "is_an_edge") {
              // Generate edge existence check
              list<argument*> argList = proc->getArgList();
              if (argList.size() >= 2) {
                // Get the two nodes to check for edge
                auto it = argList.begin();
                argument* arg1 = *it;
                ++it;
                argument* arg2 = *it;
                
                wgslOut << "findEdge(";
                if (arg1->isExpr()) {
                  generateWGSLExpr(arg1->getExpr(), wgslOut, indexVar);
                }
                wgslOut << ", ";
                if (arg2->isExpr()) {
                  generateWGSLExpr(arg2->getExpr(), wgslOut, indexVar);
                }
                wgslOut << ")";
              } else {
                wgslOut << "true"; // Fallback
              }
            } else if (methodName == "count_outNbrs") {
              // Generate neighbor count: adj_offsets[v+1] - adj_offsets[v]
              list<argument*> argList = proc->getArgList();
              if (!argList.empty()) {
                argument* arg = argList.front();
                wgslOut << "(adj_offsets[";
                if (arg->isExpr()) {
                  generateWGSLExpr(arg->getExpr(), wgslOut, indexVar);
                }
                wgslOut << " + 1] - adj_offsets[";
                if (arg->isExpr()) {
                  generateWGSLExpr(arg->getExpr(), wgslOut, indexVar);
                }
                wgslOut << "])";
              } else {
                wgslOut << "0"; // Fallback
              }
            } else if (methodName == "Min") {
              // Handle Min() function calls
              list<argument*> argList = proc->getArgList();
              if (argList.size() >= 2) {
                auto it = argList.begin();
                argument* arg1 = *it;
                ++it;
                argument* arg2 = *it;
                
                wgslOut << "min(";
                if (arg1->isExpr()) {
                  generateWGSLExpr(arg1->getExpr(), wgslOut, indexVar);
                }
                wgslOut << ", ";
                if (arg2->isExpr()) {
                  generateWGSLExpr(arg2->getExpr(), wgslOut, indexVar);
                }
                wgslOut << ")";
              } else {
                wgslOut << "0"; // Fallback
              }
            } else if (methodName == "Max") {
              // Handle Max() function calls
              list<argument*> argList = proc->getArgList();
              if (argList.size() >= 2) {
                auto it = argList.begin();
                argument* arg1 = *it;
                ++it;
                argument* arg2 = *it;
                
                wgslOut << "max(";
                if (arg1->isExpr()) {
                  generateWGSLExpr(arg1->getExpr(), wgslOut, indexVar);
                }
                wgslOut << ", ";
                if (arg2->isExpr()) {
                  generateWGSLExpr(arg2->getExpr(), wgslOut, indexVar);
                }
                wgslOut << ")";
              } else {
                wgslOut << "0"; // Fallback
              }
            } else if (methodName == "Sum") {
              // Handle Sum() function calls
              list<argument*> argList = proc->getArgList();
              if (!argList.empty()) {
                argument* arg = argList.front();
                wgslOut << "(";
                if (arg->isExpr()) {
                  generateWGSLExpr(arg->getExpr(), wgslOut, indexVar);
                }
                wgslOut << ")";
              } else {
                wgslOut << "0"; // Fallback
              }
            } else {
              wgslOut << "0"; // Other procedure calls
            }
          } else {
            wgslOut << "true"; // Fallback
          }
          break;
        }
        default: wgslOut << "0u"; break;
      }
      break;
    }
    default: wgslOut << "0u"; break;
  }
}

std::string dsl_webgpu_generator::getOpString(int opType) {
  switch (opType) {
    case OPERATOR_ADD: return "+";
    case OPERATOR_SUB: return "-";
    case OPERATOR_MUL: return "*";
    case OPERATOR_DIV: return "/";
    case OPERATOR_MOD: return "%";
    case OPERATOR_LT: return "<";
    case OPERATOR_GT: return ">";
    case OPERATOR_LE: return "<=";
    case OPERATOR_GE: return ">=";
    case OPERATOR_EQ: return "==";
    case OPERATOR_NE: return "!=";
    case OPERATOR_AND: return "&&";
    case OPERATOR_OR: return "||";
    case OPERATOR_NOT: return "!";
    case OPERATOR_ADDASSIGN: return "+=";
    case OPERATOR_SUBASSIGN: return "-=";
    case OPERATOR_MULASSIGN: return "*=";
    case OPERATOR_ORASSIGN: return "|=";
    case OPERATOR_ANDASSIGN: return "&=";
    case OPERATOR_INDEX: return "[";
    default: return "?";
  }
}

void dsl_webgpu_generator::generateProcCall(proc_callStmt* stmt, std::ofstream& out) {
  if (!stmt) return;
  
  proc_callExpr* procExpr = stmt->getProcCallExpr();
  if (!procExpr || !procExpr->getMethodId()) return;
  
  std::string methodName = procExpr->getMethodId()->getIdentifier();
  
  // Handle attachNodeProperty calls - these become buffer initialization in JS
  if (methodName == "attachNodeProperty") {
    out << "  // Property initialization: " << methodName << "\n";
  } else {
    out << "  // Unhandled proc call: " << methodName << "\n";
  }
}

void dsl_webgpu_generator::generatePropertyAccess(PropAccess* prop, std::ofstream& wgslOut, const std::string& indexVar) {
  if (!prop) return;
  
  // Property access pattern: object.property -> properties[object_id]
  Identifier* objId = prop->getIdentifier1();   // The object (e.g., 'v')
  Identifier* propId = prop->getIdentifier2();  // The property (e.g., 'deg')
  
  if (objId && propId) {
    // Map all properties to the generic 'properties' buffer
    // In the future, we can add property-specific offsets here
    wgslOut << "properties[" << objId->getIdentifier() << "]";
  }
}

void dsl_webgpu_generator::generateFixedPoint(fixedPointStmt* fp, std::ofstream& out) {
  if (!fp) return;
  
  Identifier* fixedPointId = fp->getFixedPointId();
  Expression* dependentProp = fp->getDependentProp();
  statement* body = fp->getBody();
  
  std::string fpVarName = fixedPointId ? fixedPointId->getIdentifier() : "converged";
  
  // Generate fixed point loop structure
  out << "  // Fixed point loop: " << fpVarName << "\n";
  out << "  let " << fpVarName << " = false;\n";
  out << "  let fpIterations = 0;\n";
  out << "  const maxFpIterations = 1000; // Maximum iterations to prevent infinite loops\n";
  out << "  \n";
  out << "  while (!" << fpVarName << " && fpIterations < maxFpIterations) {\n";
  out << "    " << fpVarName << " = true; // Assume convergence, will be set to false if changes occur\n";
  out << "    \n";
  
  // Generate the body (usually contains forall statements)
  if (body) {
    if (body->getTypeofNode() == NODE_BLOCKSTMT) {
      blockStatement* block = static_cast<blockStatement*>(body);
      for (statement* stmt : block->returnStatements()) {
        if (stmt->getTypeofNode() == NODE_FORALLSTMT) {
          // Generate kernel for this forall
          forallStmt* forall = static_cast<forallStmt*>(stmt);
          std::string kernelName = "kernel_" + std::to_string(kernelCounter);
          emitWGSLKernel(kernelName, forall->getBody());
          
          // Generate kernel launch
          out << "    const kernel_res_" << kernelCounter << " = await launchkernel_" << kernelCounter << "(device, adj_dataBuffer, adj_offsetsBuffer, nodeCount);\n";
          out << "    if (kernel_res_" << kernelCounter << " > 0) " << fpVarName << " = false; // Changes occurred\n";
          
          kernelCounter++;
        } else {
          // Handle other statements within the fixed point body
          generateStatement(stmt, out);
        }
      }
    } else {
      generateStatement(body, out);
    }
  }
  
  out << "    fpIterations++;\n";
  out << "  }\n";
  out << "  \n";
  out << "  console.log(`Fixed point converged after ${fpIterations} iterations`);\n";
  out << "  \n";
}

void dsl_webgpu_generator::generateDoWhile(dowhileStmt* dw, std::ofstream& out) {
  if (!dw) return;
  
  Expression* condition = dw->getCondition();
  statement* body = dw->getBody();
  
  // Generate do-while loop structure
  out << "  // Do-while loop\n";
  out << "  let doWhileIterations = 0;\n";
  out << "  const maxDoWhileIterations = 1000; // Maximum iterations to prevent infinite loops\n";
  out << "  \n";
  out << "  do {\n";
  
  // Generate the body (usually contains forall statements)
  if (body) {
    if (body->getTypeofNode() == NODE_BLOCKSTMT) {
      blockStatement* block = static_cast<blockStatement*>(body);
      for (statement* stmt : block->returnStatements()) {
        if (stmt->getTypeofNode() == NODE_FORALLSTMT) {
          // Generate kernel for this forall
          forallStmt* forall = static_cast<forallStmt*>(stmt);
          std::string kernelName = "kernel_" + std::to_string(kernelCounter);
          emitWGSLKernel(kernelName, forall->getBody());
          
          // Generate kernel launch
          out << "    const kernel_res_" << kernelCounter << " = await launchkernel_" << kernelCounter << "(device, adj_dataBuffer, adj_offsetsBuffer, nodeCount);\n";
          
          kernelCounter++;
        } else {
          // Handle other statements within the do-while body
          generateStatement(stmt, out);
        }
      }
    } else {
      generateStatement(body, out);
    }
  }
  
  out << "    doWhileIterations++;\n";
  out << "  } while (";
  
  if (condition) {
    generateExpr(condition, out);
  } else {
    out << "false"; // Default to exit
  }
  
  out << " && doWhileIterations < maxDoWhileIterations);\n";
  out << "  \n";
  out << "  console.log(`Do-while completed after ${doWhileIterations} iterations`);\n";
  out << "  \n";
}

void dsl_webgpu_generator::generateReductionStmt(reductionCallStmt* stmt, std::ofstream& out) {
  if (!stmt) return;
  
  out << "  // Reduction operation\n";
  
  if (stmt->is_reducCall()) {
    // Handle reduction call statements (like Min, Max, Sum) on integer atomics
    reductionCall* reduceCall = stmt->getReducCall();
    int reductionType = reduceCall->getReductionType();
    
    if (stmt->isLeftIdentifier()) {
      // Simple identifier reduction: variable = reduction_call(...)
      Identifier* id = stmt->getLeftId();
      out << "  " << (id ? id->getIdentifier() : "result") << " = ";
      
      // For Min/Max on scalar identifiers, use a temporary and atomicMin/Max on result
      list<argument*> argList = reduceCall->getargList();
      if (!argList.empty()) {
        argument* firstArg = argList.front();
        if (firstArg && firstArg->isExpr()) {
          if (reductionType == REDUCE_MIN) {
            out << "atomicMin(&" << (id ? id->getIdentifier() : "result") << ", ";
            generateExpr(firstArg->getExpr(), out);
            out << ")";
          } else if (reductionType == REDUCE_MAX) {
            out << "atomicMax(&" << (id ? id->getIdentifier() : "result") << ", ";
            generateExpr(firstArg->getExpr(), out);
            out << ")";
          } else {
            // Fallback: direct assignment for Sum/Product (non-atomic path not recommended)
            generateExpr(firstArg->getExpr(), out);
          }
        }
      }
      out << ";\n";
      
    } else if (stmt->getLhsType() == 2) {
      // Property reduction: object.property = reduction_call(...)
      PropAccess* prop = stmt->getPropAccess();
      if (prop && prop->getIdentifier2() && prop->getIdentifier1()) {
        out << "  " << prop->getIdentifier2()->getIdentifier() << "[" 
            << prop->getIdentifier1()->getIdentifier() << "] = ";
        
        // Property Min/Max use atomicMin/Max on properties[index]
        list<argument*> argList = reduceCall->getargList();
        if (!argList.empty()) {
          argument* firstArg = argList.front();
          if (firstArg && firstArg->isExpr()) {
            if (reductionType == REDUCE_MIN) {
              out << "atomicMin(&" << prop->getIdentifier2()->getIdentifier() << "[" 
                  << prop->getIdentifier1()->getIdentifier() << "], ";
              generateExpr(firstArg->getExpr(), out);
              out << ")";
            } else if (reductionType == REDUCE_MAX) {
              out << "atomicMax(&" << prop->getIdentifier2()->getIdentifier() << "[" 
                  << prop->getIdentifier1()->getIdentifier() << "], ";
              generateExpr(firstArg->getExpr(), out);
              out << ")";
            } else {
              generateExpr(firstArg->getExpr(), out);
            }
          }
        }
        out << ";\n";
      }
    }
    
  } else {
    // Handle reduction operations (like +=, -=, min, max)
    int reductionOp = stmt->reduction_op();
    
    if (stmt->isLeftIdentifier()) {
      // Simple identifier reduction: variable += expression
      Identifier* id = stmt->getLeftId();
      out << "  " << (id ? id->getIdentifier() : "result");
      
      // Map reduction operation to WGSL atomic operation (integers only)
      switch (reductionOp) {
        case OPERATOR_ADDASSIGN:
          out << " = atomicAdd(&" << (id ? id->getIdentifier() : "result") << ", ";
          break;
        case OPERATOR_SUBASSIGN:
          out << " = atomicSub(&" << (id ? id->getIdentifier() : "result") << ", ";
          break;
        // No atomicMul in WGSL. Product reductions need a different strategy.
        default:
          out << " += ";
          break;
      }
      
      if (stmt->getRightSide()) {
        generateExpr(stmt->getRightSide(), out);
      }
      out << ");\n";
      
    } else if (stmt->getLhsType() == 2) {
      // Property reduction: object.property += expression
      PropAccess* prop = stmt->getPropAccess();
      if (prop && prop->getIdentifier2() && prop->getIdentifier1()) {
        out << "  " << prop->getIdentifier2()->getIdentifier() << "[" 
            << prop->getIdentifier1()->getIdentifier() << "]";
        
      // Map reduction operation to WGSL atomic operation (integers only)
        switch (reductionOp) {
          case OPERATOR_ADDASSIGN:
            out << " = atomicAdd(&" << prop->getIdentifier2()->getIdentifier() << "[" 
                << prop->getIdentifier1()->getIdentifier() << "], ";
            break;
          case OPERATOR_SUBASSIGN:
            out << " = atomicSub(&" << prop->getIdentifier2()->getIdentifier() << "[" 
                << prop->getIdentifier1()->getIdentifier() << "], ";
            break;
        // No atomicMul in WGSL. Product reductions need a different strategy.
          default:
            out << " += ";
            break;
        }
        
        if (stmt->getRightSide()) {
          generateExpr(stmt->getRightSide(), out);
        }
        out << ");\n";
      }
    }
  }
}

} // namespace spwebgpu

