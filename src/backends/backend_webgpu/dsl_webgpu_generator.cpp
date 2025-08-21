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
  if (node->getTypeofNode() == NODE_FORALLSTMT) {
    forallStmt* fa = static_cast<forallStmt*>(node);
    if (fa && fa->getBody()) {
      return findFirstReductionTargetId(fa->getBody());
    }
  }
  if (node->getTypeofNode() == NODE_FIXEDPTSTMT) {
    fixedPointStmt* fp = static_cast<fixedPointStmt*>(node);
    if (fp && fp->getBody()) {
      return findFirstReductionTargetId(fp->getBody());
    }
  }
  return nullptr;
}
}

dsl_webgpu_generator::dsl_webgpu_generator() {}
dsl_webgpu_generator::~dsl_webgpu_generator() {}

void dsl_webgpu_generator::generate(ASTNode* root, const std::string& outFile) {
  std::cout << "[WebGPU] Generate called with root=" << root << ", outFile=" << outFile << std::endl;
  if (!root) {
    std::cerr << "[WebGPU] Error: root ASTNode is null!" << std::endl;
    return;
  }
  
  // Additional validation
  try {
    int nodeType = root->getTypeofNode();
    std::cout << "[WebGPU] Root node type: " << nodeType << std::endl;
    if (nodeType != NODE_FUNC) {
      std::cerr << "[WebGPU] Error: Expected function node, got type: " << nodeType << std::endl;
      return;
    }
  } catch (...) {
    std::cerr << "[WebGPU] Error: Exception when accessing root node type!" << std::endl;
    return;
  }
  
  std::ofstream out(outFile);
  if (!out.is_open()) {
    std::cerr << "[WebGPU] Failed to open output file: " << outFile << std::endl;
    return;
  }
  std::cout << "[WebGPU] Starting code generation for: " << outFile << std::endl;
  
  try {
    buildPropertyRegistry(static_cast<Function*>(root));
  generateFunc(root, out);
  } catch (const std::exception& e) {
    std::cerr << "[WebGPU] Exception during generation: " << e.what() << std::endl;
  } catch (...) {
    std::cerr << "[WebGPU] Unknown exception during generation!" << std::endl;
  }
  
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
  
  // Phase 3.18: Add modular utility imports
  out << "// StarPlat WebGPU Generated Algorithm - Modular Implementation (Phase 3.18)\n";
  out << "// Import utility functions for buffer management, pipeline caching, etc.\n";
  out << "import { WebGPUBufferUtils } from '../webgpu_utils/host_utils/webgpu_buffer_utils.js';\n";
  out << "import { WebGPUPipelineManager } from '../webgpu_utils/host_utils/webgpu_pipeline_manager.js';\n\n";
  
  out << "export async function " << funcName << "(device, adj_dataBuffer, adj_offsetsBuffer, nodeCount, props = {}, rev_adj_dataBuffer = null, rev_adj_offsetsBuffer = null) {\n";
  out << "  console.log('[WebGPU] Compute start: " << funcName << " with nodeCount=', nodeCount);\n";
  out << "  \n";
  out << "  // Phase 3.18: Use modular utilities for improved performance and maintainability\n";
  out << "  const bufferUtils = new WebGPUBufferUtils(device);\n";
  out << "  const pipelineManager = new WebGPUPipelineManager(device);\n";
  out << "  let " << resultVar << " = 0;\n";
  // Allocate shared result and properties buffers using utilities (Phase 3.18 optimization)
  out << "  const resultBuffer = bufferUtils.createEmptyStorageBuffer(4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);\n";
  if (!propInfos.empty()) { out << "  const propEntries = [];\n"; }
  if (!propInfos.empty()) {
    for (const auto &p : propInfos) {
      out << "  const " << p.name << "Buffer = (props['" << p.name << "'] && props['" << p.name << "'].buffer) ? props['" << p.name << "'].buffer : device.createBuffer({ size: (props['" << p.name << "'] && props['" << p.name << "'].data) ? props['" << p.name << "'].data.byteLength : Math.max(1, nodeCount) * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });\n";
      out << "  if (props['" << p.name << "'] && props['" << p.name << "'].data && !(props['" << p.name << "'].buffer)) { device.queue.writeBuffer(" << p.name << "Buffer, 0, props['" << p.name << "'].data); }\n";
      out << "  propEntries.push({ binding: " << p.bindingIndex << ", resource: { buffer: " << p.name << "Buffer } });\n";
    }
  } else {
  out << "  const propertyBuffer = device.createBuffer({ size: Math.max(1, nodeCount) * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });\n";
  }
  out << "  device.queue.writeBuffer(resultBuffer, 0, new Uint32Array([0]));\n";
  // Params uniform buffer for kernel constants (e.g., node_count)
  out << "  const paramsBuffer = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });\n";
  out << "  device.queue.writeBuffer(paramsBuffer, 0, new Uint32Array([nodeCount, 0, 0, 0]));\n";
  int launchIndex = 0;
  // Generate host-side sequencing for the function body
  generateHostBody(func->getBlockStatement(), out, launchIndex);
  // Optional property readback for out/inout
  if (!propInfos.empty()) {
    for (const auto &p : propInfos) {
      std::string jsCtor = (p.wgslType == "u32") ? "Uint32Array" : (p.wgslType == "i32") ? "Int32Array" : "Float32Array";
      out << "  if (props['" << p.name << "'] && (props['" << p.name << "'].usage === 'out' || props['" << p.name << "'].usage === 'inout' || props['" << p.name << "'].readback === true)) {\n";
      out << "    const sizeBytes_" << p.name << " = (props['" << p.name << "'].data) ? props['" << p.name << "'].data.byteLength : Math.max(1, nodeCount) * 4;\n";
      out << "    const rb_" << p.name << " = device.createBuffer({ size: sizeBytes_" << p.name << ", usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });\n";
      out << "    { const enc = device.createCommandEncoder(); enc.copyBufferToBuffer(" << p.name << "Buffer, 0, rb_" << p.name << ", 0, sizeBytes_" << p.name << "); device.queue.submit([enc.finish()]); }\n";
      out << "    await rb_" << p.name << ".mapAsync(GPUMapMode.READ);\n";
      out << "    const view_" << p.name << " = new " << jsCtor << "(rb_" << p.name << ".getMappedRange());\n";
      out << "    props['" << p.name << "'].dataOut = new " << jsCtor << "(view_" << p.name << ");\n";
      out << "    rb_" << p.name << ".unmap();\n";
      out << "  }\n";
    }
  }
  out << "  console.log('[WebGPU] Compute end: returning result', " << resultVar << ");\n";
  out << "  \n";
  out << "  // Phase 3.18: Cleanup utility resources\n";
  out << "  bufferUtils.cleanup();\n";
  out << "  \n";
  out << "  return " << resultVar << ";\n";
  out << "}\n\n";

  // Generate the kernel launch functions  
  // Phase 3.18 Note: These functions can be further optimized using pipelineManager
  // for caching shaders, pipelines, and bind groups (Tasks 3.1, 3.2)
  for (int i = 0; i < kernelCounter; ++i) {
    out << "async function launchkernel_" << i << "(device, adj_dataBuffer, adj_offsetsBuffer, paramsBuffer, resultBuffer, " << (propInfos.empty()? std::string("propertyBuffer") : std::string("propEntries")) << ", nodeCount, rev_adj_dataBuffer = null, rev_adj_offsetsBuffer = null) {\n";
    out << "  console.log('[WebGPU] launchkernel_" << i << ": begin');\n";
    out << "  // TODO Phase 3.18: Use pipelineManager.getShaderModule() for caching\n";
    out << "  const shaderCode = await (await fetch('kernel_" << i << ".wgsl')).text();\n";
    out << "  console.log('[WebGPU] launchkernel_" << i << ": WGSL fetched, size', shaderCode.length);\n";
    out << "  const shaderModule = device.createShaderModule({ code: shaderCode });\n";
    out << "  if (shaderModule.getCompilationInfo) {\n";
    out << "    const info = await shaderModule.getCompilationInfo();\n";
    out << "    for (const m of info.messages || []) {\n";
    out << "      const s = m.lineNum !== undefined ? `${m.lineNum}:${m.linePos}` : '';\n";
    out << "      console[(m.type === 'error') ? 'error' : 'warn']('[WGSL]', m.type, s, m.message);\n";
    out << "    }\n";
    out << "  }\n";
    out << "  const bindEntries = [\n";
    out << "    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },\n";
    out << "    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },\n";
    out << "    { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },\n";
    out << "    { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }\n";
    out << "  ];\n";
    if (!propInfos.empty()) {
      for (const auto &p : propInfos) { out << "  bindEntries.push({ binding: " << p.bindingIndex << ", visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } });\n"; }
    } else {
      out << "  bindEntries.push({ binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } });\n";
    }
    out << "  const bindGroupLayout = device.createBindGroupLayout({ entries: bindEntries });\n";
    out << "  const pipeline = device.createComputePipeline({ layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }), compute: { module: shaderModule, entryPoint: 'main' } });\n";
    out << "  console.log('[WebGPU] launchkernel_" << i << ": pipeline created');\n";
    out << "  \n";
    out << "  // Using shared result/property buffers provided by caller\n";
    out << "  const readBuffer = device.createBuffer({ \n";
    out << "    size: 4, \n";
    out << "    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ \n";
    out << "  });\n";
    out << "  \n";
    out << "  const entries = [\n";
    out << "      { binding: 0, resource: { buffer: adj_offsetsBuffer } },\n";
    out << "      { binding: 1, resource: { buffer: adj_dataBuffer } },\n";
    out << "      { binding: 4, resource: { buffer: paramsBuffer } },\n";
    out << "      { binding: 5, resource: { buffer: resultBuffer } }\n";
    out << "  ];\n";
    out << "  // Add reverse CSR buffers if provided\n";
    out << "  if (rev_adj_offsetsBuffer) {\n";
    out << "    entries.push({ binding: 2, resource: { buffer: rev_adj_offsetsBuffer } });\n";
    out << "  }\n";
    out << "  if (rev_adj_dataBuffer) {\n";
    out << "    entries.push({ binding: 3, resource: { buffer: rev_adj_dataBuffer } });\n";
    out << "  }\n";
    if (!propInfos.empty()) { out << "  entries.push(...propEntries);\n"; } else { out << "  entries.push({ binding: 6, resource: { buffer: propertyBuffer } });\n"; }
    out << "  const bindGroup = device.createBindGroup({ layout: bindGroupLayout, entries });\n";
    out << "  console.log('[WebGPU] launchkernel_" << i << ": bindGroup created');\n";
    out << "  \n";
    out << "  const encoder = device.createCommandEncoder();\n";
    out << "  const pass = encoder.beginComputePass();\n";
    out << "  pass.setPipeline(pipeline);\n";
    out << "  pass.setBindGroup(0, bindGroup);\n";
    out << "  \n";
    out << "  // Dispatch one workgroup per 256 nodes (ensure at least 1 group)\n";
    out << "  let __groups = Math.ceil(nodeCount / 256);\n";
    out << "  if (__groups < 1) { __groups = 1; }\n";
    out << "  pass.dispatchWorkgroups(__groups, 1, 1);\n";
    out << "  pass.end();\n";
    out << "  console.log('[WebGPU] launchkernel_" << i << ": dispatched groups', __groups);\n";
    out << "  \n";
    out << "  encoder.copyBufferToBuffer(resultBuffer, 0, readBuffer, 0, 4);\n";
    out << "  device.queue.submit([encoder.finish()]);\n";
    out << "  console.log('[WebGPU] launchkernel_" << i << ": submitted');\n";
    out << "  \n";
    out << "  try {\n";
    out << "    await Promise.race([readBuffer.mapAsync(GPUMapMode.READ), new Promise((_,rej)=>setTimeout(()=>rej(new Error('mapAsync timeout')), 10000))]);\n";
    out << "  } catch (e) { console.error('[WebGPU] mapAsync error:', e?.message || e); throw e; }\n";
    out << "  const result = new Uint32Array(readBuffer.getMappedRange())[0];\n";
    out << "  readBuffer.unmap();\n";
    out << "  console.log('[WebGPU] launchkernel_" << i << ": result', result);\n";
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
    out << "  // Reset result before dispatch\n";
    out << "  device.queue.writeBuffer(resultBuffer, 0, new Uint32Array([0]));\n";
    out << "  const kernel_res_" << launchIndex << " = await launchkernel_" << launchIndex << "(device, adj_dataBuffer, adj_offsetsBuffer, paramsBuffer, resultBuffer, " << (propInfos.empty()? std::string("propertyBuffer") : std::string("propEntries")) << ", nodeCount, rev_adj_dataBuffer, rev_adj_offsetsBuffer);\n";
    // Assign to generic result for now
    out << "  result = kernel_res_" << launchIndex << ";\n";
    // If there is a scalar reduction target like 'triangle_count', keep it in sync
    Identifier* __rid = findFirstReductionTargetId(node);
    if (__rid && __rid->getIdentifier() != nullptr) {
      out << "  " << __rid->getIdentifier() << " = kernel_res_" << launchIndex << ";\n";
    }
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
    // Reset result once per iteration to aggregate changes across inner kernels
    out << "    device.queue.writeBuffer(resultBuffer, 0, new Uint32Array([0]));\n";
    if (fp->getBody()) {
      generateHostBody(fp->getBody(), out, launchIndex);
      // Heuristic: if any kernel produced non-zero result, mark not converged
      out << "    if (result > 0) { " << fpVarName << " = false; }\n";
    }
    out << "    fpIterations++;\n";
    out << "  }\n";
    return;
  }

  if (node->getTypeofNode() == NODE_DOWHILESTMT) {
    dowhileStmt* dw = static_cast<dowhileStmt*>(node);
    out << "  // Do-while loop (WebGPU host)\n";
    out << "  let __dwIterations = 0; const __dwMax = 1000;\n";
    out << "  do {\n";
    out << "    device.queue.writeBuffer(resultBuffer, 0, new Uint32Array([0]));\n";
    if (dw->getBody()) {
      generateHostBody(dw->getBody(), out, launchIndex);
    }
    out << "    __dwIterations++;\n";
    out << "  } while (";
    if (dw->getCondition()) { generateExpr(dw->getCondition(), out); }
    else { out << "false"; }
    out << " && __dwIterations < __dwMax);\n";
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
      // Ignore DSL return in host; final return is handled at end of function
      out << "  // [WebGPU] ignoring DSL return in host orchestration\n";
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
            } else if (methodName == "count_inNbrs") {
              // Count of in neighbors is not computed on host; placeholder 0
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
    case NODE_PROCCALLSTMT: {
      // A statement in an expression context: try to extract expr part safely
      proc_callStmt* stmt = static_cast<proc_callStmt*>(node);
      proc_callExpr* proc = stmt ? stmt->getProcCallExpr() : nullptr;
      if (proc && proc->getMethodId()) {
        // Re-dispatch through expression handling by constructing a minimal pathway
        // Handle a few known calls used within expressions
        std::string methodName = proc->getMethodId()->getIdentifier();
        if (methodName == "is_an_edge") {
          out << "true"; // placeholder on host
        } else if (methodName == "count_outNbrs") {
          out << "0"; // placeholder on host
        } else if (methodName == "count_inNbrs") {
          out << "0"; // placeholder on host
        } else {
          out << "0";
        }
      } else {
        out << "0";
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
  // Forward CSR (required)
  wgslOut << "@group(0) @binding(0) var<storage, read> adj_offsets: array<u32>;\n";
  wgslOut << "@group(0) @binding(1) var<storage, read> adj_data: array<u32>;\n";
  // Reverse CSR (optional for algorithms needing incoming neighbors)
  wgslOut << "@group(0) @binding(2) var<storage, read> rev_adj_offsets: array<u32>;\n";
  wgslOut << "@group(0) @binding(3) var<storage, read> rev_adj_data: array<u32>;\n";
  // Params and result
  wgslOut << "struct Params { node_count: u32; _pad0: u32; _pad1: u32; _pad2: u32; };\n";
  wgslOut << "@group(0) @binding(4) var<uniform> params: Params;\n";
  wgslOut << "@group(0) @binding(5) var<storage, read_write> result: atomic<u32>;\n";
  if (!propInfos.empty()) {
    for (const auto &p : propInfos) {
      // Use atomic<u32> backing for all properties to support atomic ops and float CAS via bitcast
      wgslOut << "@group(0) @binding(" << p.bindingIndex << ") var<storage, read_write> " << p.name << ": array<atomic<u32>>;\n";
    }
    wgslOut << "\n";
  } else {
    wgslOut << "@group(0) @binding(6) var<storage, read_write> properties: array<atomic<u32>>;\n\n";
  }
  
  // --- Workgroup Shared Memory for Parallel Reductions ---
  wgslOut << "var<workgroup> scratchpad: array<u32, 256>;\n";
  wgslOut << "var<workgroup> scratchpad_f32: array<f32, 256>;\n\n";
  
  // Phase 3.18: Replace inlined utilities with modular includes
  includeWGSLUtilities(wgslOut);
  
  // --- Main Entry Point ---
  wgslOut << "@compute @workgroup_size(256)\n";
  wgslOut << "fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>) {\n";
  wgslOut << "  let v = global_id.x;\n";
  wgslOut << "  let node_count = params.node_count;\n";
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
  // Replace generic edge loops for the known triangle counting pattern
  // with distinct loop indices to avoid reuse.
  // Note: best-effort textual replacement on the generated body fragment.
  // (No-op if pattern not present.)
  wgslOut << ""; // placeholder to keep structure intact
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

    // Helper to check if target is a local variable (not a property)
    auto isLocalVariable = [&](Identifier* id) -> bool {
      if (!id) return false;
      std::string name = id->getIdentifier();
      // Check if it's a known property
      for (const auto& prop : propInfos) {
        if (prop.name == name) return false;
      }
      // Check if it matches common local variable patterns or is not a global property
      return (name.find("_count") != std::string::npos || 
              name.find("_sum") != std::string::npos ||
              name.find("local_") != std::string::npos ||
              name == "count" || name == "sum" || name == "temp");
    };

    // Validation lambda for reduction targets
    auto validateReductionTarget = [&](reductionCallStmt* r) -> bool {
      if (!r) return false;
      
      if (r->isLeftIdentifier()) {
        Identifier* leftId = r->getLeftId();
        if (!leftId || !leftId->getIdentifier()) {
          std::cerr << "[WebGPU] Warning: Reduction target identifier is null" << std::endl;
          return false;
        }
        return true;
      } else if (r->getLhsType() == 2) {
        PropAccess* prop = r->getPropAccess();
        if (!prop || !prop->getIdentifier2()) {
          std::cerr << "[WebGPU] Warning: Reduction target property is invalid" << std::endl;
          return false;
        }
        return true;
      }
      
      std::cerr << "[WebGPU] Warning: Unknown reduction target type" << std::endl;
      return false;
    };

    // Validation lambda for operator types
    auto validateOperatorType = [&](int opKind, Expression* rhsExpr) -> bool {
      if (opKind < 0 || opKind > 2) {
        std::cerr << "[WebGPU] Warning: Invalid reduction operator kind: " << opKind << std::endl;
        return false;
      }
      
      // Check for division by zero in RHS expressions
      if (rhsExpr && rhsExpr->getExpressionFamily() == EXPR_INTCONSTANT) {
        if (rhsExpr->getIntegerConstant() == 0 && (opKind == 0 || opKind == 1 || opKind == 2)) {
          std::cerr << "[WebGPU] Warning: Potential division by zero in reduction" << std::endl;
          return false;
        }
      }
      
      return true;
    };

    auto emitAtomicOp = [&](const std::string &ptr) {
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
    
    auto emitNonAtomicOp = [&](const std::string &var, Expression* rhs) {
      wgslOut << indent << var;
      if (opKind == 1) {
        wgslOut << " = min(" << var << ", ";
      } else if (opKind == 2) {
        wgslOut << " = max(" << var << ", ";
      } else {
        wgslOut << " += ";
      }
      if (rhs) {
        if (rhsIsFloat) { wgslOut << "f32("; generateWGSLExpr(rhs, wgslOut, indexVar); wgslOut << ")"; }
        else { wgslOut << "u32("; generateWGSLExpr(rhs, wgslOut, indexVar); wgslOut << ")"; }
      } else {
        wgslOut << (rhsIsFloat ? "1.0" : "1u");
      }
      if (opKind == 1 || opKind == 2) wgslOut << ")";
      wgslOut << ";\n";
    };

    // Apply validation
    if (!validateReductionTarget(r)) {
      wgslOut << indent << "// Invalid reduction target - skipping\n";
      return;
    }
    
    if (!validateOperatorType(opKind, rhsExpr)) {
      wgslOut << indent << "// Invalid operator type - skipping\n";
      return;
    }

    // Decide target pointer
    if (r->isLeftIdentifier()) {
      Identifier* leftId = r->getLeftId();
      if (isLocalVariable(leftId)) {
        // Local variable - use non-atomic operations
        Expression* rhs = nullptr;
        if (r->is_reducCall()) {
          reductionCall* rc = r->getReducCall();
          auto args = rc->getargList();
          if (!args.empty()) {
            argument* a = args.front();
            if (a && a->isExpr()) rhs = a->getExpr();
          }
        } else if (r->getRightSide()) {
          rhs = r->getRightSide();
        }
        emitNonAtomicOp(leftId ? leftId->getIdentifier() : "result", rhs);
      } else {
        // Global property or result - use atomic operations
        wgslOut << indent;
        emitAtomicOp("&result");
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
      }
    } else if (r->getLhsType() == 2) {
      // Property target - validate property access
      PropAccess* prop = r->getPropAccess();
      if (!prop || (!prop->getIdentifier1() && !prop->getPropExpr())) {
        wgslOut << indent << "// Invalid property access in reduction - skipping\n";
        return;
      }
      
      wgslOut << indent;
      std::string arr = prop && prop->getIdentifier2() ? prop->getIdentifier2()->getIdentifier() : "properties";
      // Emit operator and pointer inline (properties are always atomic)
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
      // Fallback to result (always atomic)
      wgslOut << indent;
      emitAtomicOp("&result");
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
        // For atomic reductions to global counters, use atomicAdd to result
        wgslOut << indent << "atomicAdd(&result, ";
        if (asst->getExpr()) { wgslOut << "u32("; generateWGSLExpr(asst->getExpr(), wgslOut, indexVar); wgslOut << ")"; }
        else { wgslOut << "1u"; }
        wgslOut << ");\n";
      } else {
        // Check if this is a compound assignment expression
        Expression* expr = asst->getExpr();
        if (expr && expr->getTypeofNode() == NODE_EXPR) {
          Expression* e = static_cast<Expression*>(expr);
          if (e->getExpressionFamily() == EXPR_ARITHMETIC) {
            // Check if this is a compound assignment pattern: var = var OP value
            Expression* left = e->getLeft();
            Expression* right = e->getRight();
            if (left && left->getExpressionFamily() == EXPR_ID) {
              Identifier* leftId = left->getId();
              if (leftId && id && strcmp(leftId->getIdentifier(), id->getIdentifier()) == 0) {
                // This is a compound assignment: generate var OP= value
                wgslOut << indent << id->getIdentifier();
                int opType = e->getOperatorType();
                switch (opType) {
                  case OPERATOR_ADD: wgslOut << " += "; break;
                  case OPERATOR_SUB: wgslOut << " -= "; break;
                  case OPERATOR_MUL: wgslOut << " *= "; break;
                  case OPERATOR_DIV: wgslOut << " /= "; break;
                  case OPERATOR_OR: wgslOut << " |= "; break;
                  case OPERATOR_AND: wgslOut << " &= "; break;
                  default:
                    // Fallback to regular assignment
                    wgslOut << " = ";
                    generateWGSLExpr(asst->getExpr(), wgslOut, indexVar);
                    wgslOut << ";\n";
                    return;
                }
                generateWGSLExpr(right, wgslOut, indexVar);
                wgslOut << ";\n";
                return;
              }
            }
          }
        }
        // Regular assignment
        wgslOut << indent << (id ? id->getIdentifier() : "unnamed") << " = ";
        generateWGSLExpr(asst->getExpr(), wgslOut, indexVar);
        wgslOut << ";\n";
      }
    } else if (asst->lhs_isProp()) {
      PropAccess* prop = asst->getPropId();
      std::string arr = "properties";
      std::string wgslType = "u32";
      if (prop && prop->getIdentifier2() && prop->getIdentifier2()->getIdentifier()) {
        arr = prop->getIdentifier2()->getIdentifier();
        for (const auto &p : propInfos) { if (p.name == arr) { wgslType = p.wgslType; break; } }
      }
      
      // Check if this is a compound assignment pattern: prop.field = prop.field OP value
      Expression* expr = asst->getExpr();
      if (expr && expr->getTypeofNode() == NODE_EXPR) {
        Expression* e = static_cast<Expression*>(expr);
        if (e->getExpressionFamily() == EXPR_ARITHMETIC) {
          Expression* left = e->getLeft();
          Expression* right = e->getRight();
          if (left && left->getExpressionFamily() == EXPR_PROPID) {
            PropAccess* leftProp = left->getPropId();
            // Check if left property matches the assignment target
            if (leftProp && prop && 
                leftProp->getIdentifier1() && prop->getIdentifier1() &&
                leftProp->getIdentifier2() && prop->getIdentifier2() &&
                strcmp(leftProp->getIdentifier1()->getIdentifier(), prop->getIdentifier1()->getIdentifier()) == 0 &&
                strcmp(leftProp->getIdentifier2()->getIdentifier(), prop->getIdentifier2()->getIdentifier()) == 0) {
              
              // Generate atomic compound assignment
              std::string indexExpr;
              if (prop->getIdentifier1()) { 
                indexExpr = prop->getIdentifier1()->getIdentifier(); 
              } else if (prop->getPropExpr()) { 
                // For complex property expressions, we'd need to generate the expression
                indexExpr = "0u"; // Simplified for now
              } else { 
                indexExpr = "0u"; 
              }
              
              int opType = e->getOperatorType();
              switch (opType) {
                case OPERATOR_ADD:
                  if (wgslType == "f32") {
                    wgslOut << indent << "atomicAddF32(&" << arr << "[" << indexExpr << "], f32(";
                    generateWGSLExpr(right, wgslOut, indexVar);
                    wgslOut << "));\n";
                  } else {
                    wgslOut << indent << "atomicAdd(&" << arr << "[" << indexExpr << "], u32(";
                    generateWGSLExpr(right, wgslOut, indexVar);
                    wgslOut << "));\n";
                  }
                  wgslOut << indent << "atomicAdd(&result, 1u); // Signal change for convergence\n";
                  return;
                  
                case OPERATOR_SUB:
                  if (wgslType == "f32") {
                    wgslOut << indent << "atomicSubF32(&" << arr << "[" << indexExpr << "], f32(";
                    generateWGSLExpr(right, wgslOut, indexVar);
                    wgslOut << "));\n";
                  } else {
                    wgslOut << indent << "atomicSub(&" << arr << "[" << indexExpr << "], u32(";
                    generateWGSLExpr(right, wgslOut, indexVar);
                    wgslOut << "));\n";
                  }
                  wgslOut << indent << "atomicAdd(&result, 1u); // Signal change for convergence\n";
                  return;
                  
                case OPERATOR_OR:
                  // Bitwise OR assignment: prop[idx] |= value
                  wgslOut << indent << "atomicOr(&" << arr << "[" << indexExpr << "], u32(";
                  generateWGSLExpr(right, wgslOut, indexVar);
                  wgslOut << "));\n";
                  wgslOut << indent << "atomicAdd(&result, 1u); // Signal change for convergence\n";
                  return;
                  
                case OPERATOR_AND:
                  // Bitwise AND assignment: prop[idx] &= value
                  wgslOut << indent << "atomicAnd(&" << arr << "[" << indexExpr << "], u32(";
                  generateWGSLExpr(right, wgslOut, indexVar);
                  wgslOut << "));\n";
                  wgslOut << indent << "atomicAdd(&result, 1u); // Signal change for convergence\n";
                  return;
                  
                case OPERATOR_MUL:
                case OPERATOR_DIV:
                  // For multiplication and division, we need to use compare-and-swap
                  wgslOut << indent << "// Compound " << (opType == OPERATOR_MUL ? "*=" : "/=") << " for property requires CAS\n";
                  wgslOut << indent << "{\n";
                  wgslOut << indent << "  loop {\n";
                  wgslOut << indent << "    let oldBits = atomicLoad(&" << arr << "[" << indexExpr << "]);\n";
                  if (wgslType == "f32") {
                    wgslOut << indent << "    let oldVal = bitcast<f32>(oldBits);\n";
                    wgslOut << indent << "    let newVal = oldVal " << (opType == OPERATOR_MUL ? "*" : "/") << " f32(";
                    generateWGSLExpr(right, wgslOut, indexVar);
                    wgslOut << ");\n";
                    wgslOut << indent << "    let newBits = bitcast<u32>(newVal);\n";
                  } else {
                    wgslOut << indent << "    let oldVal = oldBits;\n";
                    wgslOut << indent << "    let newVal = oldVal " << (opType == OPERATOR_MUL ? "*" : "/") << " u32(";
                    generateWGSLExpr(right, wgslOut, indexVar);
                    wgslOut << ");\n";
                    wgslOut << indent << "    let newBits = newVal;\n";
                  }
                  wgslOut << indent << "    let res = atomicCompareExchangeWeak(&" << arr << "[" << indexExpr << "], oldBits, newBits);\n";
                  wgslOut << indent << "    if (res.exchanged) {\n";
                  wgslOut << indent << "      if (oldBits != newBits) { atomicAdd(&result, 1u); }\n";
                  wgslOut << indent << "      break;\n";
                  wgslOut << indent << "    }\n";
                  wgslOut << indent << "  }\n";
                  wgslOut << indent << "}\n";
                  return;
                  
                default:
                  // Fall through to regular assignment handling
                  break;
              }
            }
          }
        }
      }
      
      // Regular property assignment with compare-and-flag for convergence
      wgslOut << indent << "let __oldBits: u32 = atomicLoad(&" << arr << "[";
      if (prop && prop->getIdentifier1()) { wgslOut << prop->getIdentifier1()->getIdentifier(); }
      else if (prop && prop->getPropExpr()) { generateWGSLExpr(prop->getPropExpr(), wgslOut, indexVar); }
      else { wgslOut << "0u"; }
      wgslOut << "]);\n";
      wgslOut << indent << "let __newBits: u32 = ";
      if (wgslType == "f32") {
        wgslOut << "bitcast<u32>(f32("; generateWGSLExpr(asst->getExpr(), wgslOut, indexVar); wgslOut << "))";
      } else {
        wgslOut << "u32("; generateWGSLExpr(asst->getExpr(), wgslOut, indexVar); wgslOut << ")";
      }
      wgslOut << ";\n";
      wgslOut << indent << "if (__oldBits != __newBits) { atomicStore(&" << arr << "[";
      if (prop && prop->getIdentifier1()) { wgslOut << prop->getIdentifier1()->getIdentifier(); }
      else if (prop && prop->getPropExpr()) { generateWGSLExpr(prop->getPropExpr(), wgslOut, indexVar); }
      else { wgslOut << "0u"; }
      wgslOut << "], __newBits); atomicAdd(&result, 1u); }\n";
    } else if (asst->lhs_isIndexAccess()) {
      // Handle index access assignment: array[index] = expression
      // Check if this is a compound assignment pattern: arr[idx] = arr[idx] OP value
      Expression* expr = asst->getExpr();
      if (expr && expr->getTypeofNode() == NODE_EXPR) {
        Expression* e = static_cast<Expression*>(expr);
        if (e->getExpressionFamily() == EXPR_ARITHMETIC) {
          Expression* left = e->getLeft();
          Expression* right = e->getRight();
          if (left && left->getExpressionFamily() == EXPR_MAPGET) {
            // Check if left side matches the assignment target
            Expression* leftMapExpr = left->getMapExpr();
            Expression* leftIndexExpr = left->getIndexExpr();
            Expression* targetMapExpr = asst->getIndexAccess()->getMapExpr();
            Expression* targetIndexExpr = asst->getIndexAccess()->getIndexExpr();
            
            // For simplicity, check if both are accessing the same array name and index
            // This is a basic pattern match - could be enhanced for more complex cases
            if (leftMapExpr && targetMapExpr && leftIndexExpr && targetIndexExpr) {
              bool isCompoundAssignment = false;
              
              // Check if map expressions refer to the same identifier
              if (leftMapExpr->getExpressionFamily() == EXPR_ID && targetMapExpr->getExpressionFamily() == EXPR_ID) {
                Identifier* leftId = leftMapExpr->getId();
                Identifier* targetId = targetMapExpr->getId();
                if (leftId && targetId && 
                    strcmp(leftId->getIdentifier(), targetId->getIdentifier()) == 0) {
                  
                  // Check if index expressions are the same
                  if (leftIndexExpr->getExpressionFamily() == EXPR_ID && targetIndexExpr->getExpressionFamily() == EXPR_ID) {
                    Identifier* leftIdx = leftIndexExpr->getId();
                    Identifier* targetIdx = targetIndexExpr->getId();
                    if (leftIdx && targetIdx && 
                        strcmp(leftIdx->getIdentifier(), targetIdx->getIdentifier()) == 0) {
                      isCompoundAssignment = true;
                    }
                  }
                }
              }
              
              if (isCompoundAssignment) {
                // Generate compound assignment for index access
                std::string arrName = targetMapExpr->getId()->getIdentifier();
                std::string indexName = targetIndexExpr->getId()->getIdentifier();
                
                // Check if this is a property array (requires atomic operations)
                bool isPropertyArray = false;
                std::string wgslType = "u32";
                for (const auto &p : propInfos) {
                  if (p.name == arrName) { 
                    wgslType = p.wgslType; 
                    isPropertyArray = true;
                    break; 
                  }
                }
                
                int opType = e->getOperatorType();
                
                if (isPropertyArray) {
                  // Property arrays: use atomic operations
                switch (opType) {
                  case OPERATOR_ADD:
                    if (wgslType == "f32") {
                      wgslOut << indent << "atomicAddF32(&" << arrName << "[" << indexName << "], f32(";
                      generateWGSLExpr(right, wgslOut, indexVar);
                      wgslOut << "));\n";
                    } else {
                      wgslOut << indent << "atomicAdd(&" << arrName << "[" << indexName << "], u32(";
                      generateWGSLExpr(right, wgslOut, indexVar);
                      wgslOut << "));\n";
                    }
                    wgslOut << indent << "atomicAdd(&result, 1u); // Signal change for convergence\n";
                    return;
                    
                  case OPERATOR_SUB:
                    if (wgslType == "f32") {
                      wgslOut << indent << "atomicAddF32(&" << arrName << "[" << indexName << "], -f32(";
                      generateWGSLExpr(right, wgslOut, indexVar);
                      wgslOut << "));\n";
                    } else {
                      wgslOut << indent << "atomicSub(&" << arrName << "[" << indexName << "], u32(";
                      generateWGSLExpr(right, wgslOut, indexVar);
                      wgslOut << "));\n";
                    }
                    wgslOut << indent << "atomicAdd(&result, 1u); // Signal change for convergence\n";
                    return;
                    
                  case OPERATOR_OR:
                    // Bitwise OR assignment: arr[idx] |= value
                    wgslOut << indent << "atomicOr(&" << arrName << "[" << indexName << "], u32(";
                    generateWGSLExpr(right, wgslOut, indexVar);
                    wgslOut << "));\n";
                    wgslOut << indent << "atomicAdd(&result, 1u); // Signal change for convergence\n";
                    return;
                    
                  case OPERATOR_AND:
                    // Bitwise AND assignment: arr[idx] &= value
                    wgslOut << indent << "atomicAnd(&" << arrName << "[" << indexName << "], u32(";
                    generateWGSLExpr(right, wgslOut, indexVar);
                    wgslOut << "));\n";
                    wgslOut << indent << "atomicAdd(&result, 1u); // Signal change for convergence\n";
                    return;
                    
                  case OPERATOR_MUL:
                  case OPERATOR_DIV:
                    // For multiplication and division, use compare-and-swap
                    wgslOut << indent << "// Compound " << (opType == OPERATOR_MUL ? "*=" : "/=") << " for array requires CAS\n";
                    wgslOut << indent << "{\n";
                    wgslOut << indent << "  loop {\n";
                    wgslOut << indent << "    let oldBits = atomicLoad(&" << arrName << "[" << indexName << "]);\n";
                    if (wgslType == "f32") {
                      wgslOut << indent << "    let oldVal = bitcast<f32>(oldBits);\n";
                      wgslOut << indent << "    let newVal = oldVal " << (opType == OPERATOR_MUL ? "*" : "/") << " f32(";
                      generateWGSLExpr(right, wgslOut, indexVar);
                      wgslOut << ");\n";
                      wgslOut << indent << "    let newBits = bitcast<u32>(newVal);\n";
                    } else {
                      wgslOut << indent << "    let oldVal = oldBits;\n";
                      wgslOut << indent << "    let newVal = oldVal " << (opType == OPERATOR_MUL ? "*" : "/") << " u32(";
                      generateWGSLExpr(right, wgslOut, indexVar);
                      wgslOut << ");\n";
                      wgslOut << indent << "    let newBits = newVal;\n";
                    }
                    wgslOut << indent << "    let res = atomicCompareExchangeWeak(&" << arrName << "[" << indexName << "], oldBits, newBits);\n";
                    wgslOut << indent << "    if (res.exchanged) {\n";
                    wgslOut << indent << "      if (oldBits != newBits) { atomicAdd(&result, 1u); }\n";
                    wgslOut << indent << "      break;\n";
                    wgslOut << indent << "    }\n";
                    wgslOut << indent << "  }\n";
                    wgslOut << indent << "}\n";
                    return;
                    
                  default:
                    // Fall through to regular assignment
                    break;
                }
                } else {
                  // Regular arrays: use direct compound operators (no atomics needed)
                  switch (opType) {
                    case OPERATOR_ADD:
                      wgslOut << indent << arrName << "[" << indexName << "] += ";
                      generateWGSLExpr(right, wgslOut, indexVar);
                      wgslOut << ";\n";
                      return;
                      
                    case OPERATOR_SUB:
                      wgslOut << indent << arrName << "[" << indexName << "] -= ";
                      generateWGSLExpr(right, wgslOut, indexVar);
                      wgslOut << ";\n";
                      return;
                      
                    case OPERATOR_MUL:
                      wgslOut << indent << arrName << "[" << indexName << "] *= ";
                      generateWGSLExpr(right, wgslOut, indexVar);
                      wgslOut << ";\n";
                      return;
                      
                    case OPERATOR_DIV:
                      wgslOut << indent << arrName << "[" << indexName << "] /= ";
                      generateWGSLExpr(right, wgslOut, indexVar);
                      wgslOut << ";\n";
                      return;
                      
                    case OPERATOR_OR:
                      wgslOut << indent << arrName << "[" << indexName << "] |= ";
                      generateWGSLExpr(right, wgslOut, indexVar);
                      wgslOut << ";\n";
                      return;
                      
                    case OPERATOR_AND:
                      wgslOut << indent << arrName << "[" << indexName << "] &= ";
                      generateWGSLExpr(right, wgslOut, indexVar);
                      wgslOut << ";\n";
                      return;
                      
                    default:
                      // Fall through to regular assignment
                      break;
                  }
                }
              }
            }
          }
        }
      }
      
      // Regular index access assignment
      Expression* targetMapExpr = asst->getIndexAccess()->getMapExpr();
      if (targetMapExpr && targetMapExpr->getExpressionFamily() == EXPR_ID) {
        std::string arrName = targetMapExpr->getId()->getIdentifier();
        
        // Check if this is a property array (requires atomic operations)
        bool isPropertyArray = false;
        std::string wgslType = "u32";
        for (const auto &p : propInfos) {
          if (p.name == arrName) { 
            wgslType = p.wgslType; 
            isPropertyArray = true;
            break; 
          }
        }
        
        if (isPropertyArray) {
          // Property arrays: use atomic store with compare-and-flag for convergence
          wgslOut << indent << "let __oldBits: u32 = atomicLoad(&" << arrName << "[";
          generateWGSLExpr(asst->getIndexAccess()->getIndexExpr(), wgslOut, indexVar);
          wgslOut << "]);\n";
          wgslOut << indent << "let __newBits: u32 = ";
          if (wgslType == "f32") {
            wgslOut << "bitcast<u32>(f32("; generateWGSLExpr(asst->getExpr(), wgslOut, indexVar); wgslOut << "))";
          } else {
            wgslOut << "u32("; generateWGSLExpr(asst->getExpr(), wgslOut, indexVar); wgslOut << ")";
          }
          wgslOut << ";\n";
          wgslOut << indent << "if (__oldBits != __newBits) { atomicStore(&" << arrName << "[";
          generateWGSLExpr(asst->getIndexAccess()->getIndexExpr(), wgslOut, indexVar);
          wgslOut << "], __newBits); atomicAdd(&result, 1u); }\n";
        } else {
          // Regular arrays: direct assignment
      wgslOut << indent;
          generateWGSLExpr(asst->getIndexAccess(), wgslOut, indexVar);
      wgslOut << " = ";
      generateWGSLExpr(asst->getExpr(), wgslOut, indexVar);
      wgslOut << ";\n";
        }
      } else {
        // Fallback for complex expressions
      wgslOut << indent;
      generateWGSLExpr(asst->getIndexAccess(), wgslOut, indexVar);
      wgslOut << " = ";
      generateWGSLExpr(asst->getExpr(), wgslOut, indexVar);
      wgslOut << ";\n";
      }
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
    case NODE_BREAKSTMT: {
      // Phase 3.7: Enhanced break statement support with control flow utilities
      wgslOut << indent << "break;\n";
      break;
    }
    case NODE_CONTINUESTMT: {
      // Phase 3.7: Enhanced continue statement support with control flow utilities
      wgslOut << indent << "continue;\n";
      break;
    }
    case NODE_WHILESTMT: {
      // Phase 3.7: While loop support with nested control flow
      whileStmt* w = static_cast<whileStmt*>(node);
      wgslOut << indent << "while (";
      if (w->getCondition()) generateWGSLExpr(w->getCondition(), wgslOut, indexVar);
      else wgslOut << "false";
      wgslOut << ") {\n";
      if (w->getBody()) {
        if (w->getBody()->getTypeofNode() == NODE_BLOCKSTMT) {
          generateWGSLStatement(w->getBody(), wgslOut, indexVar, indentLevel + 1);
        } else {
          generateWGSLStatement(w->getBody(), wgslOut, indexVar, indentLevel + 1);
        }
      }
      wgslOut << indent << "}\n";
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
            wgslOut << indent << "for (var edge = adj_offsets[" << indexVar << "]; edge < adj_offsets[" << indexVar << " + 1u]; edge = edge + 1u) {\n";
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
          } else if (methodName == "nodes_to") {
            // Generate incoming neighbor iteration loop (reverse edges)
            wgslOut << indent << "for (var edge = rev_adj_offsets[" << indexVar << "]; edge < rev_adj_offsets[" << indexVar << " + 1u]; edge = edge + 1u) {\n";
            wgslOut << indent << "  let " << fa->getIterator()->getIdentifier() << " = rev_adj_data[edge];\n";
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
          std::string varName = id ? id->getIdentifier() : "var";
          
          // Use 'var' for variables that might be modified (local variables)
          bool isMutableLocal = (varName.find("_count") != std::string::npos || 
                                varName.find("_sum") != std::string::npos ||
                                varName.find("local_") != std::string::npos ||
                                varName == "count" || varName == "sum" || varName == "temp");
          
          wgslOut << indent << (isMutableLocal ? "var " : "let ") << varName << " = ";
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
      // Safely treat statement as expression by accessing its expr payload
      proc_callStmt* stmt = static_cast<proc_callStmt*>(node);
      proc_callExpr* proc = stmt ? stmt->getProcCallExpr() : nullptr;
      if (proc && proc->getMethodId()) {
        std::string methodName = proc->getMethodId()->getIdentifier();
        if (methodName == "count_outNbrs") {
          list<argument*> argList = proc->getArgList();
          if (!argList.empty()) {
            argument* arg = argList.front();
            wgslOut << "(adj_offsets[";
            if (arg->isExpr()) { generateWGSLExpr(arg->getExpr(), wgslOut, indexVar); }
            wgslOut << " + 1] - adj_offsets[";
            if (arg->isExpr()) { generateWGSLExpr(arg->getExpr(), wgslOut, indexVar); }
            wgslOut << "])";
          } else { wgslOut << "0"; }
        } else if (methodName == "count_inNbrs") {
          list<argument*> argList = proc->getArgList();
          if (!argList.empty()) {
            argument* arg = argList.front();
            wgslOut << "(rev_adj_offsets[";
            if (arg->isExpr()) { generateWGSLExpr(arg->getExpr(), wgslOut, indexVar); }
            wgslOut << " + 1] - rev_adj_offsets[";
            if (arg->isExpr()) { generateWGSLExpr(arg->getExpr(), wgslOut, indexVar); }
            wgslOut << "])";
          } else { wgslOut << "0"; }
        } else if (methodName == "is_an_edge") {
          // Map to helper
          list<argument*> argList = proc->getArgList();
          if (argList.size() >= 2) {
            auto it = argList.begin();
            argument* arg1 = *it; ++it; argument* arg2 = *it;
            wgslOut << "findEdge(";
            if (arg1->isExpr()) { generateWGSLExpr(arg1->getExpr(), wgslOut, indexVar); }
            wgslOut << ", ";
            if (arg2->isExpr()) { generateWGSLExpr(arg2->getExpr(), wgslOut, indexVar); }
            wgslOut << ")";
          } else { wgslOut << "true"; }
          } else {
            wgslOut << "0";
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
        } else if (methodName == "count_inNbrs") {
          // Generate incoming neighbor count: rev_adj_offsets[v+1] - rev_adj_offsets[v]
          list<argument*> argList = proc->getArgList();
          if (!argList.empty()) {
            argument* arg = argList.front();
            wgslOut << "(rev_adj_offsets[";
            if (arg->isExpr()) {
              generateWGSLExpr(arg->getExpr(), wgslOut, indexVar);
            }
            wgslOut << " + 1] - rev_adj_offsets[";
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
        } else if (methodName == "get_edge") {
          // Generate edge index lookup: getEdgeIndex(u, v)
          list<argument*> argList = proc->getArgList();
          if (argList.size() >= 2) {
            auto it = argList.begin();
            argument* arg1 = *it;
            ++it;
            argument* arg2 = *it;
            
            wgslOut << "getEdgeIndex(";
            if (arg1->isExpr()) {
              generateWGSLExpr(arg1->getExpr(), wgslOut, indexVar);
            }
            wgslOut << ", ";
            if (arg2->isExpr()) {
              generateWGSLExpr(arg2->getExpr(), wgslOut, indexVar);
            }
            wgslOut << ")";
          } else {
            wgslOut << "0xFFFFFFFFu"; // Fallback
          }
        } else if (methodName == "num_nodes") {
          // Generate node count from params
          wgslOut << "params.node_count";
        } else if (methodName == "num_edges") {
          // Generate edge count: total size of adj_data array
          wgslOut << "arrayLength(&adj_data)";
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
        case EXPR_LOGICAL: { 
          wgslOut << "("; 
          generateWGSLExpr(expr->getLeft(), wgslOut, indexVar); 
          wgslOut << " " << getOpString(expr->getOperatorType()) << " "; 
          generateWGSLExpr(expr->getRight(), wgslOut, indexVar); 
          wgslOut << ")"; 
          break; 
        }
        case EXPR_RELATIONAL: {
          // Relational operators with type casting support
          wgslOut << "(";
          
          Expression* left = expr->getLeft();
          Expression* right = expr->getRight();
          
          if (left && right) {
            std::string leftType = inferExprType(left);
            std::string rightType = inferExprType(right);
            
            // Determine common type for comparison (promote to float if either is float)
            std::string commonType;
            if (leftType == "f32" || rightType == "f32") {
              commonType = "f32"; // Promote to float for precision
            } else if (leftType == "u32" && rightType == "u32") {
              commonType = "u32"; // Both bool/unsigned
            } else {
              commonType = "i32"; // Default to signed integer
            }
            
            // Generate left operand with casting if needed
            generateWithCast(left, commonType, wgslOut, indexVar);
            wgslOut << " " << getOpString(expr->getOperatorType()) << " ";
            
            // Generate right operand with casting if needed  
            generateWithCast(right, commonType, wgslOut, indexVar);
          } else {
            // Fallback for null operands
            generateWGSLExpr(left, wgslOut, indexVar); 
            wgslOut << " " << getOpString(expr->getOperatorType()) << " "; 
            generateWGSLExpr(right, wgslOut, indexVar); 
          }
          
          wgslOut << ")"; 
          break; 
        }
        case EXPR_UNARY: { 
          int opType = expr->getOperatorType();
          if (opType == OPERATOR_INC || opType == OPERATOR_DEC) {
            // Handle increment/decrement: convert to compound assignment
            Expression* inner = expr->getUnaryExpr();
            if (inner && inner->isIdentifierExpr()) {
              Identifier* id = inner->getId();
              if (id) {
                wgslOut << "(";
                generateWGSLExpr(inner, wgslOut, indexVar);
                if (opType == OPERATOR_INC) wgslOut << " += 1)";
                else wgslOut << " -= 1)";
                break;
              }
            }
            // Fallback for complex expressions
            wgslOut << "(" << (opType == OPERATOR_INC ? "++" : "--");
          generateWGSLExpr(expr->getUnaryExpr(), wgslOut, indexVar); 
          wgslOut << ")"; 
          } else {
            // Regular unary operators (!, -, +)
            wgslOut << "(" << getOpString(opType); 
            generateWGSLExpr(expr->getUnaryExpr(), wgslOut, indexVar); 
            wgslOut << ")";
          }
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
            } else if (methodName == "count_inNbrs") {
              // Generate incoming neighbor count: rev_adj_offsets[v+1] - rev_adj_offsets[v]
              list<argument*> argList = proc->getArgList();
              if (!argList.empty()) {
                argument* arg = argList.front();
                wgslOut << "(rev_adj_offsets[";
                if (arg->isExpr()) {
                  generateWGSLExpr(arg->getExpr(), wgslOut, indexVar);
                }
                wgslOut << " + 1] - rev_adj_offsets[";
                if (arg->isExpr()) {
                  generateWGSLExpr(arg->getExpr(), wgslOut, indexVar);
                }
                wgslOut << "])";
              } else {
                wgslOut << "0"; // Fallback
              }
            } else if (methodName == "get_edge") {
              // Generate edge index lookup: getEdgeIndex(u, v)
              list<argument*> argList = proc->getArgList();
              if (argList.size() >= 2) {
                auto it = argList.begin();
                argument* arg1 = *it;
                ++it;
                argument* arg2 = *it;
                
                wgslOut << "getEdgeIndex(";
                if (arg1->isExpr()) {
                  generateWGSLExpr(arg1->getExpr(), wgslOut, indexVar);
                }
                wgslOut << ", ";
                if (arg2->isExpr()) {
                  generateWGSLExpr(arg2->getExpr(), wgslOut, indexVar);
                }
                wgslOut << ")";
              } else {
                wgslOut << "0xFFFFFFFFu"; // Fallback
              }
            } else if (methodName == "num_nodes") {
              // Generate node count from params
              wgslOut << "params.node_count";
            } else if (methodName == "num_edges") {
              // Generate edge count: total size of adj_data array
              wgslOut << "arrayLength(&adj_data)";
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
    case OPERATOR_DIVASSIGN: return "/=";
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
    out << "  // Property initialization: attachNodeProperty\n";
    // For each assignment argument like prop = value, fill the corresponding GPU buffer
    for (argument* a : procExpr->getArgList()) {
      if (!a) continue;
      if (a->isAssignExpr()) {
        assignment* asg = a->getAssignExpr();
        if (asg && asg->lhs_isIdentifier()) {
          Identifier* id = asg->getId();
          if (!id || !id->getIdentifier()) continue;
          std::string propName = id->getIdentifier();
          // Find WGSL type for JS ctor selection
          std::string jsCtor = "Uint32Array";
          for (const auto &p : propInfos) {
            if (p.name == propName) {
              jsCtor = (p.wgslType == "u32") ? "Uint32Array" : (p.wgslType == "i32") ? "Int32Array" : "Float32Array";
              break;
            }
          }
          out << "  { const N = nodeCount; const initArr = new " << jsCtor << "(N);\n";
          out << "    for (let i = 0; i < N; i++) { initArr[i] = ";
          if (asg->getExpr()) { generateExpr(asg->getExpr(), out); } else { out << "0"; }
          out << "; }\n";
          out << "    device.queue.writeBuffer(" << propName << "Buffer, 0, initArr); }\n";
        }
      }
    }
  } else {
    out << "  // Unhandled proc call: " << methodName << "\n";
  }
}

void dsl_webgpu_generator::generatePropertyAccess(PropAccess* prop, std::ofstream& wgslOut, const std::string& indexVar) {
  if (!prop) return;
  
  // Property access pattern: object.property -> map to per-property buffer when available
  Identifier* objId = prop->getIdentifier1();   // The object (e.g., 'v')
  Identifier* propId = prop->getIdentifier2();  // The property (e.g., 'deg')
  
  if (objId && propId) {
    // Lookup in registry
    std::string propName = propId->getIdentifier() ? propId->getIdentifier() : "properties";
    bool mapped = false;
    for (const auto& p : propInfos) {
      if (p.name == propName) {
        wgslOut << p.name << "[" << objId->getIdentifier() << "]";
        mapped = true;
        break;
      }
    }
    if (!mapped) {
      // Fallback to monolithic buffer (to be removed in Phase 1)
    wgslOut << "properties[" << objId->getIdentifier() << "]";
    }
  }
}

void dsl_webgpu_generator::generateDoWhile(dowhileStmt* dw, std::ofstream& out) {
  if (!dw) return;
  out << "  // Do-while loop (webgpu host)\n";
  out << "  let __dwIterations = 0; const __dwMax = 1000;\n";
  out << "  do {\n";
  // Emit body statements in host context
  if (dw->getBody()) {
    if (dw->getBody()->getTypeofNode() == NODE_BLOCKSTMT) {
      blockStatement* block = static_cast<blockStatement*>(dw->getBody());
      for (statement* stmt : block->returnStatements()) {
        generateStatement(stmt, out);
      }
    } else {
      generateStatement(dw->getBody(), out);
    }
  }
  out << "    __dwIterations++;\n";
  out << "  } while (";
  if (dw->getCondition()) {
    generateExpr(dw->getCondition(), out);
  } else {
    out << "false";
  }
  out << " && __dwIterations < __dwMax);\n";
}

void dsl_webgpu_generator::generateReductionStmt(reductionCallStmt* stmt, std::ofstream& out) {
  if (!stmt) return;
  out << "  // [WebGPU] Reduction placeholder emitted on host; actual reduction occurs in kernels\n";
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
          out << "    const kernel_res_" << kernelCounter << " = await launchkernel_" << kernelCounter << "(device, adj_dataBuffer, adj_offsetsBuffer, paramsBuffer, resultBuffer, propertyBuffer, nodeCount, rev_adj_dataBuffer, rev_adj_offsetsBuffer);\n";
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

// Phase 0 helpers
void dsl_webgpu_generator::buildPropertyRegistry(Function* func) {
  propInfos.clear();
  if (!func) return;
  int nextBinding = 6; // Updated: bindings 0-1 forward CSR, 2-3 reverse CSR, 4 params, 5 result
  for (formalParam* fp : func->getParamList()) {
    if (!fp) continue;
    Type* t = fp->getType();
    Identifier* id = fp->getIdentifier();
    if (!t || !id) continue;
    if (t->isPropType() || t->isPropNodeType() || t->isPropEdgeType()) {
      PropInfo pi;
      pi.name = id->getIdentifier() ? id->getIdentifier() : std::string("prop") + std::to_string(nextBinding);
      pi.wgslType = mapTypeToWGSL(t->getInnerTargetType());
      pi.bindingIndex = nextBinding++;
      pi.isReadWrite = true;
      pi.isEdgeProperty = t->isPropEdgeType(); // true for edge properties, false for node properties
      propInfos.push_back(pi);
    }
  }
}

std::string dsl_webgpu_generator::mapTypeToWGSL(Type* type) {
  if (!type) return "u32";
  
  // Use gettypeId() like CUDA backend for proper type detection
  int typeId = type->gettypeId();
  switch (typeId) {
    case TYPE_INT: 
    case TYPE_LONG: 
      return "i32"; // no i64 in WGSL, map to i32
    case TYPE_BOOL: 
      return "u32"; // 0/1 representation
    case TYPE_FLOAT: 
    case TYPE_DOUBLE: 
      return "f32"; // no f64 in WGSL, map to f32
        default:
      return "u32"; // fallback
  }
}

bool dsl_webgpu_generator::isNumericIntegerType(Type* type) {
  if (!type) return true;
  return type->isIntegerType();
}

std::string dsl_webgpu_generator::inferExprType(Expression* expr) {
  if (!expr) return "u32";
  
  switch (expr->getExpressionFamily()) {
    case EXPR_INTCONSTANT:
      return "i32";
    case EXPR_FLOATCONSTANT:
    case EXPR_DOUBLECONSTANT:
      return "f32";
    case EXPR_BOOLCONSTANT:
      return "u32"; // bool as u32 in WGSL
    case EXPR_ID: {
      // For identifiers, we'd need symbol table lookup - basic heuristic for now
      return "i32"; // default assumption
    }
    case EXPR_PROPID: {
      // For property access, look up in propInfos
      PropAccess* prop = expr->getPropId();
      if (prop && prop->getIdentifier2()) {
        std::string propName = prop->getIdentifier2()->getIdentifier();
        for (const auto &p : propInfos) {
          if (p.name == propName) return p.wgslType;
        }
      }
      return "u32"; // fallback
    }
    case EXPR_ARITHMETIC: {
      // For arithmetic, use type promotion rules
      std::string leftType = inferExprType(expr->getLeft());
      std::string rightType = inferExprType(expr->getRight());
      // Promote to float if either operand is float
      if (leftType == "f32" || rightType == "f32") return "f32";
      return "i32"; // both integer types
    }
        default:
      return "u32"; // fallback
  }
}

void dsl_webgpu_generator::generateWithCast(Expression* expr, const std::string& targetType, std::ofstream& wgslOut, const std::string& indexVar) {
  std::string exprType = inferExprType(expr);
  
  if (exprType != targetType) {
    // Need casting
    wgslOut << targetType << "(";
    generateWGSLExpr(expr, wgslOut, indexVar);
    wgslOut << ")";
  } else {
    // No casting needed
    generateWGSLExpr(expr, wgslOut, indexVar);
  }
}

// Phase 3.18: Utility file inclusion helpers
std::string dsl_webgpu_generator::readUtilityFile(const std::string& relativePath) {
  std::string basePath = "../graphcode/webgpu_utils/";
  std::string fullPath = basePath + relativePath;
  
  std::ifstream file(fullPath);
  if (!file.is_open()) {
    std::cerr << "[WebGPU] Warning: Could not open utility file: " << fullPath << std::endl;
    return ""; // Return empty string if file cannot be read
  }
  
  std::string content;
  std::string line;
  while (std::getline(file, line)) {
    content += line + "\n";
  }
  file.close();
  
  return content;
}

void dsl_webgpu_generator::includeWGSLUtilities(std::ofstream& wgslOut) {
  wgslOut << "// ============================================================================\n";
  wgslOut << "// StarPlat WebGPU Utilities - Modular Implementation (Phase 3.14-3.16)\n";
  wgslOut << "// ============================================================================\n\n";
  
  // Include atomic operations utilities
  std::string atomicsContent = readUtilityFile("wgsl_kernels/webgpu_atomics.wgsl");
  if (!atomicsContent.empty()) {
    wgslOut << "// Atomic Operations Utilities (Task 3.14)\n";
    wgslOut << atomicsContent << "\n";
  } else {
    // Fallback: include minimal atomic functions inline for backward compatibility
    wgslOut << "// Fallback atomic operations (utility file not found)\n";
    wgslOut << "fn atomicAddF32(ptr: ptr<storage, atomic<u32>>, val: f32) -> f32 {\n";
    wgslOut << "  loop {\n";
    wgslOut << "    let oldBits: u32 = atomicLoad(ptr);\n";
    wgslOut << "    let oldVal: f32 = bitcast<f32>(oldBits);\n";
    wgslOut << "    let newVal: f32 = oldVal + val;\n";
    wgslOut << "    let newBits: u32 = bitcast<u32>(newVal);\n";
    wgslOut << "    let res = atomicCompareExchangeWeak(ptr, oldBits, newBits);\n";
    wgslOut << "    if (res.exchanged) { return oldVal; }\n";
    wgslOut << "  }\n";
    wgslOut << "}\n\n";
  }
  
  // Include graph methods utilities
  std::string graphContent = readUtilityFile("wgsl_kernels/webgpu_graph_methods.wgsl");
  if (!graphContent.empty()) {
    wgslOut << "// Graph Methods Utilities (Task 3.15)\n";
    wgslOut << graphContent << "\n";
  } else {
    // Fallback: include minimal graph functions inline
    wgslOut << "// Fallback graph methods (utility file not found)\n";
    wgslOut << "fn findEdge(u: u32, w: u32) -> bool {\n";
    wgslOut << "  let start = adj_offsets[u];\n";
    wgslOut << "  let end = adj_offsets[u + 1u];\n";
    wgslOut << "  for (var e = start; e < end; e = e + 1u) {\n";
    wgslOut << "    if (adj_data[e] == w) { return true; }\n";
    wgslOut << "  }\n";
    wgslOut << "  return false;\n";
    wgslOut << "}\n\n";
  }
  
  // Include workgroup reductions utilities  
  std::string reductionsContent = readUtilityFile("wgsl_kernels/webgpu_reductions.wgsl");
  if (!reductionsContent.empty()) {
    wgslOut << "// Workgroup Reductions Utilities (Task 3.16)\n";
    wgslOut << reductionsContent << "\n";
  } else {
    // Fallback: include minimal reduction functions inline
    wgslOut << "// Fallback workgroup reductions (utility file not found)\n";
    wgslOut << "fn workgroupReduceSum(local_id: u32, value: u32) -> u32 {\n";
    wgslOut << "  scratchpad[local_id] = value;\n";
    wgslOut << "  workgroupBarrier();\n";
    wgslOut << "  var stride = 128u;\n";
    wgslOut << "  while (stride > 0u) {\n";
    wgslOut << "    if (local_id < stride) {\n";
    wgslOut << "      scratchpad[local_id] += scratchpad[local_id + stride];\n";
    wgslOut << "    }\n";
    wgslOut << "    workgroupBarrier();\n";
    wgslOut << "    stride = stride >> 1u;\n";
    wgslOut << "  }\n";
    wgslOut << "  return scratchpad[0];\n";
    wgslOut << "}\n\n";
  }
  
  // Phase 3.7: Include control flow utilities for break/continue support
  std::string controlFlowContent = readUtilityFile("wgsl_kernels/webgpu_control_flow.wgsl");
  if (!controlFlowContent.empty()) {
    wgslOut << "// Control Flow and Nested Context Utilities (Task 3.7)\n";
    wgslOut << controlFlowContent << "\n";
  }
  
  // Phase 3.5: Include convergence detection utilities
  std::string convergenceContent = readUtilityFile("wgsl_kernels/webgpu_convergence.wgsl");
  if (!convergenceContent.empty()) {
    wgslOut << "// Enhanced Convergence Detection Utilities (Task 3.5)\n";
    wgslOut << convergenceContent << "\n";
  }
  
  // Phase 3.10: Include error handling utilities
  std::string errorContent = readUtilityFile("wgsl_kernels/webgpu_error_handling.wgsl");
  if (!errorContent.empty()) {
    wgslOut << "// Error Handling and Validation Utilities (Task 3.10)\n";
    wgslOut << errorContent << "\n";
  }
  
  // Phase 3.9/3.11: Include dynamic property management utilities
  std::string dynamicPropsContent = readUtilityFile("wgsl_kernels/webgpu_dynamic_properties.wgsl");
  if (!dynamicPropsContent.empty()) {
    wgslOut << "// Dynamic Property Management Utilities (Tasks 3.9, 3.11)\n";
    wgslOut << dynamicPropsContent << "\n";
  }
  
  // Phase 3.6: Include loop optimization utilities
  std::string loopOptContent = readUtilityFile("wgsl_kernels/webgpu_loop_optimization.wgsl");
  if (!loopOptContent.empty()) {
    wgslOut << "// Loop Optimization and Kernel Fusion Utilities (Task 3.6)\n";
    wgslOut << loopOptContent << "\n";
  }
  
  wgslOut << "// ============================================================================\n";
  wgslOut << "// End of Utility Includes\n";  
  wgslOut << "// ============================================================================\n\n";
}

} // namespace spwebgpu

