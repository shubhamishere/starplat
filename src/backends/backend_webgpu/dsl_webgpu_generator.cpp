#include "dsl_webgpu_generator.h"
#include <iostream>
#include "../../ast/ASTNodeTypes.hpp"

namespace spwebgpu {

// Constructor for the WebGPU backend code generator.
// Handles any necessary initialization.
dsl_webgpu_generator::dsl_webgpu_generator() {
    // Initialization if needed
}

// Destructor for the WebGPU backend code generator.
// Handles any necessary cleanup.
dsl_webgpu_generator::~dsl_webgpu_generator() {
    // Cleanup if needed
}

// Entry point for code generation. Takes the AST root and output JS filename.
// Opens the output file and starts function code generation.
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

// Generates the main JS function for the DSL program.
// Checks node type, emits async JS function, and emits kernel launchers.
void dsl_webgpu_generator::generateFunc(ASTNode* node, std::ofstream& out) {
    if (!node) {
        out << "// Not a function node\n";
        return;
    }
    if (node->getTypeofNode() != NODE_FUNC) {
        out << "// Not a function node (type: " << node->getTypeofNode() << ")\n";
        std::cerr << "[WebGPU] Error: generateFunc called with non-function node. Type: " << node->getTypeofNode() << std::endl;
        return;
    }
    Function* func = static_cast<Function*>(node);
    if (!func) {
        out << "// Null function pointer after cast\n";
        std::cerr << "[WebGPU] Error: Null function pointer after cast in generateFunc." << std::endl;
        return;
    }
    std::string funcName = func->getIdentifier() ? func->getIdentifier()->getIdentifier() : "unnamed";
    out << "async function " << funcName << "(device, inputBuffer, outputBuffer, N) {\n";
    out << "  // Main function body\n";
    generateBlock(func->getBlockStatement(), out);
    out << "}\n\n";
    // Emit all kernel launchers after the main function
    for (int i = 0; i < kernelCounter; ++i) {
        out << "async function launchkernel_" << i << "(device, inputBuffer, outputBuffer, N) {\n";
        out << "  // 1. Load WGSL\n";
        out << "  const shaderCode = await (await fetch('kernel_" << i << ".wgsl')).text();\n";
        out << "  const shaderModule = device.createShaderModule({ code: shaderCode });\n";
        out << "  // 2. Create pipeline\n";
        out << "  const pipeline = device.createComputePipeline({\n";
        out << "    layout: 'auto',\n";
        out << "    compute: { module: shaderModule, entryPoint: 'main' }\n";
        out << "  });\n";
        out << "  // 3. Set up bind group\n";
        out << "  const bindGroup = device.createBindGroup({\n";
        out << "    layout: pipeline.getBindGroupLayout(0),\n";
        out << "    entries: [\n";
        out << "      { binding: 0, resource: { buffer: inputBuffer } }\n";
        out << "    ]\n";
        out << "  });\n";
        out << "  // 4. Encode and dispatch\n";
        out << "  const commandEncoder = device.createCommandEncoder();\n";
        out << "  const passEncoder = commandEncoder.beginComputePass();\n";
        out << "  passEncoder.setPipeline(pipeline);\n";
        out << "  passEncoder.setBindGroup(0, bindGroup);\n";
        out << "  passEncoder.dispatchWorkgroups(Math.ceil(N / 64));\n";
        out << "  passEncoder.end();\n";
        out << "  device.queue.submit([commandEncoder.finish()]);\n";
        out << "  // 5. Read back results (if needed)\n";
        out << "  // TODO: Map outputBuffer and read results\n";
        out << "}\n\n";
    }
}

// Recursively traverses a blockStatement AST node and emits code for each statement.
void dsl_webgpu_generator::generateBlock(ASTNode* node, std::ofstream& out) {
    blockStatement* block = static_cast<blockStatement*>(node);
    if (!block) {
        out << "  // Not a blockStatement\n";
        return;
    }
    for (statement* stmt : block->returnStatements()) {
        generateStatement(stmt, out);
    }
}

// Dispatches code generation for a single statement node (declaration, assignment, control flow, parallel, etc.).
// Handles both JS and triggers WGSL kernel emission for parallel constructs.
void dsl_webgpu_generator::generateStatement(ASTNode* node, std::ofstream& out) {
    if (!node) {
        out << "  // Null statement\n";
        return;
    }
    switch (node->getTypeofNode()) {
        case NODE_BLOCKSTMT:
            out << "  // Block statement\n";
            generateBlock(node, out);
            break;
        case NODE_DECL: {
            declaration* decl = static_cast<declaration*>(node);
            Identifier* id = decl->getdeclId();
            out << "  let " << (id ? id->getIdentifier() : "unnamed") ;
            if (decl->isInitialized()) {
                out << " = ";
                generateExpr(decl->getExpressionAssigned(), out);
            }
            out << ";\n";
            break;
        }
        case NODE_ASSIGN: {
            assignment* asst = static_cast<assignment*>(node);
            // Only handle identifier lhs for now
            if (asst->lhs_isIdentifier()) {
                Identifier* id = asst->getId();
                out << "  " << (id ? id->getIdentifier() : "unnamed") << " = ";
                generateExpr(asst->getExpr(), out);
                out << ";\n";
            } else {
                out << "  // Assignment (non-identifier lhs)\n";
            }
            break;
        }
        case NODE_WHILESTMT: {
            whileStmt* whilestmt = static_cast<whileStmt*>(node);
            out << "  while (";
            generateExpr(whilestmt->getCondition(), out);
            out << ") {\n";
            if (whilestmt->getBody()) {
                if (whilestmt->getBody()->getTypeofNode() == NODE_BLOCKSTMT) {
                    generateBlock(whilestmt->getBody(), out);
                } else {
                    generateStatement(whilestmt->getBody(), out);
                }
            }
            out << "  }\n";
            break;
        }
        case NODE_DOWHILESTMT: {
            dowhileStmt* dowhilestmt = static_cast<dowhileStmt*>(node);
            out << "  do {\n";
            if (dowhilestmt->getBody()) {
                if (dowhilestmt->getBody()->getTypeofNode() == NODE_BLOCKSTMT) {
                    generateBlock(dowhilestmt->getBody(), out);
                } else {
                    generateStatement(dowhilestmt->getBody(), out);
                }
            }
            out << "  } while (";
            generateExpr(dowhilestmt->getCondition(), out);
            out << ");\n";
            break;
        }
        case NODE_SIMPLEFORSTMT: {
            simpleForStmt* forstmt = static_cast<simpleForStmt*>(node);
            out << "  for (let ";
            Identifier* var = forstmt->getLoopVariable();
            if (var) {
                out << var->getIdentifier();
            } else {
                out << "unnamed";
            }
            out << " = ";
            if (forstmt->getRhs()) {
                generateExpr(forstmt->getRhs(), out);
            } else {
                out << "0";
            }
            out << "; ";
            if (forstmt->getIterCondition()) {
                generateExpr(forstmt->getIterCondition(), out);
            }
            out << "; ";
            if (forstmt->getUpdateExpression()) {
                generateExpr(forstmt->getUpdateExpression(), out);
            }
            out << ") {\n";
            if (forstmt->getBody()) {
                if (forstmt->getBody()->getTypeofNode() == NODE_BLOCKSTMT) {
                    generateBlock(forstmt->getBody(), out);
                } else {
                    generateStatement(forstmt->getBody(), out);
                }
            }
            out << "  }\n";
            break;
        }
        case NODE_IFSTMT: {
            ifStmt* ifstmt = static_cast<ifStmt*>(node);
            out << "  if (";
            generateExpr(ifstmt->getCondition(), out);
            out << ") {\n";
            if (ifstmt->getIfBody()) {
                if (ifstmt->getIfBody()->getTypeofNode() == NODE_BLOCKSTMT) {
                    generateBlock(ifstmt->getIfBody(), out);
                } else {
                    generateStatement(ifstmt->getIfBody(), out);
                }
            }
            out << "  }";
            if (ifstmt->getElseBody()) {
                out << " else {\n";
                if (ifstmt->getElseBody()->getTypeofNode() == NODE_BLOCKSTMT) {
                    generateBlock(ifstmt->getElseBody(), out);
                } else {
                    generateStatement(ifstmt->getElseBody(), out);
                }
                out << "  }";
            }
            out << "\n";
            break;
        }
        case NODE_FORALLSTMT: {
            // Emit WGSL kernel for this forall
            std::string kernelName = "kernel_" + std::to_string(kernelCounter);
            forallStmt* forall = static_cast<forallStmt*>(node);
            emitWGSLKernel(kernelName, forall->getBody());
            out << "  // PARALLEL FORALL (WebGPU kernel launch)\n";
            out << "  // See " << kernelName << ".wgsl for kernel code\n";
            out << "  await launchkernel_" << kernelCounter << "(device, inputBuffer, outputBuffer, N);\n";
            kernelCounter++;
            break;
        }
        case NODE_LOOPSTMT: {
            loopStmt* loop = static_cast<loopStmt*>(node);
            Identifier* iter = loop->getIterator();
            out << "  // Advanced parallel/graph loop\n";
            out << "  for (let ";
            if (iter) {
                out << iter->getIdentifier();
            } else {
                out << "i";
            }
            out << " = ";
            if (loop->getStartValue()) {
                generateExpr(loop->getStartValue(), out);
            } else {
                out << "0";
            }
            out << "; ";
            if (iter) {
                out << iter->getIdentifier() << " < ";
            }
            if (loop->getEndValue()) {
                generateExpr(loop->getEndValue(), out);
            } else {
                out << "N";
            }
            out << "; ";
            if (iter) {
                out << iter->getIdentifier() << " += ";
            } else {
                out << "i += ";
            }
            if (loop->getStepValue()) {
                generateExpr(loop->getStepValue(), out);
            } else {
                out << "1";
            }
            out << ") {\n";
            if (loop->getBody()) {
                if (loop->getBody()->getTypeofNode() == NODE_BLOCKSTMT) {
                    generateBlock(loop->getBody(), out);
                } else {
                    generateStatement(loop->getBody(), out);
                }
            }
            out << "  }\n";
            break;
        }
        case NODE_REDUCTIONCALLSTMT: {
            out << "  // Reduction pattern (to be replaced with real reduction)\n";
            out << "  let result = arr.reduce((acc, x) => acc + x, 0);\n";
            break;
        }
        case NODE_BREAKSTMT:
            out << "    break;\n";
            break;
        case NODE_CONTINUESTMT:
            out << "    continue;\n";
            break;
        case NODE_FIXEDPTSTMT: {
            out << "  // Fixed-point iteration (to be replaced with real convergence check)\n";
            out << "  while (notConverged) {\n";
            fixedPointStmt* fp = static_cast<fixedPointStmt*>(node);
            if (fp->getBody()) {
                if (fp->getBody()->getTypeofNode() == NODE_BLOCKSTMT) {
                    generateBlock(fp->getBody(), out);
                } else {
                    generateStatement(fp->getBody(), out);
                }
            }
            out << "  }\n";
            break;
        }
        case NODE_ITRBFS: {
            out << "  // BFS iteration (to be replaced with real BFS order)\n";
            out << "  for (let node of bfsOrder) {\n";
            iterateBFS* bfs = static_cast<iterateBFS*>(node);
            if (bfs->getBody()) {
                if (bfs->getBody()->getTypeofNode() == NODE_BLOCKSTMT) {
                    generateBlock(bfs->getBody(), out);
                } else {
                    generateStatement(bfs->getBody(), out);
                }
            }
            out << "  }\n";
            break;
        }
        case NODE_PROCCALLSTMT: {
            proc_callStmt* procStmt = static_cast<proc_callStmt*>(node);
            proc_callExpr* callExpr = procStmt->getProcCallExpr();
            if (callExpr && callExpr->getMethodId()) {
                out << "  " << callExpr->getMethodId()->getIdentifier() << "(";
                bool first = true;
                for (argument* arg : callExpr->getArgList()) {
                    if (!first) out << ", ";
                    if (arg->getExpr()) {
                        generateExpr(arg->getExpr(), out);
                    } else {
                        out << "undefined";
                    }
                    first = false;
                }
                out << ");\n";
            } else {
                out << "  // Procedure call statement (unknown)\n";
            }
            break;
        }
        case NODE_UNARYSTMT:
            out << "  // Unary statement\n";
            break;
        case NODE_TRANSFERSTMT:
            out << "  // Transfer statement\n";
            break;
        default:
            out << "  // Unknown statement type\n";
            break;
    }
}

// Emits JS code for an expression node (identifiers, literals, arithmetic, property access, etc.).
void dsl_webgpu_generator::generateExpr(ASTNode* node, std::ofstream& out) {
    if (!node) {
        out << "undefined";
        return;
    }
    switch (node->getTypeofNode()) {
        case NODE_ID: {
            Identifier* id = static_cast<Identifier*>(node);
            out << id->getIdentifier();
            break;
        }
        case NODE_EXPR: {
            Expression* expr = static_cast<Expression*>(node);
            switch (expr->getExpressionFamily()) {
                case EXPR_INTCONSTANT:
                    out << expr->getIntegerConstant();
                    break;
                case EXPR_FLOATCONSTANT:
                    out << expr->getFloatConstant();
                    break;
                case EXPR_BOOLCONSTANT:
                    out << (expr->getBooleanConstant() ? "true" : "false");
                    break;
                case EXPR_STRINGCONSTANT:
                    out << '"' << expr->getStringConstant() << '"';
                    break;
                case EXPR_ARITHMETIC: {
                    out << "(";
                    generateExpr(expr->getLeft(), out);
                    out << " " << dsl_webgpu_generator::getOpString(expr->getOperatorType()) << " ";
                    generateExpr(expr->getRight(), out);
                    out << ")";
                    break;
                }
                case EXPR_RELATIONAL: {
                    out << "(";
                    generateExpr(expr->getLeft(), out);
                    out << " " << dsl_webgpu_generator::getOpString(expr->getOperatorType()) << " ";
                    generateExpr(expr->getRight(), out);
                    out << ")";
                    break;
                }
                case EXPR_LOGICAL: {
                    out << "(";
                    generateExpr(expr->getLeft(), out);
                    out << " " << dsl_webgpu_generator::getOpString(expr->getOperatorType()) << " ";
                    generateExpr(expr->getRight(), out);
                    out << ")";
                    break;
                }
                case EXPR_UNARY: {
                    out << "(" << dsl_webgpu_generator::getOpString(expr->getOperatorType());
                    generateExpr(expr->getUnaryExpr(), out);
                    out << ")";
                    break;
                }
                case EXPR_PROPID: {
                    PropAccess* prop = expr->getPropId();
                    if (prop) {
                        // A[u] or similar
                        if (prop->getIdentifier2()) {
                            out << prop->getIdentifier2()->getIdentifier() << "[";
                            if (prop->getIdentifier1()) {
                                out << prop->getIdentifier1()->getIdentifier();
                            } else if (prop->getPropExpr()) {
                                generateExpr(prop->getPropExpr(), out);
                            }
                            out << "]";
                        } else if (prop->getIdentifier1() && prop->getPropExpr()) {
                            // e.g., G.nodes[v]
                            out << prop->getIdentifier1()->getIdentifier() << "[";
                            generateExpr(prop->getPropExpr(), out);
                            out << "]";
                        } else {
                            out << "/*prop-access*/";
                        }
                    } else {
                        out << "/*prop-access*/";
                    }
                    break;
                }
                default:
                    out << "/*expr*/";
                    break;
            }
            break;
        }
        default:
            out << "/*expr*/";
            break;
    }
}

// Emits a WGSL kernel file for a parallel construct (forall).
// Translates the forall body to WGSL using generateWGSLStatement and generateWGSLExpr.
void dsl_webgpu_generator::emitWGSLKernel(const std::string& baseName, ASTNode* forallBody) {
    std::string filename = "../graphcode/generated_webgpu/" + baseName + ".wgsl";
    std::ofstream wgslOut(filename);
    if (!wgslOut.is_open()) {
        std::cerr << "[WebGPU] Failed to open WGSL file: " << filename << std::endl;
        return;
    }
    wgslOut << "@group(0) @binding(0) var<storage, read_write> data: array<u32>;\n\n";
    wgslOut << "@compute @workgroup_size(64)\nfn main(@builtin(global_invocation_id) global_id: vec3<u32>) {\n";
    wgslOut << "    let i = global_id.x;\n";
    // Translate forall body to WGSL
    if (forallBody && forallBody->getTypeofNode() == NODE_BLOCKSTMT) {
        blockStatement* block = static_cast<blockStatement*>(forallBody);
        for (statement* stmt : block->returnStatements()) {
            generateWGSLStatement(stmt, wgslOut, "i");
        }
    } else if (forallBody) {
        generateWGSLStatement(forallBody, wgslOut, "i");
    }
    wgslOut << "}\n";
    wgslOut.close();
}

// Recursively emits WGSL code for a statement node (declaration, assignment, if, block, etc.).
void dsl_webgpu_generator::generateWGSLStatement(ASTNode* node, std::ofstream& wgslOut, const std::string& indexVar) {
    if (!node) return;
    switch (node->getTypeofNode()) {
        case NODE_DECL: {
            declaration* decl = static_cast<declaration*>(node);
            Identifier* id = decl->getdeclId();
            wgslOut << "    var " << (id ? id->getIdentifier() : "unnamed");
            if (decl->isInitialized()) {
                wgslOut << " = ";
                generateWGSLExpr(decl->getExpressionAssigned(), wgslOut, indexVar);
            }
            wgslOut << ";\n";
            break;
        }
        case NODE_ASSIGN: {
            assignment* asst = static_cast<assignment*>(node);
            if (asst->lhs_isIdentifier()) {
                Identifier* id = asst->getId();
                wgslOut << "    " << (id ? id->getIdentifier() : "unnamed") << " = ";
                generateWGSLExpr(asst->getExpr(), wgslOut, indexVar);
                wgslOut << ";\n";
            } else if (asst->lhs_isProp()) {
                PropAccess* prop = asst->getPropId();
                if (prop && prop->getIdentifier2()) {
                    wgslOut << "    " << prop->getIdentifier2()->getIdentifier() << "[";
                    if (prop->getIdentifier1()) {
                        wgslOut << prop->getIdentifier1()->getIdentifier();
                    } else if (prop->getPropExpr()) {
                        generateWGSLExpr(prop->getPropExpr(), wgslOut, indexVar);
                    }
                    wgslOut << "] = ";
                    generateWGSLExpr(asst->getExpr(), wgslOut, indexVar);
                    wgslOut << ";\n";
                } else {
                    wgslOut << "    // Unsupported property assignment\n";
                }
            } else {
                wgslOut << "    // Unsupported assignment\n";
            }
            break;
        }
        case NODE_IFSTMT: {
            ifStmt* ifstmt = static_cast<ifStmt*>(node);
            wgslOut << "    if (";
            generateWGSLExpr(ifstmt->getCondition(), wgslOut, indexVar);
            wgslOut << ") {\n";
            if (ifstmt->getIfBody()) {
                generateWGSLStatement(ifstmt->getIfBody(), wgslOut, indexVar);
            }
            wgslOut << "    }";
            if (ifstmt->getElseBody()) {
                wgslOut << " else {\n";
                generateWGSLStatement(ifstmt->getElseBody(), wgslOut, indexVar);
                wgslOut << "    }";
            }
            wgslOut << "\n";
            break;
        }
        case NODE_BLOCKSTMT: {
            blockStatement* block = static_cast<blockStatement*>(node);
            for (statement* stmt : block->returnStatements()) {
                generateWGSLStatement(stmt, wgslOut, indexVar);
            }
            break;
        }
        default:
            wgslOut << "    // Unsupported or unhandled statement\n";
            break;
    }
}

// Recursively emits WGSL code for an expression node (identifiers, literals, arithmetic, property access, etc.).
void dsl_webgpu_generator::generateWGSLExpr(ASTNode* node, std::ofstream& wgslOut, const std::string& indexVar) {
    if (!node) {
        wgslOut << "undefined";
        return;
    }
    switch (node->getTypeofNode()) {
        case NODE_ID: {
            Identifier* id = static_cast<Identifier*>(node);
            if (id && std::string(id->getIdentifier()) == indexVar) {
                wgslOut << indexVar;
            } else {
                wgslOut << (id ? id->getIdentifier() : "unnamed");
            }
            break;
        }
        case NODE_EXPR: {
            Expression* expr = static_cast<Expression*>(node);
            switch (expr->getExpressionFamily()) {
                case EXPR_INTCONSTANT:
                    wgslOut << expr->getIntegerConstant();
                    break;
                case EXPR_FLOATCONSTANT:
                    wgslOut << expr->getFloatConstant();
                    break;
                case EXPR_ARITHMETIC:
                case EXPR_RELATIONAL:
                case EXPR_LOGICAL: {
                    wgslOut << "(";
                    generateWGSLExpr(expr->getLeft(), wgslOut, indexVar);
                    wgslOut << " " << dsl_webgpu_generator::getOpString(expr->getOperatorType()) << " ";
                    generateWGSLExpr(expr->getRight(), wgslOut, indexVar);
                    wgslOut << ")";
                    break;
                }
                case EXPR_UNARY: {
                    wgslOut << "(" << dsl_webgpu_generator::getOpString(expr->getOperatorType());
                    generateWGSLExpr(expr->getUnaryExpr(), wgslOut, indexVar);
                    wgslOut << ")";
                    break;
                }
                case EXPR_PROPID: {
                    PropAccess* prop = expr->getPropId();
                    if (prop && prop->getIdentifier2()) {
                        wgslOut << prop->getIdentifier2()->getIdentifier() << "[";
                        if (prop->getIdentifier1()) {
                            wgslOut << prop->getIdentifier1()->getIdentifier();
                        } else if (prop->getPropExpr()) {
                            generateWGSLExpr(prop->getPropExpr(), wgslOut, indexVar);
                        }
                        wgslOut << "]";
                    } else {
                        wgslOut << "/*prop-access*/";
                    }
                    break;
                }
                default:
                    wgslOut << "/*expr*/";
                    break;
            }
            break;
        }
        default:
            wgslOut << "/*expr*/";
            break;
    }
}

// Returns the WGSL/JS string for a given operator type (e.g., +, -, *, <, etc.).
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
        case OPERATOR_INC: return "++";
        case OPERATOR_DEC: return "--";
        default: return "?";
    }
}

} // namespace spwebgpu 