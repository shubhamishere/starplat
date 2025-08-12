#ifndef INCLUDE_HEADER_HPP
#define INCLUDE_HEADER_HPP

#include "../ast/ASTNode.hpp"
#include "../ast/ASTNodeTypes.hpp"
#include "../maincontext/MainContext.hpp"
#include "../symbolutil/SymbolTableBuilder.h"

#include "../backends/backend_cuda/dsl_cpp_generator.h"
#include "../backends/backend_omp/dsl_cpp_generator.h"
#include "../backends/backend_mpi/dsl_cpp_generator.h"
#include "../backends/backend_openACC/dsl_cpp_generator.h"
#include "../backends/backend_hip/dsl_cpp_generator.h"
#include "../backends/backend_sycl/dsl_cpp_generator.h"
#include "../backends/backend_amd/dsl_cpp_generator.h"
#include "../backends/backend_multigpu/dsl_cpp_generator.h"
#include "../backends/backend_webgpu/dsl_webgpu_generator.h"

// These utility functions are declared in parser's other translation units
void addFuncToList(ASTNode* func);
void setCurrentFuncType(int t);
void resetTemp(std::vector<Identifier*>& ids);

// Forward declarations for parser

#endif // INCLUDE_HEADER_HPP

