#!/bin/bash
set -e

# Build StarPlat for WebAssembly (WASM)
# Usage: ./build_wasm.sh

# 1. Generate Parser C files (Lex/Yacc)
echo "[WASM Build] Generating parser files..."
cd src/parser
flex lexer.l
bison -d -y lrparser.y
# Rename generated C files to CPP so emcc treats them as C++
if [ -f "lex.yy.c" ]; then mv lex.yy.c lex.yy.cpp; fi
if [ -f "y.tab.c" ]; then mv y.tab.c y.tab.cpp; fi

cd ../..

# 2. Compile to WASM using emcc
echo "[WASM Build] Compiling to WASM..."

# List of source files based on Makefile 'PROGRAMS' + 'EXPENDABLES'
# We include all backends because our audit showed they don't depend on system headers.
SOURCES="src/maincontext/MainContext.cpp \
src/ast/ASTHelper.cpp \
src/symbolutil/SymbolTableBuilder.cpp \
src/symbolutil/SymbolTableNew.cpp \
src/parser/y.tab.cpp \
src/parser/lex.yy.cpp \
src/analyser/dataRace/dataRaceAnalyser.cpp \
src/analyser/attachProp/attachPropAnalyser.cpp \
src/analyser/pushpull/pushpullAnalyser.cpp \
src/analyser/callGraph/callGraphAnalyser.cpp \
src/analyser/cudaGlobalVariables/cudaGlobalVariablesAnalyser.cpp \
src/analyser/deviceVars/deviceVarsAnalyser.cpp \
src/analyser/deviceVars/deviceVarsInit.cpp \
src/analyser/deviceVars/deviceVarsPrint.cpp \
src/analyser/deviceVars/deviceVarsTransfer.cpp \
src/analyser/deviceVars/getUsedVars.cpp \
src/analyser/blockVars/blockVarsAnalyser.cpp \
src/analyser/blockVars/blockVarsInit.cpp \
src/analyser/blockVars/getUsedVars.cpp \
src/analyser/blockVars/NodeBlockData.cpp \
src/analyser/blockVars/setVarsInParallel.cpp \
src/analyser/blockVars/analyserUtil.cpp \
src/analyser/analyserUtil.cpp \
src/backends/backend_cuda/dsl_cpp_generator.cpp \
src/backends/backend_openACC/dsl_cpp_generator.cpp \
src/backends/backend_omp/dsl_cpp_generator.cpp \
src/backends/backend_omp/dsl_dyn_cpp_generator.cpp \
src/backends/backend_hip/dsl_cpp_generator.cpp \
src/backends/backend_hip/get_used_data.cpp \
src/backends/backend_hip/auxillary_functions.cpp \
src/backends/backend_hip/hip_gen_functions.cpp \
src/backends/backend_mpi/dsl_cpp_generator.cpp \
src/backends/backend_mpi/dsl_cpp_generator_helper.cpp \
src/backends/backend_mpi/dsl_cpp_expression_generator.cpp \
src/backends/backend_mpi/dsl_cpp_statement_generator.cpp \
src/backends/backend_mpi/dsl_dyn_cpp_generator.cpp \
src/backends/backend_mpi/bAnalyzer/bAnalyzer.cc \
src/backends/backend_sycl/dsl_cpp_generator.cpp \
src/backends/backend_multigpu/dsl_cpp_generator.cpp \
src/backends/backend_amd/dsl_cpp_generator.cpp"

# Include paths
INCLUDES="-I src -I src/parser"

# Output file
OUTPUT="starplat.js"

emcc -O3 \
    $SOURCES \
    $INCLUDES \
    -o $OUTPUT \
    -s WASM=1 \
    -s "EXPORTED_RUNTIME_METHODS=['callMain', 'FS']" \
    -s FORCE_FILESYSTEM=1 \
    -s ALLOW_MEMORY_GROWTH=1 \
    -s ASSERTIONS=1 \
    -D__EMSCRIPTEN__

echo "[WASM Build] Success! Generated $OUTPUT and starplat.wasm"
