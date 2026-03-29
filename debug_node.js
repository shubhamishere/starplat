var fs = require('fs');

// Define Module object locally to capture the runtime
var Module = {
    onRuntimeInitialized: function () {
        console.log("Runtime Initialized!");
        runTests(Module);
    },
    print: function (text) { console.log("[STDOUT]", text); },
    printErr: function (text) { console.error("[STDERR]", text); }
};

// Emscripten checks for existing 'Module' object and extends it.
global.Module = Module;

try {
    require('./starplat.js');
} catch (e) {
    console.error("Error requiring starplat.js:", e);
}

function runTests(instance) {
    // Write the input file to MEMFS
    var inputCode = fs.readFileSync('reproduce_issue.sp', 'utf8');
    instance.FS.writeFile('/input.sp', inputCode);

    // Create output directories
    try { instance.FS.mkdir('/app'); } catch (e) { }
    try { instance.FS.mkdir('/app/src'); } catch (e) { }
    try { instance.FS.mkdir('/app/graphcode'); } catch (e) { }
    try { instance.FS.mkdir('/app/graphcode/generated_cuda'); } catch (e) { }
    try { instance.FS.mkdir('/app/graphcode/generated_omp'); } catch (e) { }

    instance.FS.chdir('/app/src');

    console.log("Running StarPlat with CUDA...");
    try {
        instance.callMain(['-s', '-f', '/input.sp', '-b', 'cuda']);
    } catch (e) {
        console.error("CUDA Run Failed:", e);
    }

    // Check for output
    try {
        var files = instance.FS.readdir('/app/graphcode/generated_cuda');
        console.log("CUDA Output Files:", files);

        // Read file content if any
        files.forEach(function (f) {
            if (f !== '.' && f !== '..') {
                console.log("Content of " + f + ":");
                console.log(instance.FS.readFile('/app/graphcode/generated_cuda/' + f, { encoding: 'utf8' }));
            }
        });

    } catch (e) { console.log("No CUDA output found"); }

    console.log("\nRunning StarPlat with OpenMP...");
    try {
        instance.callMain(['-s', '-f', '/input.sp', '-b', 'omp']);
    } catch (e) {
        console.error("OpenMP Run Failed:", e);
    }
    // Check for output
    try {
        var files = instance.FS.readdir('/app/graphcode/generated_omp');
        console.log("OpenMP Output Files:", files);

        // Read file content if any
        files.forEach(function (f) {
            if (f !== '.' && f !== '..') {
                console.log("Content of " + f + ":");
                console.log(instance.FS.readFile('/app/graphcode/generated_omp/' + f, { encoding: 'utf8' }));
            }
        });

    } catch (e) { console.log("No OpenMP output found"); }
}
