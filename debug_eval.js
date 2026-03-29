var fs = require('fs');
var vm = require('vm');
var backend = process.argv[2] || 'cuda';

console.log("Debugging backend: " + backend);

var Module = {
    preRun: [function () {
        console.log("preRun: Setting up filesystem...");
        var inputCode = fs.readFileSync('reproduce_issue.sp', 'utf8');
        Module.FS.writeFile('/input.sp', inputCode);

        try { Module.FS.mkdir('/app'); } catch (e) { }
        try { Module.FS.mkdir('/app/src'); } catch (e) { }
        try { Module.FS.mkdir('/app/graphcode'); } catch (e) { }
        try { Module.FS.mkdir('/app/graphcode/generated_' + backend); } catch (e) { }

        Module.FS.chdir('/app/src');
        console.log("preRun: Filesystem ready.");
    }],
    onExit: function (code) {
        console.log("Program exited with code:", code);
        try {
            var outDir = '/app/graphcode/generated_' + backend;
            var files = Module.FS.readdir(outDir);
            console.log("Output files in " + outDir + ":", files);

            files.forEach(function (f) {
                if (f !== '.' && f !== '..') {
                    console.log("--- Content of " + f + " ---");
                    console.log(Module.FS.readFile(outDir + '/' + f, { encoding: 'utf8' }));
                    console.log("--- End of " + f + " ---");
                }
            });

        } catch (e) {
            console.log("Could not list output files (maybe directory undefined):", e.message);
        }
    },
    // Capture stdout/stderr
    print: function (text) { console.log("[STDOUT]", text); },
    printErr: function (text) { console.error("[STDERR]", text); }
};

// Mock process.argv
// Original: node debug_eval.js <backend>
// We want starplat to see: node starplat.js -s -f /input.sp -b <backend>
var fakeArgv = ['node', 'starplat.js', '-s', '-f', '/input.sp', '-b', backend];

// Prepare context
var context = {
    Module: Module,
    require: require,
    process: {
        versions: process.versions,
        platform: process.platform,
        argv: fakeArgv,
        exit: process.exit,
        on: process.on,
        cwd: process.cwd,
        env: process.env,
        stdin: process.stdin,
        stdout: process.stdout,
        stderr: process.stderr,
        exitCode: 0,
        type: process.type
    },
    console: console,
    global: global, // Be careful with global
    globalThis: global,
    Buffer: Buffer,
    __dirname: __dirname,
    __filename: __dirname + '/starplat.js',
    URL: URL,
    TextDecoder: TextDecoder,
    TextEncoder: TextEncoder,
    setTimeout: setTimeout,
    clearTimeout: clearTimeout,
    setInterval: setInterval,
    clearInterval: clearInterval
};
context.global = context; // Ciruclar reference common in global scope

var code = fs.readFileSync('./starplat.js', 'utf8');

console.log("Running starplat.js in VM...");
try {
    vm.runInNewContext(code, context);
} catch (e) {
    console.error("VM Execution Error:", e);
}
