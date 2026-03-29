var fs = require('fs');
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
        // List output files
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
            console.log("Could not list output files:", e.message);
        }
    },
    print: function (text) { console.log("[STDOUT]", text); },
    printErr: function (text) { console.error("[STDERR]", text); }
};

global.Module = Module;

// Set fake arguments for the compiler
// Usage: ./StarPlat -s -f <file> -b <backend>
process.argv = ['node', 'starplat.js', '-s', '-f', '/input.sp', '-b', backend];

try {
    require('./starplat.js');
} catch (e) {
    console.error("Crash during require:", e);
}
