Triangle Counting with Deno and WebGPU
One of the biggest advantages of this program is its portability. It has very few external dependencies, making it easy to run on different machines.
Requirements
1. Deno Runtime
This is the only piece of software you need to install. Deno acts as the all-in-one environment that runs our JavaScript code, understands the WebGPU API, and manages the program.
You can install it on any Linux, Mac, or Windows machine with a single command:
curl -fsSL [https://deno.land/x/install/install.sh](https://deno.land/x/install/install.sh) | sh


2. GPU and Drivers
The machine must have a modern GPU and up-to-date graphics drivers that support one of the following APIs:
Vulkan (on Linux)
Metal (on macOS)
DirectX 12 (on Windows)
On any modern computer or supercomputing cluster, these drivers are typically pre-installed and managed by the system.
What You Don't Need
This project is self-contained and does not require any of the following:
No Node.js or npm: Our program is completely independent of the Node.js ecosystem.
No Python: All data processing is handled natively in Deno.
No C++ Compiler (like g++): We are not compiling any native code.
No CUDA or OpenMP: WebGPU is a higher-level API that works directly with the system's graphics drivers.
No Manual Package Management: The script uses one small utility from Deno's standard library (jsr:@std/streams/delimiter-stream), which Deno automatically downloads and caches the first time you run the script.
