# Host Utilities (JavaScript/TypeScript)

This directory contains JavaScript/TypeScript utilities for WebGPU host-side operations.

## Files

- **`webgpu_host_utils.js`** - Common WebGPU operations and utilities
- **`webgpu_device_manager.js`** - Device initialization and management  
- **`webgpu_buffer_utils.js`** - Buffer creation, management, and operations
- **`webgpu_pipeline_cache.js`** - Pipeline and shader module caching (Phase 3.1)

## Implementation Status

- [ ] **Phase 3.17**: Create base host utility classes
- [ ] **Phase 3.1**: Add pipeline caching utilities  
- [ ] **Phase 3.2**: Add auto-bind group generation
- [ ] **Phase 3.4**: Add selective property copy-back

## Usage

```javascript
// Import utilities
import { WebGPUDeviceManager } from "./webgpu_device_manager.js";
import { WebGPUBufferUtils } from "./webgpu_buffer_utils.js";

// Initialize device
const deviceManager = new WebGPUDeviceManager();
const device = await deviceManager.initialize();

// Create buffers
const bufferUtils = new WebGPUBufferUtils(device);
const buffer = bufferUtils.createStorageBuffer(data);
```

## Design Goals

1. **Consistency** - Unified API for WebGPU operations
2. **Reusability** - Common patterns extracted into utilities
3. **Performance** - Efficient buffer management and caching
4. **Error Handling** - Robust error handling and validation
5. **Integration** - Easy integration with existing CSR loaders

## Relationship to CSR Loaders

These utilities will complement the CSR loaders created in Phase 3.3:
- CSR loaders handle graph loading and CSR generation
- Host utilities handle WebGPU-specific operations
- Together they provide complete WebGPU algorithm infrastructure
