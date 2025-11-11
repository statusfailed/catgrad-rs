//! # MLIR LLVM Runtime
//!
//! Provides safe RAII wrappers around:
//!
//! - dlopen/dlclose and function symbol resolution (TODO: and calling).
//! - TODO: Simplified MLIR type creation/destruction
use libffi::middle::CodePtr;
use libffi::raw;
use std::collections::HashMap;
use std::ffi::{CString, c_void};
use std::path::Path;

////////////////////////////////////////////////////////////////////////////////
// Runtime

#[derive(Debug)]
pub enum RuntimeError {
    LibraryLoadError(String),
    SymbolNotFound(String),
    InvalidPath(String),
}

impl std::fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RuntimeError::LibraryLoadError(msg) => write!(f, "Library load error: {}", msg),
            RuntimeError::SymbolNotFound(msg) => write!(f, "Symbol not found: {}", msg),
            RuntimeError::InvalidPath(msg) => write!(f, "Invalid path: {}", msg),
        }
    }
}

impl std::error::Error for RuntimeError {}

pub struct LlvmRuntime {
    lib_handle: *mut c_void,
    entrypoints: HashMap<CString, (CodePtr, Entrypoint)>,
}

pub struct Entrypoint {
    pub func_name: CString,
    pub source_types: Vec<()>,
    pub target_types: Vec<()>,
}

impl Drop for LlvmRuntime {
    fn drop(&mut self) {
        unsafe {
            if !self.lib_handle.is_null() {
                libc::dlclose(self.lib_handle);
            }
        }
    }
}

impl LlvmRuntime {
    /// Create a new runtime from a path to a .so file.
    /// Loads all specified entrypoints and verifies they exist.
    pub fn new(so_path: &Path, entrypoints: Vec<Entrypoint>) -> Result<LlvmRuntime, RuntimeError> {
        // Convert path to C string
        let path_str = so_path
            .to_str()
            .ok_or_else(|| RuntimeError::InvalidPath("Path contains invalid UTF-8".to_string()))?;
        let lib_path = CString::new(path_str)
            .map_err(|_| RuntimeError::InvalidPath("Path contains null bytes".to_string()))?;

        unsafe {
            // Load the shared library, resolve missing symbols now
            let handle = libc::dlopen(lib_path.as_ptr(), libc::RTLD_NOW);
            if handle.is_null() {
                let error = libc::dlerror();
                let error_msg = if !error.is_null() {
                    std::ffi::CStr::from_ptr(error)
                        .to_string_lossy()
                        .to_string()
                } else {
                    "Unknown dlopen error".to_string()
                };
                return Err(RuntimeError::LibraryLoadError(error_msg));
            }

            // Load and verify all entrypoints
            let mut loaded_entrypoints = HashMap::new();
            for entrypoint in entrypoints {
                let func_ptr = libc::dlsym(handle, entrypoint.func_name.as_ptr());
                if func_ptr.is_null() {
                    let error = libc::dlerror();
                    let error_msg = if !error.is_null() {
                        std::ffi::CStr::from_ptr(error)
                            .to_string_lossy()
                            .to_string()
                    } else {
                        format!(
                            "Symbol '{}' not found",
                            entrypoint.func_name.to_string_lossy()
                        )
                    };
                    libc::dlclose(handle); // Clean up on error
                    return Err(RuntimeError::SymbolNotFound(error_msg));
                }

                let code_ptr = CodePtr(func_ptr);
                loaded_entrypoints.insert(entrypoint.func_name.clone(), (code_ptr, entrypoint));
            }

            Ok(LlvmRuntime {
                lib_handle: handle,
                entrypoints: loaded_entrypoints,
            })
        }
    }

    /// Create a tensor from an f32 buffer and a layout (extents and strides)
    ///
    /// The layout parameter contains tuples of (size, stride) for each dimension.
    ///
    /// WARNING: This creates a tensor that borrows from the input data.
    /// The caller must ensure the data lives as long as the tensor.
    pub fn tensor(data: &mut [f32], layout: Vec<(usize, usize)>) -> MlirTensor<f32> {
        let data_ptr = data.as_mut_ptr();

        // Extract sizes and strides from layout
        let (sizes, strides): (Vec<_>, Vec<_>) = layout.into_iter().unzip();
        let sizes: Vec<i64> = sizes.into_iter().map(|s| s as i64).collect();
        let strides: Vec<i64> = strides.into_iter().map(|s| s as i64).collect();

        MlirTensor {
            allocated: data_ptr,
            aligned: data_ptr, // For user-created tensors, allocated == aligned
            offset: 0,
            sizes,
            strides,
        }
    }

    /// Get a function pointer and its signature by name
    pub fn get_entrypoint(&self, name: &CString) -> Option<(CodePtr, &Entrypoint)> {
        self.entrypoints.get(name).map(|(ptr, ep)| (*ptr, ep))
    }

    /// List all loaded entrypoint names
    pub fn entrypoint_names(&self) -> Vec<&CString> {
        self.entrypoints.keys().collect()
    }
}

////////////////////////////////////////////////////////////////////////////////
use libffi::middle::*;

/// Helper struct to marshal bytes into the Memref format expected by LLVM
/// TODO: remove pub fields!
#[derive(Debug)]
pub struct MlirTensor<T> {
    pub allocated: *mut T,
    pub aligned: *mut T,
    pub offset: i64, // offset into ptr
    pub sizes: Vec<i64>,
    pub strides: Vec<i64>,
}

impl<T> MlirTensor<T> {
    /// Construct a `libffi::middle::Arg` for each field as if of the below C struct.
    /// ```c
    /// #[repr(C)]
    /// struct Memref {
    ///     allocated: *mut f32,
    ///     aligned: *mut f32,
    ///     offset: i64,
    ///     sizes: [i64; N],
    ///     strides: [i64; N],
    /// }
    /// ```
    pub fn to_args<'a>(&'a self) -> Vec<Arg<'a>> {
        let mut result = Vec::with_capacity(3 + 2 * self.sizes.len());
        result.push(Arg::new(&self.allocated));
        result.push(Arg::new(&self.aligned));
        result.push(Arg::new(&self.offset));
        for size in &self.sizes {
            result.push(Arg::new(size));
        }
        for stride in &self.strides {
            result.push(Arg::new(stride));
        }
        result
    }
}

#[derive(Debug)]
pub enum MlirValue {
    MlirTensor(MlirTensor<f32>), // TODO: f32 specialisation
    I64(i64),
}

impl MlirValue {
    pub fn to_type(&self) -> MlirType {
        match self {
            MlirValue::MlirTensor(tensor) => MlirType::Memref(tensor.sizes.len()),
            MlirValue::I64(_) => MlirType::I64,
        }
    }

    pub fn to_args<'a>(&'a self) -> Vec<Arg<'a>> {
        match self {
            MlirValue::MlirTensor(tensor) => tensor.to_args(),
            MlirValue::I64(val) => vec![Arg::new(val)],
        }
    }
}

#[derive(Debug, Clone)]
pub enum MlirType {
    Memref(usize), // ranked memref
    I64,
}

impl MlirType {
    // Get individual field types
    pub fn to_fields(&self) -> Vec<Type> {
        match self {
            MlirType::Memref(rank) => {
                let mut fields = vec![
                    Type::pointer(), // allocated
                    Type::pointer(), // aligned
                    Type::i64(),     // offset
                ];
                // Add rank number of sizes
                for _ in 0..*rank {
                    fields.push(Type::i64());
                }
                // Add rank number of strides
                for _ in 0..*rank {
                    fields.push(Type::i64());
                }
                fields
            }
            MlirType::I64 => vec![Type::i64()],
        }
    }

    // Get the struct type (wraps fields in a struct)
    pub fn to_type(&self) -> Type {
        let fields = self.to_fields();
        if fields.len() == 1 {
            fields[0].clone()
        } else {
            Type::structure(fields)
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// Utility to call the function pointer with specified source/target types

// TODO: MEMORY MANAGEMENT
// Returned `Value`s internally have ptrs to array data which needs to be freed manually.
// FIX: take ownership when wrapping in a more accessible catgrad type.
// TODO: remove pub modifier
pub fn call(
    ptr: CodePtr,
    source_values: Vec<MlirValue>,
    target_types: Vec<MlirType>,
) -> Vec<MlirTensor<f32>> {
    // Derive source types from the source values (flat map to get individual arg types)
    let source_types: Vec<Type> = source_values
        .iter()
        .flat_map(|v| v.to_type().to_fields())
        .collect();

    // Build args from source values (flatten all args)
    let args: Vec<Arg> = source_values.iter().flat_map(|v| v.to_args()).collect();

    // Build return type from target types
    let result_fields: Vec<Type> = target_types.iter().map(|vt| vt.to_type()).collect();
    let return_type = Type::structure(result_fields);

    unsafe {
        // Make Cif from source/return types
        let cif = Cif::new(source_types, return_type);

        // Calculate result buffer size dynamically
        let result_size = calculate_result_size(&target_types);
        let mut result = vec![0u8; result_size];

        // Call using raw ffi_call directly
        raw::ffi_call(
            cif.as_raw_ptr(),
            Some(*ptr.as_safe_fun()),
            result.as_mut_ptr().cast::<c_void>(),
            args.as_ptr() as *mut *mut c_void,
        );

        // Parse results dynamically from the buffer
        parse_results_from_buffer(&result, &target_types)
    }
}

fn calculate_result_size(target_types: &[MlirType]) -> usize {
    // TODO: this probably makes some non-portable assumptions about the memref struct layout
    target_types
        .iter()
        .map(|vt| match vt {
            MlirType::Memref(rank) => {
                // Size = 2 pointers + 1 i64 offset + rank sizes + rank strides
                2 * std::mem::size_of::<*mut f32>()
                    + std::mem::size_of::<i64>()
                    + 2 * rank * std::mem::size_of::<i64>()
            }
            MlirType::I64 => std::mem::size_of::<i64>(),
        })
        .sum()
}

fn parse_results_from_buffer(buffer: &[u8], target_types: &[MlirType]) -> Vec<MlirTensor<f32>> {
    unsafe {
        let mut offset = 0;
        let mut results = Vec::new();

        for target_type in target_types {
            let tensor = match target_type {
                MlirType::Memref(rank) => {
                    // Parse the memref struct fields manually using actual rank
                    let allocated = std::ptr::read(buffer.as_ptr().add(offset) as *const *mut f32);
                    offset += std::mem::size_of::<*mut f32>();

                    let aligned = std::ptr::read(buffer.as_ptr().add(offset) as *const *mut f32);
                    offset += std::mem::size_of::<*mut f32>();

                    let memref_offset = std::ptr::read(buffer.as_ptr().add(offset) as *const i64);
                    offset += std::mem::size_of::<i64>();

                    // Read the rank number of sizes
                    let mut sizes = Vec::with_capacity(*rank);
                    for _ in 0..*rank {
                        let size = std::ptr::read(buffer.as_ptr().add(offset) as *const i64);
                        sizes.push(size);
                        offset += std::mem::size_of::<i64>();
                    }

                    // Read the rank number of strides
                    let mut strides = Vec::with_capacity(*rank);
                    for _ in 0..*rank {
                        let stride = std::ptr::read(buffer.as_ptr().add(offset) as *const i64);
                        strides.push(stride);
                        offset += std::mem::size_of::<i64>();
                    }

                    MlirTensor {
                        allocated, // Shallow copy of pointer
                        aligned,   // Shallow copy of pointer
                        offset: memref_offset,
                        sizes,
                        strides,
                    }
                }
                MlirType::I64 => {
                    panic!("I64 not handled yet")
                }
            };
            results.push(tensor);
        }

        results
    }
}
