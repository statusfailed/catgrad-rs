//! # MLIR LLVM Runtime
//!
//! Provides safe RAII wrappers around:
//!
//! - dlopen/dlclose and function symbol resolution (TODO: and calling).
//! - TODO: Simplified MLIR type creation/destruction
use catgrad::prelude::Type;
use libffi::middle::CodePtr;
use std::collections::HashMap;
use std::ffi::{CString, c_void};
use std::path::Path;

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
    pub source_types: Vec<Type>,
    pub target_types: Vec<Type>,
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

    /// Get a function pointer and its signature by name
    pub fn get_entrypoint(&self, name: &CString) -> Option<(CodePtr, &Entrypoint)> {
        self.entrypoints.get(name).map(|(ptr, ep)| (*ptr, ep))
    }

    /// List all loaded entrypoint names
    pub fn entrypoint_names(&self) -> Vec<&CString> {
        self.entrypoints.keys().collect()
    }
}
