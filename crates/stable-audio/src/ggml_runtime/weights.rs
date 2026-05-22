use std::collections::HashMap;
use std::ffi::CString;
use std::ptr;

use llama_rs_sys as ffi;
use half::f16;

use crate::ggml_runtime::gguf::GgufModel;
use crate::{Error, Result};

pub struct GgmlWeights {
    ctx: *mut ffi::ggml_context,
    backend: ffi::ggml_backend_t,
    backend_cpu: ffi::ggml_backend_t,
    sched: ffi::ggml_backend_sched_t,
    buffer: ffi::ggml_backend_buffer_t,
    compute_meta: Vec<u8>,
    tensors: HashMap<String, *mut ffi::ggml_tensor>,
}

impl GgmlWeights {
    pub fn load_all(model: &GgufModel) -> Result<Self> {
        let specs = model
            .tensors()
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .map(|info| {
                let shape = model
                    .tensor_shape_by_name(&info.name)?
                    .ok_or_else(|| Error::MissingTensor(info.name.clone()))?;
                Ok(TensorSpec {
                    name: info.name,
                    tensor_type: info.tensor_type.raw(),
                    shape,
                    index: info.index,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let mut weights = Self::allocate_context(specs.len())?;
        for spec in &specs {
            weights.create_tensor(spec)?;
        }
        weights.allocate_buffer()?;
        weights.load_tensor_data(model, &specs)?;
        Ok(weights)
    }

    pub fn tensor(&self, name: &str) -> Result<*mut ffi::ggml_tensor> {
        self.tensors
            .get(name)
            .copied()
            .ok_or_else(|| Error::MissingTensor(name.into()))
    }

    pub fn has_tensor(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    pub fn tensor_f32(&self, name: &str) -> Result<Vec<f32>> {
        let tensor = self.tensor(name)?;
        let nbytes = unsafe { ffi::ggml_nbytes(tensor) };
        let mut bytes = vec![0_u8; nbytes];
        unsafe {
            ffi::ggml_backend_tensor_get(tensor, bytes.as_mut_ptr().cast(), 0, nbytes);
            match (*tensor).type_ {
                ffi::ggml_type_GGML_TYPE_F32 => {
                    if bytes.len() % 4 != 0 {
                        return Err(Error::Ggml(format!(
                            "bad f32 tensor byte length for {name}"
                        )));
                    }
                    Ok(bytes
                        .chunks_exact(4)
                        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                        .collect())
                }
                ffi::ggml_type_GGML_TYPE_F16 => {
                    if bytes.len() % 2 != 0 {
                        return Err(Error::Ggml(format!(
                            "bad f16 tensor byte length for {name}"
                        )));
                    }
                    Ok(bytes
                        .chunks_exact(2)
                        .map(|chunk| {
                            f16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32()
                        })
                        .collect())
                }
                other => Err(Error::UnsupportedTensorType {
                    name: name.into(),
                    tensor_type: other,
                }),
            }
        }
    }

    pub fn backend_name(&self) -> String {
        unsafe {
            let name = ffi::ggml_backend_name(self.backend);
            if name.is_null() {
                return "unknown".into();
            }
            std::ffi::CStr::from_ptr(name)
                .to_string_lossy()
                .into_owned()
        }
    }

    pub(crate) fn compute_context(&mut self) -> Result<*mut ffi::ggml_context> {
        let params = ffi::ggml_init_params {
            mem_size: self.compute_meta.len(),
            mem_buffer: self.compute_meta.as_mut_ptr().cast(),
            no_alloc: true,
        };
        let ctx = unsafe { ffi::ggml_init(params) };
        if ctx.is_null() {
            return Err(Error::Ggml("failed to create compute context".into()));
        }
        Ok(ctx)
    }

    pub(crate) fn scheduler(&self) -> ffi::ggml_backend_sched_t {
        self.sched
    }

    fn allocate_context(n_tensors: usize) -> Result<Self> {
        unsafe {
            ffi::ggml_backend_load_all();
            let backend = ffi::ggml_backend_init_best();
            if backend.is_null() {
                return Err(Error::Ggml("failed to initialize ggml backend".into()));
            }
            let params = ffi::ggml_init_params {
                mem_size: n_tensors * ffi::ggml_tensor_overhead(),
                mem_buffer: ptr::null_mut(),
                no_alloc: true,
            };
            let ctx = ffi::ggml_init(params);
            if ctx.is_null() {
                ffi::ggml_backend_free(backend);
                return Err(Error::Ggml("failed to initialize ggml context".into()));
            }
            Ok(Self {
                ctx,
                backend,
                backend_cpu: ptr::null_mut(),
                sched: ptr::null_mut(),
                buffer: ptr::null_mut(),
                compute_meta: Vec::new(),
                tensors: HashMap::new(),
            })
        }
    }

    fn create_tensor(&mut self, spec: &TensorSpec) -> Result<()> {
        let c_name = CString::new(spec.name.as_str())
            .map_err(|_| Error::InvalidMetadataKey(spec.name.clone()))?;
        let mut ne = [1_i64; 4];
        for (idx, dim) in spec.shape.iter().enumerate().take(4) {
            ne[idx] = *dim as i64;
        }
        let n_dims = spec.shape.len().max(1).min(4) as i32;
        let tensor =
            unsafe { ffi::ggml_new_tensor(self.ctx, spec.tensor_type, n_dims, ne.as_ptr()) };
        if tensor.is_null() {
            return Err(Error::Ggml(format!(
                "failed to create tensor {}",
                spec.name
            )));
        }
        unsafe {
            ffi::ggml_set_name(tensor, c_name.as_ptr());
        }
        self.tensors.insert(spec.name.clone(), tensor);
        Ok(())
    }

    fn allocate_buffer(&mut self) -> Result<()> {
        self.buffer = unsafe { ffi::ggml_backend_alloc_ctx_tensors(self.ctx, self.backend) };
        if self.buffer.is_null() {
            return Err(Error::Ggml("failed to allocate ggml weight buffer".into()));
        }
        unsafe {
            ffi::ggml_backend_buffer_set_usage(
                self.buffer,
                ffi::ggml_backend_buffer_usage_GGML_BACKEND_BUFFER_USAGE_WEIGHTS,
            );
        }
        Ok(())
    }

    fn load_tensor_data(&mut self, model: &GgufModel, specs: &[TensorSpec]) -> Result<()> {
        for spec in specs {
            let tensor = self.tensor(&spec.name)?;
            let bytes = model.tensor_bytes(spec.index)?;
            let expected = unsafe { ffi::ggml_nbytes(tensor) };
            if bytes.len() != expected {
                return Err(Error::Ggml(format!(
                    "tensor {} data has {} bytes, ggml expects {}",
                    spec.name,
                    bytes.len(),
                    expected
                )));
            }
            unsafe {
                ffi::ggml_backend_tensor_set(tensor, bytes.as_ptr().cast(), 0, bytes.len());
            }
        }
        unsafe {
            ffi::ggml_backend_synchronize(self.backend);
        }
        self.init_scheduler()?;
        Ok(())
    }

    fn init_scheduler(&mut self) -> Result<()> {
        unsafe {
            let device = ffi::ggml_backend_get_device(self.backend);
            let mut backends = vec![self.backend];
            if !device.is_null()
                && ffi::ggml_backend_dev_type(device)
                    != ffi::ggml_backend_dev_type_GGML_BACKEND_DEVICE_TYPE_CPU
            {
                self.backend_cpu = ffi::ggml_backend_init_by_type(
                    ffi::ggml_backend_dev_type_GGML_BACKEND_DEVICE_TYPE_CPU,
                    ptr::null(),
                );
                if !self.backend_cpu.is_null() {
                    backends.push(self.backend_cpu);
                }
            }
            self.sched = ffi::ggml_backend_sched_new(
                backends.as_mut_ptr(),
                ptr::null_mut(),
                backends.len() as i32,
                32768,
                false,
                true,
            );
            if self.sched.is_null() {
                return Err(Error::Ggml("failed to create ggml scheduler".into()));
            }
            self.compute_meta =
                vec![0; ffi::ggml_tensor_overhead() * 32768 + ffi::ggml_graph_overhead()];
            Ok(())
        }
    }
}

impl Drop for GgmlWeights {
    fn drop(&mut self) {
        unsafe {
            if !self.sched.is_null() {
                ffi::ggml_backend_sched_free(self.sched);
            }
            if !self.buffer.is_null() {
                ffi::ggml_backend_buffer_free(self.buffer);
            }
            if !self.ctx.is_null() {
                ffi::ggml_free(self.ctx);
            }
            if !self.backend_cpu.is_null() {
                ffi::ggml_backend_free(self.backend_cpu);
            }
            if !self.backend.is_null() {
                ffi::ggml_backend_free(self.backend);
            }
        }
    }
}

struct TensorSpec {
    name: String,
    tensor_type: u32,
    shape: Vec<usize>,
    index: i64,
}
