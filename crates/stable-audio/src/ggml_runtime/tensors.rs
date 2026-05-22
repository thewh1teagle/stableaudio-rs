#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct TensorType(pub u32);

impl TensorType {
    pub fn raw(self) -> u32 {
        self.0
    }
}
