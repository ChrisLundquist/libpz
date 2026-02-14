//! Shared abstractions for GPU backends (WebGPU).
//!
//! Types in this module are used by the `webgpu` feature-gated backend.
//! The module itself is always compiled (types are generic) so that
//! the backend can import from a single canonical location.

/// Round-robin ring of pre-allocated buffer slots for streaming GPU work.
///
/// Each slot holds backend-specific GPU resources (e.g., `Lz77BufferSlot`).
/// The ring cycles through slots so that GPU compute on one slot can overlap
/// with CPU readback/processing on a previously completed slot.
///
/// Typical depth is 2 (double buffer) or 3 (triple buffer), chosen at
/// creation time based on GPU memory budget.
pub(crate) struct BufferRing<S> {
    pub(crate) slots: Vec<S>,
    next: usize,
}

impl<S> BufferRing<S> {
    /// Create a new ring from pre-allocated slots.
    pub(crate) fn new(slots: Vec<S>) -> Self {
        Self { slots, next: 0 }
    }

    /// Acquire the next slot index, advancing the ring pointer.
    pub(crate) fn acquire(&mut self) -> usize {
        let idx = self.next;
        self.next = (self.next + 1) % self.slots.len();
        idx
    }

    /// Number of slots in the ring.
    pub(crate) fn depth(&self) -> usize {
        self.slots.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buffer_ring_acquire_round_robin() {
        let mut ring = BufferRing::new(vec!["a", "b", "c"]);
        assert_eq!(ring.depth(), 3);
        assert_eq!(ring.acquire(), 0);
        assert_eq!(ring.acquire(), 1);
        assert_eq!(ring.acquire(), 2);
        // Wraps around
        assert_eq!(ring.acquire(), 0);
        assert_eq!(ring.acquire(), 1);
    }

    #[test]
    fn test_buffer_ring_depth_two() {
        let mut ring = BufferRing::new(vec![10, 20]);
        assert_eq!(ring.depth(), 2);
        assert_eq!(ring.acquire(), 0);
        assert_eq!(ring.acquire(), 1);
        assert_eq!(ring.acquire(), 0);
    }

    #[test]
    fn test_buffer_ring_single_slot() {
        let mut ring = BufferRing::new(vec![42]);
        assert_eq!(ring.depth(), 1);
        assert_eq!(ring.acquire(), 0);
        assert_eq!(ring.acquire(), 0);
        assert_eq!(ring.acquire(), 0);
    }
}
