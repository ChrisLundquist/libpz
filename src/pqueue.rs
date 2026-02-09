//! A min-heap priority queue.
//!
//! This is a correct implementation of a binary min-heap, fixing the
//! sentinel/sift-down bug (BUG-01) present in the C reference.

/// An entry in the priority queue.
#[derive(Debug, Clone)]
struct HeapEntry<T> {
    priority: u32,
    data: T,
}

/// A min-heap priority queue that pops the lowest-priority element first.
///
/// Uses 0-indexed storage with parent = (i-1)/2, children = 2i+1, 2i+2.
#[derive(Debug, Clone)]
pub struct MinHeap<T> {
    nodes: Vec<HeapEntry<T>>,
}

impl<T> MinHeap<T> {
    /// Create a new, empty min-heap.
    pub fn new() -> Self {
        MinHeap { nodes: Vec::new() }
    }

    /// Returns the number of elements in the heap.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Returns true if the heap is empty.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Push an element onto the heap with the given priority.
    pub fn push(&mut self, priority: u32, data: T) {
        self.nodes.push(HeapEntry { priority, data });
        self.sift_up(self.nodes.len() - 1);
    }

    /// Pop the minimum-priority element from the heap.
    ///
    /// Returns `None` if the heap is empty.
    pub fn pop(&mut self) -> Option<T> {
        if self.nodes.is_empty() {
            return None;
        }
        if self.nodes.len() == 1 {
            return Some(self.nodes.pop().unwrap().data);
        }
        // Swap root with last, remove last, sift down root
        let last = self.nodes.len() - 1;
        self.nodes.swap(0, last);
        let result = self.nodes.pop().unwrap();
        if !self.nodes.is_empty() {
            self.sift_down(0);
        }
        Some(result.data)
    }

    /// Sift element at `index` up to maintain heap property.
    fn sift_up(&mut self, mut index: usize) {
        while index > 0 {
            let parent = (index - 1) / 2;
            if self.nodes[index].priority < self.nodes[parent].priority {
                self.nodes.swap(index, parent);
                index = parent;
            } else {
                break;
            }
        }
    }

    /// Sift element at `index` down to maintain heap property.
    ///
    /// This is the corrected version of the C sift-down (BUG-01 fix):
    /// we compare children against the current node rather than using
    /// an out-of-bounds sentinel.
    fn sift_down(&mut self, mut index: usize) {
        let len = self.nodes.len();
        loop {
            let left = 2 * index + 1;
            let right = 2 * index + 2;
            let mut smallest = index;

            if left < len && self.nodes[left].priority < self.nodes[smallest].priority {
                smallest = left;
            }
            if right < len && self.nodes[right].priority < self.nodes[smallest].priority {
                smallest = right;
            }

            if smallest == index {
                break;
            }

            self.nodes.swap(index, smallest);
            index = smallest;
        }
    }
}

impl<T> Default for MinHeap<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_heap() {
        let mut heap: MinHeap<i32> = MinHeap::new();
        assert!(heap.is_empty());
        assert_eq!(heap.len(), 0);
        assert!(heap.pop().is_none());
    }

    #[test]
    fn test_single_element() {
        let mut heap = MinHeap::new();
        heap.push(5, "hello");
        assert_eq!(heap.len(), 1);
        assert_eq!(heap.pop(), Some("hello"));
        assert!(heap.is_empty());
    }

    #[test]
    fn test_min_order() {
        let mut heap = MinHeap::new();
        heap.push(3, "three");
        heap.push(1, "one");
        heap.push(2, "two");

        assert_eq!(heap.pop(), Some("one"));
        assert_eq!(heap.pop(), Some("two"));
        assert_eq!(heap.pop(), Some("three"));
        assert!(heap.pop().is_none());
    }

    #[test]
    fn test_reverse_insert() {
        let mut heap = MinHeap::new();
        for i in (0..10).rev() {
            heap.push(i, i);
        }
        for i in 0..10 {
            assert_eq!(heap.pop(), Some(i));
        }
    }

    #[test]
    fn test_duplicate_priorities() {
        let mut heap = MinHeap::new();
        heap.push(1, "a");
        heap.push(1, "b");
        heap.push(1, "c");

        // All should come out (order among equal priorities is unspecified)
        let mut results = vec![];
        while let Some(v) = heap.pop() {
            results.push(v);
        }
        assert_eq!(results.len(), 3);
        results.sort();
        assert_eq!(results, vec!["a", "b", "c"]);
    }

    #[test]
    fn test_interleaved_push_pop() {
        let mut heap = MinHeap::new();
        heap.push(5, 5);
        heap.push(3, 3);
        assert_eq!(heap.pop(), Some(3));
        heap.push(1, 1);
        heap.push(4, 4);
        assert_eq!(heap.pop(), Some(1));
        assert_eq!(heap.pop(), Some(4));
        assert_eq!(heap.pop(), Some(5));
    }

    #[test]
    fn test_large_heap() {
        let mut heap = MinHeap::new();
        // Insert 1000 elements in random-ish order
        for i in 0u32..1000 {
            let priority = (i * 997) % 1000; // pseudo-shuffle
            heap.push(priority, priority);
        }
        let mut prev = 0u32;
        while let Some(val) = heap.pop() {
            assert!(val >= prev, "heap order violated: {} < {}", val, prev);
            prev = val;
        }
    }
}
