//! Byte-bounded in-memory LRU cache for hot decompressed `.osv` minipages.
//!
//! Stores `Arc<DecodedMinipage>` entries keyed by `(slab_id: u16, minipage_id: u32)`,
//! enforces a strict `max_bytes` resident memory cap via $O(1)$ LRU eviction, and
//! tracks atomic `cache_hits` and `cache_misses` telemetry.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use crate::osv_slab::DecodedMinipage;

/// Default byte capacity for [`MinipageLruCache`] (`64 MiB`).
pub const DEFAULT_MINIPAGE_CACHE_BYTES: usize = 64 * 1024 * 1024;

/// Composite cache key `(slab_id, minipage_id)` uniquely identifying a minipage across slabs.
pub type MinipageCacheKey = (u16, u32);

/// Snapshot of [`MinipageLruCache`] occupancy and hit/miss telemetry.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct MinipageCacheStats {
    /// Total cache hits recorded since creation (or last reset).
    pub cache_hits: u64,
    /// Total cache misses recorded since creation (or last reset).
    pub cache_misses: u64,
    /// Current resident byte footprint of cached minipages.
    pub current_bytes: usize,
    /// Configured maximum byte capacity.
    pub max_bytes: usize,
    /// Number of resident minipage entries.
    pub entry_count: usize,
}

#[derive(Debug)]
struct CacheNode {
    key: MinipageCacheKey,
    page: Arc<DecodedMinipage>,
    byte_len: usize,
    prev: Option<usize>,
    next: Option<usize>,
    occupied: bool,
}

#[derive(Debug)]
struct LruInner {
    index: HashMap<MinipageCacheKey, usize>,
    nodes: Vec<CacheNode>,
    free_slots: Vec<usize>,
    head: Option<usize>,
    tail: Option<usize>,
    current_bytes: usize,
}

impl LruInner {
    fn new() -> Self {
        Self {
            index: HashMap::new(),
            nodes: Vec::new(),
            free_slots: Vec::new(),
            head: None,
            tail: None,
            current_bytes: 0,
        }
    }

    #[inline]
    fn unlink(&mut self, slot: usize) {
        let prev = self.nodes[slot].prev;
        let next = self.nodes[slot].next;

        match prev {
            Some(p) => self.nodes[p].next = next,
            None => self.head = next,
        }
        match next {
            Some(n) => self.nodes[n].prev = prev,
            None => self.tail = prev,
        }

        self.nodes[slot].prev = None;
        self.nodes[slot].next = None;
    }

    #[inline]
    fn push_front(&mut self, slot: usize) {
        self.nodes[slot].prev = None;
        self.nodes[slot].next = self.head;

        match self.head {
            Some(old_head) => self.nodes[old_head].prev = Some(slot),
            None => self.tail = Some(slot),
        }
        self.head = Some(slot);
    }

    #[inline]
    fn move_to_front(&mut self, slot: usize) {
        if self.head == Some(slot) {
            return;
        }
        self.unlink(slot);
        self.push_front(slot);
    }

    fn remove_slot(&mut self, slot: usize) {
        if !self.nodes[slot].occupied {
            return;
        }
        let key = self.nodes[slot].key;
        let byte_len = self.nodes[slot].byte_len;
        self.unlink(slot);
        self.index.remove(&key);
        self.current_bytes = self.current_bytes.saturating_sub(byte_len);
        self.nodes[slot].occupied = false;
        self.free_slots.push(slot);
    }

    fn evict_lru_tail(&mut self) -> bool {
        let Some(tail_slot) = self.tail else {
            return false;
        };
        self.remove_slot(tail_slot);
        true
    }
}

/// Thread-safe, byte-bounded LRU cache of decompressed [`DecodedMinipage`] instances.
#[derive(Debug)]
pub struct MinipageLruCache {
    max_bytes: usize,
    inner: Mutex<LruInner>,
    /// Atomic count of cache hits observed by [`MinipageLruCache::get`].
    pub cache_hits: AtomicU64,
    /// Atomic count of cache misses observed by [`MinipageLruCache::get`].
    pub cache_misses: AtomicU64,
}

impl MinipageLruCache {
    /// Creates an empty [`MinipageLruCache`] bounded by `max_bytes`.
    pub fn new(max_bytes: usize) -> Self {
        Self {
            max_bytes,
            inner: Mutex::new(LruInner::new()),
            cache_hits: AtomicU64::new(0),
            cache_misses: AtomicU64::new(0),
        }
    }

    /// Creates an empty [`MinipageLruCache`] from a megabyte budget (`minipage_cache_mb * 1024 * 1024`).
    pub fn from_mb(minipage_cache_mb: usize) -> Self {
        Self::new(minipage_cache_mb.saturating_mul(1024 * 1024))
    }

    /// Looks up `(slab_id, minipage_id)`, promoting it to MRU on hit and updating atomic telemetry.
    pub fn get(&self, slab_id: u16, minipage_id: u32) -> Option<Arc<DecodedMinipage>> {
        if self.max_bytes == 0 {
            self.cache_misses.fetch_add(1, Ordering::Relaxed);
            return None;
        }

        let key = (slab_id, minipage_id);
        let mut guard = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let Some(&slot) = guard.index.get(&key) else {
            drop(guard);
            self.cache_misses.fetch_add(1, Ordering::Relaxed);
            return None;
        };

        guard.move_to_front(slot);
        let page = Arc::clone(&guard.nodes[slot].page);
        drop(guard);

        self.cache_hits.fetch_add(1, Ordering::Relaxed);
        Some(page)
    }

    /// Inserts or replaces `(slab_id, minipage_id)`, evicting LRU entries until
    /// `current_bytes + page_bytes <= max_bytes`.
    ///
    /// Entries whose uncompressed byte size exceeds `max_bytes` (or when `max_bytes == 0`)
    /// are not retained so the memory ceiling is never violated.
    pub fn insert(&self, slab_id: u16, minipage_id: u32, page: Arc<DecodedMinipage>) {
        let page_bytes = page.byte_len();
        let key = (slab_id, minipage_id);

        let mut guard = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        if self.max_bytes == 0 || page_bytes > self.max_bytes {
            if let Some(&existing_slot) = guard.index.get(&key) {
                guard.remove_slot(existing_slot);
            }
            return;
        }

        if let Some(&existing_slot) = guard.index.get(&key) {
            let old_bytes = guard.nodes[existing_slot].byte_len;
            guard.current_bytes = guard.current_bytes.saturating_sub(old_bytes);
            guard.nodes[existing_slot].page = page;
            guard.nodes[existing_slot].byte_len = page_bytes;
            guard.move_to_front(existing_slot);

            while guard.current_bytes + page_bytes > self.max_bytes {
                if guard.tail == Some(existing_slot) || !guard.evict_lru_tail() {
                    break;
                }
            }
            guard.current_bytes += page_bytes;
            return;
        }

        while guard.current_bytes + page_bytes > self.max_bytes {
            if !guard.evict_lru_tail() {
                break;
            }
        }

        let slot = if let Some(reused) = guard.free_slots.pop() {
            guard.nodes[reused] = CacheNode {
                key,
                page,
                byte_len: page_bytes,
                prev: None,
                next: None,
                occupied: true,
            };
            reused
        } else {
            let idx = guard.nodes.len();
            guard.nodes.push(CacheNode {
                key,
                page,
                byte_len: page_bytes,
                prev: None,
                next: None,
                occupied: true,
            });
            idx
        };

        guard.index.insert(key, slot);
        guard.push_front(slot);
        guard.current_bytes += page_bytes;
    }

    /// Removes all cached entries and resets `current_bytes` to `0`.
    pub fn clear(&self) {
        let mut guard = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        guard.index.clear();
        guard.nodes.clear();
        guard.free_slots.clear();
        guard.head = None;
        guard.tail = None;
        guard.current_bytes = 0;
    }

    /// Returns the configured byte limit `max_bytes`.
    #[inline]
    pub const fn max_bytes(&self) -> usize {
        self.max_bytes
    }

    /// Returns current resident bytes occupied in the cache.
    pub fn current_bytes(&self) -> usize {
        let guard = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        guard.current_bytes
    }

    /// Returns the number of cached minipages currently resident.
    pub fn len(&self) -> usize {
        let guard = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        guard.index.len()
    }

    /// Returns `true` if no minipages are currently cached.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns total atomic cache hits recorded.
    #[inline]
    pub fn hits(&self) -> u64 {
        self.cache_hits.load(Ordering::Relaxed)
    }

    /// Returns total atomic cache misses recorded.
    #[inline]
    pub fn misses(&self) -> u64 {
        self.cache_misses.load(Ordering::Relaxed)
    }

    /// Returns a point-in-time snapshot of occupancy and hit/miss telemetry.
    pub fn stats(&self) -> MinipageCacheStats {
        let guard = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        MinipageCacheStats {
            cache_hits: self.cache_hits.load(Ordering::Relaxed),
            cache_misses: self.cache_misses.load(Ordering::Relaxed),
            current_bytes: guard.current_bytes,
            max_bytes: self.max_bytes,
            entry_count: guard.index.len(),
        }
    }
}

#[cfg(test)]
mod tests {
    use half::f16;
    use osprey_core::OspreyResult;

    use super::*;
    use crate::osv_slab::{build_minipages, OsvVectorRecord};

    fn make_decoded_minipage(
        doc_id: u64,
        cluster_id: u32,
        payload_len: usize,
    ) -> OspreyResult<Arc<DecodedMinipage>> {
        let vec_f16 = [f16::from_f32(1.0), f16::from_f32(2.0)];
        let payload = vec![b'a'; payload_len];
        let records = [OsvVectorRecord {
            doc_id,
            cluster_id,
            vector_f16: &vec_f16,
            payload: &payload,
        }];
        let pages = build_minipages(&records, 2, 16_384, None)?;
        let buf = osprey_core::AlignedBuffer::from_slice(&pages[0].bytes)?;
        Ok(Arc::new(DecodedMinipage::from_aligned_buffer(buf)?))
    }

    #[test]
    fn lru_cache_evicts_least_recently_used_and_tracks_telemetry() -> OspreyResult<()> {
        let p0 = make_decoded_minipage(10, 0, 64)?;
        let p1 = make_decoded_minipage(20, 1, 64)?;
        let p2 = make_decoded_minipage(30, 2, 64)?;
        let page_size = p0.byte_len();
        assert_eq!(page_size, p1.byte_len());
        assert_eq!(page_size, p2.byte_len());

        // Capacity for exactly 2 pages.
        let cache = MinipageLruCache::new(page_size * 2);
        assert!(cache.is_empty());

        // Miss on empty cache
        assert!(cache.get(0, 0).is_none());
        assert_eq!(cache.misses(), 1);
        assert_eq!(cache.hits(), 0);

        cache.insert(0, 0, Arc::clone(&p0));
        cache.insert(0, 1, Arc::clone(&p1));
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.current_bytes(), page_size * 2);

        // Touch (0, 0) so (0, 1) becomes the LRU victim.
        let hit0 = cache.get(0, 0);
        assert!(hit0.is_some());
        assert_eq!(cache.hits(), 1);

        // Insert (0, 2): must evict (0, 1), keeping (0, 0) and (0, 2).
        cache.insert(0, 2, Arc::clone(&p2));
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.current_bytes(), page_size * 2);
        assert!(cache.get(0, 1).is_none());
        assert!(cache.get(0, 0).is_some());
        assert!(cache.get(0, 2).is_some());

        // Distinct slab_id uses distinct key even with same minipage_id.
        cache.insert(1, 0, Arc::clone(&p1));
        assert_eq!(cache.len(), 2);
        assert!(cache.get(0, 0).is_none()); // evicted as LRU
        assert!(cache.get(1, 0).is_some());

        let stats = cache.stats();
        assert_eq!(stats.current_bytes, page_size * 2);
        assert_eq!(stats.max_bytes, page_size * 2);
        assert_eq!(stats.entry_count, 2);
        assert_eq!(stats.cache_hits, cache.hits());
        assert_eq!(stats.cache_misses, cache.misses());

        cache.clear();
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.current_bytes(), 0);
        Ok(())
    }

    #[test]
    fn lru_cache_handles_oversized_zero_cap_and_updates() -> OspreyResult<()> {
        let small = make_decoded_minipage(1, 0, 16)?;
        let big = make_decoded_minipage(2, 0, 512)?;

        // Zero-capacity cache never retains entries.
        let zero_cache = MinipageLruCache::new(0);
        zero_cache.insert(0, 0, Arc::clone(&small));
        assert_eq!(zero_cache.len(), 0);
        assert!(zero_cache.get(0, 0).is_none());

        // Cache large enough for two `small` pages, smaller than `big`.
        let cap = small.byte_len() * 2;
        assert!(big.byte_len() > cap);
        let cache = MinipageLruCache::new(cap);
        cache.insert(0, 0, Arc::clone(&small));
        cache.insert(0, 1, Arc::clone(&small));
        assert_eq!(cache.current_bytes(), cap);

        // Updating (0, 0) with `big` (> max_bytes) removes (0, 0) without exceeding cap.
        cache.insert(0, 0, Arc::clone(&big));
        assert!(cache.get(0, 0).is_none());
        assert!(cache.get(0, 1).is_some());
        assert_eq!(cache.current_bytes(), small.byte_len());

        // Updating an existing key with larger-but-fitting page evicts LRU neighbor.
        let med = make_decoded_minipage(3, 0, 40)?;
        assert!(med.byte_len() <= cap);
        assert!(med.byte_len() + small.byte_len() > cap);
        cache.insert(0, 2, Arc::clone(&small));
        assert_eq!(cache.len(), 2);
        cache.insert(0, 2, Arc::clone(&med));
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.current_bytes(), med.byte_len());
        assert!(cache.get(0, 2).is_some());
        assert!(cache.get(0, 1).is_none());
        Ok(())
    }
}
