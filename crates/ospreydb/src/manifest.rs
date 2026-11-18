//! Stateless multi-node S3 cluster coordination via atomic `manifest.json` CAS
//! (`PutObject` with `If-Match` / `ETag` and `If-None-Match: *`).

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use base64::engine::general_purpose::STANDARD;
use base64::Engine as _;
use bytes::Bytes;
use object_store::path::Path as ObjectPath;
use object_store::{ObjectStore, PutMode, PutPayload, UpdateVersion};
use roaring::RoaringBitmap;
use serde::{Deserialize, Serialize};

use crate::error::{ProEngineError, ProResult};

/// Default object key for the cluster manifest inside the configured prefix.
pub const DEFAULT_MANIFEST_KEY: &str = "manifest.json";

/// Schema version for [`ClusterManifest`].
pub const MANIFEST_SCHEMA_VERSION_V1: u32 = 1;

/// Metadata for a single immutable `.osi` + `.osv` segment published in S3.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SegmentDescriptor {
    /// Monotonic segment identifier.
    pub segment_id: u64,
    /// Object key of the immutable `.osi` 1-bit index.
    pub osi_key: String,
    /// Object key of the immutable `.osv` locality-reordered Zstd slab.
    pub osv_key: String,
    /// Original unpadded embedding dimensionality.
    pub orig_dim: u16,
    /// FWHT-padded dimensionality (`padded_dim % 64 == 0`).
    pub padded_dim: u16,
    /// Deterministic Rademacher FWHT seed for this segment.
    pub fwht_seed: u64,
    /// Total vector count stored in this segment when built.
    pub num_vectors: u64,
    /// Portable `RoaringBitmap` bytes (Base64-encoded) of tombstoned `doc_id`s in this segment.
    pub tombstones_b64: String,
    /// Cached tombstone cardinality for fast GC ratio checks.
    pub tombstone_count: u64,
}

impl SegmentDescriptor {
    /// Creates a new [`SegmentDescriptor`] with an optional initial tombstone bitmap.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        segment_id: u64,
        osi_key: String,
        osv_key: String,
        orig_dim: u16,
        padded_dim: u16,
        fwht_seed: u64,
        num_vectors: u64,
        tombstones: &RoaringBitmap,
    ) -> ProResult<Self> {
        let mut desc = Self {
            segment_id,
            osi_key,
            osv_key,
            orig_dim,
            padded_dim,
            fwht_seed,
            num_vectors,
            tombstones_b64: String::new(),
            tombstone_count: 0,
        };
        desc.set_tombstones(tombstones)?;
        Ok(desc)
    }

    /// Decodes the segment's [`RoaringBitmap`] of tombstones.
    pub fn decode_tombstones(&self) -> ProResult<RoaringBitmap> {
        decode_roaring_b64(&self.tombstones_b64)
    }

    /// Encodes and updates the segment's [`RoaringBitmap`] of tombstones.
    pub fn set_tombstones(&mut self, bitmap: &RoaringBitmap) -> ProResult<()> {
        self.tombstones_b64 = encode_roaring_b64(bitmap)?;
        self.tombstone_count = bitmap.len();
        Ok(())
    }

    /// Returns the fraction of vectors in this segment marked deleted (`[0.0, 1.0]`).
    #[must_use]
    pub fn tombstone_ratio(&self) -> f64 {
        if self.num_vectors == 0 {
            return 0.0;
        }
        #[allow(clippy::cast_precision_loss)]
        let deleted = self.tombstone_count.min(self.num_vectors) as f64;
        #[allow(clippy::cast_precision_loss)]
        let total = self.num_vectors as f64;
        deleted / total
    }
}

/// Atomic cluster state stored in S3 `manifest.json` without ZooKeeper or etcd.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClusterManifest {
    /// Manifest schema version (`1`).
    pub schema_version: u32,
    /// Monotonically incremented generation counter updated on every CAS commit.
    pub epoch: u64,
    /// Unix timestamp UTC when this manifest epoch was committed.
    pub updated_at_unix: u64,
    /// Next unique segment ID to allocate.
    pub next_segment_id: u64,
    /// Active immutable segments visible to stateless reader nodes.
    pub segments: Vec<SegmentDescriptor>,
    /// Global `RoaringBitmap` (Base64-encoded) of logically deleted `doc_id`s across segments.
    pub global_tombstones_b64: String,
}

impl Default for ClusterManifest {
    fn default() -> Self {
        Self::new_empty()
    }
}

impl ClusterManifest {
    /// Creates an empty initial [`ClusterManifest`] at epoch `1`.
    #[must_use]
    pub fn new_empty() -> Self {
        Self {
            schema_version: MANIFEST_SCHEMA_VERSION_V1,
            epoch: 1,
            updated_at_unix: current_unix_secs(),
            next_segment_id: 1,
            segments: Vec::new(),
            global_tombstones_b64: String::new(),
        }
    }

    /// Allocates and returns a fresh monotonic segment identifier.
    pub fn allocate_segment_id(&mut self) -> u64 {
        let id = self.next_segment_id;
        self.next_segment_id = self.next_segment_id.saturating_add(1);
        id
    }

    /// Decodes the cluster-wide global tombstone [`RoaringBitmap`].
    pub fn decode_global_tombstones(&self) -> ProResult<RoaringBitmap> {
        decode_roaring_b64(&self.global_tombstones_b64)
    }

    /// Updates the cluster-wide global tombstone [`RoaringBitmap`].
    pub fn set_global_tombstones(&mut self, bitmap: &RoaringBitmap) -> ProResult<()> {
        self.global_tombstones_b64 = encode_roaring_b64(bitmap)?;
        Ok(())
    }

    /// Returns the union of segment-local and global tombstones for `seg`.
    pub fn combined_tombstones_for(&self, seg: &SegmentDescriptor) -> ProResult<RoaringBitmap> {
        let mut combined = seg.decode_tombstones()?;
        let global = self.decode_global_tombstones()?;
        combined |= &global;
        Ok(combined)
    }
}

/// Loaded manifest paired with its S3 `ETag` / `version` for CAS updates.
#[derive(Clone, Debug)]
pub struct ManifestSnapshot {
    /// Parsed cluster manifest.
    pub manifest: ClusterManifest,
    /// S3 object version / `ETag` used for `If-Match` conditional writes.
    pub version: UpdateVersion,
}

/// Conditional CAS coordinator for `manifest.json` on `object_store::ObjectStore`.
#[derive(Clone, Debug)]
pub struct ManifestStore {
    store: Arc<dyn ObjectStore>,
    path: ObjectPath,
}

impl ManifestStore {
    /// Creates a new [`ManifestStore`] targeting `path` inside `store`.
    #[must_use]
    pub fn new(store: Arc<dyn ObjectStore>, path: impl Into<ObjectPath>) -> Self {
        Self {
            store,
            path: path.into(),
        }
    }

    /// Returns the backing `ObjectStore`.
    #[must_use]
    pub fn store(&self) -> &Arc<dyn ObjectStore> {
        &self.store
    }

    /// Returns the manifest object path.
    #[must_use]
    pub fn path(&self) -> &ObjectPath {
        &self.path
    }

    /// Fetches the current `manifest.json` and its `ETag`/`version`, returning `None` if absent.
    pub async fn load(&self) -> ProResult<Option<ManifestSnapshot>> {
        let res = match self.store.get(&self.path).await {
            Ok(res) => res,
            Err(object_store::Error::NotFound { .. }) => return Ok(None),
            Err(err) => return Err(err.into()),
        };

        let version = UpdateVersion {
            e_tag: res.meta.e_tag.clone(),
            version: res.meta.version.clone(),
        };
        let payload = res.bytes().await?;
        let manifest: ClusterManifest = serde_json::from_slice(&payload)?;
        Ok(Some(ManifestSnapshot { manifest, version }))
    }

    /// Atomically creates `manifest.json` using `If-None-Match: *` (`PutMode::Create`).
    pub async fn create_if_absent(
        &self,
        manifest: &ClusterManifest,
    ) -> ProResult<ManifestSnapshot> {
        let json_bytes = Bytes::from(serde_json::to_vec_pretty(manifest)?);
        let res = self
            .store
            .put_opts(
                &self.path,
                PutPayload::from_bytes(json_bytes),
                PutMode::Create.into(),
            )
            .await
            .map_err(|err| match err {
                object_store::Error::AlreadyExists { .. } => {
                    ProEngineError::ManifestAlreadyExists {
                        path: self.path.to_string(),
                    }
                }
                other => ProEngineError::ObjectStore(other),
            })?;

        Ok(ManifestSnapshot {
            manifest: manifest.clone(),
            version: res.into(),
        })
    }

    /// Atomically replaces `manifest.json` only if its `ETag`/`version` matches `expected`
    /// (`PutMode::Update` -> S3 `If-Match: <ETag>`).
    pub async fn compare_and_swap(
        &self,
        expected: &UpdateVersion,
        next: &ClusterManifest,
    ) -> ProResult<ManifestSnapshot> {
        let json_bytes = Bytes::from(serde_json::to_vec_pretty(next)?);
        let res = self
            .store
            .put_opts(
                &self.path,
                PutPayload::from_bytes(json_bytes),
                PutMode::Update(expected.clone()).into(),
            )
            .await
            .map_err(|err| match err {
                object_store::Error::Precondition { .. } => {
                    ProEngineError::CasConflict { attempts: 1 }
                }
                other => ProEngineError::ObjectStore(other),
            })?;

        Ok(ManifestSnapshot {
            manifest: next.clone(),
            version: res.into(),
        })
    }

    /// Performs a read-modify-write CAS loop with up to `max_retries` attempts.
    ///
    /// Automatically initializes the manifest with `PutMode::Create` if absent.
    pub async fn update_with_cas_retry<F>(
        &self,
        max_retries: usize,
        mut mutator: F,
    ) -> ProResult<ManifestSnapshot>
    where
        F: FnMut(&mut ClusterManifest) -> ProResult<()>,
    {
        let attempts = max_retries.max(1);
        for _ in 0..attempts {
            match self.load().await? {
                None => {
                    let mut initial = ClusterManifest::new_empty();
                    mutator(&mut initial)?;
                    match self.create_if_absent(&initial).await {
                        Ok(snap) => return Ok(snap),
                        Err(ProEngineError::ManifestAlreadyExists { .. }) => continue,
                        Err(err) => return Err(err),
                    }
                }
                Some(snapshot) => {
                    let mut next = snapshot.manifest;
                    next.epoch = next.epoch.saturating_add(1);
                    next.updated_at_unix = current_unix_secs();
                    mutator(&mut next)?;
                    match self.compare_and_swap(&snapshot.version, &next).await {
                        Ok(snap) => return Ok(snap),
                        Err(ProEngineError::CasConflict { .. }) => continue,
                        Err(err) => return Err(err),
                    }
                }
            }
        }
        Err(ProEngineError::CasConflict { attempts })
    }

    /// Lightweight `HeadObject` poll for stateless reader nodes: reloads `manifest.json`
    /// only when its `ETag` differs from `known_etag`.
    pub async fn poll_if_changed(
        &self,
        known_etag: Option<&str>,
    ) -> ProResult<Option<ManifestSnapshot>> {
        let meta = match self.store.head(&self.path).await {
            Ok(meta) => meta,
            Err(object_store::Error::NotFound { .. }) => return Ok(None),
            Err(err) => return Err(err.into()),
        };

        if let (Some(known), Some(current)) = (known_etag, meta.e_tag.as_deref()) {
            if known == current {
                return Ok(None);
            }
        }
        self.load().await
    }
}

fn encode_roaring_b64(bitmap: &RoaringBitmap) -> ProResult<String> {
    if bitmap.is_empty() {
        return Ok(String::new());
    }
    let mut bytes = Vec::with_capacity(bitmap.serialized_size());
    bitmap.serialize_into(&mut bytes)?;
    Ok(STANDARD.encode(bytes))
}

fn decode_roaring_b64(b64: &str) -> ProResult<RoaringBitmap> {
    if b64.is_empty() {
        return Ok(RoaringBitmap::new());
    }
    let raw = STANDARD.decode(b64)?;
    let bitmap = RoaringBitmap::deserialize_from(&raw[..])?;
    Ok(bitmap)
}

fn current_unix_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use object_store::memory::InMemory;

    use super::*;

    #[tokio::test]
    async fn cas_create_update_conflict_and_poll() {
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let manifest_store = ManifestStore::new(Arc::clone(&store), DEFAULT_MANIFEST_KEY);

        assert!(manifest_store.load().await.unwrap().is_none());

        let m1 = ClusterManifest::new_empty();
        let snap1 = manifest_store.create_if_absent(&m1).await.unwrap();
        assert_eq!(snap1.manifest.epoch, 1);

        // Second create_if_absent must fail with ManifestAlreadyExists
        assert!(matches!(
            manifest_store.create_if_absent(&m1).await,
            Err(ProEngineError::ManifestAlreadyExists { .. })
        ));

        // Reader node polls with same ETag -> None (no download needed)
        let unchanged = manifest_store
            .poll_if_changed(snap1.version.e_tag.as_deref())
            .await
            .unwrap();
        assert!(unchanged.is_none());

        // Node A updates via CAS
        let mut next_a = snap1.manifest.clone();
        next_a.epoch = 2;
        let seg_id = next_a.allocate_segment_id();
        next_a.segments.push(
            SegmentDescriptor::new(
                seg_id,
                "segments/seg-1.osi".to_owned(),
                "segments/seg-1.osv".to_owned(),
                64,
                64,
                42,
                1000,
                &RoaringBitmap::new(),
            )
            .unwrap(),
        );
        let snap2 = manifest_store
            .compare_and_swap(&snap1.version, &next_a)
            .await
            .unwrap();
        assert_eq!(snap2.manifest.epoch, 2);

        // Stale Node B trying to CAS with snap1.version gets CasConflict
        let mut stale_b = snap1.manifest.clone();
        stale_b.epoch = 99;
        assert!(matches!(
            manifest_store
                .compare_and_swap(&snap1.version, &stale_b)
                .await,
            Err(ProEngineError::CasConflict { .. })
        ));

        // Stateless reader polling with old ETag sees the new manifest
        let updated = manifest_store
            .poll_if_changed(snap1.version.e_tag.as_deref())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(updated.manifest.epoch, 2);
        assert_eq!(updated.manifest.segments.len(), 1);
    }

    #[tokio::test]
    async fn concurrent_cas_retry_preserves_all_updates() {
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let manifest_store = ManifestStore::new(Arc::clone(&store), DEFAULT_MANIFEST_KEY);

        manifest_store
            .create_if_absent(&ClusterManifest::new_empty())
            .await
            .unwrap();

        let mut handles = Vec::new();
        for worker_id in 0..8_u64 {
            let ms = manifest_store.clone();
            handles.push(tokio::spawn(async move {
                ms.update_with_cas_retry(32, |m| {
                    let id = m.allocate_segment_id();
                    m.segments.push(SegmentDescriptor::new(
                        id,
                        format!("segments/seg-{id}.osi"),
                        format!("segments/seg-{id}.osv"),
                        64,
                        64,
                        worker_id + 1,
                        100,
                        &RoaringBitmap::new(),
                    )?);
                    Ok(())
                })
                .await
            }));
        }

        for h in handles {
            h.await.unwrap().unwrap();
        }

        let final_snap = manifest_store.load().await.unwrap().unwrap();
        assert_eq!(final_snap.manifest.segments.len(), 8);
        assert_eq!(final_snap.manifest.epoch, 9);
    }
}
