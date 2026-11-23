# `01_workspace_and_formats.md` — Especificação Técnica da Fase 3 (Workspace, Contratos `osprey-core` e Formatos Binários)

> **Status:** Especificação Fechada para Implementação (Fase 3)
> **Nota de Contexto:** Este documento integra as especificações originais de engenharia do OspreyDB; a referência canônica atualizada dos structs e offsets (ex.: `MinipageDirectoryEntry` de 48 bytes) reside em `crates/osprey-core/src/format/` e [`ARCHITECTURE.md`](../ARCHITECTURE.md).
> **Escopo:** Estrutura raiz `Cargo.toml` e contratos exatos de tipos, erros, alocadores alinhados, cabeçalhos binários `#[repr(C, align(64))]`, traits e testes em `crates/osprey-core`.
> **Referências:** [`ARCHITECTURE.md`](../ARCHITECTURE.md), [`AGENTS.md`](../AGENTS.md), [`specs/00_math_invariants.md`](./00_math_invariants.md).

---

## 1. Configuração Exata do Workspace (`Cargo.toml`)

O workspace raiz organiza os cinco crates Apache-2.0 (`ospreydb/ospreydb`) com dependências estritamente acíclicas e perfil de release otimizado para código vetorizado e *link-time optimization* global:

```toml
[workspace]
resolver = "2"
members = [
    "crates/osprey-core",
    "crates/osprey-simd",
    "crates/osprey-storage",
    "crates/osprey-cli",
    "crates/ospreydb",
]
exclude = ["fuzz"]

[workspace.package]
version = "0.1.1"
edition = "2021"
license = "Apache-2.0"
repository = "https://github.com/ospreydb/ospreydb"

[workspace.dependencies]
osprey-core = { path = "crates/osprey-core" }
osprey-simd = { path = "crates/osprey-simd" }
osprey-storage = { path = "crates/osprey-storage" }

# Núcleo & Zero-Copy
bytemuck = { version = "1.19.0", features = ["derive"] }
half = { version = "2.4.1", features = ["bytemuck"] }
xxhash-rust = { version = "0.8.12", features = ["xxh64"] }
thiserror = "2.0.11"

# Storage, Compressão & Índices
roaring = "0.10.12"
memmap2 = "0.9.5"
zstd = "0.13.2"
object_store = { version = "0.11.2", features = ["aws"] }
tokio = { version = "1.43.0", features = ["rt-multi-thread", "macros", "sync", "time"] }
arrow = "54.0.0"
parquet = "54.0.0"

# CLI & Serialização
clap = { version = "4.5.27", features = ["derive"] }
serde = { version = "1.0.217", features = ["derive"] }
serde_json = "1.0.138"
tracing = "0.1.41"

[profile.release]
opt-level = 3
lto = "fat"
codegen-units = 1
panic = "abort"
strip = "symbols"

[profile.bench]
inherits = "release"
debug = true
strip = "none"

[workspace.lints.rust]
unsafe_op_in_unsafe_fn = "deny"
missing_docs = "warn"

[workspace.lints.clippy]
unwrap_used = "deny"
expect_used = "deny"
panic = "deny"
undocumented_unsafe_blocks = "deny"
```

Em todo crate do workspace (`lib.rs` / `main.rs`), os seguintes atributos de compilação são obrigatórios:

```rust
#![deny(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    unsafe_op_in_unsafe_fn
)]
```

---

## 2. Hierarquia de Erros (`osprey_core::error::OspreyError`)

Nenhum caminho de produção pode usar `.unwrap()`, `.expect()` ou `panic!()`. Toda falha estrutural, numérica, de alinhamento ou de I/O é propagada por `OspreyError`:

```rust
use thiserror::Error;

/// Alias padrão de resultado para todo o workspace OspreyDB.
pub type OspreyResult<T> = Result<T, OspreyError>;

/// Hierarquia canônica de erros do núcleo OspreyDB.
#[derive(Debug, Error)]
pub enum OspreyError {
    /// Bytes mágicos divergentes de `"OSPR"` (`[0x4F, 0x53, 0x50, 0x52]`).
    #[error("invalid magic bytes: expected {expected:?}, found {found:?}")]
    InvalidMagic { expected: [u8; 4], found: [u8; 4] },

    /// O discriminante `format_kind` não corresponde ao parser invocado (`0x01`, `0x02`, `0x03`).
    #[error("unsupported format kind: expected 0x{expected:02x}, found 0x{found:02x}")]
    UnsupportedFormatKind { expected: u8, found: u8 },

    /// Versão do layout físico não suportada por este binário.
    #[error("unsupported format version {version} for format_kind 0x{format_kind:02x}")]
    UnsupportedVersion { format_kind: u8, version: u16 },

    /// Falha de integridade `xxHash64` em cabeçalho, minipágina `.osv` ou rodapé global.
    #[error("xxHash64 checksum mismatch in {context}: expected 0x{expected:016x}, actual 0x{actual:016x}")]
    ChecksumMismatch {
        context: &'static str,
        expected: u64,
        actual: u64,
    },

    /// Ponteiro em memória não respeita o alinhamento exigido de 64 bytes (AVX-512 / mmap).
    #[error("unaligned pointer in {context}: address 0x{addr:x} is not {required_align}-byte aligned")]
    UnalignedPointer {
        context: &'static str,
        addr: usize,
        required_align: usize,
    },

    /// Offset tabular em arquivo não é múltiplo de 64 bytes (`offset % 64 != 0`).
    #[error("unaligned section offset for {section}: {offset} is not divisible by {required_align}")]
    UnalignedSectionOffset {
        section: &'static str,
        offset: u64,
        required_align: u64,
    },

    /// Dimensionalidade inválida (`orig_dim == 0`, `padded_dim` não potência de 2, etc.).
    #[error("invalid dimension (orig_dim={orig_dim}, padded_dim={padded_dim}): {reason}")]
    InvalidDimension {
        orig_dim: u16,
        padded_dim: u16,
        reason: &'static str,
    },

    /// Vetor de entrada contém `NaN` ou `±Inf` na fronteira de ingestão/consulta.
    #[error("non-finite embedding value at index {index} (raw f32 bits: 0x{value_bits:08x})")]
    NonFiniteEmbedding { index: usize, value_bits: u32 },

    /// Buffer menor que o mínimo estrutural exigido (ex.: < 64 bytes para cabeçalho).
    #[error("truncated buffer in {context}: expected at least {expected_min} bytes, got {actual}")]
    TruncatedBuffer {
        context: &'static str,
        expected_min: usize,
        actual: usize,
    },

    /// Offset + comprimento apontam para além do fim do arquivo mapeado.
    #[error("out-of-bounds offset in {context}: [{offset}..+{len}] exceeds file_len {file_len}")]
    OutOfBoundsOffset {
        context: &'static str,
        offset: u64,
        len: u64,
        file_len: u64,
    },

    /// Falha ou overflow ao alocar `AlignedBuffer<T>` de 64 bytes.
    #[error("aligned allocation failed for {bytes} bytes (align={align})")]
    AllocationFailed { bytes: usize, align: usize },

    /// Erro em codec interno (`zstd`, `roaring`, parsing JSON de payload).
    #[error("codec error ({codec}): {detail}")]
    CodecError { codec: &'static str, detail: String },

    /// Erro de sistema operacional / `memmap2`.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Erro em chamadas `object_store` / S3 (`get_ranges`, `head`, etc.).
    #[error("object store error during {operation}: {detail}")]
    ObjectStore { operation: &'static str, detail: String },
}
```

---

## 3. Memória Alinhada a 64 Bytes (`AlignedBuffer<T>`) e Casts Zero-Copy

Para compatibilidade estrita com cargas AVX-512 (`vmovdqa64`), cache-line alignment e `memmap2` sem alocação ou cópias implícitas, `osprey-core::alloc` provê `AlignedBuffer<T>` respaldado por `std::alloc::Layout::from_size_align(bytes, 64)`:

```rust
use std::alloc::{self, Layout};
use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;

use bytemuck::Pod;

use crate::error::{OspreyError, OspreyResult};

/// Alinhamento obrigatório de linha de cache (64 bytes = ZMM / cache line x86_64 & ARM64).
pub const CACHE_LINE_ALIGN: usize = 64;

/// Buffer contíguo garantidamente alinhado a 64 bytes via `std::alloc::Layout`.
pub struct AlignedBuffer<T: Pod> {
    ptr: NonNull<T>,
    len: usize,
    cap: usize,
}

// SAFETY: `AlignedBuffer<T>` possui propriedade exclusiva dos elementos `T: Pod`.
unsafe impl<T: Pod + Send> Send for AlignedBuffer<T> {}
// SAFETY: Acesso compartilhado expõe apenas `&[T]`, seguro sempre que `T: Sync`.
unsafe impl<T: Pod + Sync> Sync for AlignedBuffer<T> {}

impl<T: Pod> AlignedBuffer<T> {
    /// Aloca um buffer zerado com `len` elementos alinhados a 64 bytes.
    pub fn zeroed(len: usize) -> OspreyResult<Self> {
        const {
            assert!(
                align_of::<T>() <= CACHE_LINE_ALIGN,
                "AlignedBuffer<T> supports element alignments <= 64 bytes"
            );
        }

        if len == 0 || size_of::<T>() == 0 {
            return Ok(Self {
                ptr: NonNull::dangling(),
                len,
                cap: 0,
            });
        }

        let bytes = len
            .checked_mul(size_of::<T>())
            .ok_or(OspreyError::AllocationFailed {
                bytes: usize::MAX,
                align: CACHE_LINE_ALIGN,
            })?;

        let layout = Layout::from_size_align(bytes, CACHE_LINE_ALIGN).map_err(|_| {
            OspreyError::AllocationFailed {
                bytes,
                align: CACHE_LINE_ALIGN,
            }
        })?;

        // SAFETY: `bytes > 0` e `CACHE_LINE_ALIGN == 64` (potência de 2 >= align_of::<T>()).
        let raw = unsafe { alloc::alloc_zeroed(layout) };
        let ptr = NonNull::new(raw.cast::<T>()).ok_or(OspreyError::AllocationFailed {
            bytes,
            align: CACHE_LINE_ALIGN,
        })?;

        Ok(Self {
            ptr,
            len,
            cap: len,
        })
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline]
    pub fn as_slice(&self) -> &[T] {
        if self.len == 0 {
            return &[];
        }
        // SAFETY: `ptr` válido para `len` elementos `T: Pod` inicializados e alinhados a 64B.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        if self.len == 0 {
            return &mut [];
        }
        // SAFETY: propriedade exclusiva `&mut self` sobre `len` elementos `T: Pod`.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

impl<T: Pod> Deref for AlignedBuffer<T> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<T: Pod> DerefMut for AlignedBuffer<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

impl<T: Pod> Drop for AlignedBuffer<T> {
    fn drop(&mut self) {
        if self.cap == 0 || size_of::<T>() == 0 {
            return;
        }
        let bytes = self.cap * size_of::<T>();
        if let Ok(layout) = Layout::from_size_align(bytes, CACHE_LINE_ALIGN) {
            // SAFETY: `self.ptr` foi alocado com o exato `layout` em `AlignedBuffer::zeroed`.
            unsafe {
                alloc::dealloc(self.ptr.as_ptr().cast::<u8>(), layout);
            }
        }
    }
}

/// Valida alinhamento a 64 bytes, tamanho mínimo e realiza cast zero-copy para cabeçalho `H`.
#[inline]
pub fn cast_aligned_header<H: Pod>(bytes: &[u8], context: &'static str) -> OspreyResult<&H> {
    if bytes.len() < size_of::<H>() {
        return Err(OspreyError::TruncatedBuffer {
            context,
            expected_min: size_of::<H>(),
            actual: bytes.len(),
        });
    }
    let addr = bytes.as_ptr() as usize;
    if !addr.is_multiple_of(CACHE_LINE_ALIGN) {
        return Err(OspreyError::UnalignedPointer {
            context,
            addr,
            required_align: CACHE_LINE_ALIGN,
        });
    }
    Ok(bytemuck::from_bytes(&bytes[..size_of::<H>()]))
}
```

---

## 4. Definições `#[repr(C, align(64))]` Byte-Exact (`osprey_core::format`)

Todas as structs mapeadas em arquivo obedecem ao layout Little-Endian de [`ARCHITECTURE.md`](../ARCHITECTURE.md), implementam `bytemuck::Pod + bytemuck::Zeroable`, proíbem padding implícito e são protegidas por asserções estáticas em tempo de compilação.

```rust
use std::mem::{align_of, offset_of, size_of};

use bytemuck::{Pod, Zeroable};

pub const OSPR_MAGIC: [u8; 4] = *b"OSPR";
pub const FORMAT_KIND_OSI: u8 = 0x01;
pub const FORMAT_KIND_OSV: u8 = 0x02;
pub const FORMAT_KIND_OSR: u8 = 0x03;

/// Cabeçalho do índice residente 1-bit `.osi` (64 bytes, `0x00..0x3F`).
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
pub struct OsiHeader {
    pub magic: [u8; 4],         // 0x00..0x03
    pub format_kind: u8,        // 0x04 (= 0x01)
    pub flags: u8,              // 0x05
    pub version: u16,           // 0x06..0x07
    pub orig_dim: u16,          // 0x08..0x09
    pub padded_dim: u16,        // 0x0A..0x0B
    pub num_clusters: u32,      // 0x0C..0x0F
    pub num_vectors: u64,       // 0x10..0x17
    pub fwht_seed: u64,         // 0x18..0x1F
    pub centroids_offset: u64,  // 0x20..0x27 (% 64 == 0)
    pub entries_offset: u64,    // 0x28..0x2F (% 64 == 0)
    pub tombstones_offset: u64, // 0x30..0x37 (% 64 == 0)
    pub header_xxh64: u64,      // 0x38..0x3F (xxh64 over [0x00..0x38], seed 0)
}

/// Descritor de alcance de cluster no diretório `.osi` Cluster-SoA (16 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
pub struct OsiClusterSpan {
    pub cluster_base_offset: u64, // 0x00..0x07
    pub vector_count: u32,        // 0x08..0x0B
    pub flags: u32,               // 0x0C..0x0F
}

/// Locator de 16 bytes por vetor indexado no `.osi` (Stream 3 do Cluster-SoA).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
pub struct RabitqEntryHeader {
    pub doc_id: u64,      // 0x00..0x07
    pub minipage_id: u32, // 0x08..0x0B
    pub slot_idx: u16,    // 0x0C..0x0D
    pub entry_flags: u16, // 0x0E..0x0F
}

/// Fatores `RaBitQ` em bits crus IEEE-754 `f16` (Stream 2 do Cluster-SoA, 4 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
pub struct RabitqFactors {
    pub res_norm_bits: u16,       // 0x00..0x01 (half::f16::to_bits)
    pub l1_norm_factor_bits: u16, // 0x02..0x03 (half::f16::to_bits)
}

/// Cabeçalho do arquivo de slabs `.osv` (64 bytes, `0x00..0x3F`).
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
pub struct OsvHeader {
    pub magic: [u8; 4],        // 0x00..0x03
    pub format_kind: u8,       // 0x04 (= 0x02)
    pub compression_flags: u8, // 0x05
    pub version: u16,          // 0x06..0x07
    pub num_minipages: u32,    // 0x08..0x0B
    pub orig_dim: u16,         // 0x0C..0x0D
    pub reserved_flags: u16,   // 0x0E..0x0F
    pub zstd_dict_offset: u64, // 0x10..0x17
    pub zstd_dict_len: u32,    // 0x18..0x1B
    pub zstd_dict_id: u32,     // 0x1C..0x1F
    pub directory_offset: u64, // 0x20..0x27 (% 64 == 0)
    pub num_vectors: u64,      // 0x28..0x2F
    pub reserved: [u8; 8],     // 0x30..0x37
    pub header_xxh64: u64,     // 0x38..0x3F
}

/// Entrada do diretório de minipáginas no `.osv` (32 bytes, `0x00..0x1F`).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
pub struct MinipageDirectoryEntry {
    pub byte_offset: u64,       // 0x00..0x07
    pub compressed_size: u32,   // 0x08..0x0B
    pub uncompressed_size: u32, // 0x0C..0x0F
    pub cluster_id: u32,        // 0x10..0x13
    pub vector_count: u16,      // 0x14..0x15
    pub flags: u16,             // 0x16..0x17
    pub page_xxh64: u64,        // 0x18..0x1F
}

/// Cabeçalho do store de reranking `.osr` (64 bytes, `0x00..0x3F`).
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct OsrHeader {
    pub magic: [u8; 4],        // 0x00..0x03
    pub format_kind: u8,       // 0x04 (= 0x03)
    pub subspace_dim: u8,      // 0x05 (= 2)
    pub num_layers: u8,        // 0x06 (= 2)
    pub flags: u8,             // 0x07
    pub token_dim: u16,        // 0x08..0x09 (= 128)
    pub version: u16,          // 0x0A..0x0B (= 1)
    pub gamma: f32,            // 0x0C..0x0F (= 0.28)
    pub fwht_seed: u64,        // 0x10..0x17
    pub num_docs: u64,         // 0x18..0x1F
    pub total_tokens: u64,     // 0x20..0x27
    pub doc_table_offset: u64, // 0x28..0x2F (% 64 == 0)
    pub tensors_offset: u64,   // 0x30..0x37 (% 64 == 0)
    pub header_xxh64: u64,     // 0x38..0x3F
}

/// Mapeamento de documento para faixa contígua de tokens de 64B no `.osr` (24 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
pub struct DocTokenRange {
    pub doc_id: u64,          // 0x00..0x07
    pub token_start_idx: u64, // 0x08..0x0F
    pub token_count: u16,     // 0x10..0x11
    pub reserved: [u8; 6],    // 0x12..0x17
}

// Asserções estáticas obrigatórias de ABI e offsets byte-a-byte
const _: () = {
    assert!(size_of::<OsiHeader>() == 64);
    assert!(align_of::<OsiHeader>() == 64);
    assert!(offset_of!(OsiHeader, format_kind) == 0x04);
    assert!(offset_of!(OsiHeader, version) == 0x06);
    assert!(offset_of!(OsiHeader, orig_dim) == 0x08);
    assert!(offset_of!(OsiHeader, padded_dim) == 0x0A);
    assert!(offset_of!(OsiHeader, num_clusters) == 0x0C);
    assert!(offset_of!(OsiHeader, num_vectors) == 0x10);
    assert!(offset_of!(OsiHeader, fwht_seed) == 0x18);
    assert!(offset_of!(OsiHeader, centroids_offset) == 0x20);
    assert!(offset_of!(OsiHeader, entries_offset) == 0x28);
    assert!(offset_of!(OsiHeader, tombstones_offset) == 0x30);
    assert!(offset_of!(OsiHeader, header_xxh64) == 0x38);

    assert!(size_of::<OsiClusterSpan>() == 16);
    assert!(size_of::<RabitqEntryHeader>() == 16);
    assert!(offset_of!(RabitqEntryHeader, minipage_id) == 0x08);
    assert!(offset_of!(RabitqEntryHeader, slot_idx) == 0x0C);
    assert!(offset_of!(RabitqEntryHeader, entry_flags) == 0x0E);
    assert!(size_of::<RabitqFactors>() == 4);

    assert!(size_of::<OsvHeader>() == 64);
    assert!(align_of::<OsvHeader>() == 64);
    assert!(offset_of!(OsvHeader, format_kind) == 0x04);
    assert!(offset_of!(OsvHeader, compression_flags) == 0x05);
    assert!(offset_of!(OsvHeader, num_minipages) == 0x08);
    assert!(offset_of!(OsvHeader, zstd_dict_offset) == 0x10);
    assert!(offset_of!(OsvHeader, zstd_dict_len) == 0x18);
    assert!(offset_of!(OsvHeader, directory_offset) == 0x20);
    assert!(offset_of!(OsvHeader, header_xxh64) == 0x38);

    assert!(size_of::<MinipageDirectoryEntry>() == 32);
    assert!(offset_of!(MinipageDirectoryEntry, byte_offset) == 0x00);
    assert!(offset_of!(MinipageDirectoryEntry, compressed_size) == 0x08);
    assert!(offset_of!(MinipageDirectoryEntry, uncompressed_size) == 0x0C);
    assert!(offset_of!(MinipageDirectoryEntry, cluster_id) == 0x10);
    assert!(offset_of!(MinipageDirectoryEntry, vector_count) == 0x14);
    assert!(offset_of!(MinipageDirectoryEntry, flags) == 0x16);
    assert!(offset_of!(MinipageDirectoryEntry, page_xxh64) == 0x18);

    assert!(size_of::<OsrHeader>() == 64);
    assert!(align_of::<OsrHeader>() == 64);
    assert!(offset_of!(OsrHeader, format_kind) == 0x04);
    assert!(offset_of!(OsrHeader, subspace_dim) == 0x05);
    assert!(offset_of!(OsrHeader, num_layers) == 0x06);
    assert!(offset_of!(OsrHeader, token_dim) == 0x08);
    assert!(offset_of!(OsrHeader, gamma) == 0x0C);
    assert!(offset_of!(OsrHeader, fwht_seed) == 0x10);
    assert!(offset_of!(OsrHeader, num_docs) == 0x18);
    assert!(offset_of!(OsrHeader, total_tokens) == 0x20);
    assert!(offset_of!(OsrHeader, doc_table_offset) == 0x28);
    assert!(offset_of!(OsrHeader, tensors_offset) == 0x30);
    assert!(offset_of!(OsrHeader, header_xxh64) == 0x38);

    assert!(size_of::<DocTokenRange>() == 24);
    assert!(offset_of!(DocTokenRange, token_start_idx) == 0x08);
    assert!(offset_of!(DocTokenRange, token_count) == 0x10);
    assert!(offset_of!(DocTokenRange, reserved) == 0x12);
};
```

---

## 5. Traits de Extensão (`osprey_core::traits`)

As interfaces abaixo separam a leitura imutável de segmentos das extensões de escrita contínua e compactação (`crates/ospreydb`):

```rust
use crate::error::OspreyResult;
use crate::format::{DocTokenRange, OsiHeader, OsrHeader};

/// Candidato retornado pela varredura assimétrica `.osi` no Estágio 1.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Stage1Candidate {
    pub doc_id: u64,
    pub minipage_id: u32,
    pub slot_idx: u16,
    pub approx_score: f32,
}

/// Contrato de leitura zero-copy para o índice residente `.osi`.
pub trait VectorIndexReader: Send + Sync {
    /// Retorna o cabeçalho validado do índice `.osi`.
    fn header(&self) -> &OsiHeader;

    /// Seleciona os `nprobe` clusters mais próximos à query rotacionada `q_rot`.
    fn probe_clusters(&self, q_rot: &[f32], nprobe: usize) -> OspreyResult<Vec<u32>>;

    /// Varre os clusters sondados aplicando `RaBitQ` assimétrico e filtragem de tombstones.
    fn scan_clusters(
        &self,
        q_rot: &[f32],
        cluster_ids: &[u32],
        top_c: usize,
    ) -> OspreyResult<Vec<Stage1Candidate>>;

    /// Verifica se um `doc_id` foi logicamente deletado no `RoaringBitmap` de tombstones.
    fn is_deleted(&self, doc_id: u64) -> bool;
}

/// Contrato de leitura zero-copy para o store de tokens `.osr` (64 bytes/token).
pub trait RerankStoreReader: Send + Sync {
    /// Retorna o cabeçalho validado do store `.osr`.
    fn header(&self) -> &OsrHeader;

    /// Localiza por busca binária a faixa de tokens de um `doc_id`.
    fn lookup_doc(&self, doc_id: u64) -> OspreyResult<Option<DocTokenRange>>;

    /// Retorna um slice contíguo de bytes alinhado a 64 bytes (`token_count * 64` bytes).
    fn token_bytes(&self, range: &DocTokenRange) -> OspreyResult<&[u8]>;
}

/// Registro bruto recebido para escrita contínua no servidor `osprey` (`crates/ospreydb`).
#[derive(Clone, Copy, Debug)]
pub struct RawVectorRecord<'a> {
    pub doc_id: u64,
    pub vector: &'a [f32],
    pub payload_json: &'a [u8],
}

/// Relatório de compactação assíncrona emitido pelo `Copy-on-Write Slab Compactor`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompactionReport {
    pub merged_segments: usize,
    pub reclaimed_tombstones: u64,
    pub new_segment_id: u64,
}

/// Hook de extensão para escrita contínua (WAL + LSM Delta Buffer + Compactor) do `osprey` (`crates/ospreydb`).
/// Corresponde à extensão `StorageEngineExt` descrita na `ARCHITECTURE.md`.
pub trait DeltaBufferExt: Send + Sync {
    type Error: std::error::Error + Send + Sync + 'static;

    /// Anexa lote mutável ao WAL com `fsync` e quantiza contra os centróides `.osi` ativos.
    fn append_batch(&self, records: &[RawVectorRecord<'_>]) -> Result<u64, Self::Error>;

    /// Consulta os delta-buffers em memória ainda não materializados em slabs `.osv`.
    fn search_deltas(
        &self,
        q_rot: &[f32],
        top_k: usize,
    ) -> Result<Vec<Stage1Candidate>, Self::Error>;

    /// Dispara flush CoW e coleta de tombstones para novos segmentos imutáveis no S3.
    fn compact(&self, segment_ids: &[u64]) -> Result<CompactionReport, Self::Error>;
}

/// Alias de compatibilidade direta com os nomes da `ARCHITECTURE.md` (§1.3).
pub use DeltaBufferExt as StorageEngineExt;
```

---

## 6. Plano de Testes Obrigatórios da Fase 3

Antes de iniciar a Fase 4 (`FWHT` e primitivas algébricas), **100%** dos seguintes testes unitários e verificações de memória sob Miri devem passar sem avisos:

### 6.1 Testes Unitários (`cargo test -p osprey-core --release`)

1. **`abi_layout_and_offsets_match_spec`:** Verifica em runtime `size_of`, `align_of` e preenchimento zero de `OsiHeader`, `OsiClusterSpan`, `RabitqEntryHeader`, `RabitqFactors`, `OsvHeader`, `MinipageDirectoryEntry`, `OsrHeader` e `DocTokenRange` após round-trip via `bytemuck::bytes_of` / `cast_aligned_header`.
2. **`header_checksum_validation_and_corruption_rejection`:** Calcula `header_xxh64` (`xxh64(&bytes[..56], 0)`), valida sucesso no parse e confirma rejeição explícita (`ChecksumMismatch`) ao alterar qualquer único bit nos bytes `0x00..0x37` ou `0x38..0x3F`.
3. **`header_magic_kind_version_rejection`:** Testa rejeição determinística com `InvalidMagic`, `UnsupportedFormatKind` e `UnsupportedVersion` sem alocação de buffers derivados.
4. **`aligned_buffer_64_byte_invariants`:** Valida `(buf.as_ptr() as usize) % 64 == 0` para comprimentos `0`, `1`, `17`, `64`, `1024` e `65536`, conteúdo zerado inicial, e retorno `AllocationFailed` em overflow `usize::MAX`.
5. **`cast_aligned_header_rejects_unaligned_and_truncated`:** Valida `UnalignedPointer` para slice deslocado `+1` byte após início de buffer de 64 bytes e `TruncatedBuffer` para slice `< 64` bytes.
6. **`non_finite_f32_and_f16_clamping_rules`:** Valida retorno `NonFiniteEmbedding` para `f32::NAN`, `f32::INFINITY` e `f32::NEG_INFINITY`, além do clamping seguro para `[6.1035e-5, 65500.0]` em `res_norm` e `l1_norm_factor`.

### 6.2 Verificação de Memória e UB sob Miri (`cargo +nightly miri test -p osprey-core`)

1. **Sanidade do Alocador `AlignedBuffer<T>`:**
   - Criação, escrita, leitura via `Deref`/`DerefMut` e `Drop` de `AlignedBuffer<u64>`, `AlignedBuffer<u8>` e `AlignedBuffer<OsiHeader>` (incluindo `len == 0` e tipos de tamanho zero) com **zero vazamentos de memória** e **zero violações de Stacked Borrows / Tree Borrows**.
2. **Sanidade de Cast Zero-Copy `bytemuck::Pod`:**
   - Cast de `AlignedBuffer<u8>` preenchido para `&OsiHeader`, `&OsvHeader` e `&OsrHeader` via `cast_aligned_header`, verificando leitura sem bytes não-inicializados (*undef*) ou violações de alinhamento estrito.
3. **Critério de Saída da Fase 3:** `cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets --all-features -- -D warnings`, `cargo test --workspace --release` e `cargo +nightly miri test -p osprey-core` 100% verdes.
