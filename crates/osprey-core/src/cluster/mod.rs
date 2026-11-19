//! Vector quantization and clustering algorithms (`1-Bit Binary K-Means`).

pub mod binary_kmeans;

pub use binary_kmeans::{
    binarize_rotated_vectors, fit_binary_kmeans, fit_binary_kmeans_from_rotated,
    fit_binary_kmeans_from_rotated_with_init, init_kmeans_plusplus_1bit,
    preprocess_and_binarize_batch, BinaryKMeansConfig, BinaryKMeansKernel, BinaryKMeansResult,
    ScalarClusterKernel,
};
