use std::path::PathBuf;

fn enable_target_simd(build: &mut cc::Build) {
    let features = std::env::var("CARGO_CFG_TARGET_FEATURE").unwrap_or_default();
    let has = |feature: &str| features.split(',').any(|item| item == feature);
    if has("avx2") {
        build.flag_if_supported("-mavx2");
    }
    if has("fma") {
        build.flag_if_supported("-mfma");
    }
}

fn main() {
    // ── C kernel: cosine similarity (SIMD via GCC auto-vectorization) ───────
    // Compiled unconditionally — it is a tiny self-contained translation unit
    // with no external C++ deps.  The resulting object is linked into the crate
    // and the Rust FFI wrapper in hubness.rs calls `sm_cosine_similarity`.
    //
    // The usearch cxx bridge (when the `usearch-backend` feature is enabled)
    // is handled by the `usearch` crate's own build script — we do not need
    // cxx-build here.  This build.rs only compiles the C SIMD kernel.
    let kernel_dir = PathBuf::from("c-kernels");
    let mut build = cc::Build::new();
    build
        .file(kernel_dir.join("similarity.c"))
        .include(&kernel_dir)
        .flag_if_supported("-O3");
    enable_target_simd(&mut build);
    build.compile("sm_similarity");

    println!("cargo:rerun-if-changed=c-kernels/similarity.c");
    println!("cargo:rerun-if-changed=c-kernels/similarity.h");
}
