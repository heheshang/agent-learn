//! 构建脚本：帮 `faiss` crate 找到系统安装的 libfaiss_c。
//!
//! `faiss-sys` 默认走动态链接，但它的 build script 只输出
//! `cargo:rustc-link-lib=faiss_c` —— **不输出链接搜索路径**。
//! 所以只要 libfaiss_c 不在链接器的默认搜索路径里，就会报
//! `ld: library 'faiss_c' not found`。
//!
//! 这里按优先级探测：
//!
//! 1. `FAISS_LIB_DIR` 环境变量（用户显式指定，最高优先级）
//! 2. `pkg-config --libs-only-L faiss`（Linux 发行版的标准做法）
//! 3. Homebrew 前缀（macOS：`/opt/homebrew/lib`、`/usr/local/lib`）
//!
//! 注意：**不**重复输出 `cargo:rustc-link-lib=faiss_c` ——
//! `faiss-sys` 已经输出过一次，重复会让链接器刷 "duplicate libraries" 警告。

use std::path::Path;
use std::process::Command;

/// 常见的前缀目录候选。
const BREW_PREFIXES: &[&str] = &[
    "/opt/homebrew",
    "/usr/local",
    "/opt/homebrew/opt/faiss",
    "/usr/local/opt/faiss",
];

fn main() {
    println!("cargo:rerun-if-env-changed=FAISS_LIB_DIR");
    println!("cargo:rerun-if-env-changed=PKG_CONFIG_PATH");

    // 1) 显式指定优先。
    if let Some(dir) = std::env::var_os("FAISS_LIB_DIR") {
        let dir = Path::new(&dir);
        if dir.is_dir() {
            println!("cargo:rustc-link-search=native={}", dir.display());
            return;
        }
        println!(
            "cargo:warning=FAISS_LIB_DIR 指向的目录不存在: {}",
            dir.display()
        );
    }

    // 2) pkg-config（Linux 上 faiss 通常由发行版打包）。
    if let Some(dir) = pkg_config_lib_dir() {
        println!("cargo:rustc-link-search=native={dir}");
        return;
    }

    // 3) Homebrew 前缀（macOS）。
    for prefix in BREW_PREFIXES {
        let lib = Path::new(prefix).join("lib");
        if lib.join("libfaiss_c.dylib").exists() || lib.join("libfaiss_c.so").exists() {
            println!("cargo:rustc-link-search=native={}", lib.display());
            return;
        }
    }

    // 找不到也不直接失败：最终由链接器报错，信息更准确。
    println!(
        "cargo:warning=未能自动定位 libfaiss_c。若链接报 `library 'faiss_c' not found`，\
         请安装系统 FAISS（macOS: `brew install faiss`）或设置 FAISS_LIB_DIR 环境变量。"
    );
}

/// 尝试用 pkg-config 拿到 faiss 的库搜索路径。
fn pkg_config_lib_dir() -> Option<String> {
    let output = Command::new("pkg-config")
        .args(["--libs-only-L", "faiss"])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let text = String::from_utf8(output.stdout).ok()?;
    // 输出形如 `-L/opt/faiss/lib -L/usr/lib`
    text.split_whitespace()
        .find_map(|token| token.strip_prefix("-L"))
        .map(str::to_string)
}
