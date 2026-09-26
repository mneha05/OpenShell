// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Doctor Docker preflight e2e tests.
//!
//! These tests verify that `openshell doctor check` reports actionable guidance
//! when Docker is not available.
//!
//! The tests do NOT require a running gateway or Docker — they intentionally
//! point `DOCKER_HOST` at a non-existent socket to simulate Docker being
//! unavailable.

use std::process::Stdio;
use std::time::Instant;
use std::{env, fs};

#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;

use openshell_e2e::harness::binary::openshell_cmd;
use openshell_e2e::harness::output::strip_ansi;

/// Run `openshell <args>` in an isolated environment where Docker is
/// guaranteed to be unreachable.
///
/// Sets `DOCKER_HOST` to a non-existent socket so the preflight check
/// fails immediately regardless of the host's Docker configuration.
async fn run_without_docker(args: &[&str]) -> (String, i32, std::time::Duration) {
    let tmpdir = tempfile::tempdir().expect("create isolated config dir");
    let bin_dir = tmpdir.path().join("bin");
    fs::create_dir(&bin_dir).expect("create fake bin dir");
    let fake_docker = bin_dir.join("docker");
    fs::write(
        &fake_docker,
        "#!/bin/sh\n\
         echo 'Cannot connect to Docker daemon. Check DOCKER_HOST and run docker info.' >&2\n\
         exit 1\n",
    )
    .expect("write fake docker");
    #[cfg(unix)]
    fs::set_permissions(&fake_docker, fs::Permissions::from_mode(0o755))
        .expect("chmod fake docker");

    let old_path = env::var("PATH").unwrap_or_default();
    let path = format!("{}:{old_path}", bin_dir.display());
    let start = Instant::now();

    let mut cmd = openshell_cmd();
    cmd.args(args)
        .env("XDG_CONFIG_HOME", tmpdir.path())
        .env("HOME", tmpdir.path())
        .env("PATH", path)
        .env("DOCKER_HOST", "unix:///tmp/openshell-e2e-nonexistent.sock")
        .env_remove("OPENSHELL_GATEWAY")
        .env_remove("OPENSHELL_GATEWAY_ENDPOINT")
        .env_remove("OPENSHELL_COMPUTE_DRIVER")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let output = cmd.output().await.expect("spawn openshell");
    let elapsed = start.elapsed();
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    let combined = format!("{stdout}{stderr}");
    let code = output.status.code().unwrap_or(-1);
    (combined, code, elapsed)
}

async fn run_with_fake_podman(
    args: &[&str],
    driver_from_env: bool,
    succeed: bool,
) -> (String, i32) {
    let tmpdir = tempfile::tempdir().expect("create isolated config dir");
    let bin_dir = tmpdir.path().join("bin");
    fs::create_dir(&bin_dir).expect("create fake bin dir");
    let fake_podman = bin_dir.join("podman");
    let script = if succeed {
        "#!/bin/sh\n\
         echo '5.4.2'\n"
    } else {
        "#!/bin/sh\n\
         echo 'Cannot connect to Podman socket.' >&2\n\
         exit 1\n"
    };
    fs::write(&fake_podman, script).expect("write fake podman");
    #[cfg(unix)]
    fs::set_permissions(&fake_podman, fs::Permissions::from_mode(0o755))
        .expect("chmod fake podman");

    let old_path = env::var("PATH").unwrap_or_default();
    let path = format!("{}:{old_path}", bin_dir.display());

    let mut cmd = openshell_cmd();
    cmd.args(args)
        .env("XDG_CONFIG_HOME", tmpdir.path())
        .env("HOME", tmpdir.path())
        .env("PATH", path)
        .env("CONTAINER_HOST", "unix:///tmp/openshell-e2e-podman.sock")
        .env_remove("OPENSHELL_GATEWAY")
        .env_remove("OPENSHELL_GATEWAY_ENDPOINT");

    if driver_from_env {
        cmd.env("OPENSHELL_COMPUTE_DRIVER", "podman");
    } else {
        cmd.env_remove("OPENSHELL_COMPUTE_DRIVER");
    }

    let output = cmd
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .await
        .expect("spawn openshell");

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    let combined = format!("{stdout}{stderr}");
    let code = output.status.code().unwrap_or(-1);
    (combined, code)
}

// -------------------------------------------------------------------
// doctor check: validates system prerequisites
// -------------------------------------------------------------------

/// `openshell doctor check` with Docker unavailable should fail fast
/// and report the Docker check as FAILED.
#[tokio::test]
async fn doctor_check_fails_without_docker() {
    let (output, code, elapsed) = run_without_docker(&["doctor", "check"]).await;

    assert_ne!(
        code, 0,
        "doctor check should fail when Docker is unavailable, output:\n{output}"
    );

    assert!(
        elapsed.as_secs() < 10,
        "doctor check should complete quickly (took {}s)",
        elapsed.as_secs()
    );

    let clean = strip_ansi(&output);
    assert!(
        clean.contains("FAILED"),
        "doctor check should report Docker as FAILED:\n{clean}"
    );
}

/// `openshell doctor check` output should include the check label
/// so the user knows what was tested.
#[tokio::test]
async fn doctor_check_output_shows_docker_label() {
    let (output, _, _) = run_without_docker(&["doctor", "check"]).await;
    let clean = strip_ansi(&output);

    assert!(
        clean.contains("Docker"),
        "doctor check output should include 'Docker' label:\n{clean}"
    );
}

/// `openshell doctor check` with Docker unavailable should include
/// actionable guidance in the error output.
#[tokio::test]
async fn doctor_check_error_includes_guidance() {
    let (output, code, _) = run_without_docker(&["doctor", "check"]).await;

    assert_ne!(code, 0);
    let clean = strip_ansi(&output);

    assert!(
        clean.contains("DOCKER_HOST"),
        "doctor check error should mention DOCKER_HOST:\n{clean}"
    );
    assert!(
        clean.contains("docker info"),
        "doctor check error should suggest 'docker info':\n{clean}"
    );
}

/// When Docker IS available, `openshell doctor check` should pass and
/// report the version.
///
/// This test only runs when Docker is actually reachable on the host
/// (i.e., it will pass in CI with Docker but be skipped locally if
/// Docker is not running). We detect this by checking if the default
/// socket exists.
#[tokio::test]
async fn doctor_check_passes_with_docker() {
    if !std::path::Path::new("/var/run/docker.sock").exists() {
        eprintln!("skipping: /var/run/docker.sock not found");
        return;
    }

    let tmpdir = tempfile::tempdir().expect("create isolated config dir");
    let mut cmd = openshell_cmd();
    cmd.args(["doctor", "check"])
        .env("XDG_CONFIG_HOME", tmpdir.path())
        .env("HOME", tmpdir.path())
        .env_remove("OPENSHELL_GATEWAY")
        .env_remove("OPENSHELL_GATEWAY_ENDPOINT")
        .env_remove("OPENSHELL_COMPUTE_DRIVER")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let output = cmd.output().await.expect("spawn openshell");
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    let combined = format!("{stdout}{stderr}");
    let code = output.status.code().unwrap_or(-1);
    let clean = strip_ansi(&combined);

    assert_eq!(
        code, 0,
        "doctor check should pass when Docker is available, output:\n{clean}"
    );
    assert!(
        clean.contains("All checks passed"),
        "doctor check should report success:\n{clean}"
    );
    assert!(
        clean.contains("ok"),
        "doctor check should show 'ok' for Docker:\n{clean}"
    );
}


#[tokio::test]
async fn doctor_check_podman_explicit_driver_reports_version() {
    let (output, code) =
        run_with_fake_podman(&["doctor", "check", "--driver", "podman"], false, true).await;
    let clean = strip_ansi(&output);

    assert_eq!(code, 0, "Podman doctor check should pass:\n{clean}");
    assert!(clean.contains("Podman"), "missing Podman label:\n{clean}");
    assert!(clean.contains("version 5.4.2"), "missing Podman version:\n{clean}");
    assert!(
        clean.contains("CONTAINER_HOST") && clean.contains("openshell-e2e-podman.sock"),
        "missing Podman socket guidance:\n{clean}"
    );
}

#[tokio::test]
async fn doctor_check_podman_uses_compute_driver_env() {
    let (output, code) = run_with_fake_podman(&["doctor", "check"], true, true).await;
    let clean = strip_ansi(&output);

    assert_eq!(code, 0, "Podman doctor check should pass:\n{clean}");
    assert!(clean.contains("Podman"), "driver env should select Podman:\n{clean}");
    assert!(!clean.contains("Docker"), "Podman selection should not run Docker:\n{clean}");
}

#[tokio::test]
async fn doctor_check_podman_failure_is_actionable() {
    let (output, code) =
        run_with_fake_podman(&["doctor", "check", "--driver", "podman"], false, false).await;
    let clean = strip_ansi(&output);

    assert_ne!(code, 0, "unreachable Podman should fail:\n{clean}");
    assert!(clean.contains("FAILED"), "failure should be labeled:\n{clean}");
    assert!(clean.contains("CONTAINER_HOST"), "error should mention CONTAINER_HOST:\n{clean}");
    assert!(clean.contains("podman info"), "error should suggest podman info:\n{clean}");
}
