"""Resumable minimal CMU Panoptic downloader for WS1."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile


ENDPOINTS = (
    "http://domedb.perception.cs.cmu.edu",
    "http://vcl.snu.ac.kr/panoptic",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _download(urls: list[str], destination: Path) -> dict:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size > 0:
        return {
            "status": "reused",
            "url": None,
            "bytes": destination.stat().st_size,
            "sha256": sha256_file(destination),
        }

    partial = destination.with_name(destination.name + ".part")
    repaired = partial.exists()
    curl = shutil.which("curl.exe") or shutil.which("curl")
    if not curl:
        raise RuntimeError("curl is required for resumable HTTP downloads")
    errors = []
    for url in urls:
        try:
            command = [
                curl,
                "--fail",
                "--location",
                "--retry",
                "3",
                "--connect-timeout",
                "30",
                "--speed-limit",
                "1024",
                "--speed-time",
                "60",
                "--continue-at",
                "-",
                # Relative --output + cwd: Windows curl.exe mangles non-ASCII
                # argv (ANSI main), while cwd is passed Unicode-safe.
                "--output",
                partial.name,
                url,
            ]
            completed = subprocess.run(
                command, capture_output=True, text=True, cwd=str(partial.parent)
            )
            if completed.returncode != 0:
                raise RuntimeError(completed.stderr.strip())
            if not partial.is_file() or partial.stat().st_size == 0:
                raise RuntimeError("empty response")
            os.replace(partial, destination)
            return {
                "status": "repaired" if repaired else "downloaded",
                "url": url,
                "bytes": destination.stat().st_size,
                "sha256": sha256_file(destination),
            }
        except (OSError, RuntimeError) as error:
            errors.append(f"{url}: {error}")
    raise RuntimeError(
        f"failed resource {destination.name}; " + " | ".join(errors)
    )


def _archive_members(archive: Path) -> list[tarfile.TarInfo]:
    try:
        with tarfile.open(archive) as handle:
            members = handle.getmembers()
    except (tarfile.TarError, OSError) as error:
        raise ValueError(f"invalid tar archive {archive}: {error}") from error
    if not members:
        raise ValueError(f"empty tar archive: {archive}")
    return members


def _safe_extract(archive: Path, destination: Path) -> str:
    root = destination.resolve()
    members = _archive_members(archive)
    for member in members:
        target = (destination / member.name).resolve()
        if os.path.commonpath([root, target]) != str(root):
            raise ValueError(f"unsafe archive member: {member.name}")
    pose_directory = destination / "hdPose3d_stage1_coco19"
    expected_json = sum(
        member.isfile() and member.name.endswith(".json") for member in members
    )
    existing_json = (
        sum(1 for _ in pose_directory.glob("body3DScene_*.json"))
        if pose_directory.exists()
        else 0
    )
    if expected_json > 0 and existing_json == expected_json:
        return "reused"
    with tarfile.open(archive) as handle:
        handle.extractall(destination, filter="data")
    extracted_json = sum(1 for _ in pose_directory.glob("body3DScene_*.json"))
    if extracted_json != expected_json:
        raise ValueError(
            f"archive extraction incomplete: {extracted_json}/{expected_json}"
        )
    return "repaired" if existing_json else "downloaded"


def _write_manifest(destination: Path, sequence: str, resources: list[dict]) -> None:
    records = []
    raw_paths = sorted(destination.glob("calibration_*.json"))
    archive = destination / "hdPose3d_stage1_coco19.tar"
    if archive.exists():
        raw_paths.append(archive)
    raw_paths.extend(sorted((destination / "hdVideos").glob("*.mp4")))
    for path in raw_paths:
        records.append(
            {
                "path": path.relative_to(destination).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    payload = {
        "sequence": sequence,
        "source": "CMU Panoptic Studio",
        "contents": "calibration and hdPose3d_stage1_coco19; optional one HD video",
        "resources": resources,
        "raw_files": records,
        "extracted_pose_frame_count": sum(
            1
            for _ in (destination / "hdPose3d_stage1_coco19").glob(
                "body3DScene_*.json"
            )
        ),
    }
    # JSON is valid YAML 1.2 and avoids a PyYAML runtime dependency.
    (destination / "manifest.yaml").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


def download_minimal(
    sequence: str,
    output_root: Path,
    *,
    with_hd_video: bool = False,
    video_camera: str = "00_00",
    keep_archive: bool = True,
) -> Path:
    destination = output_root / sequence
    base_paths = [f"{endpoint}/webdata/dataset/{sequence}" for endpoint in ENDPOINTS]
    calibration = f"calibration_{sequence}.json"
    pose_archive = "hdPose3d_stage1_coco19.tar"
    resources = []
    resources.append(
        {
            "resource": "calibration",
            **_download(
                [f"{base}/{calibration}" for base in base_paths],
                destination / calibration,
            ),
        }
    )
    archive_path = destination / pose_archive
    resources.append(
        {
            "resource": "body_pose_archive",
            **_download(
                [f"{base}/{pose_archive}" for base in base_paths], archive_path
            ),
        }
    )
    resources.append(
        {
            "resource": "body_pose_extraction",
            "status": _safe_extract(archive_path, destination),
            "url": None,
        }
    )
    if with_hd_video:
        video = f"hd_{video_camera}.mp4"
        resources.append(
            {
                "resource": "hd_video",
                **_download(
                    [f"{base}/videos/hd_shared_crf20/{video}" for base in base_paths],
                    destination / "hdVideos" / video,
                ),
            }
        )
    _write_manifest(destination, sequence, resources)
    if not keep_archive:
        archive_path.unlink(missing_ok=True)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sequence", action="append", default=None, help="repeat for multiple sequences"
    )
    parser.add_argument("--output-root", type=Path, default=Path("data/panoptic_raw"))
    parser.add_argument("--video-sequence")
    parser.add_argument("--video-camera", default="00_00")
    parser.add_argument("--discard-archive", action="store_true")
    args = parser.parse_args()
    sequences = args.sequence or ["171204_pose1_sample"]
    for sequence in sequences:
        destination = download_minimal(
            sequence,
            args.output_root,
            with_hd_video=sequence == args.video_sequence,
            video_camera=args.video_camera,
            keep_archive=not args.discard_archive,
        )
        files = [path for path in destination.rglob("*") if path.is_file()]
        size = sum(path.stat().st_size for path in files)
        print(f"{destination}: {len(files)} files, {size / 1024 / 1024:.2f} MiB")


if __name__ == "__main__":
    main()
