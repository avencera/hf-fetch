//! A low-dependency, low-complexity, blocking downloader for Hugging Face Hub model files
//!
//! It exists because hf-hub 1.0 brings in tokio, reqwest and hf-xet, far more than fetching
//! a few model files needs. This crate uses a small blocking `ureq` client with rustls, and
//! has no async runtime or Xet client
//!
//! The endpoint, token and cache resolve the way `huggingface_hub` resolves them, so Python
//! tools and this crate share one cache
//!
//! ```no_run
//! # #[cfg(feature = "online")]
//! # fn main() -> Result<(), hf_fetch::HubError> {
//! use hf_fetch::Hub;
//!
//! let repo = Hub::from_env()?.model("avencera/scriptrs-models", "main");
//! let path = repo.get("config.json")?;
//! # Ok(())
//! # }
//! # #[cfg(not(feature = "online"))]
//! # fn main() {}
//! ```
//!
//! The default `online` feature enables `Hub`, `HubRepo`, and `HubError`.
//! Disable default features to use only [`default_hub_cache_dir`], with no dependencies.
//! Connections time out after 30 seconds. Response headers and each body read time out
//! after 60 seconds without progress. There is no whole-download time limit. A failed
//! download keeps its partial file, and the next one resumes from it.
//!
//! Tokens stay on the Hub when it redirects to storage. Files use the Python cache's
//! `models--owner--repo/{blobs,refs,snapshots}` layout, with relative symlinks on Unix
//! and copies on other platforms. `HubRepo::get` uses cached files without a request;
//! `HubRepo::download` refreshes metadata even if the file is already cached.
//!
//! See the README for environment variable precedence and supported scope.

#![deny(missing_docs)]

use std::env;
use std::path::PathBuf;

/// Value of the environment variable `name`, treating an empty value as unset
fn env_var(name: &str) -> Option<String> {
    env::var(name).ok().filter(|value| !value.is_empty())
}

fn home_dir() -> PathBuf {
    env::home_dir().unwrap_or_else(env::temp_dir)
}

/// Path in the environment variable `name`, with a leading `~` expanded to the home
/// directory as `huggingface_hub` does, so `HF_HOME=~/hf` names the same cache in both
fn env_path(name: &str) -> Option<PathBuf> {
    let value = env_var(name)?;
    let path = match value.strip_prefix('~') {
        Some("") => home_dir(),
        Some(rest) if rest.starts_with(std::path::is_separator) => {
            home_dir().join(rest.trim_start_matches(std::path::is_separator))
        }
        _ => PathBuf::from(value),
    };
    Some(path)
}

/// `$HF_HOME`, else `$XDG_CACHE_HOME/huggingface`, else `~/.cache/huggingface`
fn hf_home() -> PathBuf {
    if let Some(home) = env_path("HF_HOME") {
        return home;
    }

    let cache = env_path("XDG_CACHE_HOME").unwrap_or_else(|| home_dir().join(".cache"));
    cache.join("huggingface")
}

/// The Hugging Face hub cache that downloads use unless given another directory
///
/// `$HF_HUB_CACHE`, else `$HUGGINGFACE_HUB_CACHE`, else `$HF_HOME/hub`, where `HF_HOME`
/// defaults to `$XDG_CACHE_HOME/huggingface`, else `~/.cache/huggingface`
pub fn default_hub_cache_dir() -> PathBuf {
    env_path("HF_HUB_CACHE")
        .or_else(|| env_path("HUGGINGFACE_HUB_CACHE"))
        .unwrap_or_else(|| hf_home().join("hub"))
}

#[cfg(feature = "online")]
pub use download::{Hub, HubError, HubRepo};

#[cfg(feature = "online")]
mod download {
    use std::fs;
    use std::io::{self, Seek, SeekFrom};
    use std::path::{Component, Path, PathBuf};
    use std::time::Duration;

    use ureq::http::Response;
    use ureq::typestate::WithoutBody;
    use ureq::{Agent, Body, RequestBuilder};

    use super::{default_hub_cache_dir, env_path, env_var, hf_home};

    const DEFAULT_ENDPOINT: &str = "https://huggingface.co";

    // the Hub redirects renamed repos to their new name, so a few hops are legitimate
    const MAX_REDIRECTS: usize = 5;

    // bounds on a stalled connection; there is no cap on a whole download, since a
    // multi-gigabyte file on a slow link can legitimately take hours
    const CONNECT_TIMEOUT: Duration = Duration::from_secs(30);
    const STALL_TIMEOUT: Duration = Duration::from_secs(60);

    /// Why a Hub download failed
    #[derive(Debug, thiserror::Error)]
    #[non_exhaustive]
    pub enum HubError {
        /// The Hub answered with an error status: 401 without a valid token, 403 when the
        /// token's account cannot access the repo, 404 for a missing repo, revision or file
        #[error("Hugging Face answered HTTP {status} for {url}")]
        Status {
            /// HTTP status code
            status: u16,
            /// Requested URL
            url: String,
        },
        /// The request failed before the Hub answered
        #[error("request {url}: {error}")]
        Request {
            /// Requested URL
            url: String,
            /// Transport error
            error: Box<ureq::Error>,
        },
        /// The Hub's answer lacks a header that a download needs
        #[error("Hugging Face's answer for {url} has no {header} header")]
        MissingHeader {
            /// Requested URL
            url: String,
            /// Name of the missing header
            header: &'static str,
        },
        /// The Hub's answer has a header value that cannot name a cache entry, such as a
        /// commit that is not a hash or an etag with path separators
        #[error("Hugging Face's answer for {url} has an invalid {header} header: {value:?}")]
        InvalidHeader {
            /// Requested URL
            url: String,
            /// Name of the invalid header
            header: &'static str,
            /// Value the Hub sent
            value: String,
        },
        /// The requested file is not a relative path inside the repo
        #[error("{file:?} is not a file path inside the repo")]
        InvalidFile {
            /// Requested file
            file: String,
        },
        /// The Hub kept redirecting
        #[error("Hugging Face redirected {url} more than {MAX_REDIRECTS} times")]
        TooManyRedirects {
            /// First requested URL
            url: String,
        },
        /// Reading the token, downloading into the cache or linking a snapshot file failed
        #[error("{}: {error}", path.display())]
        Io {
            /// File the operation was on
            path: PathBuf,
            /// I/O error
            error: io::Error,
        },
    }

    impl HubError {
        /// The HTTP status of a [`HubError::Status`]
        pub fn status(&self) -> Option<u16> {
            match self {
                Self::Status { status, .. } => Some(*status),
                _ => None,
            }
        }

        fn io(path: &Path, error: io::Error) -> Self {
            Self::Io {
                path: path.to_path_buf(),
                error,
            }
        }
    }

    /// Blocking Hub client, configured from the environment like `huggingface_hub`
    #[derive(Clone, Debug)]
    pub struct Hub {
        agent: Agent,
        endpoint: String,
        token: Option<String>,
        cache_dir: PathBuf,
    }

    impl Hub {
        /// Client for the cache at [`default_hub_cache_dir`]
        ///
        /// The endpoint is `$HF_ENDPOINT`, else `https://huggingface.co`. The token is
        /// `$HF_TOKEN`, else the file at `$HF_TOKEN_PATH`, else `$HF_HOME/token`, where
        /// `hf auth login` saves it
        pub fn from_env() -> Result<Self, HubError> {
            Self::with_cache_dir(default_hub_cache_dir())
        }

        /// Client for the cache at `cache_dir`, otherwise configured like [`Hub::from_env`]
        pub fn with_cache_dir(cache_dir: impl Into<PathBuf>) -> Result<Self, HubError> {
            let agent = Agent::config_builder()
                // error answers carry what callers report, such as a gated repo's 401
                .http_status_as_error(false)
                // redirects are followed by hand so the token never leaves the Hub
                .max_redirects(0)
                .timeout_connect(Some(CONNECT_TIMEOUT))
                .timeout_recv_response(Some(STALL_TIMEOUT))
                .timeout_per_read(Some(STALL_TIMEOUT))
                // a resumed range counts bytes of the file itself, not of a compressed copy,
                // even when another crate turns on ureq's gzip feature
                .accept_encoding("identity")
                .user_agent(concat!(
                    env!("CARGO_PKG_NAME"),
                    "/",
                    env!("CARGO_PKG_VERSION")
                ))
                .build()
                .new_agent();
            let endpoint = env_var("HF_ENDPOINT")
                .as_deref()
                .unwrap_or(DEFAULT_ENDPOINT)
                .trim_end_matches('/')
                .to_owned();

            Ok(Self {
                agent,
                endpoint,
                token: read_token()?,
                cache_dir: cache_dir.into(),
            })
        }

        /// Files of the model repo `id` (`owner/name`) at `revision`, which is a branch, a
        /// tag or a commit hash
        pub fn model(&self, id: impl Into<String>, revision: impl Into<String>) -> HubRepo {
            HubRepo {
                hub: self.clone(),
                id: id.into(),
                revision: revision.into(),
            }
        }

        /// A GET request for `url` that carries the token, for URLs on the Hub only
        fn authorized(&self, url: &str) -> RequestBuilder<WithoutBody> {
            let request = self.agent.get(url);
            match &self.token {
                Some(token) => request.header("Authorization", format!("Bearer {token}")),
                None => request,
            }
        }

        /// Asks for the first byte of `url` to learn the file's commit and blob etag
        ///
        /// Follows the Hub's own relative redirects with the token and stops at an absolute
        /// one, which points at the storage that serves the file
        fn metadata(&self, url: &str) -> Result<Metadata, HubError> {
            let mut current = url.to_owned();
            for _ in 0..=MAX_REDIRECTS {
                let request = self.authorized(&current).header("Range", "bytes=0-0");
                let response = call(request, &current)?;
                let status = response.status();
                let location = if status.is_redirection() {
                    let location = header(&response, &current, "location")?;
                    if location.starts_with('/') {
                        current = format!("{}{location}", origin(&current));
                        continue;
                    }
                    Download::Storage(location)
                } else if status.is_success() {
                    Download::Hub(current.clone())
                } else {
                    return Err(HubError::Status {
                        status: status.as_u16(),
                        url: current,
                    });
                };

                let etag = header(&response, &current, "x-linked-etag")
                    .or_else(|_| header(&response, &current, "etag"))?;
                // weak validators and quotes are not part of the blob name
                let etag = etag.trim_start_matches("W/").replace('"', "");
                let commit = header(&response, &current, "x-repo-commit")?;
                // both become cache paths, so a hostile endpoint must not reach outside it
                let invalid = |header, value: String| HubError::InvalidHeader {
                    url: current.clone(),
                    header,
                    value,
                };
                if !is_blob_name(&etag) {
                    return Err(invalid("etag", etag));
                }
                if !is_commit_hash(&commit) {
                    return Err(invalid("x-repo-commit", commit));
                }

                return Ok(Metadata {
                    commit,
                    etag,
                    location,
                });
            }

            Err(HubError::TooManyRedirects {
                url: url.to_owned(),
            })
        }

        /// Streams the file into `blob` through `<blob>.incomplete`, which outlives a failed
        /// attempt so the next one resumes where it stopped
        ///
        /// The lock at `lock` is the one `huggingface_hub` takes, so a Python download of the
        /// same blob waits for this one. The partial file relies on that lock: on a filesystem
        /// where `flock` silently succeeds, such as some NFS mounts, concurrent downloads of one
        /// blob can append to the same partial file, as with hf-hub's `.part` files
        fn fetch(&self, location: &Download, blob: &Path, lock: &Path) -> Result<(), HubError> {
            let blobs = blob.parent().expect("blob paths sit in blobs/");
            fs::create_dir_all(blobs).map_err(|error| HubError::io(blobs, error))?;
            let _lock = lock_file(lock)?;
            // another process may have finished the blob while this one waited for the lock
            if blob.is_file() {
                return Ok(());
            }

            let mut partial_name = blob.as_os_str().to_owned();
            partial_name.push(".incomplete");
            let partial = PathBuf::from(partial_name);
            let io_error = |error| HubError::io(&partial, error);
            // plain write access rather than append, which cannot truncate on Windows
            let mut file = fs::OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(false)
                .open(&partial)
                .map_err(io_error)?;
            let mut offset = file.seek(SeekFrom::End(0)).map_err(io_error)?;
            let restart = |file: &mut fs::File| {
                file.set_len(0)
                    .and_then(|()| file.seek(SeekFrom::Start(0)))
                    .map(drop)
                    .map_err(io_error)
            };

            // where this answer's bytes should end and the blob's size, when the server says
            // them, so a short answer is not published
            let (response, end, size) = loop {
                let (url, request) = self.blob_request(location);
                let request = match offset {
                    0 => request,
                    _ => request.header("Range", format!("bytes={offset}-")),
                };
                let response = call(request, url)?;
                match response.status().as_u16() {
                    200 => {
                        // the server ignored the range and sent the whole file
                        if offset > 0 {
                            restart(&mut file)?;
                        }
                        let size = header(&response, url, "content-length")
                            .ok()
                            .and_then(|length| length.parse().ok());
                        break (response, size, size);
                    }
                    206 if offset > 0 => {
                        let range = header(&response, url, "content-range").ok();
                        match range.as_deref().and_then(content_range) {
                            Some((start, end, size)) if start == offset => {
                                break (response, Some(end + 1), Some(size));
                            }
                            // a range this download cannot check, such as one of unknown size,
                            // starts over
                            _ => {
                                restart(&mut file)?;
                                offset = 0;
                            }
                        }
                    }
                    // the partial file is no prefix of the blob, such as a complete one whose
                    // rename was interrupted, so it starts over
                    416 if offset > 0 => {
                        restart(&mut file)?;
                        offset = 0;
                    }
                    status => {
                        return Err(HubError::Status {
                            status,
                            url: url.clone(),
                        });
                    }
                }
            };

            // a failed read keeps the bytes already written for the next attempt
            io::copy(&mut response.into_body().into_reader(), &mut file).map_err(io_error)?;
            let written = file.stream_position().map_err(io_error)?;
            // a body past its range's end leaves bytes the blob does not have there
            if let Some(end) = end
                && written > end
            {
                restart(&mut file)?;
                return Err(io_error(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("received bytes past {end}"),
                )));
            }
            // a server may answer with less than the rest of the file, which the next attempt
            // resumes
            if let Some(size) = size
                && written < size
            {
                return Err(io_error(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    format!("received {written} of {size} bytes"),
                )));
            }

            drop(file);
            fs::rename(&partial, blob).map_err(|error| HubError::io(blob, error))
        }

        /// A GET request for the bytes of a file
        fn blob_request<'a>(
            &self,
            location: &'a Download,
        ) -> (&'a String, RequestBuilder<WithoutBody>) {
            match location {
                Download::Hub(url) => (url, self.authorized(url)),
                // storage URLs are presigned, so they get no token, and may redirect again
                Download::Storage(url) => (
                    url,
                    self.agent
                        .get(url)
                        .config()
                        .max_redirects(MAX_REDIRECTS as u32)
                        .build(),
                ),
            }
        }
    }

    /// Where the bytes of a file come from
    #[derive(Debug)]
    enum Download {
        /// The Hub serves the file itself, as it does for small files kept in git
        Hub(String),
        /// The Hub redirected to object storage, as it does for LFS and Xet files
        Storage(String),
    }

    #[derive(Debug)]
    struct Metadata {
        commit: String,
        etag: String,
        location: Download,
    }

    /// One model repo at one revision, see [`Hub::model`]
    #[derive(Clone, Debug)]
    pub struct HubRepo {
        hub: Hub,
        id: String,
        revision: String,
    }

    impl HubRepo {
        /// The cached copy of `file`, found without a network request
        ///
        /// A commit hash names its snapshot directly; a branch or tag goes through the commit
        /// that its `refs/` entry recorded at the last download
        pub fn cached(&self, file: &str) -> Option<PathBuf> {
            if !is_repo_file(file) {
                return None;
            }

            let repo_dir = self.repo_dir();
            let commit = if is_commit_hash(&self.revision) {
                self.revision.clone()
            } else {
                let reference = repo_dir.join("refs").join(&self.revision);
                let commit = fs::read_to_string(reference).ok()?.trim().to_owned();
                is_commit_hash(&commit).then_some(commit)?
            };

            let path = repo_dir.join("snapshots").join(commit).join(file);
            // a link whose blob was deleted is not a cached file
            path.is_file().then_some(path)
        }

        /// The cached copy of `file`, downloaded first when the cache lacks it
        ///
        /// A cached file is used without asking the Hub whether a branch moved, so files
        /// already downloaded load offline and without a token
        pub fn get(&self, file: &str) -> Result<PathBuf, HubError> {
            match self.cached(file) {
                Some(path) => Ok(path),
                None => self.download(file),
            }
        }

        /// Downloads `file` into the cache, even when a copy is cached, and returns its
        /// snapshot path
        pub fn download(&self, file: &str) -> Result<PathBuf, HubError> {
            if !is_repo_file(file) {
                return Err(HubError::InvalidFile {
                    file: file.to_owned(),
                });
            }

            let url = format!(
                "{}/{}/resolve/{}/{file}",
                self.hub.endpoint,
                self.id,
                // a revision such as `refs/pr/1` is one path segment
                self.revision.replace('/', "%2F"),
            );
            let metadata = self.hub.metadata(&url)?;
            // a pinned commit must land in its own snapshot, not one the Hub picks
            if is_commit_hash(&self.revision) && metadata.commit != self.revision {
                return Err(HubError::InvalidHeader {
                    url,
                    header: "x-repo-commit",
                    value: metadata.commit,
                });
            }

            let repo_dir = self.repo_dir();
            let blob = repo_dir.join("blobs").join(&metadata.etag);
            if !blob.is_file() {
                let lock = self
                    .hub
                    .cache_dir
                    .join(".locks")
                    .join(self.folder_name())
                    .join(format!("{}.lock", metadata.etag));
                self.hub.fetch(&metadata.location, &blob, &lock)?;
            }

            let snapshot_file = repo_dir.join("snapshots").join(&metadata.commit).join(file);
            link_blob(&blob, &snapshot_file, file)?;
            if metadata.commit != self.revision {
                let reference = repo_dir.join("refs").join(&self.revision);
                let refs = reference.parent().expect("refs sit in refs/");
                fs::create_dir_all(refs).map_err(|error| HubError::io(refs, error))?;
                fs::write(&reference, &metadata.commit)
                    .map_err(|error| HubError::io(&reference, error))?;
            }

            Ok(snapshot_file)
        }

        fn repo_dir(&self) -> PathBuf {
            self.hub.cache_dir.join(self.folder_name())
        }

        fn folder_name(&self) -> String {
            format!("models--{}", self.id.replace('/', "--"))
        }
    }

    /// Opens and exclusively locks the file at `path`, blocking while another process holds it
    ///
    /// The lock is released when the returned file closes, including when the process dies
    fn lock_file(path: &Path) -> Result<fs::File, HubError> {
        let locks = path.parent().expect("lock paths sit in .locks/<repo>/");
        fs::create_dir_all(locks).map_err(|error| HubError::io(locks, error))?;
        let file = fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(false)
            .open(path)
            .map_err(|error| HubError::io(path, error))?;
        file.lock().map_err(|error| HubError::io(path, error))?;
        Ok(file)
    }

    fn read_token() -> Result<Option<String>, HubError> {
        if let Some(token) = env_var("HF_TOKEN").as_deref().and_then(clean_token) {
            return Ok(Some(token));
        }

        let path = env_path("HF_TOKEN_PATH").unwrap_or_else(|| hf_home().join("token"));
        match fs::read_to_string(&path) {
            Ok(token) => Ok(clean_token(&token)),
            Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(None),
            Err(error) => Err(HubError::io(&path, error)),
        }
    }

    /// Drops line breaks and surrounding whitespace, as `huggingface_hub` does, so a token
    /// pasted with a trailing newline still authenticates; blank means no token
    fn clean_token(raw: &str) -> Option<String> {
        let token = raw.replace(['\r', '\n'], "");
        let token = token.trim();
        (!token.is_empty()).then(|| token.to_owned())
    }

    /// Writes `reader` to a file of its own next to `dest`, then renames it into place, so
    /// readers never see a partial file and concurrent writers never share one
    #[cfg(not(unix))]
    fn write_into_place(reader: &mut impl io::Read, dest: &Path) -> Result<(), HubError> {
        use std::sync::atomic::{AtomicU64, Ordering};
        use std::time::{SystemTime, UNIX_EPOCH};

        static NEXT: AtomicU64 = AtomicU64::new(0);

        let name = dest.file_name().expect("cache paths end in a file name");
        // the process id and counter separate downloads within a host, and the clock
        // separates processes in other PID namespaces that share the cache
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |elapsed| elapsed.subsec_nanos());
        let (partial, mut file) = loop {
            let partial = dest.with_file_name(format!(
                "{}.{}.{}.{nanos}.incomplete",
                name.to_string_lossy(),
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed),
            ));
            match fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&partial)
            {
                Ok(file) => break (partial, file),
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(HubError::io(&partial, error)),
            }
        };

        let copied = io::copy(reader, &mut file);
        drop(file);
        if let Err(error) = copied {
            let _ = fs::remove_file(&partial);
            return Err(HubError::io(&partial, error));
        }

        fs::rename(&partial, dest).map_err(|error| {
            let _ = fs::remove_file(&partial);
            HubError::io(dest, error)
        })
    }

    fn call(request: RequestBuilder<WithoutBody>, url: &str) -> Result<Response<Body>, HubError> {
        request.call().map_err(|error| HubError::Request {
            url: url.to_owned(),
            error: Box::new(error),
        })
    }

    fn header(
        response: &Response<Body>,
        url: &str,
        name: &'static str,
    ) -> Result<String, HubError> {
        response
            .headers()
            .get(name)
            .and_then(|value| value.to_str().ok())
            .map(str::to_owned)
            .ok_or_else(|| HubError::MissingHeader {
                url: url.to_owned(),
                header: name,
            })
    }

    /// First byte, last byte and total size of a `Content-Range` such as `bytes 3-6/7`, when
    /// all three are known and consistent
    fn content_range(value: &str) -> Option<(u64, u64, u64)> {
        let (unit, range) = value.trim().split_once(' ')?;
        if !unit.eq_ignore_ascii_case("bytes") {
            return None;
        }

        let (range, size) = range.split_once('/')?;
        let (start, end) = range.split_once('-')?;
        let [start, end, size] = [start, end, size].map(|number| number.trim().parse::<u64>());
        let (start, end, size) = (start.ok()?, end.ok()?, size.ok()?);
        (start <= end && end < size).then_some((start, end, size))
    }

    /// `scheme://host[:port]` of `url`
    fn origin(url: &str) -> &str {
        let host_start = url.find("://").map_or(0, |index| index + 3);
        url[host_start..]
            .find('/')
            .map_or(url, |index| &url[..host_start + index])
    }

    fn is_commit_hash(revision: &str) -> bool {
        revision.len() == 40 && revision.bytes().all(|byte| byte.is_ascii_hexdigit())
    }

    /// Hub etags are git or SHA-256 hashes, so anything else cannot be a blob name
    fn is_blob_name(etag: &str) -> bool {
        !etag.is_empty()
            && etag
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-' || byte == b'_')
    }

    /// A relative path that stays inside the snapshot, such as `dir/model.bin`
    fn is_repo_file(file: &str) -> bool {
        !file.is_empty()
            && Path::new(file)
                .components()
                .all(|component| matches!(component, Component::Normal(_)))
    }

    /// Points `snapshot_file` at `blob` with a relative link, as `huggingface_hub` does, so
    /// the cache stays valid when it moves
    fn link_blob(blob: &Path, snapshot_file: &Path, file: &str) -> Result<(), HubError> {
        if snapshot_file.is_file() {
            return Ok(());
        }

        let parent = snapshot_file
            .parent()
            .expect("snapshot files sit in snapshots/");
        fs::create_dir_all(parent).map_err(|error| HubError::io(parent, error))?;
        // a dangling link from a deleted blob would block the new one
        if snapshot_file.symlink_metadata().is_ok() {
            fs::remove_file(snapshot_file).map_err(|error| HubError::io(snapshot_file, error))?;
        }

        #[cfg(unix)]
        {
            // up from the file's directory to the snapshot, then out of snapshots/<commit>/
            let depth = Path::new(file).components().count() + 1;
            let mut target: PathBuf = std::iter::repeat_n("..", depth).collect();
            target.push("blobs");
            target.push(blob.file_name().expect("blob paths end in the etag"));

            match std::os::unix::fs::symlink(&target, snapshot_file) {
                Ok(()) => Ok(()),
                // another process linked it first
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => Ok(()),
                Err(error) => Err(HubError::io(snapshot_file, error)),
            }
        }

        // symlinks need extra privileges on Windows, so the snapshot gets a copy there
        #[cfg(not(unix))]
        {
            let _ = file;
            let mut source = fs::File::open(blob).map_err(|error| HubError::io(blob, error))?;
            write_into_place(&mut source, snapshot_file)
        }
    }

    #[cfg(test)]
    mod tests {
        use std::io::{BufRead, BufReader, Write};
        use std::net::TcpListener;
        use std::sync::{Arc, Mutex};
        use std::thread;

        #[cfg(not(unix))]
        use super::write_into_place;
        use super::{
            Hub, HubError, Path, clean_token, content_range, fs, is_blob_name, is_commit_hash,
            is_repo_file,
        };

        const COMMIT: &str = "0123456789abcdef0123456789abcdef01234567";
        const BODY: &[u8] = b"weights";

        /// Path, `Authorization` and `Range` headers of each request the test server received
        type Requests = Arc<Mutex<Vec<(String, Option<String>, Option<String>)>>>;

        /// Serves one repo file: the Hub answers with a redirect to storage, which serves
        /// the bytes, and every request's path and token are recorded
        fn serve() -> (String, Requests) {
            serve_with(|_| {
                format!(
                    "HTTP/1.1 200 OK\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                    BODY.len(),
                    String::from_utf8_lossy(BODY)
                )
            })
        }

        /// Like [`serve`], with `answer` answering each storage request from its `Range`
        fn serve_with(answer: fn(Option<&str>) -> String) -> (String, Requests) {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let origin = format!("http://{}", listener.local_addr().unwrap());
            let requests = Arc::new(Mutex::new(Vec::new()));
            let recorded = Arc::clone(&requests);
            let storage = format!("{origin}/storage/blob");
            thread::spawn(move || {
                for stream in listener.incoming() {
                    let mut stream = stream.unwrap();
                    let mut reader = BufReader::new(stream.try_clone().unwrap());
                    let mut line = String::new();
                    reader.read_line(&mut line).unwrap();
                    let path = line.split_whitespace().nth(1).unwrap().to_owned();
                    let mut auth = None;
                    let mut range = None;
                    loop {
                        let mut header = String::new();
                        reader.read_line(&mut header).unwrap();
                        if header.trim().is_empty() {
                            break;
                        }
                        if let Some(value) = header.strip_prefix("authorization: ") {
                            auth = Some(value.trim().to_owned());
                        }
                        if let Some(value) = header.strip_prefix("range: ") {
                            range = Some(value.trim().to_owned());
                        }
                    }
                    recorded
                        .lock()
                        .unwrap()
                        .push((path.clone(), auth, range.clone()));

                    let response = if path.starts_with("/storage/") {
                        answer(range.as_deref())
                    } else if path.starts_with("/old/") {
                        // a renamed repo redirects relative to the Hub
                        let renamed = path.replacen("/old/", "/owner/", 1);
                        format!(
                            "HTTP/1.1 307 Temporary Redirect\r\nlocation: {renamed}\r\ncontent-length: 0\r\nconnection: close\r\n\r\n"
                        )
                    } else if path.contains("/missing") {
                        "HTTP/1.1 404 Not Found\r\ncontent-length: 0\r\nconnection: close\r\n\r\n"
                            .to_owned()
                    } else {
                        format!(
                            "HTTP/1.1 302 Found\r\nlocation: {storage}\r\nx-repo-commit: {COMMIT}\r\nx-linked-etag: \"abc123\"\r\ncontent-length: 0\r\nconnection: close\r\n\r\n"
                        )
                    };
                    stream.write_all(response.as_bytes()).unwrap();
                }
            });

            (origin, requests)
        }

        fn hub(origin: &str, cache_dir: &Path) -> Hub {
            let mut hub = Hub::with_cache_dir(cache_dir).unwrap();
            hub.endpoint = origin.to_owned();
            hub.token = Some("secret".to_owned());
            hub
        }

        #[test]
        fn downloads_into_hub_cache_layout_and_keeps_token_on_the_hub() {
            let (origin, requests) = serve();
            let cache = tempfile::tempdir().unwrap();
            let repo = hub(&origin, cache.path()).model("old/model", "main");

            let path = repo.get("nested/model.bin").unwrap();

            let repo_dir = cache.path().join("models--old--model");
            assert_eq!(
                path,
                repo_dir
                    .join("snapshots")
                    .join(COMMIT)
                    .join("nested/model.bin")
            );
            assert_eq!(fs::read(&path).unwrap(), BODY);
            #[cfg(unix)]
            assert_eq!(
                fs::read_link(&path).unwrap(),
                Path::new("../../../blobs/abc123")
            );
            assert_eq!(
                fs::read_to_string(repo_dir.join("refs/main")).unwrap(),
                COMMIT
            );

            let requests = requests.lock().unwrap().clone();
            let paths: Vec<_> = requests.iter().map(|(path, ..)| path.as_str()).collect();
            assert_eq!(
                paths,
                [
                    "/old/model/resolve/main/nested/model.bin",
                    "/owner/model/resolve/main/nested/model.bin",
                    "/storage/blob",
                ]
            );
            let bearer = Some("Bearer secret".to_owned());
            assert_eq!(requests[0].1, bearer);
            assert_eq!(requests[1].1, bearer);
            assert_eq!(requests[2].1, None);
        }

        #[cfg(not(unix))]
        #[test]
        fn incomplete_copy_preserves_the_old_file() {
            struct Interrupted;
            impl std::io::Read for Interrupted {
                fn read(&mut self, _: &mut [u8]) -> std::io::Result<usize> {
                    Err(std::io::Error::other("connection lost"))
                }
            }

            let cache = tempfile::tempdir().unwrap();
            let dest = cache.path().join("blob");
            fs::write(&dest, b"complete").unwrap();
            let result = write_into_place(&mut Interrupted, &dest);

            assert!(matches!(result, Err(HubError::Io { .. })));
            assert_eq!(fs::read(&dest).unwrap(), b"complete");
            assert_eq!(fs::read_dir(cache.path()).unwrap().count(), 1);
        }

        /// Storage requests' `Range` headers, in order
        fn storage_ranges(requests: &Requests) -> Vec<Option<String>> {
            let requests = requests.lock().unwrap();
            requests
                .iter()
                .filter(|(path, ..)| path.starts_with("/storage/"))
                .map(|(.., range)| range.clone())
                .collect()
        }

        #[test]
        fn interrupted_download_resumes_from_its_partial_file() {
            let (origin, requests) = serve_with(|range| {
                match range {
                // the connection drops after three of the seven bytes
                None => "HTTP/1.1 200 OK\r\ncontent-length: 7\r\nconnection: close\r\n\r\nwei"
                    .to_owned(),
                Some("bytes=3-") => "HTTP/1.1 206 Partial Content\r\ncontent-range: bytes 3-6/7\r\ncontent-length: 4\r\nconnection: close\r\n\r\nghts"
                    .to_owned(),
                Some(range) => panic!("unexpected range {range}"),
            }
            });
            let cache = tempfile::tempdir().unwrap();
            let repo = hub(&origin, cache.path()).model("owner/model", "main");
            let repo_dir = cache.path().join("models--owner--model");

            let interrupted = repo.get("model.bin");
            assert!(matches!(interrupted, Err(HubError::Io { .. })));
            assert_eq!(
                fs::read(repo_dir.join("blobs/abc123.incomplete")).unwrap(),
                b"wei"
            );
            assert!(!repo_dir.join("blobs/abc123").exists());

            let path = repo.get("model.bin").unwrap();

            assert_eq!(fs::read(path).unwrap(), BODY);
            assert!(!repo_dir.join("blobs/abc123.incomplete").exists());
            assert!(
                cache
                    .path()
                    .join(".locks/models--owner--model/abc123.lock")
                    .is_file()
            );
            assert_eq!(
                storage_ranges(&requests),
                [None, Some("bytes=3-".to_owned())]
            );
        }

        #[test]
        fn short_partial_answer_is_resumed_instead_of_published() {
            let (origin, requests) = serve_with(|range| {
                match range {
                // a server may send less than the rest of the file
                Some("bytes=3-") => "HTTP/1.1 206 Partial Content\r\ncontent-range: bytes 3-4/7\r\ncontent-length: 2\r\nconnection: close\r\n\r\ngh"
                    .to_owned(),
                Some("bytes=5-") => "HTTP/1.1 206 Partial Content\r\ncontent-range: bytes 5-6/7\r\ncontent-length: 2\r\nconnection: close\r\n\r\nts"
                    .to_owned(),
                range => panic!("unexpected range {range:?}"),
            }
            });
            let cache = tempfile::tempdir().unwrap();
            let blobs = cache.path().join("models--owner--model/blobs");
            fs::create_dir_all(&blobs).unwrap();
            fs::write(blobs.join("abc123.incomplete"), b"wei").unwrap();
            let repo = hub(&origin, cache.path()).model("owner/model", "main");

            let short = repo.get("model.bin");
            assert!(matches!(short, Err(HubError::Io { .. })));
            assert!(!blobs.join("abc123").exists());

            let path = repo.get("model.bin").unwrap();

            assert_eq!(fs::read(path).unwrap(), BODY);
            assert_eq!(
                storage_ranges(&requests),
                [Some("bytes=3-".to_owned()), Some("bytes=5-".to_owned())]
            );
        }

        #[test]
        fn unusable_range_answer_restarts_the_partial_file() {
            let ignoring = serve();
            let unsatisfiable = serve_with(|range| {
                match range {
                None => format!(
                    "HTTP/1.1 200 OK\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                    BODY.len(),
                    String::from_utf8_lossy(BODY)
                ),
                Some(_) => "HTTP/1.1 416 Range Not Satisfiable\r\ncontent-range: bytes */7\r\ncontent-length: 0\r\nconnection: close\r\n\r\n"
                    .to_owned(),
            }
            });

            let unknown_size = serve_with(|range| {
                match range {
                None => format!(
                    "HTTP/1.1 200 OK\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                    BODY.len(),
                    String::from_utf8_lossy(BODY)
                ),
                Some(_) => "HTTP/1.1 206 Partial Content\r\ncontent-range: bytes 5-6/*\r\ncontent-length: 2\r\nconnection: close\r\n\r\nts"
                    .to_owned(),
            }
            });

            for ((origin, requests), expected) in [
                (ignoring, vec![Some("bytes=5-".to_owned())]),
                (unsatisfiable, vec![Some("bytes=5-".to_owned()), None]),
                (unknown_size, vec![Some("bytes=5-".to_owned()), None]),
            ] {
                let cache = tempfile::tempdir().unwrap();
                let blobs = cache.path().join("models--owner--model/blobs");
                fs::create_dir_all(&blobs).unwrap();
                fs::write(blobs.join("abc123.incomplete"), b"stale").unwrap();

                let path = hub(&origin, cache.path())
                    .model("owner/model", "main")
                    .get("model.bin")
                    .unwrap();

                assert_eq!(fs::read(path).unwrap(), BODY);
                assert_eq!(storage_ranges(&requests), expected);
            }
        }

        #[test]
        fn rejects_invalid_cache_names() {
            for name in ["", "..", "../outside", "a/b", "a\\b"] {
                assert!(!is_blob_name(name), "{name:?}");
                assert!(!is_commit_hash(name), "{name:?}");
            }
            for file in ["", "../outside", "/absolute", "a/../b"] {
                assert!(!is_repo_file(file), "{file:?}");
            }
            assert!(is_blob_name("abc_123-DEF"));
            assert!(is_commit_hash(COMMIT));
            assert!(is_repo_file("nested/weights.bin"));
        }

        #[test]
        fn content_range_needs_a_consistent_known_range() {
            assert_eq!(content_range(" Bytes 3-6/7 "), Some((3, 6, 7)));
            for value in [
                "bytes 3-6/*",
                "bytes 3-/7",
                "bytes 3-2/7",
                "bytes 3-7/7",
                "items 3-6/7",
            ] {
                assert_eq!(content_range(value), None, "{value:?}");
            }
        }

        #[test]
        fn token_cleanup_matches_huggingface_hub() {
            assert_eq!(clean_token("hf_abc\r\n").as_deref(), Some("hf_abc"));
            assert_eq!(clean_token("  \n"), None);
        }

        #[test]
        fn cached_files_resolve_without_requests() {
            let (origin, requests) = serve();
            let cache = tempfile::tempdir().unwrap();
            let hub = hub(&origin, cache.path());
            hub.model("owner/model", "main").get("model.bin").unwrap();
            let before = requests.lock().unwrap().len();

            // through the branch's ref, and by the commit it recorded
            let by_branch = hub.model("owner/model", "main").get("model.bin").unwrap();
            let by_commit = hub.model("owner/model", COMMIT).get("model.bin").unwrap();

            assert_eq!(by_branch, by_commit);
            assert_eq!(requests.lock().unwrap().len(), before);
        }

        #[test]
        fn rejects_names_that_leave_the_cache() {
            let (origin, requests) = serve();
            let cache = tempfile::tempdir().unwrap();
            let hub = hub(&origin, cache.path());

            let escaping = hub.model("owner/model", "main").get("../../outside.bin");
            // the server answers with commit COMMIT, not the pinned one
            let other_commit = "fedcba9876543210fedcba9876543210fedcba98";
            let mismatched = hub.model("owner/model", other_commit).get("model.bin");

            assert!(matches!(escaping, Err(HubError::InvalidFile { .. })));
            assert!(matches!(
                mismatched,
                Err(HubError::InvalidHeader {
                    header: "x-repo-commit",
                    ..
                })
            ));
            assert_eq!(requests.lock().unwrap().len(), 1);
            assert!(!cache.path().join("models--owner--model/snapshots").exists());
        }

        #[test]
        fn error_status_reaches_the_caller() {
            let (origin, _) = serve();
            let cache = tempfile::tempdir().unwrap();
            let repo = hub(&origin, cache.path()).model("owner/missing", "main");

            let error = repo.get("model.bin").unwrap_err();

            assert_eq!(error.status(), Some(404));
            assert!(!cache.path().join("models--owner--missing").exists());
        }
    }
}
