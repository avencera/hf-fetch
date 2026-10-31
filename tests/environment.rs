use std::path::Path;
use std::process::Command;

fn child(case: &str, home: &Path) -> Command {
    let mut command = Command::new(std::env::current_exe().unwrap());
    command.args(["--ignored", "--exact", "environment_child", "--nocapture"]);
    for name in [
        "HF_ENDPOINT",
        "HF_TOKEN",
        "HF_TOKEN_PATH",
        "HF_HOME",
        "HF_HUB_CACHE",
        "HUGGINGFACE_HUB_CACHE",
        "XDG_CACHE_HOME",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ] {
        command.env_remove(name);
    }
    command.env("HOME", home).env("USERPROFILE", home);
    command.env("HF_FETCH_TEST_CASE", case);
    command
}

#[test]
fn resolves_cache_environment_in_a_separate_process() {
    let home = tempfile::tempdir().unwrap();
    let cases = [
        (
            "default",
            Vec::new(),
            home.path().join(".cache/huggingface/hub"),
        ),
        (
            "xdg",
            vec![("XDG_CACHE_HOME", "~/xdg")],
            home.path().join("xdg/huggingface/hub"),
        ),
        (
            "home",
            vec![("XDG_CACHE_HOME", "~/xdg"), ("HF_HOME", "~/hf")],
            home.path().join("hf/hub"),
        ),
        (
            "legacy",
            vec![("HF_HOME", "~/hf"), ("HUGGINGFACE_HUB_CACHE", "~/legacy")],
            home.path().join("legacy"),
        ),
        (
            "cache",
            vec![
                ("HUGGINGFACE_HUB_CACHE", "~/legacy"),
                ("HF_HUB_CACHE", "~/cache"),
            ],
            home.path().join("cache"),
        ),
        (
            "empty",
            vec![("HF_HUB_CACHE", ""), ("HF_HOME", "")],
            home.path().join(".cache/huggingface/hub"),
        ),
    ];
    for (label, env, expected) in cases {
        let output = child("cache", home.path())
            .envs(env)
            .env("HF_FETCH_EXPECTED", expected)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{label}: {}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

#[cfg(feature = "online")]
#[test]
fn public_client_resolves_endpoint_and_tokens() {
    use std::io::{BufRead, BufReader, Write};
    use std::net::TcpListener;
    use std::thread;
    use std::time::Duration;

    for mode in ["env", "file", "fallback"] {
        let home = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(home.path().join("hf")).unwrap();
        std::fs::write(home.path().join("hf/token"), " hf_file\r\n").unwrap();
        std::fs::write(home.path().join("token"), " hf_path\r\n").unwrap();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = format!("http://{}/", listener.local_addr().unwrap());
        let (send, receive) = std::sync::mpsc::channel();
        thread::spawn(move || {
            for index in 0..2 {
                let (mut stream, _) = listener.accept().unwrap();
                stream
                    .set_read_timeout(Some(Duration::from_secs(10)))
                    .unwrap();
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                let mut line = String::new();
                reader.read_line(&mut line).unwrap();
                assert_eq!(
                    line.trim(),
                    "GET /owner/model/resolve/main/config.json HTTP/1.1"
                );
                let mut auth = None;
                loop {
                    line.clear();
                    reader.read_line(&mut line).unwrap();
                    if line.trim().is_empty() {
                        break;
                    }
                    if let Some((name, value)) = line.split_once(':')
                        && name.eq_ignore_ascii_case("authorization")
                    {
                        auth = Some(value.trim().to_owned());
                    }
                }
                send.send(auth).unwrap();
                let body = if index == 0 { "w" } else { "weights" };
                write!(stream, "HTTP/1.1 200 OK\r\nx-repo-commit: 0123456789abcdef0123456789abcdef01234567\r\netag: \"abc123\"\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}", body.len()).unwrap();
            }
        });
        let mut command = child("download", home.path());
        command
            .env("HF_ENDPOINT", endpoint)
            .env("HF_HOME", "~/hf")
            .env("HF_HUB_CACHE", home.path().join("cache"));
        let token = match mode {
            "env" => {
                command.env("HF_TOKEN", " hf_env\r\n");
                "hf_env"
            }
            "file" => {
                command.env("HF_TOKEN_PATH", "~/token");
                "hf_path"
            }
            _ => {
                command.env("HF_TOKEN", " \r\n");
                "hf_file"
            }
        };
        let output = command.output().unwrap();
        assert!(
            output.status.success(),
            "{mode}: {}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        for _ in 0..2 {
            assert_eq!(
                receive
                    .recv_timeout(Duration::from_secs(10))
                    .unwrap()
                    .as_deref(),
                Some(format!("Bearer {token}").as_str())
            );
        }
    }
}

// each child receives its environment before it starts; no test mutates process state
#[test]
#[ignore]
fn environment_child() {
    match std::env::var("HF_FETCH_TEST_CASE").unwrap().as_str() {
        "cache" => assert_eq!(
            hf_fetch::default_hub_cache_dir(),
            std::path::PathBuf::from(std::env::var_os("HF_FETCH_EXPECTED").unwrap())
        ),
        #[cfg(feature = "online")]
        "download" => {
            let repo = hf_fetch::Hub::from_env()
                .unwrap()
                .model("owner/model", "main");
            let path = repo.get("config.json").unwrap();
            assert_eq!(std::fs::read(&path).unwrap(), b"weights");
            assert_eq!(repo.cached("config.json"), Some(path.clone()));
            assert_eq!(repo.get("config.json").unwrap(), path);
            assert_eq!(
                std::fs::read_to_string(
                    hf_fetch::default_hub_cache_dir().join("models--owner--model/refs/main")
                )
                .unwrap(),
                "0123456789abcdef0123456789abcdef01234567"
            );
        }
        case => panic!("unknown case: {case}"),
    }
}
