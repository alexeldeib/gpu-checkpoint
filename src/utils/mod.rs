pub fn format_memory(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KiB", "MiB", "GiB", "TiB", "PiB"];

    if bytes == 0 {
        return "0 B".to_string();
    }

    let mut size = bytes as f64;
    let mut unit_index = 0;

    while size >= 1024.0 && unit_index < UNITS.len() - 1 {
        size /= 1024.0;
        unit_index += 1;
    }

    if unit_index == 0 {
        format!("{} {}", bytes, UNITS[unit_index])
    } else {
        format!("{:.2} {}", size, UNITS[unit_index])
    }
}

pub fn format_duration(ms: u64) -> String {
    if ms < 1000 {
        format!("{}ms", ms)
    } else if ms < 60_000 {
        format!("{:.1}s", ms as f64 / 1000.0)
    } else {
        let minutes = ms / 60_000;
        let seconds = (ms % 60_000) / 1000;
        format!("{}m{}s", minutes, seconds)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_memory() {
        assert_eq!(format_memory(0), "0 B");
        assert_eq!(format_memory(512), "512 B");
        assert_eq!(format_memory(1024), "1.00 KiB");
        assert_eq!(format_memory(1536), "1.50 KiB");
        assert_eq!(format_memory(1024 * 1024), "1.00 MiB");
        assert_eq!(format_memory(1024 * 1024 * 1024), "1.00 GiB");
        assert_eq!(format_memory(1024u64.pow(4)), "1.00 TiB");
        assert_eq!(format_memory(1024u64.pow(5)), "1.00 PiB");
    }

    #[test]
    fn test_format_duration() {
        assert_eq!(format_duration(0), "0ms");
        assert_eq!(format_duration(500), "500ms");
        assert_eq!(format_duration(1000), "1.0s");
        assert_eq!(format_duration(1500), "1.5s");
        assert_eq!(format_duration(60_000), "1m0s");
        assert_eq!(format_duration(65_500), "1m5s");
        assert_eq!(format_duration(125_000), "2m5s");
    }
}
