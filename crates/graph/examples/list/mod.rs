//! The lists of values the timing examples take an option of, such as `--batch 1,2,4`.

use std::str::FromStr;

/// Numbers separated by commas: none for an empty string.
pub struct List(pub Vec<usize>);

impl FromStr for List {
    type Err = String;

    fn from_str(text: &str) -> Result<Self, String> {
        let values = text.split(',').filter(|v| !v.is_empty());
        let parsed = values.map(|v| v.parse().map_err(|e| format!("{v}: {e}")));
        parsed.collect::<Result<_, _>>().map(List)
    }
}
