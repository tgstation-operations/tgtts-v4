//! The segmenter against pysbd, on the corpus the fixture holds.

use dsp::sentences::segment;

#[test]
fn matches_pysbd() {
    let fixture = testkit::fixture!("dsp_sentences");
    let metadata = fixture.metadata();
    let texts: Vec<String> = serde_json::from_str(&metadata["texts"]).unwrap();
    let expected: Vec<Vec<String>> = serde_json::from_str(&metadata["sentences"]).unwrap();
    assert_eq!(texts.len(), expected.len());
    let mismatches: Vec<String> = texts
        .iter()
        .zip(&expected)
        .filter_map(|(text, expected)| {
            let actual = segment(text);
            (&actual != expected)
                .then(|| format!("{text:?}\n  pysbd: {expected:?}\n  here:  {actual:?}"))
        })
        .collect();
    assert!(
        mismatches.is_empty(),
        "{} of {} texts split otherwise:\n{}",
        mismatches.len(),
        texts.len(),
        mismatches[..mismatches.len().min(10)].join("\n")
    );
}
