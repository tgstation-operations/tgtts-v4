//! What the service refuses of the effects a request asks for, and what it says.

use std::sync::Arc;

use effects::{
    Pipeline,
    convolve::{Impulses, Response},
    schema::{self, Graph, Method, Node},
};

fn check(json: &str) -> Result<(), String> {
    let value = serde_json::from_str(json).unwrap();
    let graphs = schema::parse(&value).map_err(|e| e.to_string())?;
    let impulses = Impulses {
        synth: Arc::new(Response::new(&[1.0, 0.5], 1, 48000)),
    };
    Pipeline::new(&graphs, &impulses)
        .map(drop)
        .map_err(|e| e.to_string())
}

#[track_caller]
fn refused(json: &str, reason: &str) {
    match check(json) {
        Ok(()) => panic!("accepted {json}"),
        Err(e) => assert_eq!(e, reason, "{json}"),
    }
}

#[track_caller]
fn unparsed(json: &str, start: &str, part: &str) {
    match check(json) {
        Err(e) => assert!(e.starts_with(start) && e.contains(part), "{e}"),
        Ok(()) => panic!("accepted {json}"),
    }
}

fn wired(nodes: &str, edges: &str) -> String {
    format!(r#"[{{"nodes": {{{nodes}}}, "edges": [{edges}]}}]"#)
}

const GAIN: &str = r#"{"effect": "gain", "gain_db": 1}"#;

#[test]
fn accepts_chains_and_wired_graphs() {
    check("[]").unwrap();
    check("[[]]").unwrap();
    check(&format!("[[{GAIN}, {GAIN}], [{GAIN}]]")).unwrap();
    check(&wired("", r#"["input", "output"]"#)).unwrap();
    check(&wired(
        &format!(r#""a": {GAIN}, "m": {{"effect": "mix", "weights": {{"a": 0.5, "input": 0.5}}}}"#),
        r#"["input", "a"], ["a", "m"], ["input", "m"], ["m", "output"]"#,
    ))
    .unwrap();
}

#[test]
fn names_the_graph_the_node_and_the_reason() {
    refused(
        &format!(r#"[[{GAIN}], [{GAIN}, {{"effect": "pitch", "semitones": 30}}]]"#),
        "effects[1], node 1: semitones 30 is not in [-24, 24]",
    );
    refused(
        &wired(
            r#""low": {"effect": "lowpass", "frequency_hz": 5, "q": 1}"#,
            r#"["input", "low"], ["low", "output"]"#,
        ),
        r#"effects[0], node "low": frequency_hz 5 is not in [20, 20000]"#,
    );
    refused(
        r#"[[{"effect": "echo", "taps": []}]]"#,
        "effects[0], node 0: 0 taps, where 1 to 16 are allowed",
    );
}

#[test]
fn refuses_what_does_not_parse() {
    let node = "effects[0], node 0: ";
    unparsed(
        r#"[[{"effect": "reverse"}]]"#,
        node,
        "unknown variant `reverse`",
    );
    unparsed(
        r#"[[{"effect": "gain", "gain_db": 1, "gain": 2}]]"#,
        node,
        "unknown field `gain`",
    );
    unparsed(r#"[[{"effect": "gain"}]]"#, node, "missing field `gain_db`");
    unparsed(
        r#"[{"nodes": {"low": {"effect": "lowpass"}}, "edges": []}]"#,
        r#"effects[0], node "low": "#,
        "missing field `frequency_hz`",
    );
    refused(r#"[{"nodes": {}}]"#, "effects[0]: `edges` is missing");
    refused(
        r#"[{"nodes": {}, "edges": [], "x": 1}]"#,
        "effects[0]: unknown field `x`, expected `nodes` or `edges`",
    );
    unparsed(
        r#"[{"nodes": {}, "edges": [["input"]]}]"#,
        "effects[0]: edges: ",
        "",
    );
    refused(
        r#"[[], "gain"]"#,
        "effects[1]: neither an array of nodes nor an object of nodes and edges",
    );
    refused(r#"{"gain_db": 1}"#, "effects: not an array of graphs");
}

#[test]
fn refuses_nodes_and_taps_written_as_arrays() {
    refused(
        &format!(r#"[[{GAIN}, ["gain", 1]]]"#),
        "effects[0], node 1: invalid type: sequence, expected an object",
    );
    refused(
        &wired(r#""a": ["gain", 1]"#, r#"["input", "a"], ["a", "output"]"#),
        r#"effects[0], node "a": invalid type: sequence, expected an object"#,
    );
    refused(
        r#"[[{"effect": "echo", "taps": [{"delay_ms": 0, "weight": 1}, [100, 0.5]]}]]"#,
        "effects[0], node 0: invalid type: sequence, expected an object",
    );
}

#[test]
fn takes_keep_formants_as_true_or_false_or_as_the_game_writes_them() {
    let pitch = |keep: &str| {
        format!(r#"[[{{"effect": "pitch", "semitones": 2, "keep_formants": {keep}}}]]"#)
    };
    for (keep, kept) in [("true", true), ("false", false), ("1", true), ("0", false)] {
        let value = serde_json::from_str(&pitch(keep)).unwrap();
        let pitch = Node::Pitch {
            semitones: 2.0,
            keep_formants: kept,
            method: Method::Vocoder,
        };
        assert_eq!(
            schema::parse(&value),
            Ok(vec![Graph::Chain(vec![pitch])]),
            "{keep}"
        );
    }
    for keep in ["2", "-1", "0.5", "1.0", r#""true""#, "null"] {
        refused(
            &pitch(keep),
            &format!("effects[0], node 0: keep_formants {keep} is not true, false, 1 or 0"),
        );
    }
}

#[test]
fn refuses_graphs_wired_wrong() {
    let a = format!(r#""a": {GAIN}"#);
    let ab = format!(r#""a": {GAIN}, "b": {GAIN}"#);
    refused(
        &wired(&a, r#"["input", "a"]"#),
        "effects[0]: the output has 0 edges in, where it takes exactly one",
    );
    refused(
        &wired(
            &a,
            r#"["input", "a"], ["a", "output"], ["input", "output"]"#,
        ),
        "effects[0]: the output has 2 edges in, where it takes exactly one",
    );
    refused(
        &wired(
            &ab,
            r#"["input", "a"], ["b", "a"], ["a", "output"], ["input", "b"]"#,
        ),
        r#"effects[0], node "a": has 2 edges in, where only a mix takes more than one"#,
    );
    refused(
        &wired(&a, r#"["input", "z"], ["a", "output"]"#),
        r#"effects[0], node edge ["input", "z"]: there is no node "z""#,
    );
    refused(
        &wired(&a, r#"["input", "a"], ["input", "a"], ["a", "output"]"#),
        r#"effects[0], node edge ["input", "a"]: is given twice"#,
    );
    refused(
        &wired(&ab, r#"["input", "a"], ["a", "output"]"#),
        r#"effects[0], node "b": has no edge in"#,
    );
    refused(
        &wired(&ab, r#"["input", "a"], ["a", "output"], ["a", "b"]"#),
        r#"effects[0], node "b": does not lead to the output"#,
    );
    refused(
        &wired(
            &format!(r#"{ab}, "m": {{"effect": "mix", "weights": {{"input": 1, "b": 1}}}}"#),
            r#"["input", "m"], ["b", "m"], ["m", "a"], ["a", "b"], ["a", "output"]"#,
        ),
        r#"effects[0], node "a": is on a cycle"#,
    );
    refused(
        &wired(
            &format!(r#"{a}, "m": {{"effect": "mix", "weights": {{"input": 1}}}}"#),
            r#"["input", "a"], ["a", "m"], ["input", "m"], ["m", "output"]"#,
        ),
        r#"effects[0], node "m": is weighted for {"input"}, but has edges in from {"a", "input"}"#,
    );
    refused(
        &wired(&format!(r#""input": {GAIN}"#), r#"["input", "output"]"#),
        r#"effects[0], node "input": is not 1 to 32 letters, digits, _ or -, other than input and output"#,
    );
}

#[test]
fn refuses_requests_past_the_limits() {
    let graphs = vec!["[]"; 17].join(", ");
    refused(
        &format!("[{graphs}]"),
        "effects: 17 graphs, where at most 16 are allowed",
    );
    let chain = vec![GAIN; 33].join(", ");
    refused(
        &format!("[[{chain}]]"),
        "effects[0]: 33 nodes, where at most 32 are allowed",
    );
    let chain = vec![GAIN; 22].join(", ");
    refused(
        &format!("[[{chain}], [{chain}], [{chain}]]"),
        "effects: 66 nodes, where at most 64 are allowed in all",
    );
    let slow = r#"{"effect": "tempo", "ratio": 0.5}"#;
    check(&format!("[[{slow}, {slow}]]")).unwrap();
    refused(
        &format!("[[{slow}], [{slow}, {slow}]]"),
        "effects: the audio would be 8.00 times as long and 0.00 s longer, where 4 times and 5 s \
         are allowed",
    );
    let echo = r#"{"effect": "echo", "taps": [{"delay_ms": 2000, "weight": 1}]}"#;
    refused(
        &format!("[[{echo}, {slow}, {echo}]]"),
        "effects: the audio would be 2.00 times as long and 6.00 s longer, where 4 times and 5 s \
         are allowed",
    );
}
