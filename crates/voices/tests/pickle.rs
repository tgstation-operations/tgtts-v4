use std::collections::BTreeMap;

use voices::pickle::{Error, ErrorKind, Value, load};

const PROTO: &[u8] = b"\x80\x02";
const ENCODE: &[u8] = b"c_codecs\nencode\n";
const BYTES_IO: &[u8] = b"c_io\nBytesIO\n";

/// BINUNICODE of `string`.
fn unicode(string: &str) -> Vec<u8> {
    let len = u32::try_from(string.len()).unwrap();
    [b"X", &len.to_le_bytes()[..], string.as_bytes()].concat()
}

/// `_codecs.encode(string, encoding)`, as protocol 2 pickles a bytes.
fn encode(string: &str, encoding: &str) -> Vec<u8> {
    [ENCODE, &unicode(string), &unicode(encoding), b"\x86R"].concat()
}

fn pickle(parts: &[&[u8]]) -> Vec<u8> {
    [&[PROTO], parts, &[b"."]].concat().concat()
}

fn refusal(pickle: &[u8]) -> (usize, String) {
    let error = load(pickle).unwrap_err();
    (error.offset, error.to_string())
}

#[test]
fn bytes_io_holds_the_latin1_bytes() {
    let state = [&encode("\u{ff}\0A", "latin1")[..], b"K\x00N\x87"].concat();
    let value = load(&pickle(&[BYTES_IO, b")\x81", &state, b"b"])).unwrap();
    assert_eq!(value, Value::BytesIo(vec![0xff, 0x00, 0x41]));
}

#[test]
fn dict_keys_come_from_the_memo() {
    let value = load(&pickle(&[
        b"}q\x00(",
        &unicode("a"),
        b"q\x01}r\x2c\x01\x00\x00(h\x01K\x07uu",
    ]))
    .unwrap();
    let inner = Value::Dict(BTreeMap::from([("a".to_owned(), Value::Int(7))]));
    assert_eq!(
        value,
        Value::Dict(BTreeMap::from([("a".to_owned(), inner)]))
    );
}

#[test]
fn globals_other_than_the_two_are_refused() {
    for (module, name) in [
        ("os", "system"),
        ("posix", "system"),
        ("builtins", "eval"),
        ("__builtin__", "bytes"),
        ("io", "BytesIO"),
        ("_codecs", "decode"),
    ] {
        let global = format!("c{module}\n{name}\n");
        let call = [global.as_bytes(), &unicode("true"), b"\x85R"].concat();
        let error = load(&pickle(&[&call])).unwrap_err();
        let kind = ErrorKind::Global {
            module: module.to_owned(),
            name: name.to_owned(),
        };
        assert_eq!(error, Error { offset: 2, kind });
    }
    assert_eq!(
        refusal(&pickle(&[b"cos\nsystem\n"])).1,
        "pickle offset 2: global os.system is not allowed"
    );
}

#[test]
fn opcodes_the_files_do_not_use_are_refused() {
    let stack_global = [&unicode("os")[..], &unicode("system"), b"\x93"].concat();
    assert_eq!(
        refusal(&pickle(&[&stack_global])),
        (
            20,
            "pickle offset 20: opcode 0x93 is not supported".to_owned()
        )
    );
    // BINPERSID, INST, OBJ, LONG_BINGET, SETITEM, BINBYTES, SHORT_BINUNICODE, FRAME, TUPLE1,
    // EXT1, INT, BINFLOAT, APPEND.
    for opcode in b"QiojsB\x8c\x95\x85\x82IGa" {
        let error = load(&pickle(&[&[*opcode]])).unwrap_err();
        let kind = ErrorKind::Opcode(*opcode);
        assert_eq!(error, Error { offset: 2, kind });
    }
}

#[test]
fn proto_2_starts_the_pickle() {
    let protocol_0 = b"cos\nsystem\n(S'true'\ntR.";
    assert_eq!(
        refusal(protocol_0),
        (0, "pickle offset 0: no PROTO at the start".to_owned())
    );
    assert_eq!(
        refusal(&pickle(&[b"N\x80\x02"])).1,
        "pickle offset 3: PROTO after the start"
    );
}

#[test]
fn only_latin1_makes_bytes() {
    assert_eq!(
        refusal(&pickle(&[&encode("a", "utf-8")])),
        (
            35,
            "pickle offset 35: _codecs.encode to \"utf-8\"".to_owned()
        )
    );
    assert_eq!(
        refusal(&pickle(&[&encode("\u{100}", "latin1")])).1,
        "pickle offset 37: _codecs.encode of a str beyond latin1"
    );
}

#[test]
fn bytes_io_is_made_only_by_newobj_with_its_state() {
    assert_eq!(
        refusal(&pickle(&[BYTES_IO, b")R"])).1,
        "pickle offset 16: REDUCE of _io.BytesIO"
    );
    let state = [&encode("", "latin1")[..], b"K\x01N\x87"].concat();
    assert_eq!(
        refusal(&pickle(&[BYTES_IO, b")\x81", &state, b"b"])).1,
        "pickle offset 55: BytesIO state other than (bytes, 0, None)"
    );
    assert_eq!(
        refusal(&pickle(&[b"}N", b"b"])).1,
        "pickle offset 4: BUILD of a dict"
    );
}

#[test]
fn what_can_change_is_not_fetched_from_the_memo() {
    assert_eq!(
        refusal(&pickle(&[b"}q\x00h\x00"])).1,
        "pickle offset 5: BINGET 0 of a dict or BytesIO"
    );
    assert_eq!(
        refusal(&pickle(&[b"N}\x86q\x00h\x00"])).1,
        "pickle offset 7: BINGET 0 of a dict or BytesIO"
    );
    assert_eq!(
        refusal(&pickle(&[b"Nq\x00h\x01"])).1,
        "pickle offset 5: BINGET 1, which was never put"
    );
}

#[test]
fn stack_is_not_popped_past_a_mark() {
    assert_eq!(
        refusal(&pickle(&[b"N(N\x86"])).1,
        "pickle offset 5: TUPLE2 with fewer values on the stack"
    );
}

#[test]
fn dict_keys_are_strs_set_once() {
    assert_eq!(
        refusal(&pickle(&[b"}(K\x01Nu"])).1,
        "pickle offset 7: dict key an int"
    );
    let twice = [&b"}("[..], &unicode("a"), b"N", &unicode("a"), b"Nu"].concat();
    assert_eq!(
        refusal(&pickle(&[&twice])).1,
        "pickle offset 18: dict key \"a\" set twice"
    );
}

#[test]
fn truncated_pickle_is_refused() {
    assert_eq!(
        refusal(b"\x80\x02X\x05\x00\x00\x00ab"),
        (2, "pickle offset 2: truncated".to_owned())
    );
    assert_eq!(refusal(b"\x80\x02}").0, 3);
}

#[test]
fn stop_ends_the_pickle_with_one_value() {
    assert_eq!(
        refusal(b"\x80\x02}.."),
        (3, "pickle offset 3: bytes after STOP".to_owned())
    );
    assert_eq!(
        refusal(b"\x80\x02}}.").1,
        "pickle offset 4: STOP with other than one value on the stack"
    );
}
