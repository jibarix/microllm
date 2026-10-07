//! Minimal JSON reading (no serde), for commands that read back a finished run's own files
//! (meta.js, model.js, snaps/*.js). Only what those files contain: objects, arrays, strings,
//! numbers, booleans and null; long number arrays packed as {"f32"|"f64": base64}.

pub enum Json {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

impl Json {
    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Obj(kv) => kv.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Json::Num(x) => Some(*x),
            _ => None,
        }
    }

    pub fn as_usize(&self) -> Option<usize> {
        self.as_f64().filter(|x| *x >= 0.0 && x.fract() == 0.0).map(|x| x as usize)
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Json::Bool(b) => Some(*b),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Json::Str(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_arr(&self) -> Option<&[Json]> {
        match self {
            Json::Arr(a) => Some(a),
            _ => None,
        }
    }

    /// A number array: plain (null → NaN, as the writer turns non-finite numbers into null)
    /// or packed as {"f32": base64} / {"f64": base64} of little-endian values.
    pub fn numbers(&self) -> Option<Vec<f64>> {
        if let Some(a) = self.as_arr() {
            return a.iter().map(|v| if let Json::Null = v { Some(f64::NAN) } else { v.as_f64() }).collect();
        }
        if let Some(b) = self.get("f32").and_then(Json::as_str) {
            let bytes = base64_decode(b)?;
            return (bytes.len() % 4 == 0).then(|| bytes.chunks(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64).collect());
        }
        if let Some(b) = self.get("f64").and_then(Json::as_str) {
            let bytes = base64_decode(b)?;
            return (bytes.len() % 8 == 0).then(|| bytes.chunks(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect());
        }
        None
    }

    pub fn usizes(&self) -> Option<Vec<usize>> {
        self.as_arr()?.iter().map(Json::as_usize).collect()
    }
}

/// The JSON value inside a run file: `window.X = {...};` or `window.F({...});`.
pub fn parse_js(text: &str) -> Result<Json, String> {
    let a = text.find('{').ok_or("no JSON object in the file")?;
    let b = text.rfind('}').ok_or("no JSON object in the file")?;
    parse(&text[a..=b])
}

pub fn parse(s: &str) -> Result<Json, String> {
    let mut p = Parser { b: s.as_bytes(), i: 0 };
    let v = p.value()?;
    p.ws();
    if p.i != p.b.len() {
        return Err(p.err("trailing characters"));
    }
    Ok(v)
}

struct Parser<'a> {
    b: &'a [u8],
    i: usize,
}

impl Parser<'_> {
    fn err(&self, what: &str) -> String {
        format!("JSON: {what} at byte {}", self.i)
    }

    fn ws(&mut self) {
        while self.i < self.b.len() && matches!(self.b[self.i], b' ' | b'\t' | b'\n' | b'\r') {
            self.i += 1;
        }
    }

    fn lit(&mut self, word: &str, v: Json) -> Result<Json, String> {
        if self.b[self.i..].starts_with(word.as_bytes()) {
            self.i += word.len();
            Ok(v)
        } else {
            Err(self.err("unexpected character"))
        }
    }

    fn value(&mut self) -> Result<Json, String> {
        self.ws();
        match self.b.get(self.i) {
            None => Err(self.err("unexpected end")),
            Some(b'{') => {
                self.i += 1;
                let mut kv = Vec::new();
                self.ws();
                if self.b.get(self.i) == Some(&b'}') {
                    self.i += 1;
                    return Ok(Json::Obj(kv));
                }
                loop {
                    self.ws();
                    if self.b.get(self.i) != Some(&b'"') {
                        return Err(self.err("expected a key"));
                    }
                    let k = self.string()?;
                    self.ws();
                    if self.b.get(self.i) != Some(&b':') {
                        return Err(self.err("expected ':'"));
                    }
                    self.i += 1;
                    kv.push((k, self.value()?));
                    self.ws();
                    match self.b.get(self.i) {
                        Some(b',') => self.i += 1,
                        Some(b'}') => {
                            self.i += 1;
                            return Ok(Json::Obj(kv));
                        }
                        _ => return Err(self.err("expected ',' or '}'")),
                    }
                }
            }
            Some(b'[') => {
                self.i += 1;
                let mut a = Vec::new();
                self.ws();
                if self.b.get(self.i) == Some(&b']') {
                    self.i += 1;
                    return Ok(Json::Arr(a));
                }
                loop {
                    a.push(self.value()?);
                    self.ws();
                    match self.b.get(self.i) {
                        Some(b',') => self.i += 1,
                        Some(b']') => {
                            self.i += 1;
                            return Ok(Json::Arr(a));
                        }
                        _ => return Err(self.err("expected ',' or ']'")),
                    }
                }
            }
            Some(b'"') => Ok(Json::Str(self.string()?)),
            Some(b't') => self.lit("true", Json::Bool(true)),
            Some(b'f') => self.lit("false", Json::Bool(false)),
            Some(b'n') => self.lit("null", Json::Null),
            Some(_) => {
                let s = self.i;
                while self.i < self.b.len() && matches!(self.b[self.i], b'-' | b'+' | b'.' | b'e' | b'E' | b'0'..=b'9') {
                    self.i += 1;
                }
                let t = std::str::from_utf8(&self.b[s..self.i]).map_err(|_| self.err("bad number"))?;
                t.parse().map(Json::Num).map_err(|_| self.err("bad number"))
            }
        }
    }

    /// A string; `self.i` is at its opening quote.
    fn string(&mut self) -> Result<String, String> {
        self.i += 1;
        let s = self.i;
        // the common case, no escapes: one slice
        while self.i < self.b.len() && self.b[self.i] != b'"' && self.b[self.i] != b'\\' {
            self.i += 1;
        }
        let mut out = std::str::from_utf8(&self.b[s..self.i]).map_err(|_| self.err("bad UTF-8"))?.to_string();
        loop {
            match self.b.get(self.i) {
                None => return Err(self.err("unterminated string")),
                Some(b'"') => {
                    self.i += 1;
                    return Ok(out);
                }
                Some(b'\\') => {
                    let c = *self.b.get(self.i + 1).ok_or_else(|| self.err("unterminated string"))?;
                    self.i += 2;
                    match c {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{8}'),
                        b'f' => out.push('\u{c}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let mut cp = self.hex4()?;
                            if (0xD800..0xDC00).contains(&cp) && self.b[self.i..].starts_with(b"\\u") {
                                self.i += 2;
                                let lo = self.hex4()?;
                                cp = 0x10000 + ((cp - 0xD800) << 10) + (lo.wrapping_sub(0xDC00) & 0x3FF);
                            }
                            out.push(char::from_u32(cp).unwrap_or('\u{FFFD}'));
                        }
                        _ => return Err(self.err("bad escape")),
                    }
                }
                Some(_) => {
                    let s = self.i;
                    while self.i < self.b.len() && self.b[self.i] != b'"' && self.b[self.i] != b'\\' {
                        self.i += 1;
                    }
                    out.push_str(std::str::from_utf8(&self.b[s..self.i]).map_err(|_| self.err("bad UTF-8"))?);
                }
            }
        }
    }

    fn hex4(&mut self) -> Result<u32, String> {
        let h = self.b.get(self.i..self.i + 4).ok_or_else(|| self.err("bad \\u escape"))?;
        self.i += 4;
        u32::from_str_radix(std::str::from_utf8(h).map_err(|_| self.err("bad \\u escape"))?, 16).map_err(|_| self.err("bad \\u escape"))
    }
}

/// Standard base64 with padding (as `trace::base64` writes it).
pub fn base64_decode(s: &str) -> Option<Vec<u8>> {
    let val = |c: u8| -> Option<u32> {
        Some(match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            _ => return None,
        } as u32)
    };
    let b = s.as_bytes();
    if b.len() % 4 != 0 {
        return None;
    }
    let mut out = Vec::with_capacity(b.len() / 4 * 3);
    for ch in b.chunks(4) {
        let pad = ch.iter().rev().take_while(|&&c| c == b'=').count();
        let mut n = 0u32;
        for &c in &ch[..4 - pad] {
            n = n << 6 | val(c)?;
        }
        n <<= 6 * pad as u32;
        out.extend_from_slice(&n.to_be_bytes()[1..4 - pad]);
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trace::{base64, packed, packed64};

    #[test]
    fn parses_what_the_trace_writes() {
        let xs: Vec<f64> = (0..40).map(|i| i as f64 * 0.25 - 3.0).collect();
        let text = format!("window.X = {{\"a\":[1,-2.5e3,null,true],\"s\":\"q\\\"\\u00e9\\ud83d\\ude00\",\"p\":{},\"q\":{}}};\n", packed(&xs), packed64(&xs));
        let v = parse_js(&text).unwrap();
        let a = v.get("a").unwrap().as_arr().unwrap();
        assert_eq!(a[1].as_f64(), Some(-2500.0));
        assert!(matches!(a[2], Json::Null) && a[3].as_bool() == Some(true));
        assert_eq!(v.get("s").unwrap().as_str(), Some("q\"é😀"));
        assert_eq!(v.get("p").unwrap().numbers().unwrap(), xs);
        assert_eq!(v.get("q").unwrap().numbers().unwrap(), xs);
    }

    #[test]
    fn base64_round_trips() {
        for n in 0..10 {
            let bytes: Vec<u8> = (0..n).map(|i| (i * 37 + 11) as u8).collect();
            assert_eq!(base64_decode(&base64(&bytes)).unwrap(), bytes);
        }
    }
}
