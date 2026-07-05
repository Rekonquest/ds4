// DS4 (DwarfStar) — SSE writer + UTF-8 safe stream chunking.
//
// Port of `utf8_stream_safe_len` from `ds4_server.c`. The challenge
// is that a tokenizer can split a multi-byte UTF-8 character across
// two tokens; if an SSE delta ends mid-character, some clients
// replace the incomplete byte sequence with U+FFFD and the prefix
// round-trip corrupts KV cache matches. We hold only the trailing
// incomplete character; the next generated token will complete it.

// Several helpers here are exercised only by the lib target's
// integration tests; silence dead-code for the module.

/// Return the largest prefix of `s[..limit]` that ends on a UTF-8
/// character boundary. When `final == true` we always return
/// `limit`; when `final == false` we may return a smaller value.
pub fn utf8_stream_safe_len(s: &[u8], start: usize, limit: usize, final_chunk: bool) -> usize {
    if final_chunk || s.is_empty() || limit <= start {
        return limit;
    }
    let mut p = limit;
    let mut cont = 0usize;
    while p > start && cont < 4 && ((s[p - 1] & 0xc0) == 0x80) {
        p -= 1;
        cont += 1;
    }
    if p == limit {
        // No trailing continuation bytes.
        let last = s[limit - 1];
        if utf8_expected_len(last) > 1 {
            return limit - 1;
        }
        return limit;
    }
    if p == start && ((s[p] & 0xc0) == 0x80) {
        return start;
    }
    let lead = p - 1;
    let need = utf8_expected_len(s[lead]);
    if (limit - lead) < need {
        lead
    } else {
        limit
    }
}

fn utf8_expected_len(c: u8) -> usize {
    match c {
        0x00..=0x7f => 1,
        0xc2..=0xdf => 2,
        0xe0..=0xef => 3,
        0xf0..=0xf4 => 4,
        _ => 1,
    }
}

/// SSE writer — accumulates bytes per chunk and flushes them with
/// the SSE framing on demand. We deliberately keep this synchronous
/// so hyper's `body` channel can wrap it.
#[derive(Debug, Default)]
pub struct SseWriter {
    buffer: Vec<u8>,
}

impl SseWriter {
    pub fn new() -> Self {
        Self::default()
    }

    /// OpenAI-style: `data: {json}\n\n`.
    pub fn push_data_json(&mut self, payload: &str) {
        self.buffer.extend_from_slice(b"data: ");
        self.buffer.extend_from_slice(payload.as_bytes());
        self.buffer.extend_from_slice(b"\n\n");
    }

    /// Anthropic-style: `event: <name>\ndata: {json}\n\n`.
    pub fn push_event(&mut self, event: &str, payload: &str) {
        self.buffer.extend_from_slice(b"event: ");
        self.buffer.extend_from_slice(event.as_bytes());
        self.buffer.extend_from_slice(b"\ndata: ");
        self.buffer.extend_from_slice(payload.as_bytes());
        self.buffer.extend_from_slice(b"\n\n");
    }

    /// Append a comment (`: keep-alive\n\n`).
    pub fn push_comment(&mut self, comment: &str) {
        self.buffer.extend_from_slice(b": ");
        self.buffer.extend_from_slice(comment.as_bytes());
        self.buffer.extend_from_slice(b"\n\n");
    }

    /// `data: [DONE]\n\n` terminator (OpenAI convention).
    pub fn push_done(&mut self) {
        self.buffer.extend_from_slice(b"data: [DONE]\n\n");
    }

    /// Take the accumulated bytes, leaving the writer empty.
    pub fn take(&mut self) -> Vec<u8> {
        std::mem::take(&mut self.buffer)
    }

    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn utf8_stream_safe_len_returns_whole_when_complete() {
        let s = "hello world".as_bytes();
        assert_eq!(utf8_stream_safe_len(s, 0, s.len(), false), s.len());
    }

    #[test]
    fn utf8_stream_safe_len_holds_trailing_partial() {
        // Two-byte character: 'à' = 0xC3 0xA0. Limit at 0xC3 only.
        let s: &[u8] = b"hello \xC3\xA0";
        let limit = s.len() - 1; // chops the trailing byte
        assert_eq!(utf8_stream_safe_len(s, 0, limit, false), limit - 1);
    }

    #[test]
    fn utf8_stream_safe_len_final_returns_limit() {
        let s: &[u8] = b"x\xC3";
        assert_eq!(utf8_stream_safe_len(s, 0, s.len(), true), s.len());
    }

    #[test]
    fn utf8_stream_safe_len_four_byte() {
        // 4-byte: 😀 = F0 9F 98 80.
        let s = "hi 😀".as_bytes();
        // Find the start of 😀
        let start = 3usize;
        // Limit at the first byte (F0) only — 1 of 4.
        assert_eq!(utf8_stream_safe_len(s, start, start + 1, false), start);
    }

    #[test]
    fn sse_writer_basic_format() {
        let mut w = SseWriter::new();
        w.push_data_json("{\"x\":1}");
        let out = w.take();
        assert_eq!(out, b"data: {\"x\":1}\n\n".to_vec());
        assert!(w.is_empty());
    }

    #[test]
    fn sse_writer_event_format() {
        let mut w = SseWriter::new();
        w.push_event("ping", "{}");
        let out = w.take();
        assert_eq!(out, b"event: ping\ndata: {}\n\n".to_vec());
    }

    #[test]
    fn sse_writer_done() {
        let mut w = SseWriter::new();
        w.push_done();
        let out = w.take();
        assert_eq!(out, b"data: [DONE]\n\n".to_vec());
    }
}
