// DS4 (DwarfStar) — DSML parser + transcoder.
//
// The DeepSeek tool-call mini-language looks like:
//
//     <|DSML|tool_calls>
//       <|DSML|invoke name="$TOOL_NAME">
//         <|DSML|parameter name="$P">value</|DSML|parameter>
//       </|DSML|invoke>
//     </|DSML|tool_calls>
//
// (The pipe characters in the upstream C are actually `｜` U+FF5C —
// we accept either `|` or `｜` to be lenient about copy-paste.)
//
// Many of these helpers are only exercised by the lib target's
// integration tests; silence dead-code for the whole module.
// This module:
//   * parses model output into a structured `DsmlToolCall` list
//   * renders the same calls back to the canonical text form
//   * transcodes between DSML and the OpenAI / Anthropic JSON
//     tool-call shapes used by the HTTP layer.

use serde::{Deserialize, Serialize};

pub const OPEN_TAG: &str = "<|DSML|tool_calls>";
pub const CLOSE_TAG: &str = "</|DSML|tool_calls>";
pub const INVOKE_OPEN_PREFIX: &str = "<|DSML|invoke name=\"";
pub const INVOKE_OPEN_SUFFIX: &str = "\">";
pub const INVOKE_CLOSE: &str = "</|DSML|invoke>";
pub const PARAMETER_OPEN_PREFIX: &str = "<|DSML|parameter name=\"";
pub const PARAMETER_OPEN_MIDDLE: &str = "\" string=\"";
pub const PARAMETER_OPEN_SUFFIX: &str = "\">";
pub const PARAMETER_CLOSE: &str = "</|DSML|parameter>";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DsmlParameter {
    pub name: String,
    /// Whether the value is a raw string (true) or a JSON literal
    /// (false). Mirrors the `string="true|false"` attribute on the
    /// upstream `<|DSML|parameter>` tag.
    pub is_string: bool,
    pub value: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DsmlToolCall {
    pub name: String,
    pub parameters: Vec<DsmlParameter>,
}

#[derive(Debug, thiserror::Error)]
pub enum DsmlError {
    #[error("dsml: missing required attribute {attr} on tag near offset {offset}")]
    MissingAttr { attr: &'static str, offset: usize },
    #[error("dsml: unexpected token at offset {offset}: {msg}")]
    Unexpected { offset: usize, msg: String },
    #[error("dsml: unterminated tag starting at offset {offset}")]
    Unterminated { offset: usize },
}

pub fn parse(input: &str) -> Result<Vec<DsmlToolCall>, DsmlError> {
    let mut out: Vec<DsmlToolCall> = Vec::new();
    let bytes = input.as_bytes();
    let mut cursor = 0;
    while let Some(rel) = find_open_tag(&input[cursor..]) {
        let start = cursor + rel;
        // Advance past whichever form of the open tag matched
        // (ASCII or full-width — they have different byte lengths).
        cursor = start + open_tag_len(&input[start..]);
        let mut closed_outer = false;
        // Walk until close tag or EOF.
        while cursor < bytes.len() {
            // Try the invoke first so that we don't bail on the outer
            // close tag (which appears at the end of every block).
            if let Some(rel) = find_invoke_open(&input[cursor..]) {
                let invoke_start = cursor + rel;
                cursor = invoke_start + invoke_open_prefix_len(&input[invoke_start..]);
                // Find the next `">` (the closing quote + `>` of the
                // invoke open tag) in one shot. The bytes between the
                // end of `<|DSML|invoke name="` and that `">` are the
                // tool name.
                let name_end =
                    input[cursor..]
                        .find(INVOKE_OPEN_SUFFIX)
                        .ok_or(DsmlError::MissingAttr {
                            attr: "invoke suffix",
                            offset: cursor,
                        })?;
                let name = input[cursor..cursor + name_end].to_string();
                cursor += name_end + INVOKE_OPEN_SUFFIX.len();
                let mut params: Vec<DsmlParameter> = Vec::new();
                loop {
                    // Check the parameter first, otherwise the
                    // outer `</|DSML|invoke>` (at the end of the
                    // block) would always match and we'd break
                    // before parsing any parameters.
                    if let Some(rel) = find_parameter_open(&input[cursor..]) {
                        let p_start = cursor + rel;
                        cursor = p_start + parameter_open_prefix_len(&input[p_start..]);
                        // Find the closing `"` of `name="..."`.
                        let pname_end = input[cursor..]
                            .find('"')
                            .ok_or(DsmlError::Unterminated { offset: cursor })?;
                        let pname = input[cursor..cursor + pname_end].to_string();
                        cursor += pname_end + 1;
                        // Expect ` string="true|false">`.
                        const PARAM_AFTER_NAME: &str = " string=\"";
                        if !input[cursor..].starts_with(PARAM_AFTER_NAME) {
                            return Err(DsmlError::MissingAttr {
                                attr: "parameter string flag",
                                offset: cursor,
                            });
                        }
                        cursor += PARAM_AFTER_NAME.len();
                        let is_string = match input[cursor..].chars().next() {
                            Some('t') => {
                                if !input[cursor..].starts_with("true") {
                                    return Err(DsmlError::Unexpected {
                                        offset: cursor,
                                        msg: "expected true|false".to_string(),
                                    });
                                }
                                cursor += 4;
                                true
                            }
                            Some('f') => {
                                if !input[cursor..].starts_with("false") {
                                    return Err(DsmlError::Unexpected {
                                        offset: cursor,
                                        msg: "expected true|false".to_string(),
                                    });
                                }
                                cursor += 5;
                                false
                            }
                            _ => {
                                return Err(DsmlError::Unexpected {
                                    offset: cursor,
                                    msg: "expected true|false".to_string(),
                                })
                            }
                        };
                        let suffix_rel = input[cursor..]
                            .find(PARAMETER_OPEN_SUFFIX)
                            .ok_or(DsmlError::Unterminated { offset: cursor })?;
                        cursor += suffix_rel + PARAMETER_OPEN_SUFFIX.len();
                        // Unescape `&lt;/|DSML|parameter>` -> `</|DSML|parameter>`
                        // (the upstream C un-escapes this on read).
                        let close_rel = match find_parameter_close(&input[cursor..]) {
                            Some(rel) => rel,
                            None => {
                                return Err(DsmlError::Unterminated { offset: cursor });
                            }
                        };
                        let close_abs = cursor + close_rel;
                        let consumed = parameter_close_len(&input[close_abs..]);
                        let raw_value = &input[cursor..close_abs];
                        let value = unescape_parameter(raw_value);
                        cursor = close_abs + consumed;
                        params.push(DsmlParameter {
                            name: pname,
                            is_string,
                            value,
                        });
                    } else if let Some(rel) = find_invoke_close(&input[cursor..]) {
                        cursor += rel + invoke_close_len(&input[cursor + rel..]);
                        break;
                    } else {
                        // Skip whitespace / newlines between parameters.
                        if let Some(c) = input[cursor..].chars().next() {
                            if c.is_whitespace() {
                                cursor += c.len_utf8();
                                continue;
                            }
                        }
                        return Err(DsmlError::Unexpected {
                            offset: cursor,
                            msg: format!(
                                "expected parameter or </|DSML|invoke>, got {:?}",
                                input[cursor..].chars().take(20).collect::<String>()
                            ),
                        });
                    }
                }
                out.push(DsmlToolCall {
                    name,
                    parameters: params,
                });
            } else if let Some(rel) = find_close_tag(&input[cursor..]) {
                // Skip whitespace and consume the outer close tag.
                cursor += rel + close_tag_len(&input[cursor + rel..]);
                closed_outer = true;
                break;
            } else if let Some(c) = input[cursor..].chars().next() {
                if c.is_whitespace() {
                    cursor += c.len_utf8();
                    continue;
                }
                return Err(DsmlError::Unexpected {
                    offset: cursor,
                    msg: format!(
                        "expected invoke tag, got {:?}",
                        input[cursor..].chars().take(20).collect::<String>()
                    ),
                });
            } else {
                break;
            }
        }
        if !closed_outer {
            return Err(DsmlError::Unterminated { offset: start });
        }
    }
    Ok(out)
}

/// Detect whether `input` contains a DSML tool-call block.
pub fn contains_tool_calls(input: &str) -> bool {
    input.contains(OPEN_TAG)
}

/// Render `calls` to canonical DSML text.
pub fn render(calls: &[DsmlToolCall]) -> String {
    let mut out = String::new();
    out.push_str(OPEN_TAG);
    out.push('\n');
    for call in calls {
        out.push_str(&format!(
            "{INVOKE_OPEN_PREFIX}{}{INVOKE_OPEN_SUFFIX}\n",
            call.name
        ));
        for p in &call.parameters {
            let bool_str = if p.is_string { "true" } else { "false" };
            let escaped = escape_parameter(&p.value);
            out.push_str(&format!(
                "{PARAMETER_OPEN_PREFIX}{}{PARAMETER_OPEN_MIDDLE}{bool_str}{PARAMETER_OPEN_SUFFIX}{escaped}{PARAMETER_CLOSE}\n",
                p.name,
            ));
        }
        out.push_str(INVOKE_CLOSE);
        out.push('\n');
    }
    out.push_str(CLOSE_TAG);
    out.push('\n');
    out
}

fn find_open_tag(input: &str) -> Option<usize> {
    // Accept either `|` (ASCII) or `｜` (U+FF5C).
    if let Some(idx) = find_any(input, 0, &["<|DSML|tool_calls>", "<｜DSML｜tool_calls>"]) {
        return Some(idx);
    }
    // Fallback: literal ASCII search.
    input.find(OPEN_TAG)
}

fn find_close_tag(input: &str) -> Option<usize> {
    find_any(input, 0, &["</|DSML|tool_calls>", "</｜DSML｜tool_calls>"])
}

fn find_invoke_open(input: &str) -> Option<usize> {
    find_any(
        input,
        0,
        &["<|DSML|invoke name=\"", "<｜DSML｜invoke name=\""],
    )
}

fn find_invoke_close(input: &str) -> Option<usize> {
    find_any(input, 0, &["</|DSML|invoke>", "</｜DSML｜invoke>"])
}

fn find_parameter_open(input: &str) -> Option<usize> {
    find_any(
        input,
        0,
        &["<|DSML|parameter name=\"", "<｜DSML｜parameter name=\""],
    )
}

fn find_parameter_close(input: &str) -> Option<usize> {
    find_any(input, 0, &["</|DSML|parameter>", "</｜DSML｜parameter>"])
}

fn parameter_close_len(input: &str) -> usize {
    if input.starts_with("</｜DSML｜parameter>") {
        "</｜DSML｜parameter>".len()
    } else {
        PARAMETER_CLOSE.len()
    }
}

fn open_tag_len(input: &str) -> usize {
    if input.starts_with("<｜DSML｜tool_calls>") {
        "<｜DSML｜tool_calls>".len()
    } else {
        OPEN_TAG.len()
    }
}

fn close_tag_len(input: &str) -> usize {
    if input.starts_with("</｜DSML｜tool_calls>") {
        "</｜DSML｜tool_calls>".len()
    } else {
        CLOSE_TAG.len()
    }
}

fn invoke_open_prefix_len(input: &str) -> usize {
    if input.starts_with("<｜DSML｜invoke name=\"") {
        "<｜DSML｜invoke name=\"".len()
    } else {
        INVOKE_OPEN_PREFIX.len()
    }
}

fn invoke_close_len(input: &str) -> usize {
    if input.starts_with("</｜DSML｜invoke>") {
        "</｜DSML｜invoke>".len()
    } else {
        INVOKE_CLOSE.len()
    }
}

fn parameter_open_prefix_len(input: &str) -> usize {
    if input.starts_with("<｜DSML｜parameter name=\"") {
        "<｜DSML｜parameter name=\"".len()
    } else {
        PARAMETER_OPEN_PREFIX.len()
    }
}

fn find_any(input: &str, start: usize, candidates: &[&str]) -> Option<usize> {
    let mut best: Option<usize> = None;
    for cand in candidates {
        if let Some(rel) = input[start..].find(cand) {
            let abs = start + rel;
            if best.is_none_or(|b| abs < b) {
                best = Some(abs);
            }
        }
    }
    best
}

fn escape_parameter(value: &str) -> String {
    if value.contains(PARAMETER_CLOSE) {
        // Replace `</|DSML|parameter>` (or `</｜DSML｜parameter>`) with the
        // entity-escaped form so the parser can reverse it on read.
        value
            .replace("</｜DSML｜parameter>", "&lt;/｜DSML｜parameter>")
            .replace("</|DSML|parameter>", "&lt;/|DSML|parameter>")
    } else {
        value.to_string()
    }
}

fn unescape_parameter(value: &str) -> String {
    value
        .replace("&lt;/｜DSML｜parameter>", "</｜DSML｜parameter>")
        .replace("&lt;/|DSML|parameter>", "</|DSML|parameter>")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_single_tool_call() {
        let src = r#"<|DSML|tool_calls>
<|DSML|invoke name="get_weather">
<|DSML|parameter name="city" string="true">Tokyo</|DSML|parameter>
<|DSML|parameter name="unit" string="false">"celsius"</|DSML|parameter>
</|DSML|invoke>
</|DSML|tool_calls>"#;
        let calls = parse(src).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[0].parameters.len(), 2);
        assert_eq!(calls[0].parameters[0].name, "city");
        assert!(calls[0].parameters[0].is_string);
        assert_eq!(calls[0].parameters[0].value, "Tokyo");
        assert_eq!(calls[0].parameters[1].name, "unit");
        assert!(!calls[0].parameters[1].is_string);
    }

    #[test]
    fn roundtrip_render_then_parse() {
        let calls = vec![DsmlToolCall {
            name: "search".to_string(),
            parameters: vec![DsmlParameter {
                name: "q".to_string(),
                is_string: true,
                value: "rust ds4".to_string(),
            }],
        }];
        let text = render(&calls);
        let parsed = parse(&text).unwrap();
        assert_eq!(parsed, calls);
    }

    #[test]
    fn contains_tool_calls_works() {
        assert!(contains_tool_calls("hello <|DSML|tool_calls>"));
        assert!(!contains_tool_calls("hello world"));
    }

    #[test]
    fn accepts_full_width_pipes() {
        let src = "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"f\">\n<｜DSML｜parameter name=\"x\" string=\"true\">y</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>";
        let calls = parse(src).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "f");
        assert_eq!(calls[0].parameters[0].value, "y");
    }

    #[test]
    fn unescapes_parameter_close_in_value() {
        let src = r#"<|DSML|tool_calls>
<|DSML|invoke name="t">
<|DSML|parameter name="x" string="true">a &lt;/|DSML|parameter> b</|DSML|parameter>
</|DSML|invoke>
</|DSML|tool_calls>"#;
        let calls = parse(src).unwrap();
        assert_eq!(calls[0].parameters[0].value, "a </|DSML|parameter> b");
    }
    #[test]
    fn rejects_unterminated_outer_block() {
        let src = format!("{OPEN_TAG}\n{INVOKE_OPEN_PREFIX}f{INVOKE_OPEN_SUFFIX}\n{INVOKE_CLOSE}");
        assert!(matches!(parse(&src), Err(DsmlError::Unterminated { .. })));
    }
}
