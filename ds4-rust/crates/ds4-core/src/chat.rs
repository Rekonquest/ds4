// DS4 (DwarfStar) — DeepSeek chat template.
//
// Mirrors `ds4.c:22371..22434`. The chat template renders the full
// sentinels for system / user / assistant roles plus an optional
// reasoning-mode prefix. The exact sentinel string-to-token mapping
// lives in `tokenizer_data`.

use ds4_types::Ds4ThinkMode;

use crate::tokenizer::Ds4Tokenizer;
use crate::tokenizer_data::pre_tokenize;

/// Sentinel token IDs the chat template emits verbatim.
#[derive(Debug, Clone, Copy)]
pub struct ChatSentinels {
    pub bos: u32,
    pub user: u32,
    pub assistant: u32,
    pub think_start: u32,
    pub think_end: u32,
    pub dsml: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ds4Role {
    System,
    User,
    Assistant,
}

/// DeepSeek V4 chat template (system + user + assistant). Holds a
/// `Ds4Tokenizer` for BPE-encoding the rendered chat body.
pub struct Ds4ChatTemplate {
    tokenizer: Ds4Tokenizer,
    sentinels: ChatSentinels,
}

impl Ds4ChatTemplate {
    pub fn new(tokenizer: Ds4Tokenizer) -> Self {
        Ds4ChatTemplate {
            tokenizer,
            sentinels: ChatSentinels {
                bos: 0,
                user: 0,
                assistant: 0,
                think_start: 0,
                think_end: 0,
                dsml: 0,
            },
        }
    }

    pub fn with_sentinels(mut self, s: ChatSentinels) -> Self {
        self.sentinels = s;
        self
    }

    /// Emit the BOS token + the Assistant sentinel so the model can
    /// begin producing an answer.
    pub fn begin(&self) -> Vec<u32> {
        vec![self.sentinels.bos, self.sentinels.assistant]
    }

    /// Append `<｜role｜>\n<content>` to the token stream.
    pub fn append_message(&self, tokens: &mut Vec<u32>, role: Ds4Role, content: &str) {
        match role {
            Ds4Role::System => tokens.push(self.sentinels.dsml),
            Ds4Role::User => tokens.push(self.sentinels.user),
            Ds4Role::Assistant => tokens.push(self.sentinels.assistant),
        }
        self.append_text(tokens, "\n");
        self.append_text(tokens, content);
    }

    /// Append the assistant message prefix + optional `Think` sentinel.
    pub fn append_assistant_prefix(&self, tokens: &mut Vec<u32>, think: Ds4ThinkMode) {
        tokens.push(self.sentinels.assistant);
        self.append_text(tokens, "\n");
        if think == Ds4ThinkMode::High || think == Ds4ThinkMode::Max {
            tokens.push(self.sentinels.think_start);
            self.append_text(tokens, "\n");
        }
    }

    /// Append the DS4_REASONING_EFFORT_MAX_PREFIX so a max-effort
    /// thinking session has a known shape. Mirrors `ds4_chat_append_max_effort_prefix`.
    pub fn append_max_effort_prefix(&self, tokens: &mut Vec<u32>) {
        let prefix = crate::ds4_think_max_prefix();
        self.append_text(tokens, prefix);
    }

    fn append_text(&self, tokens: &mut Vec<u32>, text: &str) {
        match self.tokenizer.tokenize(text) {
            Ok(ids) => tokens.extend(ids),
            Err(_) => {
                for piece in pre_tokenize(text) {
                    for &b in piece.as_bytes() {
                        tokens.push(b as u32 + 1);
                    }
                }
            }
        }
    }

    /// Render a full chat: system prompt (if any), then user prompt,
    /// then the assistant turn prefix.
    pub fn encode_prompt(&self, system: &str, prompt: &str, think: Ds4ThinkMode) -> Vec<u32> {
        let mut out = self.begin();
        if !system.is_empty() {
            self.append_message(&mut out, Ds4Role::System, system);
        }
        self.append_message(&mut out, Ds4Role::User, prompt);
        if think == Ds4ThinkMode::Max {
            self.append_max_effort_prefix(&mut out);
        }
        self.append_assistant_prefix(&mut out, think);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ct() -> Ds4ChatTemplate {
        let mut map = [0u32; 256];
        for b in 0u32..256 {
            map[b as usize] = b + 1;
        }
        let t = Ds4Tokenizer::from_byte_mapping(map, 0, 1, 2, 3, 4, 5, 6, 7).expect("tokenizer");
        Ds4ChatTemplate::new(t).with_sentinels(ChatSentinels {
            bos: 1,
            user: 3,
            assistant: 4,
            think_start: 5,
            think_end: 6,
            dsml: 7,
        })
    }

    #[test]
    fn begin_emits_bos_and_assistant() {
        let c = ct();
        let out = c.begin();
        assert_eq!(out, vec![1u32, 4]);
    }

    #[test]
    fn encode_prompt_includes_role_sentinel_and_prefix() {
        let c = ct();
        let out = c.encode_prompt("", "hi", Ds4ThinkMode::None);
        // begin() = [bos=1, assistant=4]
        assert_eq!(out[0], 1);
        assert_eq!(out[1], 4); // assistant in begin()
                               // append_message User: [user=3, '\n' byte]
        assert_eq!(out[2], 3); // user sentinel
        assert_eq!(out[3], b'\n' as u32 + 1);
        // Tokenized "hi" -> byte ids for 'h' and 'i' (each `b + 1`).
        assert_eq!(out[4], b'h' as u32 + 1);
        assert_eq!(out[5], b'i' as u32 + 1);
        // Assistant prefix at the end of the prompt.
        assert_eq!(out[6], 4); // assistant sentinel
        assert_eq!(out[7], b'\n' as u32 + 1);
    }
}
