/** Events broadcast by the memvox UI bridge (memvox/observability/uibridge.py).
 *  Keep in sync with _UIBridge event construction on the Python side. */
export type BridgeEvent =
  | { type: "hello"; session_id: string }
  | { type: "user_final"; turn_id: string; text: string; language: string }
  | { type: "assistant_sentence"; turn_id: string; text: string }
  | { type: "assistant_final"; turn_id: string; text: string }
  | { type: "session_end"; session_id: string };

export interface Turn {
  id: string;
  role: "user" | "agent";
  text: string;
  language?: string;
  /** Agent turns stream sentence-by-sentence; pending until assistant_final. */
  pending?: boolean;
}

export interface Flashcard {
  id: string;
  korean: string;
  english: string;
  romanization?: string;
}
