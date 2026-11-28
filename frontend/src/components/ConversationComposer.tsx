import { useRef } from "react";
import type { ChangeEvent, FormEvent } from "react";

export interface ChatAttachment {
  name: string;
  content: string;
}

interface ComposerProps {
  prompt: string;
  setPrompt: (text: string) => void;
  attachments: ChatAttachment[];
  setAttachments: (files: ChatAttachment[]) => void;
  onError: (message: string) => void;
  onSend: () => void;
  onStop?: () => void;
  disabled: boolean;
}

export function ConversationComposer(props: ComposerProps) {
  const fileInput = useRef<HTMLInputElement>(null);

  async function attach(event: ChangeEvent<HTMLInputElement>) {
    const files = Array.from(event.target.files ?? []);
    event.target.value = "";
    if (props.attachments.length + files.length > 5 || files.some((file) => file.size > 200_000)) {
      props.onError("Attach up to five text documents, each under 200 KB.");
      return;
    }
    try {
      const next = await Promise.all(files.map(async (file) => ({
        name: file.name.slice(0, 120), content: await file.text(),
      })));
      if (next.some((file) => !file.content.trim())) {
        props.onError("Empty documents can’t be attached.");
        return;
      }
      props.setAttachments([...props.attachments, ...next]);
    } catch {
      props.onError("I couldn’t read the selected documents.");
    }
  }

  function send(event: FormEvent) {
    event.preventDefault();
    if (!props.disabled && props.prompt.trim()) props.onSend();
  }

  return <form className="conversation-composer" onSubmit={send}>
    {!!props.attachments.length && <div className="att-row">
      {props.attachments.map((file, index) => <span key={`${file.name}-${index}`} className="att-chip">
        {file.name}
        <button type="button" className="att-rm" aria-label={`Remove ${file.name}`}
          onClick={() => props.setAttachments(props.attachments.filter((_, i) => i !== index))}>×</button>
      </span>)}
    </div>}
    <div className="conversation-input">
      <input ref={fileInput} type="file" multiple accept=".txt,.md,.csv,.json,.tsv,.log"
        className="hidden" onChange={(event) => void attach(event)} />
      <button type="button" className="public-secondary attach-document" title="Attach text documents"
        aria-label="Attach text documents" onClick={() => fileInput.current?.click()}>+</button>
      <textarea aria-label="Your message" rows={2} maxLength={32000} value={props.prompt}
        placeholder="Message CrewForge" onChange={(event) => props.setPrompt(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
            event.preventDefault();
            if (!props.disabled && props.prompt.trim()) props.onSend();
          }
        }} />
      {props.onStop
        ? <button type="button" className="public-secondary" onClick={props.onStop}>Stop</button>
        : <button type="submit" className="public-primary"
          disabled={props.disabled || !props.prompt.trim()}>Send</button>}
    </div>
  </form>;
}
