import React, { useState, useRef, useEffect, useCallback } from "react";

const API_URL = "http://localhost:8000/ask/stream";
const SESSION_ID =
  Math.random().toString(36).slice(2) + Date.now().toString(36);

const DOC_LABEL = { iam:"IAM", s3:"S3", ec2:"EC2", vpc:"VPC", cfn:"CloudFormation" };

const SUGGESTIONS = [
  "What is the service prefix for AWS Certificate Manager?",
  "How do I assume an IAM role with the CLI?",
  "What permissions does CreateNotebookInstance require?",
  "Does SageMaker provide a no-code way to build models?",
];

/* ---------- rich text: fenced code, inline `tokens`, [n] citations ---------- */
function RichText({ text }) {
  const blocks = text.split(/```/g);
  return (
    <>
      {blocks.map((block, i) => {
        if (i % 2 === 1) {
          return (
            <pre key={i} className="code-block">
              <code>{block.replace(/^\w*\n/, "").replace(/\s+$/, "")}</code>
            </pre>
          );
        }
        // order matters: bold (**), inline code (`), citation ([n]), italic (*)
        const segs = block.split(
          /(\*\*[^*]+\*\*|`[^`]+`|\[\d+\]|\*[^*]+\*)/g
        );
        return (
          <span key={i}>
            {segs.map((s, j) => {
              if (/^\*\*[^*]+\*\*$/.test(s))
                return <strong key={j}>{s.slice(2, -2)}</strong>;
              if (/^`[^`]+`$/.test(s))
                return (
                  <code key={j} className="inline-code">
                    {s.slice(1, -1)}
                  </code>
                );
              if (/^\[\d+\]$/.test(s))
                return (
                  <sup key={j} className="cite">
                    {s}
                  </sup>
                );
              if (/^\*[^*]+\*$/.test(s))
                return <em key={j}>{s.slice(1, -1)}</em>;
              return <span key={j}>{s}</span>;
            })}
          </span>
        );
      })}
    </>
  );
}

/* ---------- references: quiet footnote line, no color, no boxes ---------- */
function References({ sources }) {
  if (!sources || sources.length === 0) return null;
  return (
    <div className="refs">
      <span className="refs-label">Sources</span>
      {sources.map((s, i) => (
        <span key={i} className="ref">
          {DOC_LABEL[s.source] || s.source}
          {s.pages && <span className="ref-page"> p.{s.pages}</span>}
          {i < sources.length - 1 && <span className="ref-sep">·</span>}
        </span>
      ))}
    </div>
  );
}

export default function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const scrollRef = useRef(null);
  const taRef = useRef(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({
      top: scrollRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [messages]);

  const autosize = () => {
    const ta = taRef.current;
    if (!ta) return;
    ta.style.height = "auto";
    ta.style.height = Math.min(ta.scrollHeight, 160) + "px";
  };

  const send = useCallback(
    async (text) => {
      const q = (text ?? input).trim();
      if (!q || busy) return;
      setInput("");
      setTimeout(autosize, 0);
      setMessages((m) => [
        ...m,
        { role: "user", text: q },
        { role: "assistant", text: "", sources: [], streaming: true },
      ]);
      setBusy(true);

      try {
        const res = await fetch(API_URL, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ session_id: SESSION_ID, question: q }),
        });
        const reader = res.body.getReader();
        const dec = new TextDecoder();
        let buf = "";

        for (;;) {
          const { done, value } = await reader.read();
          if (done) break;
          buf += dec.decode(value, { stream: true });
          const lines = buf.split("\n\n");
          buf = lines.pop();
          for (const line of lines) {
            if (!line.startsWith("data:")) continue;
            const ev = JSON.parse(line.slice(5).trim());
            setMessages((m) => {
              const copy = [...m];
              const last = { ...copy[copy.length - 1] };
              if (ev.type === "sources") last.sources = ev.sources;
              else if (ev.type === "token") last.text = last.text + ev.text;
              else if (ev.type === "done") last.streaming = false;
              copy[copy.length - 1] = last;
              return copy;
            });
          }
        }
      } catch {
        setMessages((m) => {
          const copy = [...m];
          const last = { ...copy[copy.length - 1] };
          last.text = "Couldn't reach the server. Is the backend running on :8000?";
          last.error = true;
          last.streaming = false;
          copy[copy.length - 1] = last;
          return copy;
        });
      } finally {
        setBusy(false);
      }
    },
    [input, busy]
  );

  const empty = messages.length === 0;

  return (
    <div className="app">
      <style>{CSS}</style>

      <header className="topbar">
        <span className="dot" />
        <span className="wordmark">AWS Documentation Assistant</span>
      </header>

      <main className="chat" ref={scrollRef}>
        {empty ? (
          <div className="hero">
            <h1>Ask the documentation</h1>
            <p>
              Answers are drawn from the AWS Documentation Guides,
              with the source page referenced below each response.
            </p>
            <div className="suggest">
              {SUGGESTIONS.map((s, i) => (
                <button key={i} className="sg" onClick={() => send(s)}>
                  {s}
                </button>
              ))}
            </div>
          </div>
        ) : (
          <div className="thread">
            {messages.map((m, i) => (
              <div key={i} className={`msg ${m.role}`}>
                <div className={`bubble ${m.role} ${m.error ? "err" : ""}`}>
                  {m.text ? (
                    <RichText text={m.text} />
                  ) : m.streaming ? (
                    <span className="thinking">
                      <span />
                      <span />
                      <span />
                    </span>
                  ) : null}
                  {m.streaming && m.text && <span className="caret" />}
                </div>
                {m.role === "assistant" && <References sources={m.sources} />}
              </div>
            ))}
          </div>
        )}
      </main>

      <div className="composer">
        <div className="composer-inner">
          <textarea
            ref={taRef}
            rows={1}
            value={input}
            placeholder="Ask me anything related to AWS Documents..."
            onChange={(e) => {
              setInput(e.target.value);
              autosize();
            }}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                send();
              }
            }}
          />
          <button
            className="send"
            onClick={() => send()}
            disabled={busy || !input.trim()}
            aria-label="Send message"
          >
            <svg viewBox="0 0 16 16" width="14" height="14">
              <path
                fill="currentColor"
                d="M1 8l13-6-4 6 4 6-13-6z"
              />
            </svg>
          </button>
        </div>
      </div>
    </div>
  );
}

const CSS = `
*{box-sizing:border-box;margin:0;padding:0}
:focus-visible{outline:1.5px solid var(--accent);outline-offset:2px}

.app{
  --bg:#0c0c0f;
  --surface:#15151a;
  --surface-2:#1a1a20;
  --border:#232329;
  --border-soft:#1c1c22;
  --text:#e7e7ea;
  --muted:#8e8e98;
  --dim:#54545c;
  --accent:#c9a063;
  --accent-soft:#8a734a;
  --code-bg:#0a0a0d;
  --code-fg:#cfcfd6;
  --user:#1d1d24;

  height:100vh; display:flex; flex-direction:column;
  background:var(--bg); color:var(--text);
  font-family:-apple-system,BlinkMacSystemFont,"Inter","Segoe UI",Roboto,sans-serif;
  font-size:15px; line-height:1.65;
}

/* topbar */
.topbar{
  display:flex; align-items:center; gap:9px;
  padding:16px 24px; border-bottom:1px solid var(--border-soft);
  flex-shrink:0;
}
.dot{width:6px;height:6px;border-radius:50%;background:var(--accent)}
.wordmark{font-size:13px;font-weight:600;letter-spacing:.3px;color:var(--muted)}

/* chat scroll area */
.chat{flex:1;overflow-y:auto}
.thread{max-width:700px;margin:0 auto;padding:36px 24px 24px}

/* empty state */
.hero{max-width:560px;margin:9vh auto 0;padding:0 24px}
.hero h1{font-size:23px;font-weight:600;letter-spacing:-.2px;margin-bottom:10px}
.hero p{color:var(--muted);font-size:14.5px;line-height:1.6;margin-bottom:28px}
.suggest{display:flex;flex-direction:column;gap:8px}
.sg{
  text-align:left;background:transparent;border:1px solid var(--border);
  color:var(--text);padding:12px 14px;border-radius:10px;font-size:13.5px;
  cursor:pointer;transition:border-color .15s,background .15s;
}
.sg:hover{border-color:var(--dim);background:var(--surface)}

/* messages */
.msg{margin-bottom:28px}
.msg.user{display:flex;justify-content:flex-end}
.bubble{padding:12px 16px;border-radius:12px;white-space:pre-wrap;word-wrap:break-word}
.bubble.assistant{background:transparent;padding:0 0 2px}
.bubble.user{background:var(--user);border:1px solid var(--border-soft);max-width:78%}
.bubble.err{color:#c98a8a}
.bubble strong{font-weight:650;color:var(--text)}
.bubble em{font-style:italic;color:var(--muted)}

/* code */
.code-block{
  background:var(--code-bg);border:1px solid var(--border);border-radius:8px;
  padding:12px 14px;margin:10px 0;overflow-x:auto;
  font-family:"JetBrains Mono","SF Mono",Menlo,monospace;font-size:12.5px;
  color:var(--code-fg);line-height:1.55;
}
.inline-code{
  background:var(--code-bg);border:1px solid var(--border);border-radius:4px;
  padding:1px 5px;font-family:"JetBrains Mono","SF Mono",Menlo,monospace;
  font-size:12.5px;color:var(--text);
}
.cite{color:var(--accent);font-weight:600;font-size:10.5px;margin-left:1px}

/* references — quiet footnote line, no boxes, no per-doc color */
.refs{
  margin-top:10px;font-size:11.5px;color:var(--dim);
  display:flex;flex-wrap:wrap;gap:5px;align-items:baseline;
  font-family:"JetBrains Mono","SF Mono",Menlo,monospace;
}
.refs-label{
  color:var(--dim);text-transform:uppercase;letter-spacing:.6px;
  font-size:10px;margin-right:5px;font-family:inherit;
}
.ref{color:var(--muted)}
.ref-page{color:var(--dim)}
.ref-sep{color:var(--dim);margin-left:5px}

/* streaming affordances */
.caret{
  display:inline-block;width:6px;height:13px;background:var(--accent);
  margin-left:2px;vertical-align:-2px;animation:blink 1s steps(2) infinite;border-radius:1px;
}
.thinking{display:inline-flex;gap:4px;padding:6px 0}
.thinking span{width:5px;height:5px;border-radius:50%;background:var(--dim);animation:bob 1.3s infinite both}
.thinking span:nth-child(2){animation-delay:.18s}
.thinking span:nth-child(3){animation-delay:.36s}
@keyframes blink{0%,50%{opacity:1}50.01%,100%{opacity:0}}
@keyframes bob{0%,80%,100%{opacity:.3;transform:translateY(0)}40%{opacity:1;transform:translateY(-2px)}}
@media (prefers-reduced-motion: reduce){
  .caret,.thinking span{animation:none;opacity:1}
}

/* composer */
.composer{padding:14px 24px 22px;flex-shrink:0}
.composer-inner{
  max-width:700px;margin:0 auto;display:flex;gap:8px;align-items:flex-end;
  background:var(--surface-2);border:1px solid var(--border);border-radius:14px;
  padding:9px 9px 9px 16px;transition:border-color .15s;
}
.composer-inner:focus-within{border-color:var(--accent-soft)}
.composer textarea{
  flex:1;resize:none;background:transparent;border:none;outline:none;
  color:var(--text);font-size:14.5px;font-family:inherit;line-height:1.5;
  padding:7px 0;max-height:160px;
}
.composer textarea::placeholder{color:var(--dim)}
.send{
  flex-shrink:0;width:34px;height:34px;border:none;border-radius:9px;
  background:var(--accent);color:#0c0c0f;display:flex;align-items:center;justify-content:center;
  cursor:pointer;transition:opacity .15s;
}
.send:hover:not(:disabled){opacity:.88}
.send:disabled{opacity:.3;cursor:default}

@media(max-width:640px){
  .thread,.hero,.composer-inner{padding-left:16px;padding-right:16px}
  .bubble.user{max-width:88%}
}
`;