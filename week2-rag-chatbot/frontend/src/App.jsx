import { useState, useRef, useEffect } from "react";

const API_URL = "http://127.0.0.1:8001";

export default function App() {
  const [messages, setMessages] = useState([
    {
      role: "assistant",
      content: "Hi! I'm your email marketing assistant. Ask me anything about Marketo, Eloqua, campaigns, or deliverability."
    }
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const bottomRef = useRef(null);

  // Auto-scroll to latest message
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const sendMessage = async () => {
    if (!input.trim() || loading) return;

    const userMessage = { role: "user", content: input };
    const history = messages.map(({ role, content }) => ({ role, content }));

    // Push the user message plus an empty assistant bubble that tokens stream into
    setMessages(prev => [...prev, userMessage, { role: "assistant", content: "" }]);
    setInput("");
    setLoading(true);
    setError(null);

    try {
      const res = await fetch(`${API_URL}/chat/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: userMessage.content, history })
      });

      if (!res.ok || !res.body) throw new Error("API error");

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let assistantText = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        assistantText += decoder.decode(value, { stream: true });

        // Replace the last (assistant) message's content in place — never append a new bubble per chunk
        setMessages(prev => {
          const next = [...prev];
          next[next.length - 1] = { role: "assistant", content: assistantText };
          return next;
        });
      }
      assistantText += decoder.decode();
      setMessages(prev => {
        const next = [...prev];
        next[next.length - 1] = { role: "assistant", content: assistantText };
        return next;
      });

    } catch (err) {
      setError("Something went wrong. Please try again.");
      // Drop the empty assistant bubble if nothing streamed in before the failure
      setMessages(prev => {
        const last = prev[prev.length - 1];
        return last?.role === "assistant" && last.content === "" ? prev.slice(0, -1) : prev;
      });
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <h1 style={styles.title}>📧 Email Marketing AI Assistant</h1>
        <p style={styles.subtitle}>Powered by RAG — answers grounded in your knowledge base</p>
      </div>

      <div style={styles.chatBox}>
        {messages.map((msg, i) => (
          <div key={i} style={{
            ...styles.messageWrap,
            justifyContent: msg.role === "user" ? "flex-end" : "flex-start"
          }}>
            <div style={{
              ...styles.bubble,
              background: msg.role === "user" ? "#4F46E5" : "#1E1E2E",
              color: "#fff",
              borderRadius: msg.role === "user"
                ? "18px 18px 4px 18px"
                : "18px 18px 18px 4px"
            }}>
              <p style={styles.bubbleText}>
                {msg.role === "assistant" && msg.content === "" && loading
                  ? "Thinking..."
                  : msg.content}
              </p>
              {msg.sources && msg.sources.length > 0 && (
                <div style={styles.sources}>
                  <p style={styles.sourcesLabel}>Sources:</p>
                  {msg.sources.map((src, j) => (
                    <p key={j} style={styles.sourceItem}>• {src}</p>
                  ))}
                </div>
              )}
            </div>
          </div>
        ))}

        {error && (
          <p style={styles.error}>{error}</p>
        )}

        <div ref={bottomRef} />
      </div>

      <div style={styles.inputRow}>
        <input
          style={styles.input}
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask about Marketo, Eloqua, campaigns, deliverability..."
          disabled={loading}
        />
        <button
          style={{
            ...styles.button,
            opacity: loading || !input.trim() ? 0.5 : 1,
            cursor: loading || !input.trim() ? "not-allowed" : "pointer"
          }}
          onClick={sendMessage}
          disabled={loading || !input.trim()}
        >
          Send
        </button>
      </div>
    </div>
  );
}

const styles = {
  container: {
    display: "flex",
    flexDirection: "column",
    height: "100vh",
    maxWidth: "800px",
    margin: "0 auto",
    fontFamily: "system-ui, sans-serif",
    background: "#0F0F1A",
    color: "#fff"
  },
  header: {
    padding: "1.5rem",
    borderBottom: "1px solid #2A2A3E"
  },
  title: {
    margin: 0,
    fontSize: "1.4rem",
    fontWeight: 600
  },
  subtitle: {
    margin: "4px 0 0",
    fontSize: "0.8rem",
    color: "#888"
  },
  chatBox: {
    flex: 1,
    overflowY: "auto",
    padding: "1.5rem",
    display: "flex",
    flexDirection: "column",
    gap: "12px"
  },
  messageWrap: {
    display: "flex",
    width: "100%"
  },
  bubble: {
    maxWidth: "75%",
    padding: "10px 14px"
  },
  bubbleText: {
    margin: 0,
    fontSize: "0.9rem",
    lineHeight: 1.5
  },
  sources: {
    marginTop: "8px",
    borderTop: "1px solid rgba(255,255,255,0.1)",
    paddingTop: "8px"
  },
  sourcesLabel: {
    margin: "0 0 4px",
    fontSize: "0.75rem",
    color: "#aaa",
    fontWeight: 600
  },
  sourceItem: {
    margin: "2px 0",
    fontSize: "0.72rem",
    color: "#bbb",
    lineHeight: 1.4
  },
  inputRow: {
    display: "flex",
    gap: "8px",
    padding: "1rem 1.5rem",
    borderTop: "1px solid #2A2A3E"
  },
  input: {
    flex: 1,
    padding: "10px 14px",
    borderRadius: "8px",
    border: "1px solid #2A2A3E",
    background: "#1E1E2E",
    color: "#fff",
    fontSize: "0.9rem",
    outline: "none"
  },
  button: {
    padding: "10px 20px",
    borderRadius: "8px",
    border: "none",
    background: "#4F46E5",
    color: "#fff",
    fontSize: "0.9rem",
    fontWeight: 500
  },
  error: {
    color: "#F87171",
    fontSize: "0.85rem",
    textAlign: "center"
  }
};