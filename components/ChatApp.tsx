"use client";

import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";

import { Icon, type IconName } from "./Icons";
import Message, { type UiMessage } from "./Message";
import { NearbyView, OfficeView, TripPlannerView } from "./Panels";
import { clipImageEmbedding } from "./clipEmbedding";
import type { ChatMessage, ChatResponse, ImageResult, PublicPlace } from "@/lib/types";

type View = "chat" | "plan" | "nearby" | "office";

const VIEWS: Array<{ id: View; label: string; short: string; icon: IconName; title: string; subtitle: string }> = [
  {
    id: "chat",
    label: "Assistant",
    short: "Ask",
    icon: "chat",
    title: "Tourism assistant",
    subtitle: "Answers grounded in official Punjab Tourism pages",
  },
  {
    id: "plan",
    label: "Trip planner",
    short: "Plan",
    icon: "calendar",
    title: "Trip planner",
    subtitle: "Build a day-by-day itinerary from official pages",
  },
  {
    id: "nearby",
    label: "Nearby places",
    short: "Nearby",
    icon: "compass",
    title: "Nearby places",
    subtitle: "Distances calculated privately in your browser",
  },
  {
    id: "office",
    label: "Visit the department",
    short: "Office",
    icon: "building",
    title: "Tourism Department",
    subtitle: "Office address and approximate location",
  },
];

const SUGGESTIONS: Array<{ tag: string; icon: IconName; text: string }> = [
  { tag: "Heritage", icon: "book", text: "Tell me about Derawar Fort" },
  { tag: "Location", icon: "pin", text: "Where is Badshahi Mosque?" },
  { tag: "Photos", icon: "image", text: "Show me pictures of Rohtas Fort" },
  { tag: "Roman Urdu", icon: "globe", text: "Murree kahan hai?" },
  { tag: "اردو", icon: "globe", text: "بادشاہی مسجد کہاں ہے؟" },
  { tag: "Department", icon: "building", text: "What does the Tourism Department do?" },
];

function TypingIndicator({ status }: { status: string }) {
  return (
    <div className="msg msg-assistant" role="status" aria-live="polite">
      <span className="avatar" aria-hidden="true">
        PT
      </span>
      <div className="typing">
        <span className="typing-dots" aria-hidden="true">
          <i />
          <i />
          <i />
        </span>
        <span>{status}</span>
      </div>
    </div>
  );
}

function Welcome({
  pageCount,
  onAsk,
  onNavigate,
  onPickImage,
}: {
  pageCount: number;
  onAsk: (question: string) => void;
  onNavigate: (view: View) => void;
  onPickImage: () => void;
}) {
  return (
    <section className="welcome" aria-labelledby="welcome-title">
      <p className="welcome-eyebrow badge badge-success">
        <Icon name="shield" size={13} />
        {`Grounded in ${pageCount} official pages`}
      </p>
      <h2 id="welcome-title">
        Discover Punjab, <em>one question</em> at a time.
      </h2>
      <p className="welcome-lead">
        Ask about forts, mosques, gardens and festivals across Punjab. Answers cite the official tourism website, and
        you can ask in English, Urdu or Roman Urdu.
      </p>

      <p className="section-label">Try asking</p>
      <div className="suggestions">
        {SUGGESTIONS.map((suggestion) => (
          <button key={suggestion.text} type="button" className="suggestion" onClick={() => onAsk(suggestion.text)}>
            <span className="suggestion-tag">
              <Icon name={suggestion.icon} size={13} />
              {suggestion.tag}
            </span>
            <span className="suggestion-text" dir="auto">
              {suggestion.text}
            </span>
          </button>
        ))}
      </div>

      <p className="section-label">Or explore</p>
      <div className="quick-actions">
        <button type="button" className="quick-action" onClick={() => onNavigate("plan")}>
          <span className="quick-action-icon">
            <Icon name="calendar" size={18} />
          </span>
          <span className="quick-action-copy">
            <strong>Plan a trip</strong>
            <small>Day-by-day itinerary</small>
          </span>
        </button>
        <button type="button" className="quick-action" onClick={onPickImage}>
          <span className="quick-action-icon">
            <Icon name="image" size={18} />
          </span>
          <span className="quick-action-copy">
            <strong>Identify a photo</strong>
            <small>Match it to tourism sites</small>
          </span>
        </button>
        <button type="button" className="quick-action" onClick={() => onNavigate("nearby")}>
          <span className="quick-action-icon">
            <Icon name="compass" size={18} />
          </span>
          <span className="quick-action-copy">
            <strong>Find nearby places</strong>
            <small>Private, in-browser distances</small>
          </span>
        </button>
      </div>
    </section>
  );
}

export default function ChatApp({ places, pageCount }: { places: PublicPlace[]; pageCount: number }) {
  const [view, setView] = useState<View>("chat");
  const [messages, setMessages] = useState<UiMessage[]>([]);
  const [question, setQuestion] = useState("");
  const [busy, setBusy] = useState(false);
  const [pendingStatus, setPendingStatus] = useState("");
  const history = useRef<ChatMessage[]>([]);
  const nextId = useRef(1);
  const threadRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const imageInputRef = useRef<HTMLInputElement>(null);
  const busyRef = useRef(false);

  const current = VIEWS.find((item) => item.id === view) ?? VIEWS[0];

  useLayoutEffect(() => {
    document.querySelector<HTMLElement>(".view:not([hidden])")?.scrollTo({ top: 0 });
    if (view === "chat") threadRef.current?.scrollTo({ top: 0 });
  }, [view]);

  // Follow the conversation: show the newest question and typing indicator,
  // then bring the start of each new answer into view (long answers with maps
  // would otherwise scroll their opening lines out of sight).
  useEffect(() => {
    const thread = threadRef.current;
    if (!thread) return;
    if (messages.length === 0 && !busy) {
      thread.scrollTo({ top: 0 });
      return;
    }
    const last = messages[messages.length - 1];
    if (last?.role === "assistant" && !busy) {
      const element = thread.querySelector<HTMLElement>(`[data-message-id="${last.id}"]`);
      if (element) {
        thread.scrollTo({ top: Math.max(0, element.offsetTop - 16), behavior: "smooth" });
        return;
      }
    }
    thread.scrollTo({ top: thread.scrollHeight, behavior: "smooth" });
  }, [messages, busy, pendingStatus]);

  // Auto-grow the composer.
  useLayoutEffect(() => {
    const element = inputRef.current;
    if (!element) return;
    element.style.height = "auto";
    element.style.height = `${Math.min(element.scrollHeight, 180)}px`;
  }, [question, view]);

  const add = useCallback((message: Omit<UiMessage, "id">) => {
    const id = nextId.current++;
    setMessages((existing) => [...existing, { ...message, id }]);
    return id;
  }, []);

  const startBusy = (status: string) => {
    busyRef.current = true;
    setBusy(true);
    setPendingStatus(status);
  };
  const stopBusy = () => {
    busyRef.current = false;
    setBusy(false);
    setPendingStatus("");
  };

  const ask = useCallback(
    async (rawQuestion: string) => {
      const text = rawQuestion.trim();
      if (!text || busyRef.current) return;
      setView("chat");
      add({ role: "user", text });
      setQuestion("");
      startBusy("Looking through the tourism pages...");
      try {
        const response = await fetch("/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question: text, history: history.current.slice(-8) }),
        });
        const result = (await response.json().catch(() => ({}))) as ChatResponse & { error?: string };
        if (!response.ok) throw new Error(result.error || "The chatbot could not answer right now.");
        add({
          role: "assistant",
          text: result.answer,
          sources: result.sources || [],
          maps: result.maps || [],
          location: result.location || null,
          metadata: result,
          images: result.images?.length ? result.images : undefined,
        });
        history.current.push({ role: "user", content: text }, { role: "assistant", content: result.answer });
        if (history.current.length > 10) history.current.splice(0, history.current.length - 10);
      } catch (error) {
        add({
          role: "assistant",
          text: (error as Error).message || "The chatbot could not answer right now.",
          error: true,
          retry: text,
        });
      } finally {
        stopBusy();
        inputRef.current?.focus();
      }
    },
    [add],
  );

  const matchTourismImage = useCallback(
    async (file: File | undefined, label: string) => {
      if (!file || busyRef.current) return;
      setView("chat");
      add({ role: "user", text: `Find a tourism-site match for: ${label}`, previewFile: file });
      startBusy("Preparing the image...");
      try {
        const embedding = await clipImageEmbedding(file, setPendingStatus);
        const response = await fetch("/api/image", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ embedding }),
        });
        const result = (await response.json().catch(() => ({}))) as {
          answer: string;
          matches?: ImageResult[];
          error?: string;
        };
        if (!response.ok) throw new Error(result.error || "The image could not be matched.");
        add({ role: "assistant", text: result.answer, images: result.matches || [] });
      } catch (error) {
        add({ role: "assistant", text: (error as Error).message || "The image could not be matched.", error: true });
      } finally {
        stopBusy();
      }
    },
    [add],
  );

  const newChat = () => {
    if (busyRef.current) return;
    setMessages([]);
    history.current = [];
    setQuestion("");
    inputRef.current?.focus();
  };

  const pickImage = () => imageInputRef.current?.click();

  const onPaste = (event: React.ClipboardEvent<HTMLTextAreaElement>) => {
    const items = [...(event.clipboardData?.items || [])];
    const imageItem = items.find((item) => item.kind === "file" && item.type.startsWith("image/"));
    if (!imageItem) return;
    const file = imageItem.getAsFile();
    if (!file) return;
    event.preventDefault();
    const extension = file.type.split("/")[1] || "png";
    matchTourismImage(file, file.name || `pasted-image.${extension}`);
  };

  return (
    <div className="shell">
      <aside className="sidebar" aria-label="Main navigation">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true">
            PT
          </span>
          <div>
            <p className="brand-name">Punjab Tourism</p>
            <p className="brand-sub">Assistant · Pakistan</p>
          </div>
        </div>
        <nav className="nav">
          <p className="nav-label">Explore</p>
          {VIEWS.map((item) => (
            <button
              key={item.id}
              type="button"
              className="nav-item"
              aria-current={view === item.id ? "page" : undefined}
              onClick={() => setView(item.id)}
            >
              <Icon name={item.icon} size={19} />
              {item.label}
            </button>
          ))}
        </nav>
        <div className="sidebar-card">
          <strong>
            <Icon name="shield" size={15} />
            Source-grounded
          </strong>
          Answers cite pages from{" "}
          <a href="https://tourism.punjab.gov.pk/" target="_blank" rel="noopener noreferrer">
            tourism.punjab.gov.pk
          </a>
          . Please confirm timings and fees before you travel.
        </div>
      </aside>

      <main className="workspace">
        <header className="topbar">
          <span className="brand-mark topbar-brand" aria-hidden="true">
            PT
          </span>
          <div className="topbar-title">
            <h1>{current.title}</h1>
            <p>{current.subtitle}</p>
          </div>
          <div className="topbar-actions">
            {view === "chat" && messages.length > 0 && (
              <button type="button" className="btn btn-secondary btn-sm" onClick={newChat} disabled={busy}>
                <Icon name="plus" size={15} />
                <span className="hide-sm">New chat</span>
              </button>
            )}
          </div>
        </header>

        <section className="view view-chat" hidden={view !== "chat"} aria-label="Conversation">
          <div className="thread" ref={threadRef}>
            <div className="thread-inner" aria-live="polite">
              {messages.length === 0 && !busy && (
                <Welcome pageCount={pageCount} onAsk={ask} onNavigate={setView} onPickImage={pickImage} />
              )}
              {messages.map((message) => (
                <Message key={message.id} message={message} onAsk={ask} busy={busy} />
              ))}
              {busy && <TypingIndicator status={pendingStatus} />}
            </div>
          </div>
          <div className="composer-wrap">
            <form
              className="composer"
              id="chat-form"
              onSubmit={(event) => {
                event.preventDefault();
                ask(question);
              }}
            >
              <button
                id="image-button"
                type="button"
                className="btn btn-ghost btn-icon"
                onClick={pickImage}
                disabled={busy}
                aria-label="Match a tourism image"
                title="Match a tourism image"
              >
                <Icon name="image" size={20} />
              </button>
              <textarea
                id="question"
                ref={inputRef}
                rows={1}
                maxLength={2000}
                placeholder="Ask about Punjab tourism..."
                aria-label="Your question"
                dir="auto"
                value={question}
                onChange={(event) => setQuestion(event.target.value)}
                onPaste={onPaste}
                onKeyDown={(event) => {
                  if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
                    event.preventDefault();
                    ask(question);
                  }
                }}
              />
              <button
                id="send"
                type="submit"
                className="btn btn-primary btn-icon"
                disabled={busy || !question.trim()}
                aria-label="Send"
                title="Send"
              >
                <Icon name="send" size={19} />
              </button>
            </form>
            <p className="composer-hint">
              <kbd>Enter</kbd> to send · <kbd>Shift</kbd> + <kbd>Enter</kbd> for a new line · paste or attach a photo
              to identify it (analysed on your device)
            </p>
            <input
              id="image-input"
              ref={imageInputRef}
              type="file"
              accept="image/jpeg,image/png,image/webp,image/gif"
              hidden
              onChange={(event) => {
                const file = event.target.files?.[0];
                event.target.value = "";
                matchTourismImage(file, file?.name || "selected image");
              }}
            />
          </div>
        </section>

        <section className="view" hidden={view !== "plan"} aria-label="Trip planner">
          <TripPlannerView places={places} />
        </section>
        <section className="view" hidden={view !== "nearby"} aria-label="Nearby places">
          <NearbyView places={places} />
        </section>
        {view === "office" && (
          <section className="view" aria-label="Tourism Department office">
            <OfficeView />
          </section>
        )}
      </main>

      <nav className="tabbar" aria-label="Main navigation">
        {VIEWS.map((item) => (
          <button
            key={item.id}
            type="button"
            className="tab"
            aria-current={view === item.id ? "page" : undefined}
            onClick={() => setView(item.id)}
          >
            <Icon name={item.icon} size={20} />
            {item.short}
          </button>
        ))}
      </nav>
    </div>
  );
}
