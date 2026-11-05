"use client";

import { useEffect, useMemo, useState } from "react";

import { Icon, type IconName } from "./Icons";
import LeafletMap from "./LeafletMap";
import {
  cachedBrowserCoordinates,
  distanceKm,
  googleMapsDirectionsUrl,
  googleMapsSearchUrl,
  requestBrowserLocation,
  type Coordinates,
} from "./geo";
import type { ChatResponse, ImageResult, LocationCandidate, LocationResult, Source } from "@/lib/types";

export interface UiMessage {
  id: number;
  role: "user" | "assistant";
  text: string;
  sources?: Source[];
  maps?: Source[];
  location?: LocationResult | null;
  metadata?: Pick<ChatResponse, "answer_origin" | "source_note" | "external_search_url"> | null;
  images?: ImageResult[];
  previewFile?: File;
  error?: boolean;
  /** Question to resend when the visitor presses "Try again". */
  retry?: string;
}

function hostOf(url: string): string {
  try {
    return new URL(url).hostname.replace(/^www\./, "");
  } catch {
    return "";
  }
}

function ExternalLink({ href, children, className }: { href: string; children: React.ReactNode; className?: string }) {
  return (
    <a href={href} target="_blank" rel="noopener noreferrer" className={className}>
      {children}
    </a>
  );
}

/* ---------- Provenance ---------- */

type Tone = "success" | "warn" | "info" | "neutral";

function describeOrigin(origin: string): { tone: Tone; icon: IconName; label: string } {
  switch (origin) {
    case "Punjab Tourism website":
      return { tone: "success", icon: "shield", label: "From official Punjab Tourism pages" };
    case "external general knowledge":
      return { tone: "warn", icon: "alert", label: "Unverified answer (no cited source)" };
    case "grounded local excerpt":
    case "local fallback":
      return { tone: "neutral", icon: "book", label: "Excerpt from tourism pages" };
    case "no matching Tourism website evidence":
      return { tone: "neutral", icon: "info", label: "Not found in tourism pages" };
    default:
      return { tone: "info", icon: "globe", label: `Answer source: ${origin}` };
  }
}

function Provenance({ metadata }: { metadata: NonNullable<UiMessage["metadata"]> }) {
  if (!metadata.answer_origin && !metadata.source_note) return null;
  const { tone, icon, label } = describeOrigin(metadata.answer_origin ?? "");
  return (
    <div className="provenance">
      {metadata.answer_origin && (
        <span className={`badge badge-${tone}`}>
          <Icon name={icon} size={13} />
          {label}
        </span>
      )}
      {metadata.source_note && <p className="provenance-note">{metadata.source_note}</p>}
    </div>
  );
}

function SourceChips({ sources, maps, searchUrl }: { sources: Source[]; maps: Source[]; searchUrl?: string }) {
  if (!sources.length && !maps.length && !searchUrl) return null;
  return (
    <nav className="chips" aria-label="Sources">
      {sources.map((source, index) => (
        <ExternalLink key={`s${index}`} href={source.url} className="chip">
          <span className="chip-index">{index + 1}</span>
          <span className="chip-text">{source.title || source.url}</span>
          <span className="chip-host">{hostOf(source.url)}</span>
        </ExternalLink>
      ))}
      {maps.map((map, index) => (
        <ExternalLink key={`m${index}`} href={map.url} className="chip">
          <span className="chip-index">
            <Icon name="pin" size={13} />
          </span>
          <span className="chip-text">{map.title || "Open location in Google Maps"}</span>
        </ExternalLink>
      ))}
      {searchUrl && (
        <ExternalLink href={searchUrl} className="chip">
          <span className="chip-index">
            <Icon name="search" size={13} />
          </span>
          <span className="chip-text">Search independent sources (not checked by assistant)</span>
        </ExternalLink>
      )}
    </nav>
  );
}

/* ---------- Location ---------- */

function DistanceControl({ destination }: { destination: Coordinates }) {
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState("");
  const [directions, setDirections] = useState<string | null>(null);

  const showDistance = (origin: Coordinates) => {
    const distance = distanceKm(origin, destination);
    setStatus(`About ${distance.toFixed(1)} km from this device (straight line), calculated in this browser.`);
    setDirections(googleMapsDirectionsUrl(`${destination.latitude},${destination.longitude}`, origin));
  };

  const onClick = () => {
    setBusy(true);
    const cached = cachedBrowserCoordinates();
    if (cached) {
      showDistance(cached);
      setBusy(false);
      return;
    }
    setStatus("Requesting location permission...");
    requestBrowserLocation(
      (origin) => {
        showDistance(origin);
        setBusy(false);
      },
      (message) => {
        setStatus(message);
        setBusy(false);
      },
    );
  };

  return (
    <>
      <div className="loc-actions">
        <button type="button" className="btn btn-secondary btn-sm" disabled={busy} onClick={onClick}>
          <Icon name="locate" size={15} />
          Distance from me
        </button>
        {directions && (
          <ExternalLink href={directions} className="btn btn-primary btn-sm">
            <Icon name="navigation" size={15} />
            Directions in Google Maps
          </ExternalLink>
        )}
      </div>
      <span className="status" role="status">
        {status}
      </span>
    </>
  );
}

function CopyCoordinates({ latitude, longitude }: { latitude: number; longitude: number }) {
  const [state, setState] = useState<"idle" | "copied" | "failed">("idle");
  const onCopy = async () => {
    try {
      await navigator.clipboard.writeText(`${latitude}, ${longitude}`);
      setState("copied");
      setTimeout(() => setState("idle"), 2000);
    } catch {
      setState("failed");
    }
  };
  return (
    <button type="button" className="btn btn-ghost btn-sm" onClick={onCopy}>
      <Icon name={state === "copied" ? "check" : "copy"} size={14} />
      {state === "copied" ? "Copied" : state === "failed" ? "Copy unavailable" : "Copy coordinates"}
    </button>
  );
}

function CandidateChoice({ candidate, onAsk }: { candidate: LocationCandidate; onAsk: (question: string) => void }) {
  const [shown, setShown] = useState(false);
  const title = [...new Set([candidate.name, candidate.city, candidate.district, candidate.country].filter(Boolean))].join(
    ", ",
  );
  const hasCoordinates = Number.isFinite(candidate.latitude) && Number.isFinite(candidate.longitude);
  const precision = candidate.precision || "approximate";
  const officialSource =
    typeof candidate.source_url === "string" && candidate.source_url.startsWith("https://tourism.punjab.gov.pk/")
      ? candidate.source_url
      : null;

  return (
    <div className="candidate">
      <div className="candidate-row">
        <strong>{title}</strong>
        <span className={`badge ${precision === "verified" ? "badge-success" : "badge-neutral"}`}>{precision}</span>
      </div>
      <div className="loc-actions">
        {hasCoordinates ? (
          <>
            <button type="button" className="btn btn-secondary btn-sm" disabled={shown} onClick={() => setShown(true)}>
              <Icon name="pin" size={14} />
              {`Show ${candidate.precision === "verified" ? "verified" : "approximate"} map`}
            </button>
            <ExternalLink
              href={googleMapsSearchUrl(`${candidate.latitude},${candidate.longitude}`)}
              className="btn btn-ghost btn-sm"
            >
              <Icon name="external" size={14} />
              Google Maps
            </ExternalLink>
          </>
        ) : (
          <button
            type="button"
            className="btn btn-secondary btn-sm"
            onClick={() => onAsk(`Where is ${[candidate.name, candidate.district].filter(Boolean).join(", ")}?`)}
          >
            <Icon name="search" size={14} />
            Search with its district
          </button>
        )}
        {officialSource && (
          <ExternalLink href={officialSource} className="btn btn-ghost btn-sm">
            <Icon name="book" size={14} />
            Official source
          </ExternalLink>
        )}
      </div>
      {shown && hasCoordinates && (
        <>
          <span className="coords">{`${candidate.latitude}, ${candidate.longitude} (${precision})`}</span>
          <LeafletMap
            className="loc-map"
            label={`Map showing ${title}`}
            latitude={candidate.latitude as number}
            longitude={candidate.longitude as number}
            zoom={candidate.precision === "verified" ? 16 : 15}
            markerLabel={`${title} (${precision})`}
            unavailableMessage="Map library or tiles could not be loaded; use the Google Maps link."
          />
        </>
      )}
    </div>
  );
}

function LocationCard({ location, onAsk }: { location: LocationResult; onAsk: (question: string) => void }) {
  const candidates = Array.isArray(location.candidates) ? location.candidates : [];
  const photonCandidates =
    location.status === "ambiguous" && candidates.some((candidate) => Number.isFinite(candidate.confidence));
  const found =
    location.status === "found" && Number.isFinite(location.latitude) && Number.isFinite(location.longitude);
  const name = location.label || "OpenStreetMap location";
  const area = [location.city, location.district, location.country].filter(Boolean).join(", ");
  const message = photonCandidates
    ? "Several places match. Select one to preview its approximate location (OpenStreetMap results, outside the Punjab Tourism website)."
    : location.message || `Location precision: ${location.precision || "search only"}.`;

  if (found) {
    const latitude = location.latitude as number;
    const longitude = location.longitude as number;
    return (
      <section className="card loc-card" aria-label={`Location: ${name}`}>
        <div className="loc-head">
          <span className="loc-icon">
            <Icon name="pin" size={18} />
          </span>
          <div className="loc-title">
            <strong>{name}</strong>
            {area && <span>{area}</span>}
          </div>
          <span className={`badge ${location.precision === "verified" ? "badge-success" : "badge-neutral"}`}>
            {location.precision}
          </span>
        </div>
        <LeafletMap
          className="loc-map"
          label={`Map showing ${name}`}
          latitude={latitude}
          longitude={longitude}
          zoom={16}
          markerLabel={`${name} (${location.precision})`}
        />
        <div className="loc-body">
          <div className="candidate-row">
            <span className="coords">{`${latitude}, ${longitude}`}</span>
            <CopyCoordinates latitude={latitude} longitude={longitude} />
          </div>
          {location.message && <span className="status">{location.message}</span>}
          <DistanceControl destination={{ latitude, longitude }} />
        </div>
      </section>
    );
  }

  return (
    <section className="card loc-card" aria-label="Location results">
      <div className="loc-head">
        <span className="loc-icon">
          <Icon name="pin" size={18} />
        </span>
        <div className="loc-title">
          <strong>{location.status === "ambiguous" ? "Which place did you mean?" : "Location"}</strong>
          <span>{message}</span>
        </div>
      </div>
      {location.status === "ambiguous" && candidates.length > 0 && (
        <div className="candidates">
          {candidates.map((candidate, index) => (
            <CandidateChoice key={index} candidate={candidate} onAsk={onAsk} />
          ))}
        </div>
      )}
    </section>
  );
}

/* ---------- Images ---------- */

function ImageMatchCard({ match }: { match: ImageResult }) {
  const [confirmed, setConfirmed] = useState(false);
  const label = match.place_name || match.title;
  return (
    <article className="card image-card">
      <div className="image-frame">
        {/* Remote tourism-site images are shown as-is (no Next.js image optimisation). */}
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img src={match.image_url} alt={match.alt || match.title} loading="lazy" />
        {typeof match.confidence === "number" && <span className="badge">{`${match.confidence}% match`}</span>}
      </div>
      <div className="image-card-body">
        <strong>{label}</strong>
        {typeof match.confidence === "number" && (
          <div
            className="meter"
            role="meter"
            aria-label="Visual similarity"
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={match.confidence}
          >
            <span style={{ width: `${match.confidence}%` }} />
          </div>
        )}
        <div className="link-row">
          <ExternalLink href={match.page_url}>
            Source page <Icon name="external" size={12} />
          </ExternalLink>
          <ExternalLink href={match.image_url}>
            Full image <Icon name="external" size={12} />
          </ExternalLink>
        </div>
        {confirmed ? (
          <ExternalLink href={googleMapsSearchUrl(label)} className="btn btn-secondary btn-sm">
            <Icon name="pin" size={14} />
            Open unverified map search
          </ExternalLink>
        ) : (
          <button type="button" className="btn btn-ghost btn-sm" onClick={() => setConfirmed(true)}>
            <Icon name="check" size={14} />
            Confirm landmark &amp; show map
          </button>
        )}
      </div>
    </article>
  );
}

function ImagePreview({ file }: { file: File }) {
  const url = useMemo(() => URL.createObjectURL(file), [file]);
  useEffect(() => () => URL.revokeObjectURL(url), [url]);
  // eslint-disable-next-line @next/next/no-img-element
  return <img className="image-preview" src={url} alt="Image selected for matching" />;
}

/* ---------- Message ---------- */

function CopyAnswer({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <button
      type="button"
      className="btn btn-ghost btn-sm"
      onClick={async () => {
        try {
          await navigator.clipboard.writeText(text);
          setCopied(true);
          setTimeout(() => setCopied(false), 2000);
        } catch {
          // Clipboard unavailable; nothing else to do.
        }
      }}
    >
      <Icon name={copied ? "check" : "copy"} size={14} />
      {copied ? "Copied" : "Copy"}
    </button>
  );
}

export default function Message({
  message,
  onAsk,
  busy,
}: {
  message: UiMessage;
  onAsk: (question: string) => void;
  busy: boolean;
}) {
  if (message.role === "user") {
    return (
      <div className="msg msg-user" data-message-id={message.id}>
        <div className="msg-body">
          <div className="bubble" dir="auto">
            {message.text}
            {message.previewFile && <ImagePreview file={message.previewFile} />}
          </div>
        </div>
      </div>
    );
  }

  const metadata = message.metadata;
  return (
    <div className="msg msg-assistant" data-message-id={message.id}>
      <span className="avatar" aria-hidden="true">
        PT
      </span>
      <div className="msg-body">
        {message.error ? (
          <div className="answer-error" role="alert">
            <Icon name="alert" size={18} />
            <span>{message.text}</span>
          </div>
        ) : (
          <div className="answer" dir="auto">
            {message.text}
          </div>
        )}
        {metadata && <Provenance metadata={metadata} />}
        <SourceChips
          sources={message.sources ?? []}
          maps={message.maps ?? []}
          searchUrl={metadata?.external_search_url}
        />
        {message.location && <LocationCard location={message.location} onAsk={onAsk} />}
        {message.images && message.images.length > 0 && (
          <div className="image-grid">
            {message.images.map((match) => (
              <ImageMatchCard key={match.image_url} match={match} />
            ))}
          </div>
        )}
        <div className="msg-actions">
          {message.error && message.retry ? (
            <button
              type="button"
              className="btn btn-ghost btn-sm"
              disabled={busy}
              onClick={() => onAsk(message.retry as string)}
            >
              <Icon name="refresh" size={14} />
              Try again
            </button>
          ) : (
            !message.error && <CopyAnswer text={message.text} />
          )}
        </div>
      </div>
    </div>
  );
}
