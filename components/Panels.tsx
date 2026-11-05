"use client";

import { useMemo, useState } from "react";

import { Icon } from "./Icons";
import LeafletMap from "./LeafletMap";
import { distanceKm, googleMapsDirectionsUrl, requestBrowserLocation, type Coordinates } from "./geo";
import type { PublicPlace } from "@/lib/types";

/* ==========================================================================
   Nearby places
   ========================================================================== */

type NearbyPlace = PublicPlace & { latitude: number; longitude: number; aliases?: string[] };

function normalizePlaceName(value: unknown): string {
  return String(value || "")
    .normalize("NFKC")
    .toLocaleLowerCase()
    .replace(/[^\p{L}\p{N}]+/gu, " ")
    .trim();
}

function isPunjabCoordinate(latitude: unknown, longitude: unknown): boolean {
  return (
    typeof latitude === "number" &&
    typeof longitude === "number" &&
    Number.isFinite(latitude) &&
    Number.isFinite(longitude) &&
    latitude >= 27.7 &&
    latitude <= 34 &&
    longitude >= 69.3 &&
    longitude <= 75.5
  );
}

/** Approximate a registry place's coordinates in the browser with Photon. */
async function geocodeNearbyPlace(place: PublicPlace & { aliases?: string[] }): Promise<NearbyPlace | null> {
  const url = new URL("https://photon.komoot.io/api/");
  url.searchParams.set("q", [place.name_en, place.district, "Punjab, Pakistan"].filter(Boolean).join(", "));
  url.searchParams.set("limit", "5");
  url.searchParams.set("lang", "en");
  url.searchParams.set("bbox", "69.3,27.7,75.5,34.0");
  const response = await fetch(url, { signal: AbortSignal.timeout(8000) });
  if (!response.ok) throw new Error("OpenStreetMap lookup failed.");
  const result = await response.json();
  const expectedNames = [place.name_en, place.name_ur, ...(place.aliases || [])].map(normalizePlaceName);
  const features: unknown[] = Array.isArray(result?.features) ? result.features : [];
  const match = features
    .map((feature) => {
      const value = feature as { geometry?: { coordinates?: number[] }; properties?: Record<string, unknown> };
      const coordinates = value?.geometry?.coordinates;
      const properties = value?.properties || {};
      return {
        latitude: coordinates?.[1],
        longitude: coordinates?.[0],
        candidateName: normalizePlaceName(properties.name_en || properties.name || properties.localname),
        country: typeof properties.countrycode === "string" ? properties.countrycode : undefined,
      };
    })
    .find(
      (candidate) =>
        candidate.country?.toLowerCase() === "pk" &&
        expectedNames.includes(candidate.candidateName) &&
        isPunjabCoordinate(candidate.latitude, candidate.longitude),
    );
  return match
    ? { ...place, latitude: match.latitude as number, longitude: match.longitude as number, verified: false }
    : null;
}

async function geocodeNearbyPlaces(places: PublicPlace[]) {
  const found: NearbyPlace[] = [];
  let failed = 0;
  let nextIndex = 0;
  const worker = async () => {
    while (nextIndex < places.length) {
      const place = places[nextIndex++];
      try {
        const match = await geocodeNearbyPlace(place);
        if (match) found.push(match);
      } catch {
        failed += 1;
      }
    }
  };
  await Promise.all(Array.from({ length: Math.min(3, places.length) }, () => worker()));
  return { places: found, failed };
}

interface NearbyResult {
  place: NearbyPlace;
  distance: number;
  origin: Coordinates;
  approximate: boolean;
}

export function NearbyView({ places }: { places: PublicPlace[] }) {
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState("");
  const [results, setResults] = useState<NearbyResult[]>([]);

  const render = (origin: Coordinates, candidates: NearbyPlace[], approximate = false, failed = 0) => {
    setResults([]);
    if (!candidates.length) {
      setStatus(
        failed
          ? `OpenStreetMap lookup failed for ${failed} place${failed === 1 ? "" : "s"}; no nearby results are available.`
          : "OpenStreetMap found no matching Punjab place coordinates.",
      );
      return;
    }
    const nearest = candidates
      .map((place) => ({ place, distance: distanceKm(origin, place), origin, approximate }))
      .sort((first, second) => first.distance - second.distance)
      .slice(0, 5);
    setStatus(
      approximate
        ? `Approximate straight-line distances use OpenStreetMap Photon results, not verified site coordinates.${failed ? ` Lookup failed for ${failed} other place${failed === 1 ? "" : "s"}.` : ""}`
        : "Straight-line estimates calculated in this browser. Opening directions shares your location with Google Maps.",
    );
    setResults(nearest);
  };

  const onFind = () => {
    const verifiedPlaces = places.filter(
      (place): place is NearbyPlace =>
        place.verified === true && Number.isFinite(place.latitude) && Number.isFinite(place.longitude),
    );
    setStatus("Requesting location permission...");
    setBusy(true);
    requestBrowserLocation(
      async (origin) => {
        try {
          if (verifiedPlaces.length) {
            render(origin, verifiedPlaces);
          } else {
            setStatus("Looking up approximate place locations with OpenStreetMap...");
            const result = await geocodeNearbyPlaces(places);
            render(origin, result.places, true, result.failed);
          }
        } catch (error) {
          setStatus((error as Error).message || "Nearby places could not be loaded.");
        } finally {
          setBusy(false);
        }
      },
      (message) => {
        setStatus(message);
        setBusy(false);
      },
    );
  };

  return (
    <div className="page">
      <header className="page-head">
        <h2>Tourism places near you</h2>
        <p>See which registered Punjab tourism places are closest, with straight-line distances and directions.</p>
      </header>
      <section className="card hero-card">
        <span className="hero-card-icon">
          <Icon name="locate" size={26} />
        </span>
        <div className="hero-card-copy">
          <h3>Your location stays on this device</h3>
          <p>
            Distances are calculated in your browser. Place coordinates are looked up approximately with OpenStreetMap,
            and your location is shared with Google Maps only if you open directions.
          </p>
        </div>
        <button type="button" className="btn btn-primary btn-lg" disabled={busy} onClick={onFind}>
          <Icon name="compass" size={18} />
          {busy ? "Finding places..." : "Find nearby places"}
        </button>
      </section>
      <p className="status-line" role="status" aria-live="polite">
        {status}
      </p>
      {results.length > 0 && (
        <ol className="nearby-list" style={{ listStyle: "none", padding: 0 }}>
          {results.map(({ place, distance, origin, approximate }, index) => {
            const osmUrl = new URL("https://www.openstreetmap.org/");
            osmUrl.searchParams.set("mlat", String(place.latitude));
            osmUrl.searchParams.set("mlon", String(place.longitude));
            osmUrl.hash = `map=17/${place.latitude}/${place.longitude}`;
            return (
              <li className="card nearby-item" key={place.id}>
                <span className="nearby-rank">{index + 1}</span>
                <div className="nearby-info">
                  <strong>{place.name_en}</strong>
                  <span>{[place.district, place.category].filter(Boolean).join(" · ")}</span>
                  <div className="link-row" style={{ marginTop: 6 }}>
                    <a href={osmUrl.toString()} target="_blank" rel="noopener noreferrer">
                      {approximate ? "Approximate map location" : "Map location"} <Icon name="external" size={12} />
                    </a>
                    <a
                      href={googleMapsDirectionsUrl(`${place.latitude},${place.longitude}`, origin)}
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      Directions <Icon name="external" size={12} />
                    </a>
                  </div>
                </div>
                <div className="distance">
                  <b>{`${distance.toFixed(1)} km`}</b>
                  <span>straight line</span>
                </div>
              </li>
            );
          })}
        </ol>
      )}
    </div>
  );
}

/* ==========================================================================
   Trip planner
   ========================================================================== */

interface Itinerary {
  days: Array<{
    day: number;
    items: Array<{
      place_id: string;
      place_name: string;
      timing: string;
      description: string;
      source: { title: string; url: string };
    }>;
  }>;
  note: string;
}

const MAX_PLACES = 10;

export function TripPlannerView({ places }: { places: PublicPlace[] }) {
  const [selected, setSelected] = useState<string[]>([]);
  const [days, setDays] = useState(1);
  const [filter, setFilter] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [itinerary, setItinerary] = useState<Itinerary | null>(null);

  const visible = useMemo(() => {
    const query = filter.trim().toLowerCase();
    if (!query) return places;
    return places.filter((place) =>
      [place.name_en, place.name_ur, place.district, place.category].some((value) =>
        String(value || "").toLowerCase().includes(query),
      ),
    );
  }, [filter, places]);

  const byId = useMemo(() => new Map(places.map((place) => [place.id, place])), [places]);

  // Keep the selection in registry order (matches the original form order).
  const toggle = (id: string, checked: boolean) =>
    setSelected((current) =>
      checked
        ? places.map((place) => place.id).filter((placeId) => placeId === id || current.includes(placeId))
        : current.filter((placeId) => placeId !== id),
    );

  const onSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    setError("");
    setItinerary(null);
    setBusy(true);
    try {
      const response = await fetch("/api/itinerary", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ days, place_ids: selected }),
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.error || "The itinerary could not be created.");
      setItinerary(result);
      requestAnimationFrame(() =>
        document.getElementById("itinerary-result")?.scrollIntoView({ behavior: "smooth", block: "start" }),
      );
    } catch (caught) {
      setError((caught as Error).message || "The itinerary could not be created.");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="page">
      <header className="page-head">
        <h2>Plan a Punjab trip</h2>
        <p>
          Pick the places you want to see and how many days you have. Each stop is described from the official tourism
          pages; time slots are suggestions, not verified opening hours.
        </p>
      </header>

      <form className="planner" onSubmit={onSubmit} id="itinerary-form">
        <section className="card panel" aria-labelledby="choose-places">
          <div className="panel-head">
            <h3 id="choose-places">
              <span className="step">1</span>Choose places
            </h3>
            <span className="badge badge-neutral">{`${selected.length} / ${MAX_PLACES} selected`}</span>
          </div>
          <label className="search">
            <Icon name="search" size={16} />
            <span className="sr-only">Filter places</span>
            <input
              type="search"
              placeholder="Filter by name, district or type"
              value={filter}
              onChange={(event) => setFilter(event.target.value)}
            />
          </label>
          {places.length === 0 ? (
            <p className="empty-note">Trip planning is unavailable until tourism place data loads.</p>
          ) : visible.length === 0 ? (
            <p className="empty-note">No places match “{filter}”.</p>
          ) : (
            <div className="place-grid" role="group" aria-label="Choose tourism places">
              {visible.map((place) => {
                const checked = selected.includes(place.id);
                return (
                  <label className="place-option" key={place.id}>
                    <input
                      type="checkbox"
                      value={place.id}
                      checked={checked}
                      disabled={!checked && selected.length >= MAX_PLACES}
                      onChange={(event) => toggle(place.id, event.target.checked)}
                    />
                    <span className="place-option-card">
                      <strong>{place.name_en}</strong>
                      <span className="urdu" lang="ur" dir="rtl">
                        {place.name_ur}
                      </span>
                      <span className="meta">{[place.district, place.category].filter(Boolean).join(" · ")}</span>
                    </span>
                    <span className="place-check" aria-hidden="true">
                      <Icon name="check" size={13} />
                    </span>
                  </label>
                );
              })}
            </div>
          )}
        </section>

        <aside className="card panel sticky" aria-labelledby="trip-length">
          <div className="panel-head">
            <h3 id="trip-length">
              <span className="step">2</span>Trip length
            </h3>
            <span className="badge badge-accent">{`${days} day${days === 1 ? "" : "s"}`}</span>
          </div>
          <div className="segmented" role="group" aria-label="Number of itinerary days">
            {[1, 2, 3, 4, 5, 6, 7].map((day) => (
              <button key={day} type="button" aria-pressed={days === day} onClick={() => setDays(day)}>
                {day}
              </button>
            ))}
          </div>
          {selected.length > 0 ? (
            <ul className="summary-list">
              {selected.map((id) => (
                <li key={id}>
                  <span>{byId.get(id)?.name_en}</span>
                  <button type="button" aria-label={`Remove ${byId.get(id)?.name_en}`} onClick={() => toggle(id, false)}>
                    <Icon name="x" size={14} />
                  </button>
                </li>
              ))}
            </ul>
          ) : (
            <p className="helper">Select between 1 and {MAX_PLACES} places to build an itinerary.</p>
          )}
          <button type="submit" className="btn btn-primary btn-lg btn-block" disabled={busy || !selected.length}>
            <Icon name="sparkle" size={17} />
            {busy ? "Building..." : "Build itinerary"}
          </button>
          {error && (
            <p className="form-error" role="alert">
              {error}
            </p>
          )}
          <p className="helper">Days beyond the number of places are left open for rest or travel.</p>
        </aside>
      </form>

      {itinerary && (
        <section className="timeline-wrap" id="itinerary-result" aria-label="Your itinerary">
          <header className="page-head">
            <h2>Your itinerary</h2>
          </header>
          {itinerary.days.map((day) => (
            <div className="timeline-day" key={day.day}>
              <span className="timeline-dot" aria-hidden="true" />
              <h3>{`Day ${day.day}`}</h3>
              <div className="stops">
                {!day.items.length && (
                  <p className="free-day">No selected stops — keep this day free for rest or travel.</p>
                )}
                {day.items.map((stop) => (
                  <article className="card stop" key={stop.place_id}>
                    <div className="stop-head">
                      <span className="badge badge-accent">{stop.timing}</span>
                      <strong>{stop.place_name}</strong>
                    </div>
                    <p dir="auto">{stop.description}</p>
                    <div className="link-row">
                      <a href={stop.source.url} target="_blank" rel="noopener noreferrer">
                        {`Source: ${stop.source.title}`} <Icon name="external" size={12} />
                      </a>
                    </div>
                  </article>
                ))}
              </div>
            </div>
          ))}
          <div className="callout">
            <Icon name="info" size={16} />
            <span>{itinerary.note}</span>
          </div>
        </section>
      )}
    </div>
  );
}

/* ==========================================================================
   Department office
   ========================================================================== */

const OFFICE_ADDRESS = "8th Floor, Tower C, Old P&D Building, Lahore";

export function OfficeView() {
  return (
    <div className="page">
      <header className="page-head">
        <h2>Visit the Tourism Department</h2>
        <p>Tourism, Archaeology and Museums Department, Government of the Punjab.</p>
      </header>
      <div className="office">
        <section className="card office-map">
          <LeafletMap
            id="site-map"
            label="Map showing the approximate Tourism Department location"
            latitude={31.566667}
            longitude={74.3042613}
            zoom={15}
          />
        </section>
        <aside className="card panel">
          <dl className="info-list">
            <div>
              <dt>Address</dt>
              <dd>{OFFICE_ADDRESS}</dd>
            </div>
            <div>
              <dt>Map accuracy</dt>
              <dd>
                <span className="badge badge-warn">
                  <Icon name="alert" size={13} />
                  Approximate vicinity
                </span>
                <p className="helper">The exact building pin is not verified, so no marker is shown.</p>
              </dd>
            </div>
            <div>
              <dt>Website</dt>
              <dd>
                <a href="https://tourism.punjab.gov.pk/" target="_blank" rel="noopener noreferrer">
                  tourism.punjab.gov.pk
                </a>
              </dd>
            </div>
          </dl>
          <a
            className="btn btn-primary btn-block"
            style={{ marginTop: 18 }}
            href={`https://www.google.com/maps/search/?api=1&query=${encodeURIComponent(OFFICE_ADDRESS)}`}
            target="_blank"
            rel="noopener noreferrer"
          >
            <Icon name="navigation" size={16} />
            Search this address in Google Maps
          </a>
        </aside>
      </div>
    </div>
  );
}
