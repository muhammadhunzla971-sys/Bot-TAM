"use client";

import { useEffect, useRef, useState } from "react";

interface LeafletMapProps {
  latitude: number;
  longitude: number;
  zoom: number;
  className?: string;
  id?: string;
  label: string;
  /** Popup text for a marker; omit to render the map without a marker. */
  markerLabel?: string;
  /** Message shown when the Leaflet library itself fails to load. */
  unavailableMessage?: string;
}

const TILE_ERROR =
  "Map tiles are unavailable. Check your internet connection; place details and links still work.";

export default function LeafletMap({
  latitude,
  longitude,
  zoom,
  className,
  id,
  label,
  markerLabel,
  unavailableMessage = "Map library or tiles could not be loaded; place details and links still work.",
}: LeafletMapProps) {
  const container = useRef<HTMLDivElement>(null);
  const [status, setStatus] = useState("");

  useEffect(() => {
    let disposed = false;
    let map: import("leaflet").Map | null = null;
    let resizeObserver: ResizeObserver | null = null;
    import("leaflet")
      .then(({ default: L }) => {
        if (disposed || !container.current) return;
        L.Icon.Default.mergeOptions({
          iconUrl: "/leaflet/marker-icon.png",
          iconRetinaUrl: "/leaflet/marker-icon-2x.png",
          shadowUrl: "/leaflet/marker-shadow.png",
        });
        map = L.map(container.current, { scrollWheelZoom: false, zoomControl: true }).setView(
          [latitude, longitude],
          zoom,
        );
        let tileError = false;
        const tiles = L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
          attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
          maxZoom: 19,
        });
        tiles.on("loading", () => {
          tileError = false;
        });
        tiles.on("tileerror", () => {
          tileError = true;
          if (!disposed) setStatus(TILE_ERROR);
        });
        tiles.on("load", () => {
          if (!tileError && !disposed) setStatus("");
        });
        tiles.addTo(map);
        // Keep tiles aligned when the card or layout resizes.
        resizeObserver = new ResizeObserver(() => map?.invalidateSize());
        resizeObserver.observe(container.current);
        if (markerLabel) {
          const popup = document.createElement("span");
          popup.textContent = markerLabel;
          L.marker([latitude, longitude]).addTo(map).bindPopup(popup);
        }
      })
      .catch(() => {
        if (!disposed) setStatus(unavailableMessage);
      });
    return () => {
      disposed = true;
      resizeObserver?.disconnect();
      map?.remove();
    };
  }, [latitude, longitude, zoom, markerLabel, unavailableMessage]);

  return (
    <>
      <div ref={container} id={id} className={className} aria-label={label} />
      <span className="map-status" role="status" aria-live="polite" hidden={!status}>
        {status}
      </span>
    </>
  );
}
