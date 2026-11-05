export interface Coordinates {
  latitude: number;
  longitude: number;
}

// Browser geolocation stays in page memory only; it is never sent to the server.
let browserCoordinates: Coordinates | null = null;

export function cachedBrowserCoordinates(): Coordinates | null {
  return browserCoordinates;
}

export function distanceKm(first: Coordinates, second: Coordinates): number {
  const radians = (degrees: number) => (degrees * Math.PI) / 180;
  const latitudeDelta = radians(second.latitude - first.latitude);
  const longitudeDelta = radians(second.longitude - first.longitude);
  const value =
    Math.sin(latitudeDelta / 2) ** 2 +
    Math.cos(radians(first.latitude)) * Math.cos(radians(second.latitude)) * Math.sin(longitudeDelta / 2) ** 2;
  return 6371 * 2 * Math.atan2(Math.sqrt(value), Math.sqrt(1 - value));
}

function geolocationErrorMessage(error: GeolocationPositionError): string {
  if (error.code === error.PERMISSION_DENIED) {
    return "Location permission was denied. You can still browse places without sharing your location.";
  }
  if (error.code === error.TIMEOUT) return "Your location request timed out. Please try again.";
  return "Your location is unavailable on this device right now.";
}

export function requestBrowserLocation(
  onSuccess: (coordinates: Coordinates) => void,
  onFailure: (message: string) => void,
): void {
  if (!navigator.geolocation) {
    onFailure("This browser does not support location access.");
    return;
  }
  navigator.geolocation.getCurrentPosition(
    (position) => {
      browserCoordinates = { latitude: position.coords.latitude, longitude: position.coords.longitude };
      onSuccess(browserCoordinates);
    },
    (error) => onFailure(geolocationErrorMessage(error)),
    { enableHighAccuracy: false, maximumAge: 60000, timeout: 12000 },
  );
}

export function googleMapsDirectionsUrl(destination: string, origin: Coordinates | null = null): string {
  const url = new URL("https://www.google.com/maps/dir/");
  url.searchParams.set("api", "1");
  url.searchParams.set("destination", destination);
  if (origin) url.searchParams.set("origin", `${origin.latitude},${origin.longitude}`);
  return url.toString();
}

export function googleMapsSearchUrl(query: string): string {
  const url = new URL("https://www.google.com/maps/search/");
  url.searchParams.set("api", "1");
  url.searchParams.set("query", query);
  return url.toString();
}
