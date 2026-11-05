export interface Heading {
  level?: string;
  text?: string;
}

export interface PageImage {
  url?: string;
  alt?: string;
  dhash?: string;
  embedding?: number[];
  label_embedding?: number[];
}

export interface PageDocument {
  url: string;
  text: string;
  canonical_url?: string | null;
  title?: string;
  description?: string;
  headings?: Heading[];
  images?: PageImage[];
  text_embedding?: number[];
  content_sha256?: string;
  crawled_at?: string;
}

export interface Place {
  id: string;
  name_en: string;
  name_ur: string;
  aliases: string[];
  lat: number | null;
  lng: number | null;
  district: string;
  category: string;
  source_url: string;
  verified: boolean;
  note: string;
}

export interface Source {
  title: string;
  url: string;
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

export interface LocationCandidate {
  name: string;
  district?: string;
  city?: string;
  country?: string;
  state?: string;
  latitude?: number;
  longitude?: number;
  confidence?: number;
  precision?: string;
  source_url?: string;
}

export interface LocationResult {
  status: "found" | "ambiguous" | "not_found" | "error";
  precision: string;
  message?: string;
  latitude?: number;
  longitude?: number;
  label?: string;
  city?: string;
  district?: string;
  country?: string;
  source_url?: string;
  candidates?: LocationCandidate[];
  sources?: Source[];
}

export interface ImageResult {
  title: string;
  place_name: string;
  page_url: string;
  image_url: string;
  alt: string;
  confidence?: number;
}

export interface ChatResponse {
  answer: string;
  sources: Source[];
  maps?: Source[];
  location?: LocationResult;
  images?: ImageResult[];
  answer_origin?: string;
  source_note?: string;
  external_search_url?: string;
}

export interface PublicPlace {
  id: string;
  name_en: string;
  name_ur: string;
  district: string;
  category: string;
  source_url: string;
  verified: boolean;
  latitude: number | null;
  longitude: number | null;
}
