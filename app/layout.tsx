import type { Metadata, Viewport } from "next";
import { Fraunces, Inter, Noto_Naskh_Arabic } from "next/font/google";
import "leaflet/dist/leaflet.css";
import "./globals.css";

const sans = Inter({ subsets: ["latin"], variable: "--font-sans", display: "swap" });
const display = Fraunces({ subsets: ["latin"], variable: "--font-display", display: "swap" });
const urdu = Noto_Naskh_Arabic({
  subsets: ["arabic"],
  weight: ["400", "600"],
  variable: "--font-urdu",
  display: "swap",
});

export const metadata: Metadata = {
  title: "Punjab Tourism Assistant",
  description:
    "Discover places in Punjab, plan a trip, and get answers grounded in official Punjab Tourism website pages.",
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  viewportFit: "cover",
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#0e3b2b" },
    { media: "(prefers-color-scheme: dark)", color: "#0a1a14" },
  ],
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${sans.variable} ${display.variable} ${urdu.variable}`}>
      <body>{children}</body>
    </html>
  );
}
