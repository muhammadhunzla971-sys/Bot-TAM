import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // The API routes read the corpus and place registry from disk at runtime.
  // Include them in every serverless function bundle, and keep the large
  // local ONNX models (used only by the offline Python tools) out of it.
  outputFileTracingIncludes: {
    "/api/**": ["./data/tourism_pages.jsonl", "./data/places.json"],
  },
  outputFileTracingExcludes: {
    "*": ["./data/models/**", "./**/*.py", "./tests/**", "./docs/**"],
  },
  poweredByHeader: false,
  async headers() {
    return [
      {
        source: "/:path*",
        headers: [
          { key: "X-Content-Type-Options", value: "nosniff" },
          { key: "Referrer-Policy", value: "strict-origin-when-cross-origin" },
        ],
      },
      {
        source: "/api/:path*",
        headers: [{ key: "Cache-Control", value: "no-store" }],
      },
    ];
  },
};

export default nextConfig;
