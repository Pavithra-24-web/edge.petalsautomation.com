/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'export',
  trailingSlash: true,
  /* Default is unchanged. Set NEXT_DIST_DIR=.next-build to run `next build`
     without wiping the .next a running `next dev` is serving from — two writers
     in one build dir means 404 hot-updates and phantom chunk errors in the
     browser that already has the app open.

     Note when you do: under `output: "export"` Next only writes the exported
     site to out/ when distDir is the default .next. With any other distDir the
     export lands in that directory instead, so finish an isolated build with
     `mv .next-build out`. */
  distDir: process.env.NEXT_DIST_DIR || '.next',
  reactStrictMode: true,
  env: {
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL || "http://localhost:8010",
    NEXT_PUBLIC_WS_URL: process.env.NEXT_PUBLIC_WS_URL || "http://localhost:8010",
    NEXT_PUBLIC_GOOGLE_CLIENT_ID: process.env.NEXT_PUBLIC_GOOGLE_CLIENT_ID || "",
  },
};

module.exports = nextConfig;
