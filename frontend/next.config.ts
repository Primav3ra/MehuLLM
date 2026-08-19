import type { NextConfig } from "next";

const config: NextConfig = {
  env: { NEXT_PUBLIC_API: process.env.NEXT_PUBLIC_API ?? "http://127.0.0.1:8010" },
};
export default config;
