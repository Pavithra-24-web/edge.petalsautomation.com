import type { Metadata } from "next";
import SolutionsHub from "@/components/marketing/solutions/SolutionsHub";
import { HUB, SITE_NAME, SITE_URL } from "@/lib/solutions.config";

/* /solutions — hub page. Server component so the metadata below can be static
   while the shell underneath stays a client component. */
export const metadata: Metadata = {
  title: HUB.seo.title,
  description: HUB.seo.description,
  alternates: { canonical: `${SITE_URL}/solutions/` },
  openGraph: {
    type: "website",
    siteName: SITE_NAME,
    title: HUB.seo.title,
    description: HUB.seo.description,
    url: `${SITE_URL}/solutions/`,
  },
};

export default function SolutionsPage() {
  return <SolutionsHub />;
}
