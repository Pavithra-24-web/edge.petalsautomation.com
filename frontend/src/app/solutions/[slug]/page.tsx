import type { Metadata } from "next";
import { notFound } from "next/navigation";
import SolutionPage from "@/components/marketing/solutions/SolutionPage";
import {
  SOLUTION_SLUGS,
  SITE_NAME,
  SITE_URL,
  getSolution,
  solutionHref,
} from "@/lib/solutions.config";

/* Static export (next.config.js: output "export") — every solution route is
   pre-rendered from the config, so adding a solution there adds a page here. */
export function generateStaticParams() {
  return SOLUTION_SLUGS.map((slug) => ({ slug }));
}

/* No `dynamicParams = false` here, deliberately — do not re-add it.

   Under `output: "export"` it buys nothing: the export emits HTML only for the
   slugs above, so an unknown slug has no file and the host serves 404. The
   notFound() guard below covers it in-app.

   It also breaks `next dev` outright on Next 14.2.3. buildAppStaticPaths derives
   `fallback = !generateParams.some(g => g.config?.dynamicParams === false)`, the
   dev server maps anything other than `true` to a non-"static" fallbackMode, and
   base-server then throws — reporting this page as "missing exported function
   generateStaticParams()" even though it is right here and was just called to
   produce that fallback value. Every /solutions/* route 500s in dev as a result,
   while `next build` (which takes the export worker path) passes. */

/* Metadata strings live in solutions.config.ts, not here, so every word a page
   says is authored in one file. OG image is attached only when the solution
   actually has a hero asset — a link preview pointing at a missing file is
   worse than no image tag at all. */
export function generateMetadata({ params }: { params: { slug: string } }): Metadata {
  const solution = getSolution(params.slug);
  if (!solution) return {};

  const { title, description } = solution.seo;
  const url = `${SITE_URL}${solutionHref(solution.slug)}`;
  const hero = solution.media?.hero;

  return {
    title,
    description,
    alternates: { canonical: url },
    openGraph: {
      type: "website",
      siteName: SITE_NAME,
      title,
      description,
      url,
      ...(hero ? { images: [{ url: `${SITE_URL}${hero.src}`, alt: hero.alt }] } : {}),
    },
  };
}

export default function Page({ params }: { params: { slug: string } }) {
  if (!getSolution(params.slug)) notFound();
  return <SolutionPage slug={params.slug} />;
}
