// Returns a JSON 404 for OAuth discovery probes (e.g.
// `/.well-known/oauth-protected-resource`, `/.well-known/oauth-authorization-server`).
// Per RFC 9728 / the MCP authorization spec a 404 here means "this resource
// doesn't require OAuth — connect anonymously." Without this handler the
// Docusaurus catch-all serves an HTML 404, which trips MCP clients that try to
// JSON-parse the body to extract an OAuth error.
import { VercelRequest, VercelResponse } from "@vercel/node";

export default function handler(_req: VercelRequest, res: VercelResponse) {
  res.setHeader("Content-Type", "application/json");
  res.setHeader("Cache-Control", "public, max-age=3600");
  return res.status(404).end("{}");
}
